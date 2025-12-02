import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datetime import datetime
import time
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import json
import mlflow.pytorch  

from mlflow import MlflowClient
from mlflow.pytorch import log_model
from config import Config
from dataset import ISICDataModule
from unet import AttentionUNet
from metrics import (
    DiceLoss, BCEDiceLoss, TverskyLoss, FocalTverskyLoss, CombinedLoss,
    dice_coefficient, iou_score, calculate_metrics, DeepSupervisionLoss 
)
from visualization import Visualizer, TestEvaluator 

# --- MetricsLogger class is no longer needed, MLflow will handle this ---

def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net model for ISIC 2018')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=None, help='Image size')
    parser.add_argument('--loss', type=str, default='bce_dice', 
                        choices=['dice', 'bce_dice', 'tversky', 'focal_tversky', 'combined'],
                        help='Loss function to use')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='Disable data augmentation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # <-- MLflow specific arguments (optional but good practice) -->
    parser.add_argument('--experiment_name', type=str, default="DS-AttentionUNet-Skin-Lesion", help="Name of the MLflow experiment")
    parser.add_argument('--run_name', type=str, default=None, help="Name for this specific MLflow run")
    
    return parser.parse_args()

def setup_config(args):
    """Update config with command line arguments"""
    config = Config()
    
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.img_size is not None:
        config.IMG_WIDTH = args.img_size
        config.IMG_HEIGHT = args.img_size
    if args.augment is not None:
        config.AUGMENTATION = args.augment
        
    # <-- This timestamped output dir is now less important for metrics, -->
    # <-- but still useful for saving local checkpoints and visualizations. -->
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config.LOCAL_RUN_DIR = os.path.join('outputs', f"run_{timestamp}")
    config.MODEL_DIR = os.path.join(config.LOCAL_RUN_DIR, 'models')
    config.LOG_DIR = os.path.join(config.LOCAL_RUN_DIR, 'logs')
    config.VISUALIZATION_DIR = os.path.join(config.LOCAL_RUN_DIR, 'visualizations')
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    
    # <-- Store MLflow info in config -->
    config.EXPERIMENT_NAME = args.experiment_name
    config.RUN_NAME = args.run_name if args.run_name else f"run_{timestamp}"
    
    return config

def get_loss_function(loss_type, config):
    """Get loss function based on name"""
    if loss_type == 'dice':
        return DiceLoss(smooth=config.DICE_SMOOTH)
    elif loss_type == 'bce_dice':
        return BCEDiceLoss(smooth=config.DICE_SMOOTH)
    elif loss_type == 'tversky':
        return TverskyLoss(alpha=0.7, beta=0.3, smooth=config.DICE_SMOOTH)
    elif loss_type == 'focal_tversky':
        return FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75, smooth=config.DICE_SMOOTH)
    elif loss_type == 'combined':
        return CombinedLoss()
    else:
        print(f"Warning: Unknown loss type {loss_type}, using BCE_Dice")
        return BCEDiceLoss(smooth=config.DICE_SMOOTH)
        

def train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, device, epoch):
    """Train model for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    batch_count = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(device).float()
            target = target.to(device).float()
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=config.MIXED_PRECISION):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            # Calculate metrics
            with torch.no_grad():
                if isinstance(output, list):
                    pred = output[0].detach()
                else:
                    pred = output.detach()
                
                pred_binary = (pred > 0.5).float()
                dice = dice_coefficient(pred_binary, target).item()
                iou = iou_score(pred_binary, target).item()
            
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            batch_count += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # <-- Log batch-level metrics to MLflow -->
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metrics({
                "batch_train_loss": loss.item(),
                "batch_train_dice": dice,
                "batch_lr": current_lr
            }, step=step)
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice, 'iou': iou})
    
    avg_loss = running_loss / batch_count
    avg_dice = running_dice / batch_count
    avg_iou = running_iou / batch_count
    
    return {'loss': avg_loss, 'dice': avg_dice, 'iou': avg_iou}


def validate(model, val_loader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    with tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]") as pbar:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data = data.to(device).float()
                target = target.to(device).float()
                
                output = model(data)
                loss = criterion(output, target)
                
                if isinstance(output, list):
                    pred = output[0]
                else:
                    pred = output
                
                metrics = calculate_metrics(pred, target)
                all_metrics.append(metrics)
                
                running_loss += loss.item()
                
                pbar.set_postfix({'loss': loss.item(), 'dice': metrics['dice']})
    
    avg_loss = running_loss / len(val_loader)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics

def save_last_checkpoint(model, optimizer, scheduler, epoch, config):
    """Save the latest checkpoint for resuming training"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, os.path.join(config.MODEL_DIR, 'last_checkpoint.pth'))

def train(config, args):
    """Train the model"""
    print(f"Using device: {config.DEVICE}")
    
    # --- MLflow Setup ---
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    # <-- Start the MLflow Run -->
    with mlflow.start_run(run_name=config.RUN_NAME) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # <-- Log all parameters -->
        mlflow.log_param("batch_size", config.BATCH_SIZE)
        mlflow.log_param("epochs", config.EPOCHS)
        mlflow.log_param("learning_rate", config.LEARNING_RATE)
        mlflow.log_param("img_size", config.IMG_WIDTH)
        mlflow.log_param("augmentation", config.AUGMENTATION)
        mlflow.log_param("loss_function", args.loss)
        mlflow.log_param("initial_filters", config.INITIAL_FILTERS)
        mlflow.log_param("dropout_rate", config.DROPOUT_RATE)
        mlflow.log_param("deep_supervision", config.DEEP_SUPERVISION)
        mlflow.log_param("mixed_precision", config.MIXED_PRECISION)
        
        # --- End MLflow Setup ---

        data_module = ISICDataModule(config)
        
        if args.debug:
            print("Debug mode: using small subset of data")
            data_module.load_paths()
            data_module.train_img_paths = data_module.train_img_paths[:100]
            data_module.train_mask_paths = data_module.train_mask_paths[:100]
            data_module.val_img_paths = data_module.val_img_paths[:20]
            data_module.val_mask_paths = data_module.val_mask_paths[:20]
            data_module.test_img_paths = data_module.test_img_paths[:20]
            data_module.test_mask_paths = data_module.test_mask_paths[:20]
        
        data_module.setup()
        
        # <-- Log dataset stats as artifacts -->
        stats_path = os.path.join(config.OUTPUT_DIR, 'dataset_stats.json')
        if os.path.exists(stats_path):
             mlflow.log_artifact(stats_path)
        mlflow.log_artifact(os.path.join(config.VISUALIZATION_DIR, 'lesion_size_distribution.png'))
        mlflow.log_artifact(os.path.join(config.VISUALIZATION_DIR, 'dataset_samples.png'))
        
        train_loader, val_loader, test_loader = data_module.get_dataloaders()
        
        model = AttentionUNet(
            in_channels=config.IMG_CHANNELS,
            out_channels=1,
            features=config.INITIAL_FILTERS,
            dropout=config.DROPOUT_RATE,
            bilinear=True,
            deep_supervision=config.DEEP_SUPERVISION
        ).to(config.DEVICE)
        
        # <-- Log model summary as text artifact -->
        model_summary = str(model)
        mlflow.log_text(model_summary, "model_summary.txt")
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_parameters", total_params)
        print(f"Total parameters: {total_params}")

        base_criterion = get_loss_function(args.loss, config)
        criterion = DeepSupervisionLoss(base_criterion)
        
        optimizer = AdamW(  
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )  # type: ignore
        
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=config.LEARNING_RATE, 
            epochs=config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000
        )

        scaler = GradScaler(enabled=config.MIXED_PRECISION)
        

        # --- Visualizer is now only for saving sample images, not plots ---
        visualizer = Visualizer(config, data_module) 
        
        best_dice = 0.0
        best_epoch = 0
        
        print(f"Starting training for {config.EPOCHS} epochs...")
        for epoch in range(config.EPOCHS):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, config.DEVICE, epoch)
            val_metrics = validate(model, val_loader, criterion, config.DEVICE, epoch)
            
            # <-- Log epoch-level metrics to MLflow -->
            mlflow.log_metrics({
                "epoch_train_loss": train_metrics['loss'],
                "epoch_train_dice": train_metrics['dice'],
                "epoch_train_iou": train_metrics['iou'],
                "epoch_val_loss": val_metrics['loss'],
                "epoch_val_dice": val_metrics['dice'],
                "epoch_val_iou": val_metrics['iou'],
                "epoch_val_precision": val_metrics['precision'],
                "epoch_val_recall": val_metrics['recall'],
                "epoch_val_f1": val_metrics['f1']
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{config.EPOCHS}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}")
            
            # Visualize predictions and log as artifact
            if epoch % config.VISUALIZE_FREQUENCY == 0:
                val_iter = iter(val_loader)
                batch_imgs, batch_masks = next(val_iter)
                batch_imgs, batch_masks = batch_imgs.to(config.DEVICE), batch_masks.to(config.DEVICE)
                
                model.eval()
                with torch.no_grad():
                    batch_preds = model(batch_imgs)
                
                # This saves images locally
                visualizer.visualize_batch(batch_imgs, batch_masks, batch_preds, epoch)
                # <-- Log the saved images to MLflow -->
                epoch_viz_dir = os.path.join(config.VISUALIZATION_DIR, 'epochs', f'epoch_{epoch}')
                if os.path.exists(epoch_viz_dir):
                    mlflow.log_artifacts(epoch_viz_dir, artifact_path=f"epoch_visualizations/epoch_{epoch}")

            
            # Check if this is the best model
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
                best_epoch = epoch
                print(f"  New best model! Dice: {best_dice:.4f}")
                
                # 1. Get the input example using the correct 'val_loader' variable
                print("Logging new best model to MLflow...")
                # --- NEW LINE ---
                batch = next(iter(val_loader))[0]
                input_example = batch.float().cpu().numpy()
                
                
                # <-- Log the best model to MLflow Model Registry -->
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="model",  # <-- Correct new parameter
                    input_example=input_example,
                    registered_model_name=config.EXPERIMENT_NAME # This registers the model
                )
                # <-- Log the best metrics -->
                mlflow.log_metrics({
                    "best_val_dice": best_dice,
                    "best_val_iou": val_metrics['iou'],
                    "best_epoch": best_epoch
                })
            
            # Save last checkpoint for resume
            save_last_checkpoint(model, optimizer, scheduler, epoch, config)
            
            # Early stopping
            if epoch - best_epoch >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break
        
        print(f"Training completed! Best validation Dice: {best_dice:.4f} at epoch {best_epoch+1}")
        
        # --- Final Test Evaluation ---
        print("\nEvaluating best model on test set...")
        
        # <-- Load the best model *from MLflow* for final eval -->
        # This ensures we're testing the exact model we registered
        best_model_uri = f"runs:/{run.info.run_id}/model"
        best_model = mlflow.pytorch.load_model(best_model_uri, map_location=config.DEVICE)
        
        test_evaluator = TestEvaluator(best_model, test_loader, config, data_module)
        test_metrics = test_evaluator.evaluate() # This saves artifacts locally
        
        # <-- Log final test metrics -->
        test_metrics_mean = {f"test_{k}": v for k, v in test_metrics.items() if not ('std' in k or 'min' in k or 'max' in k)}
        mlflow.log_metrics(test_metrics_mean)
        
        # <-- Log all test artifacts (report, plots, etc.) -->
        test_viz_dir = os.path.join(config.VISUALIZATION_DIR, 'test_evaluation')
        if os.path.exists(test_viz_dir):
            mlflow.log_artifacts(test_viz_dir, artifact_path="test_evaluation_report")
            
        # --- Save and log final config ---
        config_path = os.path.join(config.LOCAL_RUN_DIR, 'final_config.txt')
        with open(config_path, 'w') as f:
            for attr in dir(config):
                if not attr.startswith('__') and not callable(getattr(config, attr)):
                    f.write(f"{attr}: {getattr(config, attr)}\n")
        mlflow.log_artifact(config_path)

        return model, best_dice, test_metrics

if __name__ == "__main__":
    args = parse_args()
    config = setup_config(args)
    train(config, args)
    print("\nTraining complete! Check the MLflow UI for results.")
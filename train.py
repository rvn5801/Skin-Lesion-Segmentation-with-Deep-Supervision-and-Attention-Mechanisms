import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import time
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
import json

from config import Config
from dataset import ISICDataModule
from unet import AttentionUNet
from metrics import (
    DiceLoss, BCEDiceLoss, TverskyLoss, FocalTverskyLoss, CombinedLoss,
    dice_coefficient, iou_score, calculate_metrics, DeepSupervisionLoss 
)
from visualization import Visualizer, TestEvaluator



class MetricsLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = {'train_loss': [], 'train_dice': [], 'train_iou': [], 
                       'lr': [], 'steps': [], 'epochs': []}
        os.makedirs(log_dir, exist_ok=True)
        
    def log_batch(self, epoch, batch_idx, total_batches, metrics, lr):
        step = epoch * total_batches + batch_idx
        self.metrics['steps'].append(step)
        self.metrics['epochs'].append(epoch + batch_idx / total_batches)
        self.metrics['train_loss'].append(metrics['loss'])
        self.metrics['train_dice'].append(metrics['dice'])
        self.metrics['train_iou'].append(metrics['iou'])
        self.metrics['lr'].append(lr)
        
        # Save every 20 batches
        if batch_idx % 20 == 0:
            self.save_metrics()
    
    def save_metrics(self):
        with open(os.path.join(self.log_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(self.metrics, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net model for ISIC 2018 skin lesion segmentation')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=None, help='Image size')
    parser.add_argument('--loss', type=str, default='bce_dice', 
                        choices=['dice', 'bce_dice', 'tversky', 'focal_tversky', 'combined'],
                        help='Loss function to use')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='Disable data augmentation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (small dataset)')
    
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
    
    # Create an experiment folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config.EXPERIMENT_NAME = f"unet_isic_{timestamp}"
    config.OUTPUT_DIR = os.path.join('outputs', config.EXPERIMENT_NAME)
    config.MODEL_DIR = os.path.join(config.OUTPUT_DIR, 'models')
    config.LOG_DIR = os.path.join(config.OUTPUT_DIR, 'logs')
    config.VISUALIZATION_DIR = os.path.join(config.OUTPUT_DIR, 'visualizations')
    
    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    
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
    

def train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, device, epoch, metrics_logger=None):
    """Train model for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    batch_count = 0
    last_dice = 0.0
    last_iou = 0.0
    
    # Training loop with progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with autocast for mixed precision
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Step the scheduler after each batch
            scheduler.step()
            
            # Calculate metrics (properly handling deep supervision)
            with torch.no_grad():
                if isinstance(output, list):
                    pred = output[0].detach()
                else:
                    pred = output.detach()
                
                pred_binary = (pred > 0.5).float()
                dice = dice_coefficient(pred_binary, target).item()
                iou = iou_score(pred_binary, target).item()
                
                last_dice = dice
                last_iou = iou
            
            # Update running metrics
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            batch_count += 1
            
            # Log metrics if logger is provided
            current_lr = optimizer.param_groups[0]['lr']
            if metrics_logger is not None and batch_idx % 5 == 0:
                metrics_logger.log_batch(
                    epoch, 
                    batch_idx, 
                    len(train_loader),
                    {'loss': loss.item(), 'dice': dice, 'iou': iou},
                    current_lr
                )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(), 
                'dice': dice, 
                'iou': iou
            })
    
    # Calculate average metrics
    avg_loss = running_loss / batch_count
    avg_dice = running_dice / batch_count
    avg_iou = running_iou / batch_count
    
    # Add normalization check and fallback
    if avg_dice < 0.001 and last_dice > 0.05:
        print(f"Warning: Dice averaging issue detected. Using last batch dice: {last_dice:.4f}")
        avg_dice = last_dice
        avg_iou = last_iou
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'iou': avg_iou
    }


def validate(model, val_loader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    # Validation loop with progress bar
    with tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]") as pbar:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Handle deep supervision output format
                if isinstance(output, list):
                    pred = output[0]  # Use the main output for metrics
                else:
                    pred = output
                
                # Calculate metrics using the main prediction only
                metrics = calculate_metrics(pred, target)
                all_metrics.append(metrics)
                
                # Update running loss
                running_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(), 
                    'dice': metrics['dice']
                })
    
    # Calculate average metrics
    avg_loss = running_loss / len(val_loader)
    
    # Average all metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Add loss to metrics
    avg_metrics['loss'] = avg_loss
    
    return avg_metrics

# In the save_checkpoint function:
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    # Save last checkpoint
    torch.save(checkpoint, os.path.join(config.MODEL_DIR, 'last_checkpoint.pth'))
    
    # Save best checkpoint if this is the best model
    if is_best:
        torch.save(checkpoint, os.path.join(config.MODEL_DIR, 'best_model.pth'))
        
        # Also save just the model state dict for easier loading
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model_state_dict.pth'))

# def save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, is_best=False):
#     """Save model checkpoint"""
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
#         'metrics': metrics
#     }
    
#     # Save last checkpoint
#     torch.save(checkpoint, os.path.join(config.MODEL_DIR, 'last_checkpoint.pth'))
    
#     # Save best checkpoint if this is the best model
#     if is_best:
#         torch.save(checkpoint, os.path.join(config.MODEL_DIR, 'best_model.pth'))
        
#         # Also save just the model state dict for easier loading
#         torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model_state_dict.pth'))




def train(config, loss_type='focal_tversky', debug=False):
    """Train the model"""
    print(f"Using device: {config.DEVICE}")
    
    # Initialize dataset
    data_module = ISICDataModule(config)
    
    # If debug mode, use a small subset of data
    if debug:
        print("Debug mode: using small subset of data")
        data_module.load_paths()
        data_module.train_img_paths = data_module.train_img_paths[:100]
        data_module.train_mask_paths = data_module.train_mask_paths[:100]
        data_module.val_img_paths = data_module.val_img_paths[:20]
        data_module.val_mask_paths = data_module.val_mask_paths[:20]
        data_module.test_img_paths = data_module.test_img_paths[:20]
        data_module.test_mask_paths = data_module.test_mask_paths[:20]
    
    # Setup dataset
    data_module.setup()
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    # Initialize model
    model = AttentionUNet(
        in_channels=config.IMG_CHANNELS,
        out_channels=1,
        features=config.INITIAL_FILTERS,
        dropout=config.DROPOUT_RATE,
        bilinear=True,
        deep_supervision=True 

    ).to(config.DEVICE)

    if hasattr(model, 'deep_supervision') and model.deep_supervision:
        print("Model is using deep supervision")
    else:
        print("Model is NOT using deep supervision")
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize loss function with deep supervision wrapper
    base_criterion = get_loss_function(loss_type, config)
    criterion = DeepSupervisionLoss(base_criterion)
    
    # # Initialize optimizer
    # optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Initialize learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='max', 
    #     factor=config.SCHEDULER_FACTOR, 
    #     patience=config.SCHEDULER_PATIENCE, 
    #     verbose=True
    # )

    
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-5
    )
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=3e-4,  # Higher peak learning rate
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3, # Spend 30% of training warming up
        div_factor=25,
        final_div_factor= 1000
    )

    # Initialize AMP scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    metrics_logger = MetricsLogger(config.LOG_DIR)
    # Initialize visualizer
    visualizer = Visualizer(config,data_module)
    
    # Training loop
    best_dice = 0.0
    best_epoch = 0
    early_stop_counter = 0
    
    print(f"Starting training for {config.EPOCHS} epochs...")
    for epoch in range(config.EPOCHS):
        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler, config.DEVICE, epoch, metrics_logger)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config.DEVICE, epoch)
        
        # Update learning rate
        # scheduler.step(val_metrics['dice'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update visualizations
        visualizer.update_metrics(epoch, train_metrics, val_metrics, current_lr)
        visualizer.plot_detailed_metrics()
        
        # Log metrics
        print(f"Epoch {epoch+1}/{config.EPOCHS}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Visualize predictions periodically
        if epoch % config.VISUALIZE_FREQUENCY == 0:
            # Sample a batch from validation set
            val_iter = iter(val_loader)
            batch_imgs, batch_masks = next(val_iter)
            batch_imgs, batch_masks = batch_imgs.to(config.DEVICE), batch_masks.to(config.DEVICE)
            
            # Generate predictions
            model.eval()
            with torch.no_grad():
                batch_preds = model(batch_imgs)
            
            # Visualize batch
            visualizer.visualize_batch(batch_imgs, batch_masks, batch_preds, epoch)
        
        # Check if this is the best model
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            early_stop_counter = 0
            print(f"  New best model! Dice: {best_dice:.4f}")
        else:
            early_stop_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, is_best)
        
        # Early stopping
        if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} with Dice: {best_dice:.4f}")
            break
    
    print(f"Training completed! Best validation Dice: {best_dice:.4f} at epoch {best_epoch+1}")
    
    # Load best model for evaluation
    checkpoint = torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    test_evaluator = TestEvaluator(model, test_loader, config, data_module)
    test_metrics = test_evaluator.evaluate()
    
    # Save config and results to file
    with open(os.path.join(config.OUTPUT_DIR, 'config.txt'), 'w') as f:
        for attr in dir(config):
            if not attr.startswith('__') and not callable(getattr(config, attr)):
                value = getattr(config, attr)
                if not isinstance(value, torch.device):
                    f.write(f"{attr}: {value}\n")
                else:
                    f.write(f"{attr}: {value.__str__()}\n")
        
        f.write("\nBest Model Performance:\n")
        f.write(f"Best validation Dice: {best_dice:.4f} at epoch {best_epoch+1}\n")
        for key, value in test_metrics.items():
            if not key.endswith('_std') and not key.endswith('_min') and not key.endswith('_max'):
                f.write(f"Test {key}: {value:.4f}\n")
    
    return model, best_dice, test_metrics

def batch_end_callback(epoch, batch_idx, loss, dice, iou):
    # Log every 50 batches
    if batch_idx % 50 == 0:
        print(f"  Batch {batch_idx}: Loss={loss:.4f}, Dice={dice:.4f}, IoU={iou:.4f}")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set up configuration
    config = setup_config(args)
    
    # Train model
    model, best_dice, test_metrics = train(config, loss_type=args.loss, debug=args.debug)
    
    print("\nTraining complete!")



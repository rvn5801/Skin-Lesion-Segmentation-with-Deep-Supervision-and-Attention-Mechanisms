import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
import seaborn as sns
from datetime import datetime
import json

from metrics import dice_coefficient, calculate_metrics

class MetricTracker:
    def __init__(self):
        self.train_metrics = {'loss': [], 'dice': [], 'iou': []}
        self.val_metrics = {'loss': [], 'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': []}
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch, train_metrics, val_metrics, lr):
        self.epochs.append(epoch)
        self.learning_rates.append(lr)
        
        for k, v in train_metrics.items():
            if k in self.train_metrics:
                self.train_metrics[k].append(v)
        
        for k, v in val_metrics.items():
            if k in self.val_metrics:
                self.val_metrics[k].append(v)
    
    def plot_metrics(self, save_path):
        plt.figure(figsize=(20, 15))
        
        # Plot loss
        plt.subplot(3, 2, 1)
        plt.plot(self.epochs, self.train_metrics['loss'], label='Train Loss')
        plt.plot(self.epochs, self.val_metrics['loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot dice coefficient
        plt.subplot(3, 2, 2)
        plt.plot(self.epochs, self.train_metrics['dice'], label='Train Dice')
        plt.plot(self.epochs, self.val_metrics['dice'], label='Val Dice')
        plt.title('Dice Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.grid(True)
        
        # Plot IoU
        plt.subplot(3, 2, 3)
        plt.plot(self.epochs, self.train_metrics['iou'], label='Train IoU')
        plt.plot(self.epochs, self.val_metrics['iou'], label='Val IoU')
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True)
        
        # Plot precision, recall, f1
        plt.subplot(3, 2, 4)
        plt.plot(self.epochs, self.val_metrics['precision'], label='Precision')
        plt.plot(self.epochs, self.val_metrics['recall'], label='Recall')
        plt.plot(self.epochs, self.val_metrics['f1'], label='F1 Score')
        plt.title('Precision, Recall, F1')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(3, 2, 5)
        plt.plot(self.epochs, self.learning_rates)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid(True)
        
        # If we have more than 2 epochs, compute improvement rates
        if len(self.epochs) > 2:
            plt.subplot(3, 2, 6)
            dice_improvement = [0] + [self.val_metrics['dice'][i] - self.val_metrics['dice'][i-1] 
                                     for i in range(1, len(self.val_metrics['dice']))]
            plt.bar(self.epochs, dice_improvement)
            plt.title('Dice Improvement per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Improvement')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class Visualizer:
    def __init__(self, config, data_module=None):
        self.config = config
        self.visualization_dir = self.config.VISUALIZATION_DIR
        self.epoch_dir = os.path.join(self.visualization_dir, 'epochs')
        os.makedirs(self.epoch_dir, exist_ok=True)
        
        # Store mean and std for denormalization if available
        if data_module is not None and hasattr(data_module, 'mean') and hasattr(data_module, 'std'):
            self.mean = data_module.mean
            self.std = data_module.std
        else:
            # Default mean and std from your dataset
            self.mean = np.array([0.71122827, 0.57738177, 0.53471777])
            self.std = np.array([0.15258599, 0.16130802, 0.17394956])
        
        # Create metric tracker
        self.metric_tracker = MetricTracker()

    def plot_detailed_metrics(self, save_path=None):
        """Plot detailed metrics from logged data"""
        if save_path is None:
            save_path = self.visualization_dir
        
        # Load detailed metrics if available
        metrics_path = os.path.join(self.config.LOG_DIR, 'detailed_metrics.json') 
        if not os.path.exists(metrics_path):
            print("No detailed metrics found. Skipping detailed plot.")
            return
        
        try:
            with open(metrics_path, 'r') as f:
                detailed = json.load(f)
            
            plt.figure(figsize=(15, 10))
            
            # Plot learning rate with many points
            plt.subplot(2, 2, 1)
            plt.plot(detailed['epochs'], detailed['lr'])
            plt.title('Learning Rate (Detailed)')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            
            # Plot loss with many points
            plt.subplot(2, 2, 2)
            plt.plot(detailed['epochs'], detailed['train_loss'])
            plt.title('Training Loss (Detailed)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Plot Dice with many points
            plt.subplot(2, 2, 3)
            plt.plot(detailed['epochs'], detailed['train_dice'])
            plt.title('Training Dice (Detailed)')
            plt.xlabel('Epoch')
            plt.ylabel('Dice')
            plt.grid(True)
            
            # Plot IoU with many points
            plt.subplot(2, 2, 4)
            plt.plot(detailed['epochs'], detailed['train_iou'])
            plt.title('Training IoU (Detailed)')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'detailed_metrics.png'))
            plt.close()
            
            print(f"Detailed metrics plot saved to {os.path.join(save_path, 'detailed_metrics.png')}")
        except Exception as e:
            print(f"Error plotting detailed metrics: {str(e)}")
        
    def denormalize_image(self, img_tensor):
        """Properly denormalize a normalized image tensor for visualization"""
        # Convert tensor to numpy and move channels to the end
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Get mean and std from the dataset if available
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            mean = self.mean
            std = self.std
        else:
            # Use the dataset's statistics from your output
            mean = np.array([0.71122827, 0.57738177, 0.53471777])
            std = np.array([0.15258599, 0.16130802, 0.17394956])
        
        # Denormalize
        img_np = img_np * std + mean
        
        # Ensure values are within valid range
        img_np = np.clip(img_np, 0, 1)
        
        return img_np
    
    def visualize_batch(self, batch_imgs, batch_masks, batch_preds, epoch, sample_indices=None, current_epoch_dir=None):
        """Visualize a batch of predictions"""
        if current_epoch_dir is None:
            current_epoch_dir = os.path.join(self.epoch_dir, f'epoch_{epoch}')
            os.makedirs(current_epoch_dir, exist_ok=True)
        
        batch_size = batch_imgs.size(0)
        
        # If no sample indices provided, use all batch examples
        if sample_indices is None:
            sample_indices = range(min(batch_size, self.config.SAMPLES_TO_VISUALIZE))
        
        for i, idx in enumerate(sample_indices):
            if idx >= batch_size:
                continue
                
            # Get single example
            img = batch_imgs[idx]
            mask = batch_masks[idx]
            
            # Handle deep supervision output if needed
            if isinstance(batch_preds, list):
                pred = batch_preds[0][idx]  # Use the main prediction
            else:
                pred = batch_preds[idx]
            
            # Denormalize the image for visualization
            img_np = self.denormalize_image(img)
            
            # Convert mask and prediction to numpy
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')
            
            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pred_np, cmap='gray')
            plt.title(f"Prediction (Dice: {dice_coefficient(pred, mask):.4f})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(current_epoch_dir, f'sample_{i}_epoch_{epoch}.png'))
            plt.close()
            
            # Create overlay visualization with denormalized image
            self.visualize_overlay(img_np, mask_np, pred_np, os.path.join(current_epoch_dir, f'overlay_{i}_epoch_{epoch}.png'))

    
    def visualize_overlay(self, img, mask, pred, save_path, threshold=0.5):
        """Create overlay visualization of prediction vs ground truth"""
        # Apply threshold to prediction (if not already binary)
        if pred.dtype != np.uint8:
            pred_binary = (pred > threshold).astype(np.float32)
        else:
            pred_binary = pred / 255.0
        
        # Image should already be denormalized at this point
        
        # Create RGB mask overlays (ground truth: red, prediction: green)
        gt_overlay = np.zeros_like(img)
        gt_overlay[:,:,0] = mask * 255  # Red channel
        
        pred_overlay = np.zeros_like(img)
        pred_overlay[:,:,1] = pred_binary * 255  # Green channel
        
        # Combined overlay to show all classes
        combined = np.zeros_like(img)
        combined[:,:,0] = mask * 255  # Red: ground truth
        combined[:,:,1] = pred_binary * 255  # Green: prediction
        # Yellow will be the overlap
        
        plt.figure(figsize=(15, 5))
        
        # Ground truth overlay
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.addWeighted(img, 1, gt_overlay, 0.5, 0))
        plt.title("Ground Truth (Red)")
        plt.axis('off')
        
        # Prediction overlay
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.addWeighted(img, 1, pred_overlay, 0.5, 0))
        plt.title("Prediction (Green)")
        plt.axis('off')
        
        # Combined overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.addWeighted(img, 1, combined, 0.5, 0))
        plt.title("Combined (Yellow=Overlap)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def update_metrics(self, epoch, train_metrics, val_metrics, lr):
        """Update metrics tracker and plot metrics"""
        self.metric_tracker.update(epoch, train_metrics, val_metrics, lr)
        self.metric_tracker.plot_metrics(os.path.join(self.visualization_dir, 'metrics_history.png'))

class TestEvaluator:
    def __init__(self, model, test_loader, config, dataset):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.dataset = dataset
        self.device = self.config.DEVICE
        self.visualization_dir = os.path.join(self.config.VISUALIZATION_DIR, 'test_evaluation')
        os.makedirs(self.visualization_dir, exist_ok=True)



    def evaluate(self):
        """Evaluate model on test set and generate visualizations"""
        self.model.eval()
        
        # Lists to store metrics for each image
        all_metrics = []
        all_dice_scores = []
        all_images = []
        all_masks = []
        all_preds = []
        
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(self.test_loader):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                preds = self.model(imgs)
                
                # Calculate metrics for each image in batch
                for i in range(imgs.size(0)):
                    img = imgs[i]
                    mask = masks[i]
                    pred = preds[i]
                    
                    # Calculate metrics
                    metrics = calculate_metrics(pred, mask)
                    # Get the global index to find the image name
                    global_idx = (batch_idx * self.test_loader.batch_size) + i
                    if global_idx < len(self.dataset.test_img_paths):
                        metrics['image_name'] = os.path.basename(self.dataset.test_img_paths[global_idx])
                    else:
                        metrics['image_name'] = f"unknown_idx_{global_idx}"
                    all_metrics.append(metrics)
                    all_dice_scores.append(metrics['dice'])
                    
                    # Store for visualization (first 50 samples)
                    if len(all_images) < 50:  # Limit number of stored images
                        all_images.append(img.cpu())
                        all_masks.append(mask.cpu())
                        all_preds.append(pred.cpu())

        # Save all individual metrics to a new CSV file
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_df = all_metrics_df.sort_values(by='dice', ascending=True) # Sort worst to best
        csv_path = os.path.join(self.visualization_dir, 'all_test_scores.csv')
        all_metrics_df.to_csv(csv_path, index=False)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != 'image_name':
                values = [m[key] for m in all_metrics]
            # values = [m[key] for m in all_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
                avg_metrics[f'{key}_min'] = np.min(values)
                avg_metrics[f'{key}_max'] = np.max(values)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([avg_metrics])
        metrics_df.to_csv(os.path.join(self.visualization_dir, 'test_metrics.csv'), index=False)
        
        # Print metrics
        print("\nTest metrics:")
        for key, value in avg_metrics.items():
            if '_std' not in key and '_min' not in key and '_max' not in key:
                print(f"{key}: {value:.4f} ± {avg_metrics[f'{key}_std']:.4f} (min: {avg_metrics[f'{key}_min']:.4f}, max: {avg_metrics[f'{key}_max']:.4f})")
        
        # Visualize best and worst predictions if we have enough images
        if len(all_images) > 0:
            try:
                self.visualize_best_worst(all_images, all_masks, all_preds, all_dice_scores)
            except Exception as e:
                print(f"Error visualizing best/worst predictions: {e}")
        
        # Generate detailed report
        try:
            self.generate_report(avg_metrics)
        except Exception as e:
            print(f"Error generating report: {e}")
        
        return avg_metrics
    
    # def evaluate(self):
    #     """Evaluate model on test set and generate visualizations"""
    #     self.model.eval()
        
    #     # Lists to store metrics for each image
    #     all_metrics = []
    #     all_dice_scores = []
    #     all_images = []
    #     all_masks = []
    #     all_preds = []
        
    #     with torch.no_grad():
    #         for batch_idx, (imgs, masks) in enumerate(self.test_loader):
    #             imgs = imgs.to(self.device)
    #             masks = masks.to(self.device)
                
    #             # Forward pass
    #             preds = self.model(imgs)
                
    #             # Calculate metrics for each image in batch
    #             for i in range(imgs.size(0)):
    #                 img = imgs[i]
    #                 mask = masks[i]
    #                 pred = preds[i]
                    
    #                 # Calculate metrics
    #                 metrics = calculate_metrics(pred, mask)
    #                 all_metrics.append(metrics)
    #                 all_dice_scores.append(metrics['dice'])
                    
    #                 # Store for visualization (first N samples)
    #                 if len(all_images) < 50:  # Limit number of stored images
    #                     all_images.append(img.cpu())
    #                     all_masks.append(mask.cpu())
    #                     all_preds.append(pred.cpu())
        
    #     # Calculate average metrics
    #     avg_metrics = {}
    #     for key in all_metrics[0].keys():
    #         values = [m[key] for m in all_metrics]
    #         avg_metrics[key] = np.mean(values)
    #         avg_metrics[f'{key}_std'] = np.std(values)
    #         avg_metrics[f'{key}_min'] = np.min(values)
    #         avg_metrics[f'{key}_max'] = np.max(values)
        
    #     # Save metrics to CSV
    #     metrics_df = pd.DataFrame([avg_metrics])
    #     metrics_df.to_csv(os.path.join(self.visualization_dir, 'test_metrics.csv'), index=False)
        
    #     # Print metrics
    #     print("\nTest metrics:")
    #     for key, value in avg_metrics.items():
    #         if '_std' not in key and '_min' not in key and '_max' not in key:
    #             print(f"{key}: {value:.4f} ± {avg_metrics[f'{key}_std']:.4f} (min: {avg_metrics[f'{key}_min']:.4f}, max: {avg_metrics[f'{key}_max']:.4f})")
        
    #     # Visualize best and worst predictions
    #     self.visualize_best_worst(all_images, all_masks, all_preds, all_dice_scores)
        
    #     # Generate detailed report
    #     self.generate_report(avg_metrics)
        
    #     return avg_metrics

    def visualize_best_worst(self, images, masks, preds, scores, n_samples=5):
        """Visualize best and worst predictions based on Dice scores"""
        # Sort indices by dice score
        indices = np.argsort(scores)
        
        # Ensure indices are within the range of available images
        indices = [idx for idx in indices if idx < len(images)]
        
        # Ensure we don't try to access beyond available samples
        n_samples = min(n_samples, len(images) // 2)  # Make sure we have enough for both best and worst
        
        if n_samples == 0:
            print("Warning: Not enough samples to visualize best/worst cases")
            return
        
        # Get worst and best indices
        worst_indices = indices[:n_samples]
        best_indices = indices[-n_samples:]
        
        # Create visualization directories
        os.makedirs(os.path.join(self.visualization_dir, 'best'), exist_ok=True)
        os.makedirs(os.path.join(self.visualization_dir, 'worst'), exist_ok=True)
        
        # Visualize worst predictions
        plt.figure(figsize=(15, 5 * n_samples))
        for i, idx in enumerate(worst_indices):
            if idx >= len(images):
                print(f"Warning: Index {idx} out of range for images list of length {len(images)}")
                continue
                
            # img = images[idx].permute(1, 2, 0).numpy()
            # --- START OF NEW DENORMALIZATION FIX ---
            img_tensor = images[idx].cpu()
            
            # Get the correct stats from the live dataset object
            mean = self.dataset.mean
            std = self.dataset.std

            # Ensure they are tensors for broadcasting
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, dtype=img_tensor.dtype)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std, dtype=img_tensor.dtype)

            # Reshape for C, H, W format
            mean = mean.reshape(3, 1, 1)
            std = std.reshape(3, 1, 1)

            # Denormalize and clip
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1) # This clip fixes the warning
            img = img_denorm.permute(1, 2, 0).numpy() # Convert to numpy for plotting
            mask = masks[idx].squeeze().numpy()
            pred = preds[idx].squeeze().numpy()
            dice = scores[idx]
            
            # Original image
            plt.subplot(n_samples, 3, i*3+1)
            plt.imshow(img)
            plt.title(f"Image (Dice: {dice:.4f})")
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(n_samples, 3, i*3+2)
            plt.imshow(mask, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')
            
            # Predicted mask
            plt.subplot(n_samples, 3, i*3+3)
            plt.imshow(pred, cmap='gray')
            plt.title("Prediction")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'worst_predictions.png'))
        plt.close()
        
        # Visualize best predictions
        plt.figure(figsize=(15, 5 * n_samples))
        for i, idx in enumerate(best_indices):
            if idx >= len(images):
                print(f"Warning: Index {idx} out of range for images list of length {len(images)}")
                continue
                
            img = images[idx].permute(1, 2, 0).numpy()
            mask = masks[idx].squeeze().numpy()
            pred = preds[idx].squeeze().numpy()
            dice = scores[idx]
            
            # Original image
            plt.subplot(n_samples, 3, i*3+1)
            plt.imshow(img)
            plt.title(f"Image (Dice: {dice:.4f})")
            plt.axis('off')
            
            # Ground truth mask
            plt.subplot(n_samples, 3, i*3+2)
            plt.imshow(mask, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')
            
            # Predicted mask
            plt.subplot(n_samples, 3, i*3+3)
            plt.imshow(pred, cmap='gray')
            plt.title("Prediction")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, 'best_predictions.png'))
        plt.close()
        
        # We'll skip individual overlays if there are index problems
        try:
            # Save individual images for best and worst cases
            for i, idx in enumerate(worst_indices):
                if idx >= len(images):
                    continue
                    
                img = images[idx].permute(1, 2, 0).numpy()
                mask = masks[idx].squeeze().numpy()
                pred = preds[idx].squeeze().numpy()
                
                # Create overlay
                overlay = self.create_overlay(img, mask, pred)
                
                # Save the overlay
                plt.figure(figsize=(8, 8))
                plt.imshow(overlay)
                plt.title(f"Worst Case {i+1} (Dice: {scores[idx]:.4f})")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.visualization_dir, 'worst', f'worst_case_{i+1}.png'))
                plt.close()
            
            for i, idx in enumerate(best_indices):
                if idx >= len(images):
                    continue
                    
                img = images[idx].permute(1, 2, 0).numpy()
                mask = masks[idx].squeeze().numpy()
                pred = preds[idx].squeeze().numpy()
                
                # Create overlay
                overlay = self.create_overlay(img, mask, pred)
                
                # Save the overlay
                plt.figure(figsize=(8, 8))
                plt.imshow(overlay)
                plt.title(f"Best Case {i+1} (Dice: {scores[idx]:.4f})")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.visualization_dir, 'best', f'best_case_{i+1}.png'))
                plt.close()
        except Exception as e:
            print(f"Warning: Could not create individual overlays: {e}")
    

    
    def create_overlay(self, img, mask, pred, threshold=0.5):
        """Create overlay of ground truth and prediction on image"""
        # Apply threshold to prediction
        pred_binary = (pred > threshold).astype(np.float32)
        
        # Create combined overlay
        combined = np.zeros_like(img)
        combined[:,:,0] = mask * 255  # Red: ground truth
        combined[:,:,1] = pred_binary * 255  # Green: prediction
        # Yellow will be the overlap
        
        # Add overlay to original image
        overlay = cv2.addWeighted(img, 1, combined, 0.5, 0)
        
        return overlay
    
    def generate_report(self, metrics):
        """Generate a comprehensive HTML report"""
        # Create report title and date
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # HTML report
        report_html = f"""
        <html>
        <head>
            <title>ISIC 2018 Segmentation Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-card {{ background-color: #f9f9f9; border-radius: 5px; 
                               padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                .images {{ display: flex; flex-wrap: wrap; justify-content: center; }}
                .image-container {{ margin: 10px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>ISIC 2018 Skin Lesion Segmentation Model Evaluation Report</h1>
            <p><strong>Date:</strong> {report_date}</p>
            
            <h2>Model Performance Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>Dice Coefficient</h3>
                    <div class="metric-value">{metrics['dice']:.4f}</div>
                    <p>Min: {metrics['dice_min']:.4f} | Max: {metrics['dice_max']:.4f} | StdDev: {metrics['dice_std']:.4f}</p>
                </div>
                <div class="metric-card">
                    <h3>IoU (Jaccard)</h3>
                    <div class="metric-value">{metrics['iou']:.4f}</div>
                    <p>Min: {metrics['iou_min']:.4f} | Max: {metrics['iou_max']:.4f} | StdDev: {metrics['iou_std']:.4f}</p>
                </div>
                <div class="metric-card">
                    <h3>Precision</h3>
                    <div class="metric-value">{metrics['precision']:.4f}</div>
                    <p>Min: {metrics['precision_min']:.4f} | Max: {metrics['precision_max']:.4f}</p>
                </div>
                <div class="metric-card">
                    <h3>Recall</h3>
                    <div class="metric-value">{metrics['recall']:.4f}</div>
                    <p>Min: {metrics['recall_min']:.4f} | Max: {metrics['recall_max']:.4f}</p>
                </div>
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <div class="metric-value">{metrics['f1']:.4f}</div>
                    <p>Min: {metrics['f1_min']:.4f} | Max: {metrics['f1_max']:.4f}</p>
                </div>
            </div>
            
            <h2>Visualization Results</h2>
            
            <h3>Best Predictions</h3>
            <div class="images">
                <div class="image-container">
                    <img src="best_predictions.png" alt="Best Predictions">
                </div>
            </div>
            
            <h3>Worst Predictions</h3>
            <div class="images">
                <div class="image-container">
                    <img src="worst_predictions.png" alt="Worst Predictions">
                </div>
            </div>
            
            <h2>Conclusion</h2>
            <p>The model achieves a mean Dice coefficient of {metrics['dice']:.4f} on the test set,
               with an IoU of {metrics['iou']:.4f}. The precision is {metrics['precision']:.4f} and
               the recall is {metrics['recall']:.4f}, resulting in an F1 score of {metrics['f1']:.4f}.</p>
               
            <p>Areas for improvement:</p>
            <ul>
                <li>Better handling of boundary regions to reduce false positives</li>
                <li>Improved detection of small lesions to reduce false negatives</li>
                <li>More robust segmentation for cases with low contrast</li>
            </ul>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(self.visualization_dir, 'test_report.html'), 'w') as f:
            f.write(report_html)
        
        print(f"Test evaluation report saved to {os.path.join(self.visualization_dir, 'test_report.html')}")
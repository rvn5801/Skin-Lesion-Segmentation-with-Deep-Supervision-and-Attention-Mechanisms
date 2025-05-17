import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A
import json

from config import Config

class ISICDataset(Dataset):
    def __init__(self, config, img_paths, mask_paths, transform=None, normalize=True):
        self.config = config
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.normalize = normalize
        self.mean = None
        self.std = None
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
        
        # Apply transformations if specified
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Normalize image
        if self.normalize:
            img = img / 255.0
            if self.mean is not None and self.std is not None:
                img = (img - self.mean) / (self.std + 1e-7)
        
        # Convert mask to binary
        mask = (mask > 127).astype(np.float32)
        
        # Convert to PyTorch tensors
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img, mask

class ISICDataModule:
    def __init__(self, config):
        self.config = config
        self.train_img_paths = []
        self.train_mask_paths = []
        self.val_img_paths = []
        self.val_mask_paths = []
        self.test_img_paths = []
        self.test_mask_paths = []
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.train_transform = None
        self.val_transform = None
    
    def load_paths(self):
        print("Loading dataset paths...")
        # Get training data paths
        self.train_img_paths = sorted(glob(os.path.join(self.config.TRAIN_IMG_DIR, 'ISIC_*.jpg')))
        
        # Match masks to images
        self.train_mask_paths = []
        valid_train_img_paths = []
        for img_path in self.train_img_paths:
            img_filename = os.path.basename(img_path)
            isic_id = img_filename.split('.')[0]
            mask_filename = f"{isic_id}_segmentation.png"
            mask_path = os.path.join(self.config.TRAIN_MASK_DIR, mask_filename)
            
            if os.path.exists(mask_path):
                self.train_mask_paths.append(mask_path)
                valid_train_img_paths.append(img_path)
            else:
                print(f"Warning: No mask found for {img_filename}")
        
        self.train_img_paths = valid_train_img_paths
        
        # Get validation data paths
        self.val_img_paths = sorted(glob(os.path.join(self.config.VAL_IMG_DIR, 'ISIC_*.jpg')))
        
        # Match masks to validation images
        self.val_mask_paths = []
        valid_val_img_paths = []
        for img_path in self.val_img_paths:
            img_filename = os.path.basename(img_path)
            isic_id = img_filename.split('.')[0]
            mask_filename = f"{isic_id}_segmentation.png"
            mask_path = os.path.join(self.config.VAL_MASK_DIR, mask_filename)
            
            if os.path.exists(mask_path):
                self.val_mask_paths.append(mask_path)
                valid_val_img_paths.append(img_path)
            else:
                print(f"Warning: No validation mask found for {img_filename}")
        
        self.val_img_paths = valid_val_img_paths
        
        # Get test data paths
        self.test_img_paths = sorted(glob(os.path.join(self.config.TEST_IMG_DIR, 'ISIC_*.jpg')))
        
        # Match masks to test images
        self.test_mask_paths = []
        valid_test_img_paths = []
        for img_path in self.test_img_paths:
            img_filename = os.path.basename(img_path)
            isic_id = img_filename.split('.')[0]
            mask_filename = f"{isic_id}_segmentation.png"
            mask_path = os.path.join(self.config.TEST_MASK_DIR, mask_filename)
            
            if os.path.exists(mask_path):
                self.test_mask_paths.append(mask_path)
                valid_test_img_paths.append(img_path)
            else:
                print(f"Warning: No test mask found for {img_filename}")
        
        self.test_img_paths = valid_test_img_paths
        
        print(f"Training images: {len(self.train_img_paths)}")
        print(f"Validation images: {len(self.val_img_paths)}")
        print(f"Test images: {len(self.test_img_paths)}")
    
    def _calculate_stats(self):
        """Calculate dataset mean and std for normalization"""
        print("Calculating dataset statistics...")
        # Take a sample of images to calculate mean and std
        sample_size = min(100, len(self.train_img_paths))
        sample_paths = np.random.choice(self.train_img_paths, sample_size, replace=False)
        
        samples = []
        for path in tqdm(sample_paths, desc="Loading samples for stats"):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            img = img / 255.0  # Scale to [0, 1]
            samples.append(img)
        
        samples = np.array(samples)
        self.mean = np.mean(samples, axis=(0, 1, 2))
        self.std = np.std(samples, axis=(0, 1, 2))
        
        print(f"Dataset mean: {self.mean}")
        print(f"Dataset std: {self.std}")

    def save_stats(self):
        """Save calculated mean and std to disk"""
        stats = {'mean': self.mean.tolist(), 'std': self.std.tolist()}
        with open(os.path.join(self.config.OUTPUT_DIR, 'dataset_stats.json'), 'w') as f:
            json.dump(stats, f)

    def load_stats(self):
        """Load precalculated mean and std"""
        stats_path = os.path.join(self.config.OUTPUT_DIR, 'dataset_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.mean = np.array(stats['mean'])
            self.std = np.array(stats['std'])
            print(f"Loaded dataset stats: mean={self.mean}, std={self.std}")
            return True
        return False
    
    def setup(self):
        # Load data paths
        self.load_paths()
        
        # Calculate stats for normalization
        self._calculate_stats()
        
        # Set up transformations
        if self.config.AUGMENTATION:
        
            self.train_transform = A.Compose([
                
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Enhanced augmentations for hard cases
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                A.GridDistortion(p=0.2),
                
                # Better contrast handling (crucial for low-contrast lesions)
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                
                # Color transformations for different imaging conditions
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
                
                # Geometric transformations
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
            ])
        
        # Simple transformation for validation and test (just resize)
        self.val_transform = A.Compose([])
        
        # Analyze class distribution
        self._analyze_class_distribution()
        
        # Visualize samples
        self.visualize_samples()
        
        if self.config.AUGMENTATION:
            self.visualize_augmentations()
    
    def _analyze_class_distribution(self):
        """Analyze class distribution in the dataset"""
        # Load a sample of masks to analyze
        sample_size = min(100, len(self.train_mask_paths))
        sample_paths = np.random.choice(self.train_mask_paths, sample_size, replace=False)
        
        foreground_percentages = []
        for path in tqdm(sample_paths, desc="Analyzing class distribution"):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            # Calculate percentage of foreground pixels
            foreground_percent = np.mean(mask > 127) * 100
            foreground_percentages.append(foreground_percent)
        
        # Print statistics
        mean_fg = np.mean(foreground_percentages)
        median_fg = np.median(foreground_percentages)
        min_fg = np.min(foreground_percentages)
        max_fg = np.max(foreground_percentages)
        
        print(f"Foreground statistics:")
        print(f"  Mean: {mean_fg:.2f}%")
        print(f"  Median: {median_fg:.2f}%")
        print(f"  Min: {min_fg:.2f}%")
        print(f"  Max: {max_fg:.2f}%")
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(foreground_percentages, bins=20)
        plt.title('Distribution of Lesion Sizes (% of Image)')
        plt.xlabel('Foreground Percentage')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.config.VISUALIZATION_DIR, 'lesion_size_distribution.png'))
        plt.close()
    
    def visualize_samples(self, num_samples=5):
        """Visualize random samples from the dataset"""
        # Randomly select samples
        indices = np.random.choice(len(self.train_img_paths), num_samples, replace=False)
        
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i, idx in enumerate(indices):
            # Load image and mask
            img = cv2.imread(self.train_img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            
            mask = cv2.imread(self.train_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            
            # Original image
            plt.subplot(num_samples, 3, i*3+1)
            plt.imshow(img)
            plt.title(f"Original Image")
            plt.axis('off')
            
            # Mask
            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Segmentation Mask")
            plt.axis('off')
            
            # Overlay
            mask_rgb = np.zeros_like(img)
            mask_rgb[:,:,1] = (mask > 127).astype(np.uint8) * 255  # Green channel
            overlay = cv2.addWeighted(img, 1, mask_rgb, 0.5, 0)
            
            plt.subplot(num_samples, 3, i*3+3)
            plt.imshow(overlay)
            plt.title(f"Overlay")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.VISUALIZATION_DIR, 'dataset_samples.png'))
        plt.close()
    
    def visualize_augmentations(self, num_samples=3, num_augmentations=4):
        """Visualize augmentations for a few samples"""
        if not self.config.AUGMENTATION or self.train_transform is None:
            print("Augmentation is disabled in config.")
            return
        
        # Randomly select samples
        indices = np.random.choice(len(self.train_img_paths), num_samples, replace=False)
        
        plt.figure(figsize=(12, 4 * num_samples))
        
        for i, idx in enumerate(indices):
            # Load image and mask
            img = cv2.imread(self.train_img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            
            mask = cv2.imread(self.train_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            
            # Show original
            plt.subplot(num_samples, num_augmentations+1, i*(num_augmentations+1)+1)
            plt.imshow(img)
            plt.title("Original")
            plt.axis('off')
            
            # Apply different augmentations
            for j in range(num_augmentations):
                augmented = self.train_transform(image=img, mask=mask)
                aug_img = augmented['image']
                aug_mask = augmented['mask']
                
                # Create overlay
                aug_mask_rgb = np.zeros_like(aug_img)
                aug_mask_rgb[:,:,1] = (aug_mask > 127).astype(np.uint8) * 255
                overlay = cv2.addWeighted(aug_img, 1, aug_mask_rgb, 0.5, 0)
                
                plt.subplot(num_samples, num_augmentations+1, i*(num_augmentations+1)+j+2)
                plt.imshow(overlay)
                plt.title(f"Augmented {j+1}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.VISUALIZATION_DIR, 'augmentation_examples.png'))
        plt.close()
    
    def get_dataloaders(self):
        """Get PyTorch DataLoaders for training, validation, and testing"""
        train_dataset = ISICDataset(
            self.config,
            self.train_img_paths,
            self.train_mask_paths,
            transform=self.train_transform,
            normalize=True
        )
        train_dataset.mean = self.mean
        train_dataset.std = self.std
        
        val_dataset = ISICDataset(
            self.config,
            self.val_img_paths, 
            self.val_mask_paths,
            transform=self.val_transform,
            normalize=True
        )
        val_dataset.mean = self.mean
        val_dataset.std = self.std
        
        test_dataset = ISICDataset(
            self.config,
            self.test_img_paths,
            self.test_mask_paths,
            transform=self.val_transform,
            normalize=True
        )
        test_dataset.mean = self.mean
        test_dataset.std = self.std
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
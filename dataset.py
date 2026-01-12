# import os
# import numpy as np
# import cv2
# import torch
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from glob import glob
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import json
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# from config import Config

# class ISICDataset(Dataset):
#     def __init__(self, config, img_paths, mask_paths, 
#                  tiny_transform=None, normal_transform=None, val_transform=None, 
#                  category_map=None, mode='train', normalize=True):
        
#         self.config = config
#         self.img_paths = img_paths
#         self.mask_paths = mask_paths
        
#         # Transforms
#         self.tiny_transform = tiny_transform
#         self.normal_transform = normal_transform
#         self.val_transform = val_transform
        
#         # Metadata dictionary
#         self.category_map = category_map
#         self.mode = mode 
#         self.normalize = normalize
        
#         # Stats
#         self.mean = np.array([0.70809584, 0.58178357, 0.53571441])
#         self.std = np.array([0.15733581, 0.16560281, 0.18079209])
    
#     def __len__(self):
#         return len(self.img_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         mask_path = self.mask_paths[idx]
        
#         # 1. Load Image
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # 2. Load Mask
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
#         # 3. SELECT TRANSFORM BASED ON METADATA
#         selected_transform = None
        
#         if self.mode == 'train' and self.config.AUGMENTATION:
#             mask_filename = os.path.basename(mask_path)
#             category = "Medium (5-30%)" # Default
            
#             if self.category_map:
#                 category = self.category_map.get(mask_filename, "Medium (5-30%)")
            
#             # Select Strategy
#             if category == "Tiny (<5%)" and self.tiny_transform:
#                 selected_transform = self.tiny_transform
#             elif self.normal_transform:
#                 selected_transform = self.normal_transform
#         else:
#             selected_transform = self.val_transform

#         # 4. Apply Transform
#         if selected_transform:
#             augmented = selected_transform(image=img, mask=mask)
#             img = augmented['image']
#             mask = augmented['mask']
#         else:
#             # Fallback
#             img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
#             mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
#             img = img.transpose(2, 0, 1).astype('float32') / 255.0
#             img = torch.from_numpy(img)
#             mask = torch.from_numpy(mask).float()

#         # 5. Safeguard Normalization (if not in transform)
#         if self.normalize and not isinstance(img, torch.Tensor):
#              img = img / 255.0
#              if self.mean is not None:
#                  img = (img - self.mean) / (self.std + 1e-7)

#         # 6. Ensure Binary Mask
#         if isinstance(mask, torch.Tensor):
#              mask = (mask > 0.5).float()
#              if mask.ndim == 2:
#                  mask = mask.unsqueeze(0)
#         else:
#              mask = (mask > 127).astype(np.float32)
#              mask = torch.from_numpy(mask).unsqueeze(0)
        
#         if not isinstance(img, torch.Tensor):
#             img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        
#         return img, mask

# class ISICDataModule:
#     def __init__(self, config):
#         self.config = config
#         self.train_img_paths = []
#         self.train_mask_paths = []
#         self.val_img_paths = []
#         self.val_mask_paths = []
#         self.test_img_paths = []
#         self.test_mask_paths = []
        
#         self.mean = np.array([0.70809584, 0.58178357, 0.53571441])
#         self.std = np.array([0.15733581, 0.16560281, 0.18079209])
        
#         self.tiny_transform = None
#         self.normal_transform = None
#         self.val_transform = None
#         self.category_map = {} 

#     def load_paths(self):
#         print("Loading dataset paths...")
#         def get_images_in_dir(path):
#             imgs = sorted(glob(os.path.join(path, '*.jpg')))
#             if not imgs: imgs = sorted(glob(os.path.join(path, '*.png')))
#             return imgs

#         # 1. Train
#         self.train_img_paths = get_images_in_dir(self.config.TRAIN_IMG_DIR)
#         valid_train = []
#         for p in self.train_img_paths:
#             m = os.path.join(self.config.TRAIN_MASK_DIR, f"{os.path.splitext(os.path.basename(p))[0]}_segmentation.png")
#             if os.path.exists(m):
#                 self.train_mask_paths.append(m)
#                 valid_train.append(p)
#         self.train_img_paths = valid_train

#         # 2. Val
#         self.val_img_paths = get_images_in_dir(self.config.VAL_IMG_DIR)
#         valid_val = []
#         for p in self.val_img_paths:
#             m = os.path.join(self.config.VAL_MASK_DIR, f"{os.path.splitext(os.path.basename(p))[0]}_segmentation.png")
#             if os.path.exists(m):
#                 self.val_mask_paths.append(m)
#                 valid_val.append(p)
#         self.val_img_paths = valid_val

#         # 3. Test
#         self.test_img_paths = get_images_in_dir(self.config.TEST_IMG_DIR)
#         valid_test = []
#         for p in self.test_img_paths:
#             m = os.path.join(self.config.TEST_MASK_DIR, f"{os.path.splitext(os.path.basename(p))[0]}_segmentation.png")
#             if os.path.exists(m):
#                 self.test_mask_paths.append(m)
#                 valid_test.append(p)
#         self.test_img_paths = valid_test
        
#         print(f"Dataset Loaded - Train: {len(self.train_img_paths)}, Val: {len(self.val_img_paths)}, Test: {len(self.test_img_paths)}")

#     def load_metadata_csv(self):
#         csv_path = os.path.join(self.config.OUTPUT_DIR, 'data_analysis', 'lesion_ratios.csv')
        
#         if os.path.exists(csv_path):
#             print(f"Loading metadata from {csv_path}...")
#             df = pd.read_csv(csv_path)
#             self.category_map = dict(zip(df['filename'], df['size_category']))
#             print("Metadata loaded successfully.")
#         else:
#             print("Warning: lesion_ratios.csv not found. All images will use 'Normal' strategy.")
#             self.category_map = {}

#     def setup(self):
#         self.load_paths()
#         self.load_metadata_csv() 
        
#         # --- STRATEGY 1: TINY LESIONS (Aggressive Zoom) ---
#         self.tiny_transform = A.Compose([
#             # FIX: Updated for Albumentations v1.4.0+ (Use 'size' instead of 'height'/'width')
#             A.RandomResizedCrop(
#                 size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH), 
#                 scale=(0.2, 0.5), 
#                 ratio=(0.75, 1.33), 
#                 p=1.0 
#             ),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Rotate(limit=90, p=0.5),
#             A.RandomBrightnessContrast(p=0.5),
#             # FIX: Use .tolist()
#             A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
#             ToTensorV2()
#         ]) # type: ignore
        
#         # --- STRATEGY 2: NORMAL/LARGE LESIONS (Standard) ---
#         self.normal_transform = A.Compose([
#             A.Resize(height=self.config.IMG_HEIGHT, width=self.config.IMG_WIDTH),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.Rotate(limit=90, p=0.5),
#             A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
#             ToTensorV2()
#         ]) # type: ignore
        
#         # --- VALIDATION (No Augmentation) ---
#         self.val_transform = A.Compose([
#             A.Resize(height=self.config.IMG_HEIGHT, width=self.config.IMG_WIDTH),
#             A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
#             ToTensorV2(),
#         ]) # type: ignore
        
#         self._analyze_class_distribution()

#     def get_dataloaders(self):
#         train_dataset = ISICDataset(
#             self.config, 
#             self.train_img_paths, 
#             self.train_mask_paths, 
#             tiny_transform=self.tiny_transform,    
#             normal_transform=self.normal_transform, 
#             category_map=self.category_map,        
#             mode='train',
#             normalize=False
#         )
        
#         val_dataset = ISICDataset(
#             self.config, 
#             self.val_img_paths, 
#             self.val_mask_paths, 
#             val_transform=self.val_transform,
#             mode='val',
#             normalize=False
#         )
        
#         test_dataset = ISICDataset(
#             self.config, 
#             self.test_img_paths, 
#             self.test_mask_paths, 
#             val_transform=self.val_transform,
#             mode='test',
#             normalize=False
#         )
        
#         train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
#         val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
#         test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
#         return train_loader, val_loader, test_loader

#     def _analyze_class_distribution(self):
#         """Keep your visualization logic"""
#         sample_size = min(100, len(self.train_mask_paths))
#         if sample_size == 0: return
#         sample_paths = np.random.choice(self.train_mask_paths, sample_size, replace=False)
#         foreground_percentages = []
#         for path in sample_paths:
#             mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
#             fg_pct = np.mean(mask > 127) * 100
#             foreground_percentages.append(fg_pct)
        
#         plt.figure(figsize=(10, 6))
#         plt.hist(foreground_percentages, bins=20, color='skyblue', edgecolor='black')
#         plt.title('Distribution of Lesion Sizes')
#         plt.savefig(os.path.join(self.config.VISUALIZATION_DIR, 'lesion_size_distribution.png'))
#         plt.close()
    

##########################################################################################################################################

import os
import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from glob import glob

# --- FIX: Matplotlib Headless Mode ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# -------------------------------------

from tqdm import tqdm
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config

class ISICDataset(Dataset):
    def __init__(self, config, img_paths, mask_paths, transform=None, normalize=True):
        self.config = config
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.normalize = normalize
        
        # Stats
        self.mean = np.array([0.70809584, 0.58178357, 0.53571441])
        self.std = np.array([0.15733581, 0.16560281, 0.18079209])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Transform
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback
            img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            img = img.transpose(2, 0, 1).astype('float32') / 255.0
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask).float()

        # Safeguard Normalization
        if self.normalize and not isinstance(img, torch.Tensor):
             img = img / 255.0
             if self.mean is not None:
                 img = (img - self.mean) / (self.std + 1e-7)

        # Ensure Binary Mask
        if isinstance(mask, torch.Tensor):
             mask = (mask > 0.5).float()
             if mask.ndim == 2: mask = mask.unsqueeze(0)
        else:
             mask = (mask > 127).astype(np.float32)
             mask = torch.from_numpy(mask).unsqueeze(0)
        
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        
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
        
        self.mean = np.array([0.70809584, 0.58178357, 0.53571441])
        self.std = np.array([0.15733581, 0.16560281, 0.18079209])
        
        self.train_transform = None
        self.val_transform = None

    def load_paths(self):
        print("Loading dataset paths...")
        def get_images_in_dir(path):
            imgs = sorted(glob(os.path.join(path, '*.jpg')))
            if not imgs: imgs = sorted(glob(os.path.join(path, '*.png')))
            return imgs

        # 1. Train
        self.train_img_paths = get_images_in_dir(self.config.TRAIN_IMG_DIR)
        valid_train = []
        for p in self.train_img_paths:
            m = os.path.join(self.config.TRAIN_MASK_DIR, f"{os.path.splitext(os.path.basename(p))[0]}_segmentation.png")
            if os.path.exists(m):
                self.train_mask_paths.append(m)
                valid_train.append(p)
        self.train_img_paths = valid_train

        # 2. Val
        self.val_img_paths = get_images_in_dir(self.config.VAL_IMG_DIR)
        valid_val = []
        for p in self.val_img_paths:
            m = os.path.join(self.config.VAL_MASK_DIR, f"{os.path.splitext(os.path.basename(p))[0]}_segmentation.png")
            if os.path.exists(m):
                self.val_mask_paths.append(m)
                valid_val.append(p)
        self.val_img_paths = valid_val

        # 3. Test
        self.test_img_paths = get_images_in_dir(self.config.TEST_IMG_DIR)
        valid_test = []
        for p in self.test_img_paths:
            m = os.path.join(self.config.TEST_MASK_DIR, f"{os.path.splitext(os.path.basename(p))[0]}_segmentation.png")
            if os.path.exists(m):
                self.test_mask_paths.append(m)
                valid_test.append(p)
        self.test_img_paths = valid_test
        
        print(f"Dataset Loaded - Train: {len(self.train_img_paths)}, Val: {len(self.val_img_paths)}, Test: {len(self.test_img_paths)}")

    def setup(self):
        self.load_paths()
        
        # --- THE ZOOM STRATEGY ---
        if self.config.AUGMENTATION:
            self.train_transform = A.Compose([
                # 1. Random Resized Crop (The Zoom)
                A.RandomResizedCrop(
                    size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH), 
                    scale=(0.5, 1.0), 
                    ratio=(0.75, 1.33),
                    p=0.5
                ),
                
                # 2. Standard Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5),
                
                # 3. Color & Distortion
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                A.GridDistortion(p=0.2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
                
                # 4. Normalize & Convert
                A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
                ToTensorV2(),
            ]) 
        else:
            self.train_transform = A.Compose([
                A.Resize(height=self.config.IMG_HEIGHT, width=self.config.IMG_WIDTH),
                A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
                ToTensorV2(),
            ]) 
        
        self.val_transform = A.Compose([
            A.Resize(height=self.config.IMG_HEIGHT, width=self.config.IMG_WIDTH),
            A.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
            ToTensorV2(),
        ]) 
        
        self._analyze_class_distribution()
        self.visualize_samples()
        if self.config.AUGMENTATION:
            self.visualize_augmentations()

    def get_dataloaders(self):
        train_dataset = ISICDataset(self.config, self.train_img_paths, self.train_mask_paths, transform=self.train_transform, normalize=False)
        val_dataset = ISICDataset(self.config, self.val_img_paths, self.val_mask_paths, transform=self.val_transform, normalize=False)
        test_dataset = ISICDataset(self.config, self.test_img_paths, self.test_mask_paths, transform=self.val_transform, normalize=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader, test_loader

    def _analyze_class_distribution(self):
        """Simple visualization logic"""
        sample_size = min(100, len(self.train_mask_paths))
        if sample_size == 0: return
        sample_paths = np.random.choice(self.train_mask_paths, sample_size, replace=False)
        foreground_percentages = []
        for path in sample_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            fg_pct = np.mean(mask > 127) * 100
            foreground_percentages.append(fg_pct)
        
        plt.figure(figsize=(10, 6))
        plt.hist(foreground_percentages, bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Lesion Sizes')
        
        # --- FIX: Ensure Directory Exists ---
        os.makedirs(self.config.VISUALIZATION_DIR, exist_ok=True)
        # ------------------------------------
        
        plt.savefig(os.path.join(self.config.VISUALIZATION_DIR, 'lesion_size_distribution.png'))
        plt.close()

    def visualize_samples(self, num_samples=5):
        """Visualize random samples from the dataset"""
        if len(self.train_img_paths) == 0: return
        
        indices = np.random.choice(len(self.train_img_paths), min(num_samples, len(self.train_img_paths)), replace=False)
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i, idx in enumerate(indices):
            # Load raw image/mask
            img = cv2.imread(self.train_img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            mask = cv2.imread(self.train_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            
            # 1. Original
            plt.subplot(num_samples, 3, i*3+1)
            plt.imshow(img)
            plt.title("Original")
            plt.axis('off')
            
            # 2. Mask
            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.axis('off')
            
            # 3. Overlay
            mask_rgb = np.zeros_like(img)
            mask_rgb[:,:,1] = (mask > 127).astype(np.uint8) * 255
            overlay = cv2.addWeighted(img, 1, mask_rgb, 0.5, 0)
            
            plt.subplot(num_samples, 3, i*3+3)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis('off')
            
        plt.tight_layout()
        
        # --- FIX: Ensure Directory Exists ---
        os.makedirs(self.config.VISUALIZATION_DIR, exist_ok=True)
        # ------------------------------------
        
        plt.savefig(os.path.join(self.config.VISUALIZATION_DIR, 'dataset_samples.png'))
        plt.close()

    def visualize_augmentations(self, num_samples=3):
        """Visualize how augmentation changes the images"""
        if not self.config.AUGMENTATION or self.train_transform is None: return
        if len(self.train_img_paths) == 0: return

        indices = np.random.choice(len(self.train_img_paths), min(num_samples, len(self.train_img_paths)), replace=False)
        plt.figure(figsize=(12, 4 * num_samples))
        
        for i, idx in enumerate(indices):
            img = cv2.imread(self.train_img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.train_mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            
            # Show Original
            img_resized = cv2.resize(img, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
            plt.subplot(num_samples, 5, i*5+1)
            plt.imshow(img_resized)
            plt.title("Original")
            plt.axis('off')
            
            # Show 4 Random Augmentations
            for j in range(4):
                augmented = self.train_transform(image=img, mask=mask)
                aug_img = augmented['image'].permute(1, 2, 0).numpy()
                
                # --- Convert back to Viewable Format ---
                aug_img = (aug_img * self.std + self.mean)
                aug_img = np.clip(aug_img, 0, 1)
                aug_img = (aug_img * 255).astype(np.uint8)
                
                plt.subplot(num_samples, 5, i*5+j+2)
                plt.imshow(aug_img)
                plt.title(f"Aug {j+1}")
                plt.axis('off')
                
        plt.tight_layout()
        
        # --- FIX: Ensure Directory Exists ---
        os.makedirs(self.config.VISUALIZATION_DIR, exist_ok=True)
        # ------------------------------------
        
        plt.savefig(os.path.join(self.config.VISUALIZATION_DIR, 'augmentation_examples.png'))
        plt.close()
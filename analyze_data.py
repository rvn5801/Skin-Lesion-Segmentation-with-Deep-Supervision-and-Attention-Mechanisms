import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob
from config import Config

def analyze_dataset():
    # 1. Setup
    config = Config()
    print(f"Analyzing data in: {config.TRAIN_MASK_DIR}")
    
    # Create analysis output folder
    analysis_dir = os.path.join(config.OUTPUT_DIR, 'data_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 2. Find all masks (Ground Truth)
    # We use masks because they define exactly where the lesion is
    mask_paths = sorted(glob(os.path.join(config.TRAIN_MASK_DIR, '*.png')))
    
    if not mask_paths:
        print("ERROR: No masks found. Check your config paths.")
        return

    stats = []

    print("Calculating lesion ratios...")
    for path in tqdm(mask_paths):
        # Read mask in grayscale (0=Background, 255=Lesion)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate areas
        total_pixels = mask.size
        lesion_pixels = np.count_nonzero(mask)
        
        # Calculate Ratio (0.0 to 1.0)
        ratio = lesion_pixels / total_pixels
        
        # Categorize size
        if ratio < 0.05: size_cat = "Tiny (<5%)"
        elif ratio < 0.30: size_cat = "Medium (5-30%)"
        else: size_cat = "Large (>30%)"

        stats.append({
            'filename': os.path.basename(path),
            'lesion_pixels': lesion_pixels,
            'total_pixels': total_pixels,
            'lesion_ratio': ratio,
            'size_category': size_cat
        })

    # 3. Create DataFrame and Save CSV
    df = pd.DataFrame(stats)
    csv_path = os.path.join(analysis_dir, 'lesion_ratios.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Analysis CSV saved to: {csv_path}")

    # 4. Generate Statistics
    print("\n--- Summary Statistics ---")
    print(df['lesion_ratio'].describe())
    
    # 5. Generate Plots
    print("\nGenerating plots...")
    
    # Plot A: Histogram of Ratios
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='lesion_ratio', bins=30, kde=True, color='blue')
    plt.title(f'Distribution of Lesion Sizes (Lesion / Image Ratio)\nTotal Images: {len(df)}')
    plt.xlabel('Lesion Ratio (0.0 = Empty, 1.0 = Full Image)')
    plt.ylabel('Count')
    plt.axvline(df['lesion_ratio'].mean(), color='red', linestyle='--', label=f"Mean: {df['lesion_ratio'].mean():.2f}")
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, 'ratio_histogram.png'))
    plt.close()

    # Plot B: Boxplot by Size Category
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='lesion_ratio', data=df, color='lightblue')
    plt.title('Boxplot of Lesion Ratios')
    plt.savefig(os.path.join(analysis_dir, 'ratio_boxplot.png'))
    plt.close()
    
    
    print(f"Plots saved to: {analysis_dir}")

if __name__ == "__main__":
    analyze_dataset()
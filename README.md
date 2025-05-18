#  DS-AttentionUNet: High-Recall Skin Lesion Segmentation with Deep Learning 🔬

**Early detection of skin cancer saves lives. This repository provides a powerful deep learning model, DS-AttentionUNet, for accurately segmenting skin lesions from dermoscopic images, designed with a critical focus on high recall for clinical reliability.**

**✨ See DS-AttentionUNet in Action! ✨**

![DS-AttentionUNet Segmentation Demo](https://github.com/user-attachments/assets/67a3f4a1-be9b-4115-b7aa-2793c4dfeeb8)



---

## 📖 Table of Contents

1.  [🎯 Overview](#-overview)
2.  [🚀 Key Features](#-key-features)
3.  [🛠️ Installation](#️-installation)
4.  [💾 Dataset](#-dataset)
5.  [📂 Project Structure](#-project-structure)
6.  [▶️ Usage](#️-usage)
    * [Training](#training)
    * [Batch Prediction](#batch-prediction)
    * [Single Image Inference](#single-image-inference)

---

## 🎯 Overview

Skin cancer is a prevalent global health issue, with melanoma being particularly aggressive. **Early and accurate detection is paramount.** This project introduces **DS-AttentionUNet**, an advanced deep learning model specifically engineered for skin lesion segmentation from dermoscopic images. This segmentation is a vital initial step in computer-aided diagnosis (CAD) systems, aiming to assist clinicians in making faster and more accurate diagnoses.

Our DS-AttentionUNet model isn't just another segmentation tool. It's built with a crucial medical requirement in mind: **minimizing false negatives.**
Achieving a **Dice coefficient of $0.6611 \pm 0.2060$** and an **IoU of $0.5251 \pm 0.2065$** on the challenging ISIC 2018 dataset, it particularly shines with a **Recall of $0.9216$**. This high recall makes it a promising candidate for clinical settings where missing a potential lesion can have serious consequences.

---

## 🚀 Key Features

* **🎯 Attention Gates:** Intelligently focus on relevant lesion features while filtering out noise and irrelevant artifacts.
* **💡 Deep Supervision:** Enhances gradient flow and learning by integrating supervision signals at multiple decoder stages.
* **⚖️ Group Normalization:** Provides stable performance, especially effective with smaller batch sizes common in medical imaging.
* **⚡ Mixed Precision Training:** Accelerates training and reduces GPU memory footprint without sacrificing accuracy.
* **📈 OneCycleLR:** Implements an efficient learning rate schedule for faster convergence and better generalization.
* **🧩 Composite Loss Function:** A powerful blend of BCE, Dice, and Focal Tversky losses to tackle class imbalance and optimize for segmentation quality.
* **📊 Detailed Visualization:** Offers comprehensive performance metrics and clear visual outputs for in-depth analysis.

---

## 🛠️ Installation

Get DS-AttentionUNet up and running on your system.

### Prerequisites

* Python 3.8+
* CUDA-compatible GPU (Highly Recommended for optimal performance)

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/skin-lesion-segmentation.git](https://github.com/yourusername/skin-lesion-segmentation.git)
    cd skin-lesion-segmentation
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Core Requirements

The `requirements.txt` file includes:
* `torch>=1.12.0`
* `torchvision>=0.13.0`
* `albumentations>=1.3.0`
* `opencv-python>=4.5.5`
* `matplotlib>=3.5.1`
* `numpy>=1.22.3`
* `pandas>=1.4.1`
* `scikit-learn>=1.0.2`
* `tqdm>=4.63.0`

---

## 💾 Dataset

This project is designed for the **ISIC 2018 Skin Lesion Analysis Towards Melanoma Detection Challenge dataset**.

### Download the Dataset

1.  Go to the [ISIC Challenge 2018 website](https://challenge.isic-archive.com/data/#2018).
2.  Download the necessary archives:
    * Training Dataset (Images)
    * Training Dataset (Ground Truth)
    * Validation Dataset (Images)
    * Validation Dataset (Ground Truth)
    * Test Dataset (Images)
    * Test Dataset (Ground Truth)
3.  Extract and place the downloaded files into the directory structure shown above.

### Configuration

❗ **Important:** Update the dataset path in `config.py` to point to your local `Data` directory:

```python
# In config.py
BASE_DIR = '/path/to/your/Data'
```

📂 Project Structure
A glimpse into the organization of the codebase:
```
isic_segmentation/
├── config.py            # ⚙️ Configuration settings (paths, hyperparameters)
├── dataset.py           # 🖼️ Dataset loading, preprocessing, and augmentation
├── unet.py              # 🧠 DS-AttentionUNet model architecture
├── metrics.py           # 📈 Loss functions and evaluation metrics
├── visualization.py     # 🎨 Utilities for generating plots and visual results
├── train.py             # 🚂 Script for training the model
├── predict.py           # 💨 Script for batch prediction on multiple images
├── inference.py         # 🔎 Script for single image inference
├── main.py              # 🚀 Main entry point for all commands (train, predict, inference)
├── requirements.txt     # 📦 Python package dependencies
└── README.md            # 📄 You are here!
```
▶️ Usage
The main.py script serves as a unified interface for all operations.

Basic Command Structure
```
python main.py <command> [options]
```
Where <command> can be train, predict, or inference.
Training
Kick off the training process for DS-AttentionUNet:
```
python main.py train --batch_size 8 --epochs 50 --lr 0.0003 --img_size 384 --loss focal_tversky --augment
```
Options:

* batch_size: Number of images per batch (default: from config.py)
* epochs: Number of training epochs (default: from config.py)
* lr: Learning rate (default: from config.py)
* img_size: Input image size (e.g., 256, 384) (default: from config.py)
* loss: Loss function. Choose from ['dice', 'bce_dice', 'tversky', 'focal_tversky', 'combined'] (default: 'bce_dice')
* augment / --no-augment: Enable/disable data augmentation (default: from config.py)
* debug: Use a small subset of data for quick testing and debugging.
Alternatively, you can directly use train.py:
```
python train.py --batch_size 8 --epochs 50 --lr 0.0003 --img_size 384 --loss focal_tversky --augment
```
Batch Prediction
Generate segmentations for a directory of images:
```
python main.py predict --model_path outputs/unet_isic_TIMESTAMP/models/best_model.pth --input_dir test_images --output_dir results --img_size 384 --use_gpu
```
Single Image Inference
Perform segmentation on a single image:
```
python main.py inference --model_path outputs/unet_isic_TIMESTAMP/models/best_model.pth --input sample.jpg --output_dir results --img_size 384 --use_gpu
```

Output Directory Structure
Training artifacts and results are saved under outputs/unet_isic_TIMESTAMP/:
```
outputs/unet_isic_TIMESTAMP/
├── models/                  # Saved model checkpoints
│   ├── best_model.pth       # Model with best validation performance
│   ├── best_model_state_dict.pth # State dictionary for easy inference
│   └── last_checkpoint.pth  # Checkpoint from the last epoch
├── logs/                    # Training logs
│   └── detailed_metrics.json# Per-batch metrics (JSON format)
├── visualizations/          # Visual outputs
│   ├── metrics_history.png  # Plots of training metrics over epochs
│   ├── detailed_metrics.png # Detailed metrics visualization
│   ├── epochs/              # Sample predictions per epoch (if enabled)
│   └── test_evaluation/     # Evaluation results on the test set
│       ├── best/            # Examples of best predictions
│       ├── worst/           # Examples of worst predictions
│       ├── test_report.html # Comprehensive HTML report of test results
│       └── test_metrics.csv # Test metrics in CSV format
|__ results/                 # after running the infernece file 
|   |_ ISIC_000000_mask.png  
|   |_ ISIC_000000_overlay.png
|   |_ ISIC_000000_result.png # gives the segmentation overlay 
└── config.txt               # Copy of the configuration used for this run

```




import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import argparse
from glob import glob
from tqdm import tqdm

from config import Config
from unet import AttentionUNet
from test_time_aguments import tta_predict

def parse_args():
    parser = argparse.ArgumentParser(description='Predict using trained U-Net model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for model input')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    try:
        # Try to load the full checkpoint first
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
    except:
        # If not a checkpoint, try loading state dict directly
        model_state = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    return model

def preprocess_image(img_path, img_size):
    """Preprocess a single image for inference"""
    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_shape = img.shape[:2]
    
    # Resize for model input
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0, 1]
    img_normalized = img_resized / 255.0
    
    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img_normalized.transpose((2, 0, 1))).float().unsqueeze(0)
    
    return img, img_tensor, original_shape

def postprocess_mask(mask, original_shape):
    """Postprocess the predicted mask"""
    # Convert to numpy
    mask_np = mask.squeeze().cpu().numpy()
    
    # Resize to original shape
    mask_resized = cv2.resize(mask_np, (original_shape[1], original_shape[0]))
    
    return mask_resized

def create_overlay(img, mask, alpha=0.5, threshold=0.5):
    """Create overlay of mask on original image"""
    # Apply threshold to create binary mask
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Create colored mask (green channel)
    mask_colored = np.zeros_like(img)
    mask_colored[:,:,1] = binary_mask
    
    # Create overlay
    overlay = cv2.addWeighted(img, 1.0, mask_colored, alpha, 0)
    
    return overlay, binary_mask

def visualize_result(original_img, mask, output_path, threshold=0.5):
    """Visualize and save result"""
    # Create overlay and binary mask
    overlay, binary_mask = create_overlay(original_img, mask, threshold=threshold)
    
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Probability map
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='jet')
    plt.title('Probability Map')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Segmentation Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def post_process_mask(mask, min_size=30):
    """Apply post-processing to improve consistency"""
    # Apply threshold to create binary mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Filter out small components
    for i in range(1, num_labels):
        component_size = np.sum(labels == i)
        if component_size < min_size:
            binary_mask[labels == i] = 0
    
    # Optional: Apply morphological operations for smoother boundaries
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return binary_mask

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Get list of images
    img_paths = sorted(glob(os.path.join(args.input_dir, '*.jpg')))
    if not img_paths:
        img_paths = sorted(glob(os.path.join(args.input_dir, '*.jpeg')))
    if not img_paths:
        img_paths = sorted(glob(os.path.join(args.input_dir, '*.png')))
    
    if not img_paths:
        print(f"No images found in {args.input_dir}")
        return
    
    print(f"Found {len(img_paths)} images")
    
    # Process each image
    for img_path in tqdm(img_paths, desc="Processing images"):
        # Get image filename
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]
        
        # Preprocess image
        original_img, img_tensor, original_shape = preprocess_image(img_path, args.img_size)
        
        # # Run inference
        # with torch.no_grad():
        #     prediction = model(img_tensor.to(device))

        prediction = tta_predict(model, img_tensor, device)
        
        # Postprocess mask
        mask = postprocess_mask(prediction, original_shape)
        
        # Create binary mask
        binary_mask = (mask > args.threshold).astype(np.uint8) * 255
        
        # Save binary mask
        cv2.imwrite(
            os.path.join(args.output_dir, f"{img_name}_mask.png"),
            binary_mask
        )
        
        # Create and save overlay
        overlay, _ = create_overlay(original_img, mask, threshold=args.threshold)
        cv2.imwrite(
            os.path.join(args.output_dir, f"{img_name}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        
        # Visualize and save
        visualize_result(
            original_img,
            mask,
            os.path.join(args.output_dir, f"{img_name}_visualization.png"),
            args.threshold
        )
    
    print(f"Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()
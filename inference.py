import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import argparse
import time
from glob import glob

from unet import AttentionUNet
from test_time_aguments import tta_predict

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on a single image or directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory to save results')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for model input')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

def load_model(model_path, device):
    """Load trained model"""
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
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
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

def run_inference(model, img_path, output_dir, img_size, threshold, device):
    """Run inference on a single image"""
    try:
        # Preprocess image
        original_img, img_tensor, original_shape = preprocess_image(img_path, img_size)
        
        # Start timer
        start_time = time.time()
        
        # Run inference
        # with torch.no_grad():
        #     prediction = model(img_tensor.to(device))

        prediction = tta_predict(model, img_tensor, device)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Convert prediction to numpy and move to CPU
        mask = prediction.squeeze().cpu().numpy()
        
        # Resize to original size
        mask_original_size = cv2.resize(mask, (original_shape[1], original_shape[0]))
        
        # Create output filename
        base_filename = os.path.basename(img_path)
        name, ext = os.path.splitext(base_filename)
        output_path = os.path.join(output_dir, f"{name}_result.png")
        
        # Save binary mask
        binary_mask = (mask_original_size > threshold).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f"{name}_mask.png"), binary_mask)
        
        # Create and save overlay
        overlay = create_overlay(original_img, mask_original_size, threshold)
        cv2.imwrite(os.path.join(output_dir, f"{name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Visualize result
        visualize_result(original_img, mask_original_size, overlay, output_path)
        
        return inference_time
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def create_overlay(img, mask, threshold=0.5):
    """Create overlay of mask on original image"""
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Create colored mask (green channel)
    mask_colored = np.zeros_like(img)
    mask_colored[:,:,1] = binary_mask * 255
    
    # Create overlay
    overlay = cv2.addWeighted(img, 1.0, mask_colored, 0.5, 0)
    
    return overlay

def visualize_result(original_img, mask, overlay, output_path):
    """Visualize and save the result"""
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

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    
    # Check if input is a directory or a single image
    if os.path.isdir(args.input):
        # Get list of images
        img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_paths.extend(glob(os.path.join(args.input, ext)))
        
        if not img_paths:
            print(f"No images found in {args.input}")
            return
        
        print(f"Found {len(img_paths)} images")
        
        # Process each image
        total_time = 0
        successful = 0
        
        for i, img_path in enumerate(img_paths):
            print(f"Processing image {i+1}/{len(img_paths)}: {os.path.basename(img_path)}")
            
            inference_time = run_inference(
                model, img_path, args.output_dir, 
                args.img_size, args.threshold, device
            )
            
            if inference_time is not None:
                total_time += inference_time
                successful += 1
                print(f"Inference time: {inference_time:.4f} seconds")
        
        if successful > 0:
            avg_time = total_time / successful
            print(f"Average inference time: {avg_time:.4f} seconds per image")
        else:
            print("No images were successfully processed")
    else:
        # Process single image
        if not os.path.exists(args.input):
            print(f"Input file {args.input} does not exist")
            return
        
        print(f"Processing single image: {args.input}")
        inference_time = run_inference(
            model, args.input, args.output_dir, 
            args.img_size, args.threshold, device
        )
        
        if inference_time is not None:
            print(f"Inference time: {inference_time:.4f} seconds")
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
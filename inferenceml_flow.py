import os
import numpy as np
import cv2
import torch
import argparse
import torch 
import time 
import mlflow 
from mlflow.pytorch
from mlflow import pyfunc  
from mlflow import MlflowClient



# We don't need to import the UNet architecture here!
# MLflow's pyfunc model format bundles the model and its logic.

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference using a model from MLflow Registry')
    
    # <-- We load the model by name and stage, not path! -->
    parser.add_argument('--model_name', type=str, default="DS-AttentionUNet-Skin-Lesion", help='Name of the registered model in MLflow')
    parser.add_argument('--model_stage', type=str, default="None", help='Stage of the model to use (e.g., "Staging", "Production"). "None" gets the latest version.')
    
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='inference_results_mlflow', help='Directory to save results')
    parser.add_argument('--img_size', type=int, default=384, help='Image size for model input')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

def load_model_from_mlflow(model_name, model_stage):
    """Load a trained model from the MLflow Model Registry."""
    print(f"Loading model '{model_name}' from stage '{model_stage}'...")
    
    # <-- This is the magic. It finds and downloads the correct model version. -->
    model_uri = f"models:/{model_name}/{model_stage}"
    if model_stage == "None":
        # Get the latest version regardless of stage
        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=None)[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
        print(f"Using latest version: {latest_version.version}")
    
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model

def preprocess_image(img_path, img_size):
    """Preprocess a single image for inference"""
    # ... (This function is identical to your existing inference.py) ...
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # IMPORTANT: You must use the SAME normalization stats as in training
    # For a real MLOps pipeline, these stats (mean/std) would be saved
    # as an artifact with the model in MLflow.
    # For now, we'll hardcode them based on your dataset.py
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_normalized = img_resized / 255.0
    img_normalized = (img_normalized - mean) / (std + 1e-7)
    
    img_tensor = torch.from_numpy(img_normalized.transpose((2, 0, 1))).float().unsqueeze(0)
    return img, img_tensor, original_shape

def run_inference(model, img_tensor, device):
    """Run inference (no TTA for this example)"""
    with torch.no_grad():
        prediction = model(img_tensor.to(device))
    
    # Handle deep supervision output if model is in train mode by mistake
    if isinstance(prediction, list):
        prediction = prediction[0]
        
    return prediction.squeeze().cpu().numpy()

# ... (include post_process_mask, create_overlay, visualize_result from your inference.py) ...
# (These functions do not need to change)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model from MLflow
    model = load_model_from_mlflow(args.model_name, args.model_stage)
    model.to(device)
    
    # ... (The rest of your main() from inference.py can go here) ...
    # (Just replace the part that processes images)

    if os.path.isdir(args.input):
        print("Directory processing not shown in this example, please adapt from inference.py")
    else:
        print(f"Processing single image: {args.input}")
        original_img, img_tensor, original_shape = preprocess_image(args.input, args.img_size)
        
        start_time = time.time()
        mask_pred = run_inference(model, img_tensor, device)
        inference_time = time.time() - start_time
        
        mask_original_size = cv2.resize(mask_pred, (original_shape[1], original_shape[0]))
        
        # ... (call visualize_result, create_overlay, etc.) ...
        
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
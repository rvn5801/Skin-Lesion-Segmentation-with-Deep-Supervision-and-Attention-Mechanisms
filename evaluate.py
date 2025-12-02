import os
import torch
import pandas as pd
import mlflow
from tqdm import tqdm
import cv2                 # --- NEW ---
import numpy as np         # --- NEW ---

# Import your own project files
# from api import MODEL_NAME
from config import Config
import dataset
import metrics

## -----------------------------------------------------------------
## 1. CONFIGURATION
## -----------------------------------------------------------------
MODEL_NAME = "DS-AttentionUNet-Skin-Lesion"
RUN_ID = "bf8497596c23482b84bf21f5b1d72116" 
MLFLOW_TRACKING_URI = "sqlite:////app/mlruns/mlflow.db"
OUTPUT_CSV_FILE = "full_test_set_scores.csv"

# --- NEW ---
# Create a folder to save our visual prediction images
PREDICTION_SAVE_DIR = "test_predictions"
os.makedirs(PREDICTION_SAVE_DIR, exist_ok=True)
# --- END NEW ---

## -----------------------------------------------------------------
## 2. MAIN SCRIPT
## -----------------------------------------------------------------

def evaluate_model():
    print(f"Starting evaluation for run: {RUN_ID}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # --- 1. Load Data ---
    print("Loading test data...")
    cfg = Config() 
    data_module = dataset.ISICDataModule(cfg)
    data_module.setup()
    
    _, _, test_loader = data_module.get_dataloaders()
    test_img_paths = data_module.test_img_paths
    test_mask_paths = data_module.test_mask_paths # --- NEW ---

    if not test_img_paths:
        print("ERROR: Could not load test image paths.")
        return
    if not test_mask_paths: # --- NEW ---
        print("ERROR: Could not load test mask paths.")
        return

    # --- 2. Load Model ---
    print("Loading trained model from MLflow...")
    model_uri = f"models:/{RUN_ID}/production"
    
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model from {model_uri}")
        print(f"Error details: {e}")
        return

    # --- 3. Run Inference & Save Images ---
    print("Running inference and saving comparison images...")
    all_results = []
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")

    with torch.no_grad():
        for batch_idx, (imgs, masks) in pbar:
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()
            
            preds = model(imgs)
            
            if isinstance(preds, list):
                preds = preds[0]
            
            for i in range(imgs.size(0)):
                pred_single = preds[i]
                mask_single = masks[i]
                
                global_idx = (batch_idx * cfg.BATCH_SIZE) + i
                
                if global_idx >= len(test_img_paths):
                    continue
                    
                file_name = os.path.basename(test_img_paths[global_idx])
                
                # --- A. Calculate Metrics ---
                metrics_dict = metrics.calculate_metrics(pred_single, mask_single)
                metrics_dict['file_name'] = file_name
                all_results.append(metrics_dict)

                # --- B. Save Comparison Image (THE NEW PART) ---
                try:
                    # 1. Get original image and ground truth path
                    img_path = test_img_paths[global_idx]
                    mask_path = test_mask_paths[global_idx]
                    
                    # 2. Load and resize original image
                    img_orig = cv2.imread(img_path)
                    img_orig = cv2.resize(img_orig, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
                    
                    # 3. Load and resize ground truth mask
                    mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask_gt = cv2.resize(mask_gt, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
                    
                    # 4. Convert prediction tensor to a visual mask
                    # Squeeze (1,H,W) to (H,W), move to CPU, convert to numpy
                    pred_mask = pred_single.squeeze().cpu().numpy()
                    # Apply threshold (0.5) to get 0s and 1s
                    pred_mask = (pred_mask > 0.5).astype(np.uint8)
                    # Convert to 0-255 range for saving
                    pred_mask = pred_mask * 255
                    
                    # 5. Create color overlays
                    # Green for Ground Truth
                    gt_color = np.zeros_like(img_orig)
                    gt_color[:, :, 1] = mask_gt # Set Green channel
                    
                    # Red for Prediction
                    pred_color = np.zeros_like(img_orig)
                    pred_color[:, :, 2] = pred_mask # Set Red channel
                    
                    # 6. Stack images side-by-side: [Original] | [Truth] | [Prediction]
                    comparison_img = np.hstack([img_orig, gt_color, pred_color])
                    
                    # 7. Save the stacked image
                    save_name = f"{os.path.splitext(file_name)[0]}_comparison.png"
                    save_path = os.path.join(PREDICTION_SAVE_DIR, save_name)
                    cv2.imwrite(save_path, comparison_img)

                except Exception as e:
                    print(f"\nWarning: Failed to save comparison image for {file_name}. Error: {e}")
                # --- END NEW PART ---


    # --- 4. Create and Save CSV ---
    print("Inference complete. Saving results to CSV...")
    
    if not all_results:
        print("ERROR: No results were generated.")
        return
        
    results_df = pd.DataFrame(all_results)
    cols = ['file_name'] + [col for col in results_df.columns if col != 'file_name']
    results_df = results_df[cols]
    results_df = results_df.sort_values(by='dice', ascending=True)
    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"Successfully saved all scores to {OUTPUT_CSV_FILE}")
    print(f" Successfully saved all comparison images to {PREDICTION_SAVE_DIR}")
    print("\n--- Worst 5 Performing Images (see folder for details) ---")
    print(results_df.head(5))

# --- This makes the script runnable ---
if __name__ == "__main__":
    evaluate_model()
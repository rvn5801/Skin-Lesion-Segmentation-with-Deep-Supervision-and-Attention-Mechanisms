import os
import torch
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from tqdm import tqdm
import cv2
import numpy as np
import argparse

# Import your own project files
from config import Config
import dataset
import metrics
from unet import AttentionUNet

## -----------------------------------------------------------------
## 1. CONFIGURATION
## -----------------------------------------------------------------
MODEL_NAME = "DS-AttentionUNet-Skin-Lesion"
MLFLOW_TRACKING_URI = "sqlite:////app/mlruns/mlflow.db"
OUTPUT_BASE_DIR = "outputs" 

## -----------------------------------------------------------------
## 2. HELPER FUNCTIONS
## -----------------------------------------------------------------

# def get_latest_run_id(experiment_name):
#     """Fetches the latest run ID (Finished OR Failed/Killed)."""
#     client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
#     experiment = client.get_experiment_by_name(experiment_name)
#     if experiment is None:
#         print(f"Error: Experiment '{experiment_name}' not found.")
#         return None
    
#     # --- FIX: REMOVED filter_string="status = 'FINISHED'" ---
#     # Now it grabs the latest run, even if you stopped it early.
#     runs = client.search_runs(
#         experiment_ids=[experiment.experiment_id],
#         order_by=["start_time DESC"],
#         max_results=1
#     )
    
#     if not runs:
#         print(f"Error: No runs found for experiment '{experiment_name}'.")
#         return None
    
#     latest_run = runs[0]
#     print(f"Found latest run: {latest_run.info.run_id} (Status: {latest_run.info.status})")
#     return latest_run.info.run_id

def get_latest_run_id(experiment_name):
    """
    1. Tries to find the latest FINISHED run.
    2. If none found, finds the latest run of ANY status (e.g. KILLED).
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found.")
        return None

    # --- ATTEMPT 1: Find a Completed (Success) Run ---
    finished_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if finished_runs:
        run = finished_runs[0]
        print(f"Found latest SUCCESSFUL run: {run.info.run_id}")
        return run.info.run_id

    # --- ATTEMPT 2: Fallback to Any Run (Killed/Failed) ---
    print("No finished runs found. Falling back to latest INCOMPLETE run...")
    any_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        # No status filter here
        order_by=["start_time DESC"],
        max_results=1
    )

    if any_runs:
        run = any_runs[0]
        print(f"Found latest (Incomplete) run: {run.info.run_id} (Status: {run.info.status})")
        return run.info.run_id
    
    print(f"Error: No runs found at all for experiment '{experiment_name}'.")
    return None

## -----------------------------------------------------------------
## 3. MAIN SCRIPT
## -----------------------------------------------------------------

def evaluate_model(run_id=None):
    print(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # --- 0. Determine Run ID ---
    if not run_id:
        print(f"No Run ID provided. Fetching latest for experiment: {MODEL_NAME}")
        run_id = get_latest_run_id(MODEL_NAME)
        if not run_id:
            print("Aborting evaluation: Could not find a valid Run ID.")
            return 

    print(f"Starting evaluation for Run ID: {run_id}")
    
    # Setup Output Directories
    eval_output_dir = os.path.join(OUTPUT_BASE_DIR, f"eval_{run_id}")
    prediction_save_dir = os.path.join(eval_output_dir, "test_predictions")
    os.makedirs(prediction_save_dir, exist_ok=True)
    
    output_csv_file = os.path.join(eval_output_dir, "full_test_set_scores.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    print("Loading test data...")
    cfg = Config() 
    
    # Check if we need to ensure visualization dir exists for dataset.py
    os.makedirs(cfg.VISUALIZATION_DIR, exist_ok=True)

    data_module = dataset.ISICDataModule(cfg)
    data_module.setup()
    
    _, _, test_loader = data_module.get_dataloaders()
    test_img_paths = data_module.test_img_paths
    test_mask_paths = data_module.test_mask_paths

    if not test_img_paths:
        print("ERROR: Could not load test image paths.")
        return
    if not test_mask_paths:
        print("ERROR: Could not load test mask paths.")
        return
    
    print(f"Found {len(test_img_paths)} test images.")

    # --- 2. Load Model ---
    print("Loading trained model from MLflow...")
    model_uri = f"runs:/{run_id}/model" 

    # model_uri = f"models:/{MODEL_NAME}/Latest"  # Loads the highest version number
    
    # OR THIS (If you tagged a model as 'Production' in MLflow UI):
    # model_uri = f"models:/{MODEL_NAME}/Production"
    
    print(f"Loading model from Registry: {model_uri}")
    
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
    
    current_idx = 0 
    pbar = tqdm(test_loader, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for imgs, masks in pbar:
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()
            
            preds = model(imgs)
            
            if isinstance(preds, list):
                preds = preds[0]
            
            batch_size = imgs.size(0)

            for i in range(batch_size):
                global_idx = current_idx + i
                if global_idx >= len(test_img_paths): break

                pred_single = preds[i]
                mask_single = masks[i]
                
                file_path = test_img_paths[global_idx]
                file_name = os.path.basename(file_path)
                
                # Metrics
                metrics_dict = metrics.calculate_metrics(pred_single, mask_single)
                metrics_dict['file_name'] = file_name
                all_results.append(metrics_dict)

                # Visualization
                try:
                    img_path = test_img_paths[global_idx]
                    mask_path = test_mask_paths[global_idx]
                    
                    img_orig = cv2.imread(img_path)
                    if img_orig is None: continue
                    img_orig = cv2.resize(img_orig, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
                    
                    mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_gt is None:
                         mask_gt = (mask_single.cpu().numpy().squeeze() * 255).astype(np.uint8)
                    else:
                        mask_gt = cv2.resize(mask_gt, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
                    
                    pred_prob = torch.sigmoid(pred_single).squeeze().cpu().numpy()
                    pred_mask = (pred_prob > 0.5).astype(np.uint8) * 255
                    
                    gt_color = np.zeros_like(img_orig)
                    gt_color[:, :, 1] = mask_gt # Green
                    
                    pred_color = np.zeros_like(img_orig)
                    pred_color[:, :, 2] = pred_mask # Red
                    
                    alpha = 0.3
                    gt_overlay = cv2.addWeighted(img_orig, 1, gt_color, alpha, 0)
                    pred_overlay = cv2.addWeighted(img_orig, 1, pred_color, alpha, 0)

                    comparison_img = np.hstack([img_orig, gt_overlay, pred_overlay])
                    
                    save_name = f"{os.path.splitext(file_name)[0]}_comparison.png"
                    save_path = os.path.join(prediction_save_dir, save_name)
                    cv2.imwrite(save_path, comparison_img)

                except Exception as e:
                    print(f"\nWarning: Failed to save comparison image for {file_name}. Error: {e}")

            current_idx += batch_size

    # --- 4. Create and Save CSV ---
    print("Inference complete. Saving results to CSV...")
    
    if not all_results:
        print("ERROR: No results were generated.")
        return
        
    results_df = pd.DataFrame(all_results)
    
    cols = ['file_name'] + [col for col in results_df.columns if col != 'file_name']
    results_df = results_df[cols]
    results_df = results_df.sort_values(by='dice', ascending=True)
    
    results_df.to_csv(output_csv_file, index=False)
    
    print(f"Successfully saved all scores to {output_csv_file}")
    print(f"Successfully saved all comparison images to {prediction_save_dir}")
    print("\n--- Worst 5 Performing Images ---")
    print(results_df.head(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--run_id", type=str, default=None, help="MLflow Run ID")
    args = parser.parse_args()
    
    evaluate_model(args.run_id)
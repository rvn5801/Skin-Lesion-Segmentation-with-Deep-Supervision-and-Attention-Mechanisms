import io
import os
import cv2
import numpy as np
import torch
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse # <--- ADDED StreamingResponse
from PIL import Image
import uvicorn

# Import Config
from config import Config

# --- PROMETHEUS SETUP ---
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False
    class Instrumentator:
        def __init__(self, *args, **kwargs): pass
        def instrument(self, app): return self
        def expose(self, app, endpoint: str = None): return self

# --- CONFIGURATION ---
MLFLOW_TRACKING_URI = "sqlite:////app/mlruns/mlflow.db"
MODEL_NAME = "DS-AttentionUNet-Skin-Lesion"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
DATA_MEAN = np.array([0.70809584, 0.58178357, 0.53571441])
DATA_STD = np.array([0.15733581, 0.16560281, 0.18079209])

model = None

app = FastAPI(
    title="Skin Lesion Visualizer API",
    description="Returns the image with a RED overlay where the lesion is detected."
)

# Enable Metrics
Instrumentator().instrument(app).expose(app)

# --- HELPER FUNCTIONS ---

def get_latest_run_id(experiment_name):
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None: return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    return runs[0].info.run_id if runs else None

def load_model():
    global model
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = get_latest_run_id(MODEL_NAME)
    
    if not run_id:
        print("CRITICAL WARNING: No Run ID found.")
        return

    try:
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model", map_location=DEVICE)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy"})
    return {"status": "healthy", "mode": "Visual (Returns PNG)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Read Image
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image_pil)

    # 2. Preprocess
    # Resize to model size (e.g., 384x384)
    img_resized = cv2.resize(image_np, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
    
    # Normalize for Model
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - DATA_MEAN) / (DATA_STD + 1e-7)
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    # 3. Inference
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, list): output = output[0]
        prob = torch.sigmoid(output)
        
        # Get binary mask (0 or 1) and move to CPU numpy
        print(f"DEBUG: Max Confidence Score for this image: {prob.max().item()}")
        # mask = (prob > 0.6).float()
        mask = (prob > 0.6).float().squeeze().cpu().numpy()

    # --- 4. VISUALIZATION LOGIC (NEW) ---
    
    # Create a red layer (Same size as resized image)
    heatmap = np.zeros_like(img_resized)
    heatmap[:, :, 0] = mask * 255  # Set Red channel to 255 where mask is 1

    # Blend: 60% Original Image + 40% Red Heatmap
    # We only apply the red color where the mask is positive to avoid darkening the background
    overlay = img_resized.copy()
    mask_bool = mask > 0
    
    # Apply red tint only to lesion pixels
    if mask_bool.any():
        overlay[mask_bool] = cv2.addWeighted(
            img_resized[mask_bool], 0.6, 
            heatmap[mask_bool], 0.4, 
            0
        )

    # 5. Encode to PNGre
    # Convert RGB back to BGR (OpenCV standard) before encoding
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".png", overlay_bgr)

    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    # Return the image stream
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
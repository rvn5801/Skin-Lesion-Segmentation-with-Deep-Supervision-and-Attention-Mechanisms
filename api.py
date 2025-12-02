import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import mlflow
import io
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for environments without the package
    PROMETHEUS_AVAILABLE = False
    # Provide a tiny no-op fallback so the API can run without Prometheus installed
    class Instrumentator:
        def __init__(self, *args, **kwargs):
            pass
        def instrument(self, app):
            return self
        def expose(self, app, endpoint: str = None):
            return self
from unet import AttentionUNet

# --- Configuration ---
MODEL_NAME = "DS-AttentionUNet-Skin-Lesion"
RUN_ID = "bf8497596c23482b84bf21f5b1d72116"
MODEL_NAME = "DS-AttentionUNet-Skin-Lesion"
MLFLOW_TRACKING_URI = "sqlite:////app/mlruns/mlflow.db" 

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-processing: Must be IDENTICAL to your validation/test set
# These are the stats from your dataset.py
DATA_MEAN = (0.70809584, 0.58178357, 0.53571441)
DATA_STD = (0.15733581, 0.16560281, 0.18079209)
IMG_HEIGHT = 256
IMG_WIDTH = 256

# --- MLflow Model Loading ---
def load_model(model_name):
    """
    Load the trained PyTorch model from the MLflow Model Registry.
    """
    print(f"Loading model from run_id: {model_name}")
    try:
        # This URI format points to the model artifact within the run
        model_uri = f"models:/{model_name}/1"
        model = mlflow.pytorch.load_model(model_uri)
        model = model.to(DEVICE)
        model.eval()
        print(f"Model successfully loaded to {DEVICE} and set to eval() mode.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("---")
        print("Troubleshooting Tips:")
        print(f"1. Is MLFLOW_TRACKING_URI correct? We are using: {MLFLOW_TRACKING_URI}")
        print(f"2. Does the model '{model_name}' exist?")
        print("3. Is the model stage ('production' or a version number) correct?")
        raise e

# --- Pre-processing Function ---
def preprocess_image(image_bytes: bytes):
    """
    Converts raw image bytes to a normalized tensor.
    """
    # 1. Decode image bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Define and apply transformations
    transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=DATA_MEAN, std=DATA_STD),
        ToTensorV2(),
    ])
    
    augmented = transform(image=img)
    image_tensor = augmented['image']
    
    # 3. Add batch dimension and send to device
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    return image_tensor

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Skin Lesion Segmentation API",
    description="Serves a DS-AttentionUNet model trained with MLflow."
)

# This line creates the /metrics endpoint
Instrumentator().instrument(app).expose(app)

# Load the model on startup
model = load_model(MODEL_NAME)

@app.get("/")
def read_root():
    return {"message": "Model API is running. Post an image to /predict"}

@app.post("/predict", response_class=StreamingResponse)
async def predict(file: UploadFile = File(...)):
    """
    Receives an image, performs segmentation, and returns the binary mask.
    """
    # 1. Read and preprocess the image
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    
    # 2. Run model inference
    with torch.no_grad():
        pred_mask = model(image_tensor)
        
    # 3. Post-process the output
    # Apply sigmoid, threshold, and convert to NumPy
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float() # Threshold to binary
    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255 # 0 or 255
    
    # 4. Encode the mask as a PNG
    _, img_encoded = cv2.imencode(".png", pred_mask_np)
    
    # 5. Return the image as a stream
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/png")
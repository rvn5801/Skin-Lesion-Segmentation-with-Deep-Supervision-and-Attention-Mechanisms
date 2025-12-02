import os
import torch
from typing import List, Tuple # <-- FIX: Import typing for better hints

class Config:
    # Paths
    BASE_DIR: str = r'/app/SkinLession_data' # <-- FIX: Added : str
    TRAIN_IMG_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Training_Input') # <-- FIX: Added : str
    TRAIN_MASK_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1_Training_GroundTruth') # <-- FIX: Added : str
    VAL_IMG_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Validation_Input') # <-- FIX: Added : str
    VAL_MASK_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1_Validation_GroundTruth') # <-- FIX: Added : str
    TEST_IMG_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Test_Input') # <-- FIX: Added : str
    TEST_MASK_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1_Test_GroundTruth') # <-- FIX: Added : str
    
    # Output directories (these are the defaults)
    OUTPUT_DIR: str = 'outputs' # <-- FIX: Added : str
    MODEL_DIR: str = os.path.join(OUTPUT_DIR, 'models') # <-- FIX: Added : str
    LOG_DIR: str = os.path.join(OUTPUT_DIR, 'logs') # <-- FIX: Added : str
    VISUALIZATION_DIR: str = os.path.join(OUTPUT_DIR, 'visualizations') # <-- FIX: Added : str
    
    # --- FIX: Add new attributes for MLflow-based runs ---
    # These will be set at runtime in train.py
    LOCAL_RUN_DIR: str | None = None
    EXPERIMENT_NAME: str | None = None
    RUN_NAME: str | None = None
    
    # Create directories if they don't exist
    # This logic is fine, but it's better to do this in train.py
    # We'll leave it for now.
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Image parameters
    IMG_WIDTH: int = 384
    IMG_HEIGHT: int = 384
    IMG_CHANNELS: int = 3
    
    # Training parameters
    BATCH_SIZE: int = 8
    EPOCHS: int = 50
    LEARNING_RATE: float = 3e-4
    VALIDATION_SPLIT: float = 0.2
    
    # Augmentation parameters
    AUGMENTATION: bool = True
    ROTATION_RANGE: int = 20
    HORIZONTAL_FLIP: bool = True
    VERTICAL_FLIP: bool = True
    BRIGHTNESS_RANGE: List[float] = [0.8, 1.2] # <-- FIX: Used List for typing
    
    # Model parameters
    INITIAL_FILTERS: int = 32
    DROPOUT_RATE: float = 0.4
    DEEP_SUPERVISION: bool = True
    USE_GROUP_NORM: bool = True
    MIXED_PRECISION: bool = True
    
    # Callbacks
    EARLY_STOPPING_PATIENCE: int = 10
    SCHEDULER_PATIENCE: int = 5
    SCHEDULER_FACTOR: float = 0.2
    
    # Device
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Metrics
    DICE_SMOOTH: float = 1e-5
    
    # Visualization
    SAMPLES_TO_VISUALIZE: int = 4
    VISUALIZE_FREQUENCY: int = 5
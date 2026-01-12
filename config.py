import os
import torch
from datetime import datetime
from typing import List

class Config:
    # --- 1. PATHS & ORGANIZATION ---
    BASE_DIR: str = r'/app/SkinLession_data'
    
    # Input Directories
    TRAIN_IMG_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Training_Input')
    TRAIN_MASK_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1_Training_GroundTruth')
    VAL_IMG_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Validation_Input')
    VAL_MASK_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1_Validation_GroundTruth')
    TEST_IMG_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Test_Input')
    TEST_MASK_DIR: str = os.path.join(BASE_DIR, 'ISIC2018_Task1_Test_GroundTruth')
    
    # --- THE FIX IS HERE ---
    # We calculate the strings, but we REMOVED os.makedirs() from the global scope.
    TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_DIR: str = os.path.join('outputs', TIMESTAMP)
    
    MODEL_DIR: str = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR: str = os.path.join(OUTPUT_DIR, 'logs')
    VISUALIZATION_DIR: str = os.path.join(OUTPUT_DIR, 'visualizations')
    
    # MLflow Settings
    LOCAL_RUN_DIR: str = OUTPUT_DIR
    EXPERIMENT_NAME: str = "DS-AttentionUNet-Skin-Lesion"
    RUN_NAME: str = f"run_{TIMESTAMP}"
    
    # --- 2. PARAMETERS (Your Original Settings) ---
    IMG_WIDTH: int = 256
    IMG_HEIGHT: int = 256
    IMG_CHANNELS: int = 3
    BATCH_SIZE: int = 16
    EPOCHS: int = 100
    LEARNING_RATE: float = 3e-4
    VALIDATION_SPLIT: float = 0.2
    AUGMENTATION: bool = True
    ROTATION_RANGE: int = 20
    HORIZONTAL_FLIP: bool = True
    VERTICAL_FLIP: bool = True
    BRIGHTNESS_RANGE: List[float] = [0.8, 1.2]
    INITIAL_FILTERS: int = 32
    DROPOUT_RATE: float = 0.4
    DEEP_SUPERVISION: bool = True
    USE_GROUP_NORM: bool = True
    MIXED_PRECISION: bool = True
    EARLY_STOPPING_PATIENCE: int = 10
    SCHEDULER_PATIENCE: int = 5
    SCHEDULER_FACTOR: float = 0.2
    DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DICE_SMOOTH: float = 1e-5
    SAMPLES_TO_VISUALIZE: int = 4
    VISUALIZE_FREQUENCY: int = 5

    # --- NEW FUNCTION ---
    # This acts as the "Trigger". Only the service that calls this will create folders.
    def create_directories(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.VISUALIZATION_DIR, exist_ok=True)
        print(f"Directory structure created at: {self.OUTPUT_DIR}")
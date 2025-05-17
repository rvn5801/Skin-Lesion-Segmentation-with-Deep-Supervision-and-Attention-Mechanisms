import os
import torch

class Config:
    # Paths
    BASE_DIR = r'C:\\Users\\alone\\OneDrive\\Desktop\\MEd\\Data'
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Training_Input')
    TRAIN_MASK_DIR = os.path.join(BASE_DIR, 'ISIC2018_Task1_Training_GroundTruth')
    VAL_IMG_DIR = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Validation_Input')
    VAL_MASK_DIR = os.path.join(BASE_DIR, 'ISIC2018_Task1_Validation_GroundTruth')
    TEST_IMG_DIR = os.path.join(BASE_DIR, 'ISIC2018_Task1-2_Test_Input')
    TEST_MASK_DIR = os.path.join(BASE_DIR, 'ISIC2018_Task1_Test_GroundTruth')
    
    # Output directories
    OUTPUT_DIR = 'outputs'
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    
    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Image parameters
    IMG_WIDTH = 384
    IMG_HEIGHT = 384
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    VALIDATION_SPLIT = 0.2
    
    # Augmentation parameters
    AUGMENTATION = True
    ROTATION_RANGE = 20
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    BRIGHTNESS_RANGE = [0.8, 1.2]
    
    # Model parameters
    INITIAL_FILTERS = 32
    DROPOUT_RATE = 0.4
    DEEP_SUPERVISION = True
    USE_GROUP_NORM = True
    MIXED_PRECISION = True
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 10
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.2
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Metrics
    DICE_SMOOTH = 1e-5
    
    # Visualization
    SAMPLES_TO_VISUALIZE = 4
    VISUALIZE_FREQUENCY = 5  # Visualize every N epochs
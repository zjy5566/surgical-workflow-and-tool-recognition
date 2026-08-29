"""
Statement on GenAI: 
This code was developed with the assistance of Gemini. 
GenAI assisted in centralizing hyperparameters and path configurations 
for the manual image-based data pipeline.
"""

class Config:
    # --- Dataset Basic Configuration ---
    NUM_PHASES = 7
    NUM_TOOLS = 7
    
    # Phase names defined according to Cholec80 paper standards (for txt parsing)
    PHASE_NAMES = [
        "Preparation", "CalotTriangleDissection", "ClippingCutting", 
        "GallbladderDissection", "GallbladderPackaging", "CleaningCoagulation", 
        "GallbladderRetraction"
    ]
    
    TOOL_NAMES = [
        'Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag'
    ]

    # --- Temporal Model Configuration (Task A) ---
    SEQ_LENGTH = 10         # Length of sliding window (e.g., observing 10 frames per step)
    BATCH_SIZE = 64        # Image reading is memory-intensive; adjust based on hardware
    NUM_WORKERS = 16        # Number of DataLoader subprocesses

    # --- Image Processing Configuration ---
    # Standardize resolution for ResNet-based models
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    
    # --- Training Hyperparameters ---
    LEARNING_RATE_A = 1e-4
    LEARNING_RATE_B = 1e-5
    NUM_EPOCHS_A = 50
    NUM_EPOCHS_B = 50
    
    # --- Task Specific Configuration ---
    END_PHASE_ID = 7        # Identifier for surgery end or unannotated state
    
cfg = Config()
# Global configuration variables

BATCH_SIZE = 16
MAX_EPOCHS = 50
DATASET_PATH = r"E:\github repos\porn_ai_analyser\app\datasets\pornstar_images"
PERFORMER_DATA_PATH = r"E:\github repos\porn_ai_analyser\app\datasets\performers_details_data.json"
OUTPUT_DATASET_PATH = r"E:\github repos\porn_ai_analyser\app\datasets\performer_images_with_metadata.npy"
MODEL_SAVE_PATH = "performer_recognition_model"
CHECKPOINT_DIR = "model_checkpoints"
UNFREEZE_COUNT = 8
DATA_AUGMENTATION_EPOCH_THRESHOLD = 75  # threshold percentage

# Data pipeline settings
SHUFFLE_BUFFER_SIZE = 1000
NUM_PARALLEL_CALLS = 2
PREFETCH_BUFFER_SIZE = 2

# Learning rate schedule settings
INITIAL_LR = 0.003
WARMUP_STEPS = 5000
DECAY_STEPS = 100000
DECAY_RATE = 0.001

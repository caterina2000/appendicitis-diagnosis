# === File Path ===
DATA_PATH = 'data/raw/app_data.xlsx'
IMAGE_FOLDER = 'data/raw/US_Pictures'
OUTPUT_FOLDER = 'data/processed/US_Pictures_new'

# === Images Path ===
IMAGE_SIZE = (224,224)

# === Training parameters ===
BATCH_SIZE = 32
EPOCHS_INITIAL = 7
EPOCHS_FINE_TUNE = 5
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
PATIENCE = 3
CLASS_WEIGHT = None #lo gestisco dinamicamente

# === Test parameters ===
TEST_SIZE = 0.4
RANDOM_STATE = 42

# === Path to save model ===
MODEL_PATH = 'outputs/models/best_model.h5'
LOG_DIR = 'outputs/logs'
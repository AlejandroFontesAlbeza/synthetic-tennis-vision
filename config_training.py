from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


### TRAINING PATHS
# Define dataset paths
DATA_DIR = ROOT_DIR / "data"

# Paths UE dataset -> Preprocessed dataset; pre_processed_dataset.bat

UE_IMAGES_DIR = DATA_DIR / "renderUE" / "images"
UE_MASKS_DIR = DATA_DIR / "renderUE" / "masks"
PREPROCESSED_IMAGES_DIR = DATA_DIR / "datasetPreProcessed" / "images"
PREPROCESSED_MASKS_DIR = DATA_DIR / "datasetPreProcessed" / "masks"


# Paths dataset Structured -> train & valid; dataset_splitter.bat
DATASET_IMAGES_TRAIN_DIR = DATA_DIR / "dataset" / "train" / "images"
DATASET_MASKS_TRAIN_DIR = DATA_DIR / "dataset" / "train" / "masks"
DATASET_IMAGES_VALID_DIR = DATA_DIR / "dataset" / "valid" / "images"
DATASET_MASKS_VALID_DIR = DATA_DIR / "dataset" / "valid" / "masks"


#Define model paths finetuning and saving new model
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "unet_modelV0.pth"
NEW_MODEL_PATH = MODELS_DIR / "unet_modelNew.pth"





### TESTING PATHS

TEST_DATASET_IMAGES_TRAIN_DIR = DATA_DIR / "test_rsc" / "train" / "images"
TEST_DATASET_MASKS_TRAIN_DIR = DATA_DIR / "test_rsc" / "train" / "masks"
TEST_DATASET_IMAGES_VALID_DIR = DATA_DIR / "test_rsc" / "valid" / "images"
TEST_DATASET_MASKS_VALID_DIR = DATA_DIR / "test_rsc" / "valid" / "masks"

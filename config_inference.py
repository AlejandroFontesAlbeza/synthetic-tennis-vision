from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

### INFERENCE PATHS
INFERENCE_VIDEO_PATH = ROOT_DIR / "data" / "tennisMatch" / "clips" / "clip4.mp4"
INFERENCE_MODEL_PATH = ROOT_DIR / "models" / "unet_modelV1.pth"


OUTPUT_DIR = ROOT_DIR / "data" / "inference_info"
OUTPUT_GIF_NAME = "inference_output.gif"

IP = "127.0.0.1"
PORT = 5005



# config.py — lightweight configuration
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = BASE_DIR / "samples"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

SR = 44100
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024

BATCH_SIZE = 8
LR = 1e-4
NUM_EPOCHS = 100

DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
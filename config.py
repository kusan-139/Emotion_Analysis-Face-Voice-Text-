import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMBINED_DATASET_DIR = os.path.join(BASE_DIR, "dataset_combined")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.pth")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
TRAINING_PLOTS_DIR = os.path.join(BASE_DIR, "training_plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-5
NUM_CLASSES = 7
INPUT_SIZE = 224

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion mapping (index -> label)
IDX2EMOTION = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}
EMOTION2IDX = {v: k for k, v in IDX2EMOTION.items()}

# ==========================
# Text emotion (MELD) config
# ==========================
TEXT_DATA_DIR = os.path.join(DATA_DIR, "text_meld")
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(TEXT_MODEL_DIR, exist_ok=True)

# ==========================
# Audio emotion (SAVEE) config
# ==========================
AUDIO_DATA_DIR = os.path.join(DATA_DIR, "savee")
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(AUDIO_MODEL_DIR, exist_ok=True)
AUDIO_SAMPLE_RATE = 16000

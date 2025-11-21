import os
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import librosa

from config import AUDIO_SAMPLE_RATE, AUDIO_MODEL_DIR

# SAVEE filename emotion codes:
# 'a' = anger, 'd' = disgust, 'f' = fear,
# 'h' = happiness, 'n' = neutral, 'sa' = sadness, 'su' = surprise

EMOTION_MAP_AUDIO = {
    "a": 0,   # angry
    "d": 1,   # disgust
    "f": 2,   # fear
    "h": 3,   # happy
    "sa": 4,  # sad
    "su": 5,  # surprise
    "n": 6,   # neutral
}

IDX2EMO_AUDIO = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


import re

def extract_emotion_from_filename(fname: str) -> Optional[int]:
    """
    Extract emotion from SAVEE filename.
    Expected pattern:  <speaker>_<code><number>.wav
    where code in {a, d, f, h, n, sa, su}
    Examples:
        DC_a01.wav  -> a  (anger)
        KL_sa02.wav -> sa (sad)
        JE_su07.wav -> su (surprise)
    """
    name = fname.lower()
    # Look for underscore + code + digit: _(sa|su|a|d|f|h|n)digit
    m = re.search(r"_(sa|su|a|d|f|h|n)\d", name)
    if not m:
        return None
    code = m.group(1)
    return EMOTION_MAP_AUDIO.get(code)



class SAVEEAudioDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        sr: int = AUDIO_SAMPLE_RATE,
        n_mels: int = 64,
        duration: float = 3.0,
    ):
        """
        file_paths: list of wav paths
        labels: list of integer emotion indices (0..6)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.max_len = int(sr * duration)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        y, sr = librosa.load(path, sr=self.sr)
        if len(y) > self.max_len:
            y = y[: self.max_len]
        else:
            pad_len = self.max_len - len(y)
            y = np.pad(y, (0, pad_len))

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_mels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # shape [n_mels, time] -> [1, n_mels, time]
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()

        return mel_tensor, torch.tensor(label, dtype=torch.long)


class AudioCNN(nn.Module):
    """
    Simple CNN on log-mel spectrograms.
    Output order matches IDX2EMO_AUDIO.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [B, 1, n_mels, time]
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_audio_model(device: Optional[torch.device] = None):
    """
    Used at inference time in app.py when you want to
    use your OWN trained model instead of HF fallback.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=7)
    state_dict = torch.load(os.path.join(AUDIO_MODEL_DIR, "audio_model.pth"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

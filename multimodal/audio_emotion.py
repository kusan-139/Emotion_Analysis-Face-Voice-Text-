# multimodal/audio_emotion.py

"""
Audio emotion classification with mode switching (OFFLINE ONLY).

Modes:
- "fast"     -> prefer your local CNN (models/audio_model.pth), fallback to HF local
- "accurate" -> prefer HF local (models/hf_audio_model/), fallback to your CNN

No online models are used.
"""

import os
from functools import lru_cache
from typing import Dict

import numpy as np
import av 
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline

from config import DEVICE, AUDIO_SAMPLE_RATE
from multimodal.audio_model import AudioCNN  # your CNN architecture


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

LOCAL_CNN_PATH = os.path.join("models", "audio_model.pth")
LOCAL_HF_DIR   = os.path.join("models", "hf_audio_model")  # saved HF model directory


# ----------------------------
# Helpers: audio preprocessing
# ----------------------------
def _load_and_mel(file_path: str) -> torch.Tensor:
    """
    Load audio file using PyAV (no system FFmpeg required) and convert to mel spectrogram.
    """
    # --- FIXED PYAV LOADING LOGIC ---
    try:
        container = av.open(file_path)
        # Resample immediately to target rate, mono, float32
        resampler = av.audio.resampler.AudioResampler(format='flt', layout='mono', rate=AUDIO_SAMPLE_RATE)
        
        frames = []
        for frame in container.decode(audio=0):
            frame.pts = None 
            # resample() returns a LIST of frames. We must iterate over it.
            resampled_frames = resampler.resample(frame)
            for r_frame in resampled_frames:
                frames.append(r_frame.to_ndarray()[0])
            
        if not frames:
            raise ValueError("Audio file is empty")
            
        y = np.concatenate(frames)
    except Exception as e:
        print(f"Error reading audio with av: {e}")
        raise RuntimeError(f"Could not decode audio file: {file_path}")
    # -------------------------------

    max_len = int(AUDIO_SAMPLE_RATE * 3.0)  # 3 seconds
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=AUDIO_SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return mel_tensor


# ----------------------------
# Model loaders
# ----------------------------
@lru_cache(maxsize=1)
def _load_cnn_model() -> torch.nn.Module | None:
    if not os.path.exists(LOCAL_CNN_PATH):
        return None

    model = AudioCNN(num_classes=len(EMOTIONS))
    state = torch.load(LOCAL_CNN_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("[Audio] Using local CNN model:", LOCAL_CNN_PATH)
    return model


@lru_cache(maxsize=1)
def _load_hf_local_pipeline():
    if not os.path.isdir(LOCAL_HF_DIR):
        return None

    extractor = AutoFeatureExtractor.from_pretrained(LOCAL_HF_DIR)
    model = AutoModelForAudioClassification.from_pretrained(LOCAL_HF_DIR)

    device_idx = 0 if DEVICE.type == "cuda" else -1
    clf = pipeline(
        "audio-classification",
        model=model,
        feature_extractor=extractor,
        top_k=None,
        device=device_idx,
    )
    print("[Audio] Using HF LOCAL model from", LOCAL_HF_DIR)
    return clf


# ----------------------------
# Inference helpers
# ----------------------------
def _run_cnn(file_path: str) -> Dict:
    model = _load_cnn_model()
    if model is None:
        raise RuntimeError("Local CNN audio model not found at models/audio_model.pth")

    mel_tensor = _load_and_mel(file_path)

    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = EMOTIONS[pred_idx]

    scores = {emo: float(probs[i]) for i, emo in enumerate(EMOTIONS)}
    return {
        "predicted_label": pred_label,
        "scores": scores,
    }


def _run_hf_local(file_path: str) -> Dict:
    clf = _load_hf_local_pipeline()
    if clf is None:
        raise RuntimeError(
            "HF local audio model not found at models/hf_audio_model/. "
            "Please download and save it there."
        )

    # --- FIX START ---
    # Don't take [0] yet. 'out' should be the list of predictions.
    raw_output = clf(file_path) 

    # Handle nested lists (pipelines sometimes return [[{...}, {...}]])
    if isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], list):
        raw_output = raw_output[0]
    
    out = raw_output
    # --- FIX END ---

    # Map HF labels -> our 7 labels
    scores = {emo: 0.0 for emo in EMOTIONS}
    for item in out:
        label = item["label"].lower()
        score = float(item["score"])

        # Very simple mapping logic
        if "ang" in label:
            scores["angry"] += score
        elif "dis" in label:
            scores["disgust"] += score
        elif "fear" in label or "fea" in label:
            scores["fear"] += score
        elif "hap" in label or "joy" in label:
            scores["happy"] += score
        elif "sad" in label:
            scores["sad"] += score
        elif "sur" in label:
            scores["surprise"] += score
        elif "neu" in label or "neutral" in label:
            scores["neutral"] += score

    total = sum(scores.values()) or 1.0
    for k in scores:
        scores[k] /= total

    pred_label = max(scores, key=scores.get)
    return {
        "predicted_label": pred_label,
        "scores": scores,
    }


# ----------------------------
# PUBLIC API
# ----------------------------
def classify_audio(file_path: str, mode: str = "fast") -> Dict:
    """
    mode = "fast":
        - Prefer local CNN audio model
        - If missing, fallback to HF-local model

    mode = "accurate":
        - Prefer HF-local model
        - If missing, fallback to local CNN

    Raises RuntimeError if no audio model is available.
    """
    mode = mode.lower().strip()

    cnn_available = _load_cnn_model() is not None
    hf_available = _load_hf_local_pipeline() is not None

    if mode == "fast":
        if cnn_available:
            return _run_cnn(file_path)
        if hf_available:
            return _run_hf_local(file_path)
        raise RuntimeError("No audio model available (CNN or HF-local).")

    elif mode == "accurate":
        if hf_available:
            return _run_hf_local(file_path)
        if cnn_available:
            return _run_cnn(file_path)
        raise RuntimeError("No audio model available (HF-local or CNN).")

    else:
        raise ValueError("Invalid mode for classify_audio. Use 'fast' or 'accurate'.")
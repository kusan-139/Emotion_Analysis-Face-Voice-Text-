# multimodal/text_emotion.py

"""
Text emotion classification (OFFLINE ONLY).

Priority:
1) Your fine-tuned model:      models/text_model/
2) Local HF model (offline):   models/hf_text_model/

No online model download is used.

Later, when you fine-tune your own text model, just save it to models/text_model/
and this code will automatically start using it.
"""

import os
from functools import lru_cache
from typing import Dict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# Paths for local models
LOCAL_FINETUNED_DIR = "models/text_model"      # your future fine-tuned model
LOCAL_HF_DIR = "models/hf_text_model"         # offline copy of HF model

# Map HF labels -> our 7 emotion labels
TEXT_LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}


@lru_cache(maxsize=1)
def _load_finetuned_local():
    """Load your fine-tuned text model from models/text_model/ if it exists."""
    if os.path.isdir(LOCAL_FINETUNED_DIR):
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_FINETUNED_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_FINETUNED_DIR)
        print("[Text] Using YOUR fine-tuned model from", LOCAL_FINETUNED_DIR)
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )
        return clf
    return None


@lru_cache(maxsize=1)
def _load_hf_local():
    """Load locally saved HF model from models/hf_text_model/ if present."""
    if os.path.isdir(LOCAL_HF_DIR):
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_HF_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_HF_DIR)
        print("[Text] Using LOCAL HF model from", LOCAL_HF_DIR)
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )
        return clf
    return None


def _get_text_pipeline(allow_hf_fallback: bool = True):
    """
    Returns a pipeline according to priority:
      1) your fine-tuned model
      2) local HF model (if allow_hf_fallback=True)

    If neither exists:
      - if allow_hf_fallback=False -> raise RuntimeError
      - if allow_hf_fallback=True  -> also raise RuntimeError (no models available)
    """
    # 1) Your own fine-tuned model
    clf = _load_finetuned_local()
    if clf is not None:
        return clf

    if not allow_hf_fallback:
        raise RuntimeError(
            "No local fine-tuned text model found at models/text_model "
            "and HF fallback is disabled."
        )

    # 2) Local HF model
    clf = _load_hf_local()
    if clf is not None:
        return clf

    # No models available at all
    raise RuntimeError(
        "No text emotion model available. "
        "Expected either models/text_model/ or models/hf_text_model/ to exist."
    )


def classify_text(text: str, allow_hf_fallback: bool = True) -> Dict:
    """
    Classify a piece of text into our 7 emotion classes.

    Returns:
        {
          "predicted_label": "<angry|disgust|fear|happy|sad|surprise|neutral>",
          "scores": {emotion: probability}
        }
    """
    clf = _get_text_pipeline(allow_hf_fallback=allow_hf_fallback)
    outputs = clf(text)[0]  # list of dicts: {"label":..., "score":...}

    mapped_scores = {emo: 0.0 for emo in TEXT_LABEL_MAP.values()}

    for item in outputs:
        hf_label = item["label"].lower()
        score = float(item["score"])
        if hf_label in TEXT_LABEL_MAP:
            mapped_label = TEXT_LABEL_MAP[hf_label]
            mapped_scores[mapped_label] += score

    total = sum(mapped_scores.values()) or 1.0
    for k in mapped_scores:
        mapped_scores[k] /= total

    predicted_label = max(mapped_scores, key=mapped_scores.get)

    return {
        "predicted_label": predicted_label,
        "scores": mapped_scores,
    }

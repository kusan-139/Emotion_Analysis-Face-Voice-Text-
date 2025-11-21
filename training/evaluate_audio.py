import os
import argparse
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import AUDIO_DATA_DIR, TRAINING_PLOTS_DIR, AUDIO_MODEL_DIR, AUDIO_SAMPLE_RATE
from multimodal.audio_model import (
    SAVEEAudioDataset,
    extract_emotion_from_filename,
    AudioCNN,
    IDX2EMO_AUDIO,
)

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline


LOCAL_CNN_PATH = "models/audio_model.pth"
LOCAL_HF_MODEL_DIR = "models/hf_audio_model"


def build_savee_filelist():
    """Scan WAV files and extract emotion labels."""
    wav_paths = glob(os.path.join(AUDIO_DATA_DIR, "**", "*.wav"), recursive=True)
    labels, valid_paths = [], []

    for p in wav_paths:
        fname = os.path.basename(p)
        emo = extract_emotion_from_filename(fname)
        if emo is None:
            continue
        valid_paths.append(p)
        labels.append(emo)

    return valid_paths, labels


def evaluate_local_cnn(batch_size: int, out_dir: str):
    """Evaluate your trained AudioCNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels = build_savee_filelist()
    print(f"[Eval local_cnn] Files: {len(paths)}")

    dataset = SAVEEAudioDataset(paths, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if not os.path.exists(LOCAL_CNN_PATH):
        raise FileNotFoundError(f"Local CNN model not found: {LOCAL_CNN_PATH}")

    model = AudioCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(LOCAL_CNN_PATH, map_location=device))
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for mel, lbls in tqdm(loader, desc="[local_cnn]"):
            mel, lbls = mel.to(device), lbls.to(device)
            logits = model(mel)
            preds = logits.argmax(dim=1)

            all_labels.extend(lbls.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    save_results(np.array(all_labels), np.array(all_preds), out_dir, prefix="audio_local_cnn")


def evaluate_hf_local(out_dir: str):
    """Evaluate LOCAL offline HF model."""
    if not os.path.isdir(LOCAL_HF_MODEL_DIR):
        raise FileNotFoundError(
            f"Local HF model not found at {LOCAL_HF_MODEL_DIR}.\n"
            "Download it using:\n"
            "from transformers import AutoFeatureExtractor, AutoModelForAudioClassification\n"
            "extractor.save_pretrained('models/hf_audio_model')\n"
            "model.save_pretrained('models/hf_audio_model')"
        )

    print("[Eval hf_local] Loading local HF model...")

    extractor = AutoFeatureExtractor.from_pretrained(LOCAL_HF_MODEL_DIR)
    model = AutoModelForAudioClassification.from_pretrained(LOCAL_HF_MODEL_DIR)
    clf = pipeline("audio-classification", model=model, feature_extractor=extractor)

    paths, labels = build_savee_filelist()
    print(f"[Eval hf_local] Files: {len(paths)}")

    emo_map = {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fearful": "fear",
        "disgust": "disgust",
        "surprised": "surprise",
    }
    emo2idx = {v: k for k, v in IDX2EMO_AUDIO.items()}

    all_labels, all_preds = [], []

    for p, true_lbl in tqdm(zip(paths, labels), total=len(paths), desc="[hf_local]"):
        out = clf(p)

        scores = {emo: 0.0 for emo in emo_map.values()}
        for item in out:
            lab = item["label"].lower()
            if lab in emo_map:
                scores[emo_map[lab]] += float(item["score"])

        pred = max(scores, key=scores.get)
        pred_idx = emo2idx[pred]

        all_labels.append(true_lbl)
        all_preds.append(pred_idx)

    save_results(
        np.array(all_labels),
        np.array(all_preds),
        out_dir,
        prefix="audio_hf_local"
    )


def save_results(all_labels, all_preds, out_dir, prefix):
    """Save confusion matrix, report, and per-class accuracy."""
    os.makedirs(out_dir, exist_ok=True)

    class_names = [IDX2EMO_AUDIO[i] for i in range(len(IDX2EMO_AUDIO))]

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Save report
    with open(os.path.join(out_dir, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(prefix)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    with open(os.path.join(out_dir, f"{prefix}_per_class_accuracy.txt"), "w") as f:
        for i, emo in IDX2EMO_AUDIO.items():
            f.write(f"{emo}: {per_class_acc[i]:.4f}\n")

    print(f"Saved results to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["local_cnn", "hf_local"],
        help="Which audio model to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = os.path.join(TRAINING_PLOTS_DIR, f"audio_{args.model_type}")

    if args.model_type == "local_cnn":
        evaluate_local_cnn(batch_size=args.batch_size, out_dir=out_dir)

    elif args.model_type == "hf_local":
        evaluate_hf_local(out_dir=out_dir)


if __name__ == "__main__":
    main()

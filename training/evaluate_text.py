import os
import argparse
from typing import List

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from config import TRAINING_PLOTS_DIR
from multimodal.text_emotion import (
    LOCAL_TRAINED_TEXT_DIR,
    LOCAL_HF_TEXT_DIR,
    HF_TEXT_MODEL_NAME,
    IDX2EMO_TEXT,
)


# --------------------------------------------------------
# Load datasets for evaluation
# --------------------------------------------------------

def load_text_dataset() -> (List[str], List[int]):
    """
    Loads text dataset for evaluation.
    Expected folder structure:
        data/text_dataset/
              angry/*.txt
              joy/*.txt
              sad/*.txt
              ...
    Each file contains raw text.
    """

    base_dir = "data/text_dataset"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            "Dataset not found: data/text_dataset/.\n"
            "Create structure:\n"
            "data/text_dataset/<emotion>/<text files>"
        )

    texts, labels = [], []
    emo_to_idx = {v: k for k, v in IDX2EMO_TEXT.items()}  # anger->0, ... neutral->6

    for emo in emo_to_idx:
        emo_dir = os.path.join(base_dir, emo)
        if not os.path.isdir(emo_dir):
            continue

        for fname in os.listdir(emo_dir):
            if fname.endswith(".txt"):
                path = os.path.join(emo_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                texts.append(text)
                labels.append(emo_to_idx[emo])

    print(f"[Text Eval] Loaded {len(texts)} samples for evaluation")
    return texts, labels


# --------------------------------------------------------
# Save evaluation results
# --------------------------------------------------------

def save_results(all_labels, all_preds, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    class_names = [IDX2EMO_TEXT[i] for i in range(len(IDX2EMO_TEXT))]

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    # Save classification report
    with open(os.path.join(out_dir, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix image
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
        for i, emo in IDX2EMO_TEXT.items():
            f.write(f"{emo}: {per_class_acc[i]:.4f}\n")

    print(f"[Text Eval] Results saved to {out_dir}")


# --------------------------------------------------------
# Evaluate local trained TEXT model
# --------------------------------------------------------

def evaluate_local_text(out_dir: str):
    if not os.path.isdir(LOCAL_TRAINED_TEXT_DIR):
        raise FileNotFoundError(
            f"No local trained text model found at: {LOCAL_TRAINED_TEXT_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TRAINED_TEXT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_TRAINED_TEXT_DIR)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

    texts, labels = load_text_dataset()

    print("[Text Eval] Evaluating local trained model...")
    preds = []
    for text in tqdm(texts):
        out = pipe(text)[0]
        preds.append(_label_to_idx(out["label"]))

    save_results(np.array(labels), np.array(preds), out_dir, "text_local")


# --------------------------------------------------------
# Evaluate HF LOCAL (offline) TEXT model
# --------------------------------------------------------

def evaluate_hf_local(out_dir: str):
    if not os.path.isdir(LOCAL_HF_TEXT_DIR):
        raise FileNotFoundError(
            f"Local HF model not found: {LOCAL_HF_TEXT_DIR}\n"
            "Download it first."
        )

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_HF_TEXT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_HF_TEXT_DIR)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

    texts, labels = load_text_dataset()

    print("[Text Eval] Evaluating HF local model...")
    preds = []
    for text in tqdm(texts):
        out = pipe(text)[0]
        preds.append(_label_to_idx(out["label"]))

    save_results(np.array(labels), np.array(preds), out_dir, "text_hf_local")


# --------------------------------------------------------
# Helper: Convert HF label to our index
# --------------------------------------------------------

def _label_to_idx(label: str) -> int:
    label = label.lower()
    idx_map = {v: k for k, v in IDX2EMO_TEXT.items()}
    return idx_map.get(label, idx_map["neutral"])


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["local_text", "hf_local"],
        help="Which text model to evaluate.",
    )
    args = parser.parse_args()

    out_dir = os.path.join(TRAINING_PLOTS_DIR, f"text_{args.model_type}")

    if args.model_type == "local_text":
        evaluate_local_text(out_dir)

    elif args.model_type == "hf_local":
        evaluate_hf_local(out_dir)


if __name__ == "__main__":
    main()

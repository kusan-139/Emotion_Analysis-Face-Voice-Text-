import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE, BEST_MODEL_PATH, RESULTS_DIR, IDX2EMOTION
from models.emotion_model import EmotionResNet18
from training.utils import get_data_loaders


def evaluate():
    _, _, test_loader = get_data_loaders()
    model = EmotionResNet18(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(IDX2EMOTION.values()))
    print(report)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=list(IDX2EMOTION.values()),
                yticklabels=list(IDX2EMOTION.values()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    with open(os.path.join(RESULTS_DIR, "per_class_accuracy.txt"), "w") as f:
        for i, emo in IDX2EMOTION.items():
            f.write(f"{emo}: {per_class_acc[i]:.4f}\n")

    print("Evaluation artifacts saved in", RESULTS_DIR)


if __name__ == "__main__":
    evaluate()

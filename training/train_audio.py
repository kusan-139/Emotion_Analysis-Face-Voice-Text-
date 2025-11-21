import os
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import AUDIO_DATA_DIR, AUDIO_MODEL_DIR
from multimodal.audio_model import SAVEEAudioDataset, extract_emotion_from_filename, AudioCNN


def build_savee_filelist():
    """
    Scan data/savee recursively for .wav files,
    extract emotion from filename using SAVEE naming scheme.
    """
    wav_paths = glob(os.path.join(AUDIO_DATA_DIR, "**", "*.wav"), recursive=True)
    labels = []
    valid_paths = []

    for p in wav_paths:
        fname = os.path.basename(p)
        emo = extract_emotion_from_filename(fname)
        if emo is None:
            continue
        valid_paths.append(p)
        labels.append(emo)

    return valid_paths, labels


def train_audio_model(epochs: int = 25, batch_size: int = 8, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels = build_savee_filelist()
    print(f"[Audio] Found {len(paths)} valid SAVEE audio files")
    from collections import Counter
    print("[Audio] Label distribution:", Counter(labels))


    dataset = SAVEEAudioDataset(paths, labels)

    # 80/20 train/val split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AudioCNN(num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --------- Train ----------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for mel, labels in tqdm(train_loader, desc=f"Audio Epoch {epoch}/{epochs} [Train]"):
            mel, labels = mel.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --------- Val ----------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for mel, labels in tqdm(val_loader, desc=f"Audio Epoch {epoch}/{epochs} [Val]"):
                mel, labels = mel.to(device), labels.to(device)
                logits = model(mel)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[Audio] Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(AUDIO_MODEL_DIR, exist_ok=True)
            save_path = os.path.join(AUDIO_MODEL_DIR, "audio_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[Audio] New best val acc: {best_val_acc:.4f}, saved to {save_path}")

    print("Finished training audio model.")


if __name__ == "__main__":
    train_audio_model()

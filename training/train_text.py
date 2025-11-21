# training/train_text.py
import os
import random
from typing import List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

from tqdm import tqdm

from config import TEXT_DATA_DIR, TEXT_MODEL_DIR
from text_models.text_model import (
    TextExample,
    MELDDataset,
    create_text_model,
    EMOTION2IDX_TEXT,
    IDX2EMOTION_TEXT,
)


def load_meld_split(csv_name: str) -> List[TextExample]:
    path = os.path.join(TEXT_DATA_DIR, csv_name)
    df = pd.read_csv(path)

    examples = []
    for _, row in df.iterrows():
        emo = str(row["Emotion"]).lower()
        if emo not in EMOTION2IDX_TEXT:
            continue
        text = str(row["Utterance"])
        label = EMOTION2IDX_TEXT[emo]
        examples.append(TextExample(text=text, label=label))
    return examples


def train_text_model(
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ex = load_meld_split("train_sent_emo.csv")
    val_ex = load_meld_split("dev_sent_emo.csv")

    print(f"Loaded {len(train_ex)} train, {len(val_ex)} val text examples")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_ds = MELDDataset(train_ex, tokenizer, max_len=max_len)
    val_ds = MELDDataset(val_ex, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = create_text_model().to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"Text Epoch {epoch}/{epochs} [Train]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == batch["labels"]).sum().item()
            train_total += batch["labels"].size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Text Epoch {epoch}/{epochs} [Val]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item() * batch["labels"].size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch["labels"]).sum().item()
                val_total += batch["labels"].size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[Text] Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best val acc: {best_val_acc:.4f}")
            # Save tokenizer + model in TEXT_MODEL_DIR
            tokenizer.save_pretrained(TEXT_MODEL_DIR)
            model.save_pretrained(TEXT_MODEL_DIR)

    print("Finished training text model.")


if __name__ == "__main__":
    train_text_model()

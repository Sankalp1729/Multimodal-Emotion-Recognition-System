import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from utils.logger import get_logger
from utils.config import DATA_DIR, MODELS_DIR, EMOTIONS

logger = get_logger("train_text")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Text dataset CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: text,label")
    label_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    df = df[df["label"].str.lower().isin(label_to_idx.keys())]
    texts = df["text"].astype(str).tolist()
    labels = [label_to_idx[str(l).lower()] for l in df["label"].tolist()]
    return texts, np.array(labels, dtype=np.int64), label_to_idx


def tokenize_dataset(tokenizer, texts, labels):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)
    enc["labels"] = labels.tolist()
    return enc


def main():
    csv_path = DATA_DIR / "text" / "dataset.csv"
    try:
        texts, labels, label_to_idx = load_dataset(csv_path)
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        logger.info("Expected dataset.csv with columns text,label under data/text.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(EMOTIONS))

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    train_enc = tokenize_dataset(tokenizer, X_train, y_train)
    val_enc = tokenize_dataset(tokenizer, X_val, y_val)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return len(self.enc["labels"])
        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in self.enc.items()}

    train_ds = SimpleDataset(train_enc)
    val_ds = SimpleDataset(val_enc)

    out_dir = MODELS_DIR / "text_model"

    args = TrainingArguments(
        output_dir=str(out_dir / "trainer_out"),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "label_map.json", "w") as f:
        json.dump(label_to_idx, f)

    logger.info(f"Saved text model and tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
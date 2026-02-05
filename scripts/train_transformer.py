#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from preprocess import clean_pq  # your existing cleaner

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ML_training_data.csv"
OUT_DIR = ROOT / "artifacts" / "models_transformer" / "distilbert_pq_heading"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
SEED = 42
MIN_HEADING_FREQ = 8  # collapse rare headings into "Other"
OTHER_LABEL = "Other"

# Determinism-ish
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    print("Loading data:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    required_cols = {"question", "heading", "department"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df["question"] = df["question"].astype(str)
    df["department"] = df["department"].astype(str)
    df["heading"] = df["heading"].astype(str)

    # Clean text if needed
    if "question_clean" not in df.columns:
        print("Cleaning questions...")
        df["question_clean"] = df["question"].map(clean_pq)

    # -----------------------------
    # COLLAPSE RARE HEADINGS
    # -----------------------------
    freq = Counter(df["heading"])
    rare = {h for h, c in freq.items() if c < MIN_HEADING_FREQ}

    before = df["heading"].nunique()
    df["heading_collapsed"] = df["heading"].apply(lambda h: OTHER_LABEL if h in rare else h)
    after = df["heading_collapsed"].nunique()
    other_count = int((df["heading_collapsed"] == OTHER_LABEL).sum())

    print(f"Unique headings before: {before}")
    print(f"Unique headings after : {after}")
    print(f"Headings collapsed    : {before - after}")
    print(f"Rows now labelled Other: {other_count}")

    # -----------------------------
    # BUILD TRANSFORMER INPUT TEXT
    # -----------------------------
    # We embed department as text to avoid a second input branch
    def build_text(row) -> str:
        return f"[DEPT] {row['department']} [SEP] {row['question_clean']}"

    df["text"] = df.apply(build_text, axis=1)

    # -----------------------------
    # LABEL ENCODING
    # -----------------------------
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["heading_collapsed"])
    num_labels = int(len(label_encoder.classes_))
    print("Number of labels:", num_labels)

    # -----------------------------
    # TRAIN / VALIDATION SPLIT
    # -----------------------------
    # NOTE: This is a random split; your 2026 holdout evaluation remains your "true" temporal test.
    train_df, val_df = train_test_split(
        df,
        test_size=0.10,
        random_state=SEED,
        stratify=df["label"],
    )

    print("Train size:", len(train_df))
    print("Val size  :", len(val_df))

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    # -----------------------------
    # TOKENIZER
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    # Remove the raw text column (we keep label + token fields)
    for col in ["text"]:
        if col in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns([col])
        if col in val_dataset.column_names:
            val_dataset = val_dataset.remove_columns([col])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # -----------------------------
    # MODEL
    # -----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    # -----------------------------
    # METRICS (TOP-1 + TOP-3)
    # -----------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()

        top1 = float((probs.argmax(axis=1) == labels).mean())

        # "top-3 hit": true label appears anywhere in the top 3 predicted classes
        top3_hits = 0
        top3 = np.argsort(-probs, axis=1)[:, :3]
        for i in range(len(labels)):
            if labels[i] in top3[i]:
                top3_hits += 1
        top3_acc = float(top3_hits / len(labels))

        return {"top1_accuracy": top1, "top3_accuracy": top3_acc}

    # -----------------------------
    # TRAINING ARGS
    # -----------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Transformers versions differ: newer versions use eval_strategy instead of evaluation_strategy.
    # This script uses eval_strategy to match your installed version.
    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_top1_accuracy",  # Trainer prefixes metrics with "eval_"
        greater_is_better=True,
        seed=SEED,
        report_to=[],  # avoids trying to auto-configure wandb etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # ok to include; handles some padding edge cases
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    print("Training transformer model...")
    trainer.train()

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    # -----------------------------
    # SAVE ARTIFACTS
    # -----------------------------
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    # Save label map (id -> heading string)
    label_map = {int(i): cls for i, cls in enumerate(label_encoder.classes_)}
    (OUT_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    meta = {
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "num_labels": num_labels,
        "min_heading_freq": MIN_HEADING_FREQ,
        "other_label": OTHER_LABEL,
        "text_format": "[DEPT] {department} [SEP] {question_clean}",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "seed": SEED,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved transformer model to:", OUT_DIR)


if __name__ == "__main__":
    main()
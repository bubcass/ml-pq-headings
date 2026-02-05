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
    DebertaV2Tokenizer,   # <-- important: slow tokenizer
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from preprocess import clean_pq  # your existing cleaner

# -----------------------------
# CONFIG (LIKE-FOR-LIKE)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ML_training_data.csv"

MODEL_NAME = "microsoft/deberta-v3-small"
OUT_DIR = ROOT / "artifacts" / "models_transformer" / "deberta_v3_small_pq_heading"

MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
SEED = 42
MIN_HEADING_FREQ = 8
OTHER_LABEL = "Other"

# Early stopping behaviour
EARLY_STOP_PATIENCE = 1  # stop after 1 epoch with no improvement
EARLY_STOP_THRESHOLD = 0.0  # require > 0.0 improvement to reset patience

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def main() -> None:
    print("Loading data:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    required_cols = {"question", "heading", "department"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df["question"] = df["question"].astype(str)
    df["department"] = df["department"].astype(str)
    df["heading"] = df["heading"].astype(str)

    if "question_clean" not in df.columns:
        print("Cleaning questions...")
        df["question_clean"] = df["question"].map(clean_pq)

    # ---- collapse rare headings ----
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

    # ---- build input text (must match inference later) ----
    df["text"] = df.apply(lambda r: f"[DEPT] {r['department']} [SEP] {r['question_clean']}", axis=1)

    # ---- label encoding ----
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["heading_collapsed"])
    num_labels = int(len(label_encoder.classes_))
    print("Number of labels:", num_labels)

    # ---- split ----
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

    # ---- tokenizer ----
    # IMPORTANT: bypass AutoTokenizer fast-tokenizer bug by explicitly using the slow tokenizer.
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    # remove raw text
    for col in ["text"]:
        if col in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns([col])
        if col in val_dataset.column_names:
            val_dataset = val_dataset.remove_columns([col])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # ---- model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    # ---- metrics ----
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits: (N, C) as numpy
        probs = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()

        top1 = float((probs.argmax(axis=1) == labels).mean())

        top3 = np.argsort(-probs, axis=1)[:, :3]
        hits = 0
        for i in range(len(labels)):
            if labels[i] in top3[i]:
                hits += 1
        top3_acc = float(hits / len(labels))

        return {"top1_accuracy": top1, "top3_accuracy": top3_acc}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),

        # train/eval cadence
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,

        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,

        # keep only a couple of checkpoints
        save_total_limit=2,

        # pick and reload best checkpoint at end (needed for early stopping)
        load_best_model_at_end=True,
        metric_for_best_model="eval_top1_accuracy",
        greater_is_better=True,

        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # warning-only in your version; ok
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOP_PATIENCE,
                early_stopping_threshold=EARLY_STOP_THRESHOLD,
            )
        ],
    )

    print("Training DeBERTa v3 small model...")
    trainer.train()

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    # ---- save artifacts ----
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

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
        "tokenizer_fast": False,
        "tokenizer_class": "DebertaV2Tokenizer",
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "early_stopping_threshold": EARLY_STOP_THRESHOLD,
        "best_model_metric": "eval_top1_accuracy",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved DeBERTa model to:", OUT_DIR)


if __name__ == "__main__":
    main()
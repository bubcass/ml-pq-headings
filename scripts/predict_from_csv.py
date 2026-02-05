#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Uses the same cleaning logic as your training corpus build step
from preprocess import clean_pq

ROOT = Path(__file__).resolve().parents[1]

# Artifacts produced by scripts/train_model.py
MODEL_PATH = ROOT / "artifacts" / "models" / "heading_model.keras"
ENC_DIR = ROOT / "artifacts" / "encoders"
TOKENIZER_PATH = ENC_DIR / "tokenizer.pkl"
DEPT_ENCODER_PATH = ENC_DIR / "department_encoder.pkl"
HEADING_ENCODER_PATH = ENC_DIR / "heading_encoder.pkl"

# Must match training
MAXLEN = 300
OTHER_LABEL = "Other"

# Operational control: how strict to be about review
REVIEW_THRESHOLD = 0.25  # tune: 0.25 fewer flagged, 0.35 more flagged


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(in_csv: str, out_csv: str, top_k: int = 3) -> None:
    # --- Load artifacts ---
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")

    for p in (TOKENIZER_PATH, DEPT_ENCODER_PATH, HEADING_ENCODER_PATH):
        if not p.exists():
            raise SystemExit(f"Encoder/tokenizer not found: {p}")

    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = load_pickle(TOKENIZER_PATH)
    dept_encoder = load_pickle(DEPT_ENCODER_PATH)
    heading_encoder = load_pickle(HEADING_ENCODER_PATH)

    # --- Load inputs ---
    df = pd.read_csv(in_csv)

    # Expect: question,department
    required = {"question", "department"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    # Clean text if needed
    if "question_clean" not in df.columns:
        df["question_clean"] = df["question"].map(clean_pq)

    # Encode departments (fail loudly if unseen)
    dept_series = df["department"].fillna("").astype(str)
    known_depts = set(dept_encoder.classes_)
    unseen = sorted(set(dept_series) - known_depts)
    if unseen:
        raise SystemExit(
            "Unseen department label(s) in input CSV:\n"
            + "\n".join(f"  - {d}" for d in unseen)
            + "\n\nFix: use the exact department labels from training."
        )
    dept_ids = dept_encoder.transform(dept_series).astype(np.int32)

    # Tokenize + pad
    texts = df["question_clean"].fillna("").astype(str).tolist()
    seq = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    # --- Predict ---
    probs = model.predict([X, dept_ids], verbose=0)

    # Top-k candidates (for alternates / QA)
    k = max(1, int(top_k))
    topk_idx = np.argsort(-probs, axis=1)[:, :k]          # (N, k)
    topk_scores = np.take_along_axis(probs, topk_idx, 1)  # (N, k)
    topk_labels = heading_encoder.inverse_transform(topk_idx.reshape(-1)).reshape(-1, k)

    # Never output "Other": if top-1 is Other, choose best non-Other across all classes
    classes = heading_encoder.classes_
    other_idx = int(np.where(classes == OTHER_LABEL)[0][0]) if OTHER_LABEL in classes else None

    final_idx = np.empty((probs.shape[0],), dtype=int)
    final_conf = np.empty((probs.shape[0],), dtype=float)

    for i in range(probs.shape[0]):
        if other_idx is not None and topk_idx[i, 0] == other_idx:
            p2 = probs[i].copy()
            p2[other_idx] = -1.0
            j = int(p2.argmax())
            final_idx[i] = j
            final_conf[i] = float(probs[i, j])
        else:
            j = int(topk_idx[i, 0])
            final_idx[i] = j
            final_conf[i] = float(topk_scores[i, 0])

    df["pred_heading"] = heading_encoder.inverse_transform(final_idx)
    df["pred_confidence"] = final_conf
    df["needs_review"] = df["pred_confidence"] < REVIEW_THRESHOLD

    # Add alternates (default top 3)
    for j in range(min(k, 3)):
        df[f"alt_heading_{j+1}"] = topk_labels[:, j]
        df[f"alt_conf_{j+1}"] = topk_scores[:, j].astype(float)

    # Write output
    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)
    print(f"Review threshold: {REVIEW_THRESHOLD}  |  Flagged for review: {int(df['needs_review'].sum())}/{len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict PQ headings from CSV (question,department).")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (question,department)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    ap.add_argument("--topk", dest="topk", type=int, default=3, help="Number of alternates to compute (default 3)")
    args = ap.parse_args()

    main(args.inp, args.out, top_k=args.topk)
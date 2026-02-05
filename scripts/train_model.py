#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# -----------------------
# Paths / config
# -----------------------
PROJECT_ROOT = Path("/Users/david/Developer/2026-01-24_ML_PQs")
TRAIN_CSV = PROJECT_ROOT / "data" / "ML_training_data_clean.csv"

ART_MODELS = PROJECT_ROOT / "artifacts" / "models"
ART_ENC = PROJECT_ROOT / "artifacts" / "encoders"
ART_META = PROJECT_ROOT / "artifacts" / "metadata"
ART_MODELS.mkdir(parents=True, exist_ok=True)
ART_ENC.mkdir(parents=True, exist_ok=True)
ART_META.mkdir(parents=True, exist_ok=True)

MODEL_OUT = ART_MODELS / "heading_model.keras"

# Model/training params
NUM_WORDS = 20000
MAXLEN = 300
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 30
VAL_SPLIT = 0.1
SEED = 42

# Rare-class collapse (Fix C)
MIN_HEADING_COUNT = 8
OTHER_LABEL = "Other"


def build_model(num_words: int, maxlen: int, n_depts: int, n_heads: int) -> tf.keras.Model:
    """
    Strong baseline for text+department classification:
      - text sequence -> Embedding -> GlobalAveragePooling
      - dept id -> small embedding
      - concat -> dense -> softmax
    """
    text_in = tf.keras.Input(shape=(maxlen,), name="text")
    dept_in = tf.keras.Input(shape=(), dtype=tf.int32, name="dept")  # scalar per row

    x = tf.keras.layers.Embedding(num_words, EMBED_DIM, name="text_emb")(text_in)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    d = tf.keras.layers.Embedding(n_depts, 8, name="dept_emb")(dept_in)
    d = tf.keras.layers.Flatten()(d)

    h = tf.keras.layers.Concatenate()([x, d])
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dropout(0.25)(h)
    out = tf.keras.layers.Dense(n_heads, activation="softmax")(h)

    model = tf.keras.Model(inputs=[text_in, dept_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    df = pd.read_csv(TRAIN_CSV)

    # Validate required columns
    required = {"question_clean", "department", "heading"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Training CSV missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    # -----------------------
    # Fix C: collapse rare headings into "Other"
    # -----------------------
    heading_counts = df["heading"].fillna("").astype(str).value_counts()
    rare_headings = set(heading_counts[heading_counts < MIN_HEADING_COUNT].index)

    df["heading_collapsed"] = df["heading"].fillna("").astype(str).apply(
        lambda h: OTHER_LABEL if h in rare_headings else h
    )

    print(f"Unique headings before: {df['heading'].nunique()}")
    print(f"Unique headings after : {df['heading_collapsed'].nunique()}")
    print(f"Headings collapsed    : {len(rare_headings)}")
    print(f"Rows now labelled Other: {(df['heading_collapsed'] == OTHER_LABEL).sum()}")

    # -----------------------
    # Encoders
    # -----------------------
    dept_enc = LabelEncoder()
    head_enc = LabelEncoder()

    df["department_encoded"] = dept_enc.fit_transform(df["department"].astype(str))
    df["heading_encoded"] = head_enc.fit_transform(df["heading_collapsed"].astype(str))

    # -----------------------
    # Tokenizer trained on CLEAN text
    # -----------------------
    tok = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(df["question_clean"].fillna("").astype(str).tolist())

    seq = tok.texts_to_sequences(df["question_clean"].fillna("").astype(str).tolist())
    X = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    D = df["department_encoded"].to_numpy().astype(np.int32)
    y = df["heading_encoded"].to_numpy().astype(np.int32)

    # -----------------------
    # Train/val split (stratify should now work after collapsing rare labels)
    # -----------------------
    try:
        X_train, X_val, D_train, D_val, y_train, y_val = train_test_split(
            X, D, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y
        )
    except ValueError as e:
        print("\n[WARN] Stratified split failed (still have singleton classes).")
        print("Falling back to non-stratified split. Consider increasing MIN_HEADING_COUNT.\n")
        print("Error was:", e, "\n")
        X_train, X_val, D_train, D_val, y_train, y_val = train_test_split(
            X, D, y, test_size=VAL_SPLIT, random_state=SEED, shuffle=True
        )

    n_depts = int(D.max()) + 1
    n_heads = int(y.max()) + 1

    model = build_model(NUM_WORDS, MAXLEN, n_depts, n_heads)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
    ]

    history = model.fit(
        x={"text": X_train, "dept": D_train},
        y=y_train,
        validation_data=({"text": X_val, "dept": D_val}, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # -----------------------
    # Save artifacts
    # -----------------------
    model.save(MODEL_OUT)

    with open(ART_ENC / "tokenizer.pkl", "wb") as f:
        pickle.dump(tok, f)

    with open(ART_ENC / "department_encoder.pkl", "wb") as f:
        pickle.dump(dept_enc, f)

    with open(ART_ENC / "heading_encoder.pkl", "wb") as f:
        pickle.dump(head_enc, f)

    meta = {
        "train_csv": str(TRAIN_CSV),
        "num_words": NUM_WORDS,
        "maxlen": MAXLEN,
        "embed_dim": EMBED_DIM,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "val_split": VAL_SPLIT,
        "seed": SEED,
        "min_heading_count": MIN_HEADING_COUNT,
        "other_label": OTHER_LABEL,
        "n_departments": n_depts,
        "n_headings": n_heads,
        "model_out": str(MODEL_OUT),
        "keras_version": tf.keras.__version__,
        "tf_version": tf.__version__,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    (ART_META / "train_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved model to: {MODEL_OUT}")
    print(f"Saved tokenizer/encoders to: {ART_ENC}")
    print(f"Saved metadata to: {ART_META / 'train_config.json'}")


if __name__ == "__main__":
    main()
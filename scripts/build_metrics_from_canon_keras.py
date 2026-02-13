#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


# -------------------------
# Helpers (copied from your HF script)
# -------------------------
def safe_get(d: Any, path: List[str]) -> Optional[Any]:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def first_non_empty(*vals: Any) -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def parse_possible_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ----------------------------
# PQ-aware cleaning (match training)
# ----------------------------
PATTERNS = [
    (r"^\s*\d+\.\s*", " "),
    (r"^\s*deputy\s+[a-záéíóú\-\'\.\s]+\s+asked\s+", " "),
    (r"\basked\s+the\s+minister\s+for\s+[^,]+,?\s*", " "),
    (r"\basked\s+the\s+minister\s+of\s+state\s+at\s+[^,]+,?\s*", " "),
    (r"\bthe\s+taoiseach\b", " "),
    (r"\bthe\s+tánaiste\b", " "),
    (r"\btaoiseach\b", " "),
    (r"\btánaiste\b", " "),
    (r"\b(if|whether)\s+he\s+will\s+make\s+a\s+statement\s+on\s+the\s+matter\b", " "),
    (r"\b(if|whether)\s+she\s+will\s+make\s+a\s+statement\s+on\s+the\s+matter\b", " "),
    (r"\band\s+if\s+he\s+will\s+make\s+a\s+statement\b", " "),
    (r"\band\s+if\s+she\s+will\s+make\s+a\s+statement\b", " "),
    (r"\band\s+if\s+so\b", " "),
    (r"\band\s+if\s+not\b", " "),
    (r"\bto\s+outline\b", " "),
    (r"\bto\s+provide\s+details\b", " "),
    (r"\bin\s+relation\s+to\b", " "),
]
COMPILED = [(re.compile(pat, flags=re.IGNORECASE), repl) for pat, repl in PATTERNS]


def clean_pq(text: str) -> str:
    """PQ-aware text cleaning. Keeps acronyms and digits; removes procedural boilerplate."""
    if text is None:
        return ""
    t = str(text).strip().lower()

    for rx, repl in COMPILED:
        t = rx.sub(repl, t)

    # Preserve word chars, digits, underscores, hyphen; remove other punctuation.
    t = re.sub(r"[^\w\s\-]", " ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


# -------------------------
# Extract PQ rows from canon
# -------------------------
@dataclass
class PQRow:
    question: str
    heading: str
    department: str
    date: datetime


def extract_rows_from_canon(obj: Any) -> List[PQRow]:
    if isinstance(obj, dict):
        for key in ("items", "results", "data", "pqs", "questions"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break

    if not isinstance(obj, list):
        raise ValueError("Canon JSON is not a list (or dict containing a list).")

    rows: List[PQRow] = []
    for item in obj:
        if not isinstance(item, dict):
            continue

        q_text = first_non_empty(
            safe_get(item, ["question", "showAs"]),
            item.get("question"),
            safe_get(item, ["showAs"]),
        )

        heading = first_non_empty(
            safe_get(item, ["question", "debateSection", "showAs"]),
            item.get("heading"),
            safe_get(item, ["debateSection", "showAs"]),
        )

        dept = first_non_empty(
            safe_get(item, ["question", "to", "showAs"]),
            item.get("department"),
            safe_get(item, ["to", "showAs"]),
        )

        date_s = first_non_empty(
            safe_get(item, ["question", "date"]),
            item.get("date"),
            safe_get(item, ["contextDate"]),
            item.get("contextDate"),
            safe_get(item, ["question", "replyDate"]),
            item.get("replyDate"),
        )
        dt = parse_possible_date(date_s)
        if dt is None:
            continue

        if q_text and heading and dept:
            rows.append(PQRow(question=q_text, heading=heading, department=dept, date=dt))

    return rows


# -------------------------
# Metrics (top-1 only)
# -------------------------
def compute_day_top1_metrics(df_pred: pd.DataFrame, truth_col: str = "heading_eval") -> pd.DataFrame:
    df = df_pred.copy()
    df[truth_col] = df[truth_col].fillna("").astype(str)
    df["pred_heading"] = df["pred_heading"].fillna("").astype(str)
    df["pred_confidence"] = pd.to_numeric(df["pred_confidence"], errors="coerce").fillna(0.0)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date.astype(str)

    df["top1_correct"] = (df["pred_heading"] == df[truth_col])

    rows = []
    for day, g in df.groupby("date", sort=True):
        n = len(g)
        if n == 0:
            continue
        rows.append(
            {
                "date": day,
                "n": int(n),
                "avg_conf": float(g["pred_confidence"].mean()),
                "top1": float(g["top1_correct"].mean()),
                "n_top1": int(g["top1_correct"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["date"]).reset_index(drop=True)


def upsert_metrics(existing_csv: Path, new_metrics: pd.DataFrame, key: str = "date") -> None:
    if not existing_csv.exists():
        new_metrics.to_csv(existing_csv, index=False)
        return
    existing = pd.read_csv(existing_csv)
    if existing.empty:
        new_metrics.to_csv(existing_csv, index=False)
        return
    merged = existing[~existing[key].isin(set(new_metrics[key]))]
    merged = pd.concat([merged, new_metrics], ignore_index=True)
    merged = merged.sort_values([key]).reset_index(drop=True)
    merged.to_csv(existing_csv, index=False)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--heading-encoder", required=True)
    ap.add_argument("--department-encoder", required=True)
    ap.add_argument("--out-metrics", default="outputs/metrics_daily_keras.csv")
    ap.add_argument("--out-preds", default="outputs/predictions_keras_latest.csv")
    ap.add_argument("--maxlen", type=int, default=300)
    ap.add_argument(
        "--text-format",
        default="{question}",
        help="Format used to build text for the tokenizer. Training used question_clean.",
    )
    ap.add_argument(
        "--other-label",
        default="Other",
        help="Label used during training to collapse rare/unseen headings.",
    )
    args = ap.parse_args()

    out_metrics = Path(args.out_metrics)
    out_preds = Path(args.out_preds)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_preds.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching canon:", args.canon_url)
    r = requests.get(args.canon_url, timeout=60)
    r.raise_for_status()
    canon = r.json()

    rows = extract_rows_from_canon(canon)
    if not rows:
        raise SystemExit("No rows extracted from canon JSON (check field paths).")

    df = pd.DataFrame(
        [{"question": x.question, "heading": x.heading, "department": x.department, "date": x.date.isoformat()} for x in rows]
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    print(f"Extracted rows: {len(df):,}")

    print("Loading model:", args.model)
    model = tf.keras.models.load_model(args.model)

    print("Loading encoders/tokenizer...")
    with open(args.tokenizer, "rb") as f:
        tok = pickle.load(f)
    with open(args.heading_encoder, "rb") as f:
        heading_enc = pickle.load(f)  # expects inverse_transform, has .classes_
    with open(args.department_encoder, "rb") as f:
        dept_enc = pickle.load(f)  # expects transform, has .classes_

    # ---- Align inference text with training: use CLEANED canon question text ----
    df["question_clean"] = df["question"].map(clean_pq)

    texts = [
        args.text_format.format(department=d, question=q)
        for d, q in zip(df["department"].astype(str), df["question_clean"].astype(str))
    ]

    seq = tok.texts_to_sequences(texts)
    X_text = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=args.maxlen, padding="post", truncating="post")

    # Department as scalar int32 (match training: Input(shape=(), name="dept"))
    X_dept = dept_enc.transform(df["department"].astype(str).tolist()).astype("int32")

    # Predict
    probs = model.predict([X_text, X_dept], batch_size=256, verbose=1)
    pred_ids = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)

    # Decode headings
    pred_headings = heading_enc.inverse_transform(pred_ids)

    # ---- Align evaluation label space with training ----
    other_label = str(args.other_label)
    known_headings = set(getattr(heading_enc, "classes_", []))
    if other_label not in known_headings:
        print(
            f"[WARN] other_label='{other_label}' not found in heading_encoder.classes_. "
            "Truth-collapsing may not match training."
        )

    df["heading_eval"] = df["heading"].fillna("").astype(str).apply(
        lambda h: h if h in known_headings else other_label
    )

    out = df.copy()
    out["pred_heading"] = pred_headings
    out["pred_confidence"] = pred_conf.astype(float)

    out.to_csv(out_preds, index=False)
    print("Wrote predictions:", out_preds)

    metrics = compute_day_top1_metrics(out, truth_col="heading_eval")
    upsert_metrics(out_metrics, metrics, key="date")
    print("Updated daily metrics:", out_metrics)

    top1 = float((out["pred_heading"] == out["heading_eval"]).mean())
    print(f"Headline (all rows): top1={top1:.3f}")

    # Optional quick diagnostics
    unseen_headings_rate = float((out["heading_eval"] == other_label).mean())
    print(f"Truth mapped to '{other_label}': {unseen_headings_rate:.3f} of rows")


if __name__ == "__main__":
    main()
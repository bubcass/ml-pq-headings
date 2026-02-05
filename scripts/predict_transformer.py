#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer


# -----------------------------
# Text formatting MUST match training
# -----------------------------
TEXT_FORMAT = "[DEPT] {department} [SEP] {question_clean}"


def _try_clean_pq(text: str) -> str:
    """
    Best-effort cleaner:
    - If your project has preprocess.clean_pq, use it.
    - Otherwise, fall back to a light normalisation (no dependency).
    """
    try:
        from preprocess import clean_pq  # type: ignore
        return clean_pq(text)
    except Exception:
        # very light fallback – avoids breaking runs if preprocess.py isn't importable
        return " ".join(str(text).split())


def load_label_map(model_dir: Path) -> Dict[int, str]:
    """
    Expects label_map.json written by training (id -> label).
    """
    lm_path = model_dir / "label_map.json"
    if not lm_path.exists():
        raise SystemExit(
            f"Missing label_map.json in model dir: {model_dir}\n"
            "Your training script writes this file at the end; ensure it exists."
        )
    data = json.loads(lm_path.read_text(encoding="utf-8"))
    # JSON keys may be strings; normalise
    return {int(k): str(v) for k, v in data.items()}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def predict_batch(
    model: AutoModelForSequenceClassification,
    tokenizer: DebertaV2Tokenizer,
    texts: List[str],
    device: torch.device,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      topk_ids: (B, topk) int
      topk_probs: (B, topk) float
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length > 0 else 256,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    probs = torch.softmax(out.logits, dim=1)  # (B, C)
    topk_probs, topk_ids = torch.topk(probs, k=topk, dim=1)

    return topk_ids.detach().cpu().numpy(), topk_probs.detach().cpu().numpy()


def main(model_dir: str, inp: str, out: str, topk: int, batch_size: int) -> None:
    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model directory not found: {model_path}")

    in_path = Path(inp).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Model:", model_path)
    print("Input:", in_path)
    print("Output:", out_path)

    # Load label map
    id2label = load_label_map(model_path)

    # IMPORTANT: DeBERTa slow tokenizer (avoids fast-tokenizer edge cases)
    tokenizer = DebertaV2Tokenizer.from_pretrained(str(model_path))
    # Make sure max_length behaviour is predictable
    # (Training used MAX_LEN; here we respect tokenizer max unless it’s unset.)
    if not tokenizer.model_max_length or tokenizer.model_max_length > 1_000_000:
        tokenizer.model_max_length = 256  # sane default; you can override via saved config if needed

    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = pick_device()
    model.to(device)
    model.eval()

    df = pd.read_csv(in_path)

    if "question" not in df.columns:
        raise SystemExit(f"Input CSV must include a 'question' column. Found: {list(df.columns)}")

    # Department is optional, but your training used it, so use it if present.
    if "department" not in df.columns:
        df["department"] = ""

    # Clean + format
    questions = df["question"].fillna("").astype(str).tolist()
    departments = df["department"].fillna("").astype(str).tolist()

    question_clean = [_try_clean_pq(q) for q in questions]
    texts = [TEXT_FORMAT.format(department=d, question_clean=qc) for d, qc in zip(departments, question_clean)]

    # Predict
    all_pred_heading: List[str] = []
    all_pred_conf: List[float] = []

    # Alternates
    alt_headings: List[List[str]] = [[] for _ in range(topk - 1)]
    alt_confs: List[List[float]] = [[] for _ in range(topk - 1)]

    n = len(texts)
    print(f"Rows: {n}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Top-k: {topk}")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]

        topk_ids, topk_probs = predict_batch(model, tokenizer, batch_texts, device, topk=topk)

        # Map ids -> labels
        for i in range(len(batch_texts)):
            ids = topk_ids[i].tolist()
            probs = topk_probs[i].tolist()

            labels = [id2label.get(int(idx), f"__UNK_{idx}__") for idx in ids]

            all_pred_heading.append(labels[0])
            all_pred_conf.append(float(probs[0]))

            for j in range(1, topk):
                alt_headings[j - 1].append(labels[j])
                alt_confs[j - 1].append(float(probs[j]))

        if (start // batch_size) % 20 == 0:
            print(f"  processed {end}/{n}")

    # Write output
    out_df = df.copy()

    out_df["pred_heading"] = all_pred_heading
    out_df["pred_confidence"] = all_pred_conf

    for j in range(1, topk):
        out_df[f"alt_heading_{j}"] = alt_headings[j - 1]
        out_df[f"alt_conf_{j}"] = alt_confs[j - 1]

    out_df.to_csv(out_path, index=False)
    print("Wrote:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict PQ headings with a saved transformer model directory.")
    ap.add_argument(
        "--model",
        required=True,
        help="Path to saved model directory (must contain config + weights + label_map.json).",
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (must include 'question'; 'department' optional).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--topk", type=int, default=4, help="Number of suggestions to output (default: 4).")
    ap.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    args = ap.parse_args()

    if args.topk < 1:
        raise SystemExit("--topk must be >= 1")

    main(
        model_dir=args.model,
        inp=args.inp,
        out=args.out,
        topk=args.topk,
        batch_size=args.batch,
    )
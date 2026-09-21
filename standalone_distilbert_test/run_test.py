#!/usr/bin/env python3
"""Run the saved DistilBERT PQ-heading model on questions_ml_input.csv.

With no arguments, this script uses the CSV packaged beside it and writes a
top-10 DistilBERT prediction CSV to its output folder. The output schema matches
outputs/predictions_distilbert_latest.csv exactly. Progress includes elapsed time,
estimated remaining time, throughput, batches, and processed row count.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = Path("/Users/david/Developer/2026-01-24_ML_PQs")
DEFAULT_INPUT = SCRIPT_DIR / "questions_ml_input.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "questions_ml_predictions_distilbert_top10.csv"
TEXT_FORMAT = "[DEPT] {department} [SEP] {question_clean}"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log(message: str, start: float | None = None) -> None:
    timing = f" (elapsed {format_duration(time.perf_counter() - start)})" if start else ""
    print(f"[{time.strftime('%H:%M:%S')}] {message}{timing}", flush=True)


def load_cleaner(repo: Path):
    path = repo / "scripts" / "preprocess.py"
    if not path.exists():
        raise SystemExit(f"Existing preprocessing script not found: {path}")
    spec = importlib.util.spec_from_file_location("ml_pq_preprocess", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load preprocessing script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean_pq


def load_label_map(model_dir: Path) -> dict[int, str]:
    path = model_dir / "label_map.json"
    if not path.exists():
        raise SystemExit(f"DistilBERT label map not found: {path}")
    return {int(key): str(value).strip() for key, value in json.loads(path.read_text()).items()}


def load_max_length(model_dir: Path) -> int:
    path = model_dir / "metadata.json"
    if path.exists():
        return int(json.loads(path.read_text()).get("max_len", 192))
    return 192


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign PQ headings to the extracted CSV using the saved DistilBERT model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--model",
        type=Path,
        help="Model directory; defaults to artifacts/models_transformer/distilbert_pq_heading",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--review-threshold", type=float, default=0.30)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, help="Process only the first N rows for a smoke test")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    overall_start = time.perf_counter()
    repo = args.repo.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model_dir = (
        args.model.expanduser().resolve()
        if args.model
        else repo / "artifacts" / "models_transformer" / "distilbert_pq_heading"
    )

    if args.batch < 1 or args.top_k < 1:
        raise SystemExit("--batch and --top-k must be positive integers.")
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")
    if not model_dir.exists():
        raise SystemExit(f"DistilBERT model directory not found: {model_dir}")

    log("Starting DistilBERT standalone CSV test")
    log(f"Input:  {input_path}")
    log(f"Model:  {model_dir}")
    log(f"Output: {output_path}")

    stage_start = time.perf_counter()
    log("Reading and validating CSV")
    data = pd.read_csv(input_path)
    missing = {"question", "department"} - set(data.columns)
    if missing:
        raise SystemExit(f"Input is missing required columns: {sorted(missing)}")
    if args.limit is not None:
        data = data.head(args.limit).copy()
        log(f"Smoke-test limit applied: {len(data):,} rows")
    if data.empty:
        raise SystemExit("Input CSV has no rows to process.")
    if data[["question", "department"]].isna().any().any():
        raise SystemExit("Input has missing question or department values.")

    trained_departments = set(
        pd.read_csv(repo / "data" / "ML_training_data.csv", usecols=["department"])[
            "department"
        ]
        .dropna()
        .astype(str)
        .unique()
    )
    unseen_departments = sorted(set(data["department"].astype(str)) - trained_departments)
    if unseen_departments:
        raise SystemExit(
            "Input contains department labels not seen in training:\n  - "
            + "\n  - ".join(unseen_departments)
        )

    # Recompute with the repository's current cleaner so the test uses the same path
    # as established model runs, even if the input CSV already contains question_clean.
    clean_pq = load_cleaner(repo)
    data["question_clean"] = data["question"].fillna("").astype(str).map(clean_pq)
    if data["question_clean"].str.strip().eq("").any():
        raise SystemExit("At least one question became empty after preprocessing.")
    log(
        f"Validated {len(data):,} rows across {data['department'].nunique()} departments",
        stage_start,
    )

    labels = load_label_map(model_dir)
    max_length = load_max_length(model_dir)
    device = select_device(args.device)

    stage_start = time.perf_counter()
    log(f"Loading DistilBERT on {device.type}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )
    model.to(device)
    model.eval()
    if args.top_k > model.config.num_labels:
        raise SystemExit(
            f"Requested top-k {args.top_k} exceeds {model.config.num_labels} model labels."
        )
    log(
        f"Model loaded; batch={args.batch}, top-k={args.top_k}, max_length={max_length}",
        stage_start,
    )

    texts = [
        TEXT_FORMAT.format(department=department, question_clean=question_clean)
        for department, question_clean in zip(data["department"], data["question_clean"])
    ]
    prediction_ids: list[list[int]] = []
    prediction_scores: list[list[float]] = []
    starts = range(0, len(texts), args.batch)
    total_batches = (len(texts) + args.batch - 1) // args.batch
    inference_start = time.perf_counter()
    log(f"Assigning headings to {len(texts):,} questions")

    progress = tqdm(
        starts,
        total=total_batches,
        desc="Assigning headings",
        unit="batch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    for start in progress:
        end = min(start + args.batch, len(texts))
        encoded = tokenizer(
            texts[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        probabilities = torch.softmax(model(**encoded).logits, dim=1)
        scores, ids = torch.topk(probabilities, k=args.top_k, dim=1)
        prediction_ids.extend(ids.cpu().tolist())
        prediction_scores.extend(scores.cpu().tolist())
        progress.set_postfix_str(f"rows={end:,}/{len(texts):,}")

    log("Heading assignment complete", inference_start)

    results = data.copy()
    results["pred_heading"] = [labels[row[0]] for row in prediction_ids]
    results["pred_confidence"] = [row[0] for row in prediction_scores]
    # Match the established 2026 output convention: alt_heading_1 is the
    # top-ranked result (and therefore duplicates pred_heading), through rank 10.
    for rank in range(args.top_k):
        results[f"alt_heading_{rank + 1}"] = [labels[row[rank]] for row in prediction_ids]
        results[f"alt_conf_{rank + 1}"] = [row[rank] for row in prediction_scores]
    needs_review = (
        results["pred_confidence"].lt(args.review_threshold)
        | results["pred_heading"].eq("Other")
    )

    # The Word-derived input predates publication of the official headings. Keep
    # the formal ground-truth field empty rather than substituting a prediction.
    if "heading" not in results.columns:
        results["heading"] = ""
    if "date" not in results.columns:
        results["date"] = ""

    formal_columns = [
        "question",
        "heading",
        "department",
        "date",
        "pred_confidence",
        "pred_heading",
    ]
    for rank in range(1, args.top_k + 1):
        formal_columns.extend([f"alt_heading_{rank}", f"alt_conf_{rank}"])
    output = results[formal_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")

    confidence = output["pred_confidence"]
    review_count = int(needs_review.sum())
    other_count = int(output["pred_heading"].eq("Other").sum())
    print("\nPrediction summary")
    print("------------------")
    print(f"Rows processed:        {len(output):,}")
    print(f"Distinct headings:     {output['pred_heading'].nunique():,}")
    print(f"Mean confidence:       {confidence.mean():.3f}")
    print(f"Median confidence:     {confidence.median():.3f}")
    print(
        f"Needs review:          {review_count:,} ({review_count / len(output):.1%})"
    )
    print(f"Predicted as Other:    {other_count:,}")
    print(f"Inference elapsed:     {format_duration(time.perf_counter() - inference_start)}")
    print(f"Total elapsed:         {format_duration(time.perf_counter() - overall_start)}")
    print(f"Output:                {output_path}")
    print("\nMost frequent predicted headings")
    print(output["pred_heading"].value_counts().head(15).to_string())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user; no partial output was written.", file=sys.stderr)
        raise SystemExit(130)

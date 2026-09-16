#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_pq


# -------------------------
# Helpers
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


# -------------------------
# Extract PQ rows from canon
# -------------------------
@dataclass
class PQRow:
    question: str
    heading: str
    department: str
    date: datetime


def extract_rows_from_canon(obj: Any, stats: Optional[Dict[str, int]] = None) -> List[PQRow]:
    if isinstance(obj, dict):
        for key in ("items", "results", "data", "pqs", "questions"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break

    if not isinstance(obj, list):
        raise ValueError("Canon JSON is not a list (or dict containing a list).")

    counts = stats if stats is not None else {}
    counts.update(
        {
            "source_items": len(obj),
            "accepted": 0,
            "non_object": 0,
            "missing_date": 0,
            "missing_question": 0,
            "missing_heading": 0,
            "missing_department": 0,
        }
    )

    rows: List[PQRow] = []
    for item in obj:
        if not isinstance(item, dict):
            counts["non_object"] += 1
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
            counts["missing_date"] += 1
            continue

        if not q_text:
            counts["missing_question"] += 1
        if not heading:
            counts["missing_heading"] += 1
        if not dept:
            counts["missing_department"] += 1
        if not (q_text and heading and dept):
            continue

        rows.append(PQRow(question=q_text, heading=heading, department=dept, date=dt))
        counts["accepted"] += 1

    return rows


# -------------------------
# Model inference
# -------------------------
def predict_topk(
    df: pd.DataFrame,
    model_id_or_path: str,
    text_format: str,
    topk: int = 10,
    batch_size: int = 16,
    max_len: int = 256,
    tokenizer_fast: bool = True,
    device: str = "auto",
) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=tokenizer_fast)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    model.eval()

    if device == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    model.to(dev)

    # Load label_map.json if present
    label_map: Dict[int, str] = {}
    label_map_path = Path(model_id_or_path) / "label_map.json"
    if label_map_path.exists():
        label_map = {
            int(k): str(v).strip()
            for k, v in json.loads(label_map_path.read_text(encoding="utf-8")).items()
        }
    else:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            p = hf_hub_download(repo_id=model_id_or_path, filename="label_map.json")
            label_map = {
                int(k): str(v).strip()
                for k, v in json.loads(Path(p).read_text(encoding="utf-8")).items()
            }
        except Exception:
            if getattr(model.config, "id2label", None):
                label_map = {int(i): str(s).strip() for i, s in model.config.id2label.items()}

    def build_text(row: pd.Series) -> str:
        return text_format.format(
            department=row["department"],
            question=row["question"],
            question_clean=row["question_clean"],
        )

    texts = [build_text(r) for _, r in df.iterrows()]

    all_top_idx: List[np.ndarray] = []
    all_top_prob: List[np.ndarray] = []

    with torch.no_grad():
        batches = range(0, len(texts), batch_size)
        for i in tqdm(batches, desc=f"Predicting on {dev}", unit="batch"):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(dev)

            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = torch.topk(probs, k=topk, dim=-1)

            all_top_idx.append(top_idx.detach().cpu().numpy())
            all_top_prob.append(top_prob.detach().cpu().numpy())

    top_idx = np.vstack(all_top_idx)
    top_prob = np.vstack(all_top_prob)

    out = df.copy()
    out["pred_label_id"] = top_idx[:, 0].astype(int)
    out["pred_confidence"] = top_prob[:, 0].astype(float)
    out["pred_heading"] = out["pred_label_id"].map(lambda i: label_map.get(i, str(i)))

    for k in range(topk):
        out[f"alt_heading_{k+1}"] = top_idx[:, k].astype(int)
        out[f"alt_conf_{k+1}"] = top_prob[:, k].astype(float)
        out[f"alt_heading_{k+1}"] = out[f"alt_heading_{k+1}"].map(lambda i: label_map.get(int(i), str(i)))

    return out.drop(columns=["pred_label_id"])


# -------------------------
# Metrics
# -------------------------
def topk_hit_row(row: pd.Series, truth_col: str, k: int) -> bool:
    truth = str(row[truth_col]).strip()
    if not truth:
        return False
    for i in range(1, k + 1):
        c = f"alt_heading_{i}"
        if c in row and pd.notna(row[c]) and str(row[c]).strip() == truth:
            return True
    return False


def compute_day_metrics_multi(
    df_pred: pd.DataFrame,
    truth_col: str = "heading",
    ks: Optional[List[int]] = None,
    model_labels: Optional[set[str]] = None,
) -> pd.DataFrame:
    df = df_pred.copy()
    ks = ks or [1, 4, 10]

    # Types
    df[truth_col] = df[truth_col].fillna("").astype(str).str.strip()
    df["pred_heading"] = df["pred_heading"].fillna("").astype(str).str.strip()
    df["pred_confidence"] = pd.to_numeric(df["pred_confidence"], errors="coerce").fillna(0.0)

    # Date -> YYYY-MM-DD string
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date.astype(str)

    # Per-row flags
    df["top1_correct"] = (df["pred_heading"] == df[truth_col])
    if model_labels is not None:
        df["heading_in_model"] = df[truth_col].isin(model_labels)

    # Ensure k list is unique and sorted
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks)

    for k in ks:
        if k == 1:
            df["top1"] = df["top1_correct"]
        else:
            df[f"top{k}"] = df.apply(lambda r: topk_hit_row(r, truth_col, k=k), axis=1)

    # outside of max_k (e.g. outside top-10)
    df[f"outside_top{max_k}"] = ~df[f"top{max_k}"] if max_k != 1 else ~df["top1"]

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    rows = []
    for day, g in df.groupby("date", sort=True):
        n = len(g)
        if n == 0:
            continue

        base: Dict[str, Any] = {
            "date": day,
            "n": int(n),
            "avg_conf": float(g["pred_confidence"].mean()),
        }

        if "heading_in_model" in g:
            known = g[g["heading_in_model"]]
            base["n_heading_in_model"] = int(len(known))
            base["heading_coverage"] = float(len(known) / n)
            base["top1_known_headings"] = float(known["top1_correct"].mean()) if len(known) else np.nan

        # headline rates + counts
        base["top1"] = float(g["top1_correct"].mean())
        base["top4"] = float(g["top4"].mean()) if "top4" in g else np.nan
        base["top10"] = float(g["top10"].mean()) if "top10" in g else np.nan
        base[f"outside_top{max_k}"] = float(g[f"outside_top{max_k}"].mean())

        base["n_top1"] = int(g["top1_correct"].sum())
        if "top4" in g:
            base["n_top4"] = int(g["top4"].sum())
            base["n_top4_only"] = int((g["top4"] & ~g["top1_correct"]).sum())
            if "heading_in_model" in g:
                base["top4_known_headings"] = float(known["top4"].mean()) if len(known) else np.nan
        if "top10" in g:
            base["n_top10"] = int(g["top10"].sum())
            base["n_top10_only"] = int((g["top10"] & ~g.get("top4", g["top1_correct"])).sum())
            if "heading_in_model" in g:
                base["top10_known_headings"] = float(known["top10"].mean()) if len(known) else np.nan
        base[f"n_outside_top{max_k}"] = int(g[f"outside_top{max_k}"].sum())

        # confidence threshold automation stats
        for t in thresholds:
            auto = g[g["pred_confidence"] >= t]
            review = g[g["pred_confidence"] < t]

            base[f"review_rate_t{t:.2f}"] = float(len(review) / n)
            base[f"auto_coverage_t{t:.2f}"] = float(len(auto) / n)

            if len(auto) > 0:
                # wrong at top-1 among auto items
                base[f"auto_error_rate_t{t:.2f}"] = float((~auto["top1_correct"]).mean())

                # “rescue” rates: wrong at top-1 but fixable if a human picks from top-4/top-10
                if "top4" in g:
                    base[f"auto_top4_rescue_t{t:.2f}"] = float((~auto["top1_correct"] & auto["top4"]).mean())
                if "top10" in g:
                    base[f"auto_top10_rescue_t{t:.2f}"] = float((~auto["top1_correct"] & auto["top10"]).mean())
            else:
                base[f"auto_error_rate_t{t:.2f}"] = np.nan
                if "top4" in g:
                    base[f"auto_top4_rescue_t{t:.2f}"] = np.nan
                if "top10" in g:
                    base[f"auto_top10_rescue_t{t:.2f}"] = np.nan

        rows.append(base)

    return pd.DataFrame(rows).sort_values(["date"]).reset_index(drop=True)


def write_csv_atomic(df: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_model_contract(model_id_or_path: str) -> Dict[str, Any]:
    model_path = Path(model_id_or_path)
    metadata_path = model_path / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing model metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in ("max_len", "text_format"):
        if field not in metadata:
            raise SystemExit(f"Model metadata is missing required field {field!r}: {metadata_path}")

    fields = {
        name
        for _, name, _, _ in string.Formatter().parse(str(metadata["text_format"]))
        if name is not None
    }
    supported = {"department", "question", "question_clean"}
    unsupported = fields - supported
    if unsupported:
        raise SystemExit(f"Unsupported fields in model text_format: {sorted(unsupported)}")

    # DistilBERT training used AutoTokenizer's default fast tokenizer. DeBERTa's
    # metadata explicitly records false because it was trained with the slow tokenizer.
    metadata.setdefault("tokenizer_fast", True)
    return metadata


def load_canon(args: argparse.Namespace) -> tuple[Any, str]:
    if args.canon_path:
        source_path = Path(args.canon_path)
        raw = source_path.read_bytes()
        source = str(source_path)
    else:
        print("Fetching canon:", args.canon_url)
        response = requests.get(args.canon_url, timeout=60)
        response.raise_for_status()
        raw = response.content
        source = args.canon_url

    digest = hashlib.sha256(raw).hexdigest()
    print(f"Canon source: {source}")
    print(f"Canon SHA-256: {digest}")
    return json.loads(raw), digest


def validate_results(
    source_df: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    topk: int,
) -> None:
    if len(predictions) != len(source_df):
        raise ValueError(f"Prediction row count changed: {len(source_df)} -> {len(predictions)}")
    if predictions.empty:
        raise ValueError("Prediction output is empty.")
    if predictions[["question", "heading", "department", "date"]].isna().any().any():
        raise ValueError("Prediction output contains missing core fields.")
    if not predictions["pred_confidence"].between(0.0, 1.0).all():
        raise ValueError("Prediction confidence is outside [0, 1].")
    if not predictions["pred_heading"].equals(predictions["alt_heading_1"]):
        raise ValueError("pred_heading and alt_heading_1 do not match.")
    if not np.allclose(predictions["pred_confidence"], predictions["alt_conf_1"], rtol=0, atol=1e-7):
        raise ValueError("pred_confidence and alt_conf_1 do not match.")
    required_alts = {f"alt_heading_{i}" for i in range(1, topk + 1)} | {
        f"alt_conf_{i}" for i in range(1, topk + 1)
    }
    missing_alts = required_alts - set(predictions.columns)
    if missing_alts:
        raise ValueError(f"Missing top-k output columns: {sorted(missing_alts)}")
    if int(metrics["n"].sum()) != len(predictions):
        raise ValueError("Daily metric counts do not sum to the prediction row count.")
    if metrics["date"].duplicated().any():
        raise ValueError("Daily metrics contain duplicate dates.")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--canon-url", help="Raw URL to PQs_2026_paginated.json")
    source.add_argument("--canon-path", help="Local snapshot of PQs_2026_paginated.json")
    ap.add_argument("--model", required=True, help="HF repo id or local model path")
    ap.add_argument("--out-metrics", default="outputs/metrics_daily.csv")
    ap.add_argument("--out-preds", default="outputs/predictions_latest.csv")

    # Run inference once to max-k; compute metrics for 1/4/10 from the same outputs
    ap.add_argument("--max-k", type=int, default=10, help="How many alternatives to generate (alt_heading_1..max-k)")
    ap.add_argument("--report-ks", default="1,4,10", help="Comma-separated ks to report (subset of max-k)")

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, help="Override the model metadata max_len")
    ap.add_argument(
        "--text-format",
        help="Override the model metadata text_format",
    )
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.report_ks.split(",") if x.strip()]
    if not ks:
        raise SystemExit("report-ks parsed to empty list.")
    if max(ks) > args.max_k:
        raise SystemExit(f"max(report-ks)={max(ks)} exceeds --max-k={args.max_k}. Increase --max-k.")

    out_metrics = Path(args.out_metrics)
    out_preds = Path(args.out_preds)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_preds.parent.mkdir(parents=True, exist_ok=True)

    contract = read_model_contract(args.model)
    max_len = args.max_len if args.max_len is not None else int(contract["max_len"])
    text_format = args.text_format if args.text_format is not None else str(contract["text_format"])
    tokenizer_fast = bool(contract["tokenizer_fast"])
    print(f"Inference contract: max_len={max_len}, tokenizer_fast={tokenizer_fast}")
    print(f"Inference text format: {text_format}")

    canon, _canon_sha256 = load_canon(args)

    extraction_stats: Dict[str, int] = {}
    rows = extract_rows_from_canon(canon, extraction_stats)
    if not rows:
        raise SystemExit("No rows extracted from canon JSON (check field paths).")

    df = pd.DataFrame(
        [{"question": x.question, "heading": x.heading, "department": x.department, "date": x.date.isoformat()} for x in rows]
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["question_clean"] = df["question"].map(clean_pq)

    print(f"Extracted rows: {len(df):,}")
    print("Extraction audit:", json.dumps(extraction_stats, sort_keys=True))

    print("Running inference:", args.model)
    df_pred = predict_topk(
        df,
        model_id_or_path=args.model,
        text_format=text_format,
        topk=args.max_k,
        batch_size=args.batch,
        max_len=max_len,
        tokenizer_fast=tokenizer_fast,
        device="auto",
    )

    label_map_path = Path(args.model) / "label_map.json"
    if not label_map_path.exists():
        raise SystemExit(f"Missing local label map required for coverage metrics: {label_map_path}")
    model_labels = {
        str(value).strip()
        for value in json.loads(label_map_path.read_text(encoding="utf-8")).values()
    }
    metrics = compute_day_metrics_multi(df_pred, truth_col="heading", ks=ks, model_labels=model_labels)
    validate_results(df, df_pred, metrics, args.max_k)

    # Keep the established prediction schema: question_clean is an internal inference field.
    df_pred = df_pred.drop(columns=["question_clean"])
    write_csv_atomic(df_pred, out_preds)
    write_csv_atomic(metrics, out_metrics)
    print("Wrote validated predictions:", out_preds)
    print("Wrote validated full-year metrics:", out_metrics)

    # Headline for the whole corpus
    top1 = float((df_pred["pred_heading"] == df_pred["heading"]).mean())
    print(f"Headline (all rows): top1={top1:.3f}")
    known = df_pred["heading"].isin(model_labels)
    print(f"Heading vocabulary coverage: {known.mean():.3f} ({known.sum():,}/{len(known):,})")
    if known.any():
        print(f"Headline (known headings): top1={(df_pred.loc[known, 'pred_heading'] == df_pred.loc[known, 'heading']).mean():.3f}")
    for k in ks:
        if k == 1:
            continue
        rate = float(df_pred.apply(lambda r: topk_hit_row(r, "heading", k), axis=1).mean())
        print(f"Headline (all rows): top{k}={rate:.3f}")
    max_k = max(ks)
    outside = 1.0 - float(df_pred.apply(lambda r: topk_hit_row(r, "heading", max_k), axis=1).mean())
    print(f"Headline (all rows): outside_top{max_k}={outside:.3f}")


if __name__ == "__main__":
    main()

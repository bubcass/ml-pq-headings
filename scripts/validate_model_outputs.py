#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = ["question", "heading", "department", "date"]


def validate_one(predictions_path: Path, metrics_path: Path, name: str) -> pd.DataFrame:
    # Heading text can legitimately resemble a conventional CSV missing-value token.
    predictions = pd.read_csv(predictions_path, low_memory=False, keep_default_na=False)
    metrics = pd.read_csv(metrics_path)

    required_predictions = set(CORE_COLUMNS) | {"pred_heading", "pred_confidence"}
    required_predictions |= {f"alt_heading_{i}" for i in range(1, 11)}
    required_predictions |= {f"alt_conf_{i}" for i in range(1, 11)}
    missing = required_predictions - set(predictions.columns)
    if missing:
        raise SystemExit(f"{name}: missing prediction columns: {sorted(missing)}")

    required_metrics = {
        "date",
        "n",
        "top1",
        "top4",
        "top10",
        "n_heading_in_model",
        "heading_coverage",
        "top1_known_headings",
        "top4_known_headings",
        "top10_known_headings",
    }
    missing = required_metrics - set(metrics.columns)
    if missing:
        raise SystemExit(f"{name}: missing metric columns: {sorted(missing)}")

    if predictions.empty or metrics.empty:
        raise SystemExit(f"{name}: predictions or metrics are empty")
    if predictions[CORE_COLUMNS].isna().any().any():
        raise SystemExit(f"{name}: missing values in core prediction columns")
    if not predictions["pred_confidence"].between(0.0, 1.0).all():
        raise SystemExit(f"{name}: confidence outside [0, 1]")
    if not predictions["pred_heading"].equals(predictions["alt_heading_1"]):
        raise SystemExit(f"{name}: pred_heading differs from alt_heading_1")
    if not np.allclose(predictions["pred_confidence"], predictions["alt_conf_1"], rtol=0, atol=1e-7):
        raise SystemExit(f"{name}: pred_confidence differs from alt_conf_1")
    if int(metrics["n"].sum()) != len(predictions):
        raise SystemExit(f"{name}: daily counts do not sum to prediction rows")
    if metrics["date"].duplicated().any():
        raise SystemExit(f"{name}: duplicate metric dates")
    for column in ("top1", "top4", "top10", "heading_coverage"):
        if not metrics[column].between(0.0, 1.0).all():
            raise SystemExit(f"{name}: {column} outside [0, 1]")

    print(
        f"{name}: {len(predictions):,} predictions, {len(metrics):,} dates, "
        f"{metrics['n_heading_in_model'].sum():,.0f} headings in model vocabulary"
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paired DistilBERT and DeBERTa annual outputs.")
    parser.add_argument("--distil-preds", type=Path, required=True)
    parser.add_argument("--distil-metrics", type=Path, required=True)
    parser.add_argument("--deberta-preds", type=Path, required=True)
    parser.add_argument("--deberta-metrics", type=Path, required=True)
    args = parser.parse_args()

    distil = validate_one(args.distil_preds, args.distil_metrics, "DistilBERT")
    deberta = validate_one(args.deberta_preds, args.deberta_metrics, "DeBERTa")

    if len(distil) != len(deberta):
        raise SystemExit("Models were evaluated on different row counts")
    if not distil[CORE_COLUMNS].fillna("").astype(str).equals(
        deberta[CORE_COLUMNS].fillna("").astype(str)
    ):
        raise SystemExit("Models were not evaluated on the same questions in the same order")

    print("Paired validation passed: both models used the identical annual question set.")


if __name__ == "__main__":
    main()

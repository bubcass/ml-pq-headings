#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def topk_hit(row: pd.Series, truth_col: str, alt_cols: list[str]) -> bool:
    truth = str(row[truth_col]) if pd.notna(row[truth_col]) else ""
    if not truth:
        return False
    for c in alt_cols:
        if c in row and pd.notna(row[c]) and str(row[c]) == truth:
            return True
    return False


def main(pred_csv: str, truth_col: str = "heading") -> None:
    df = pd.read_csv(pred_csv)

    required = {"question", "department", truth_col, "pred_heading", "pred_confidence"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in predictions file: {sorted(missing)}\nFound: {list(df.columns)}")

    n = len(df)
    if n == 0:
        raise SystemExit("No rows found.")

    # Clean types
    df[truth_col] = df[truth_col].fillna("").astype(str)
    df["pred_heading"] = df["pred_heading"].fillna("").astype(str)
    df["pred_confidence"] = pd.to_numeric(df["pred_confidence"], errors="coerce")

    # Accuracy (top-1 exact match)
    top1 = (df["pred_heading"] == df[truth_col]).mean()

    # Top-3 / Top-5 depending on what you output
    alt_cols = [c for c in ["alt_heading_1", "alt_heading_2", "alt_heading_3", "alt_heading_4", "alt_heading_5"] if c in df.columns]

    if alt_cols:
        # alt_heading_1 is usually the same as pred_heading, but we handle either case.
        topk = df.apply(lambda r: topk_hit(r, truth_col, ["pred_heading"] + alt_cols), axis=1).mean()
    else:
        topk = np.nan

    # Review queue rates at some candidate thresholds
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]
    review_rates = []
    for t in thresholds:
        review_rates.append((t, float((df["pred_confidence"] < t).mean())))

    # Quick “Other” stats (whether predicted heading ever equals "Other")
    other_rate = float((df["pred_heading"].str.lower() == "other".lower()).mean())

    # By-department accuracy (top-1)
    dept_summary = (
        df.assign(correct=(df["pred_heading"] == df[truth_col]))
          .groupby("department", dropna=False)
          .agg(
              n=("department", "size"),
              top1=("correct", "mean"),
              avg_conf=("pred_confidence", "mean"),
          )
          .sort_values(["n"], ascending=False)
          .head(20)
          .reset_index()
    )

    # Print report
    print("\n=== Holdout evaluation ===")
    print(f"Rows: {n}")
    print(f"Top-1 exact match: {top1:.4f}  ({top1*100:.1f}%)")

    if not np.isnan(topk):
        k = len(alt_cols) + 1  # + pred_heading
        print(f"Top-{k} hit rate:    {topk:.4f}  ({topk*100:.1f}%)  [pred_heading + {len(alt_cols)} alternates]")
    else:
        print("Top-k hit rate:     (no alt_heading_* columns found)")

    print(f'Predicted "Other":  {other_rate:.4f}  ({other_rate*100:.2f}%)')

    print("\nReview queue rate by confidence threshold (pred_confidence < t):")
    for t, r in review_rates:
        print(f"  t={t:.2f}: {r:.4f}  ({r*100:.1f}%)  -> ~{int(round(r*n))} items out of {n}")

    print("\nTop departments by volume (top-1 accuracy + avg confidence):")
    # Print as a neat table
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 120):
        print(dept_summary.to_string(index=False))

    # Save a small “errors” file for inspection
    out_dir = Path(pred_csv).resolve().parent
    errors_path = out_dir / "2026_errors_sample.csv"

    errors = df[df["pred_heading"] != df[truth_col]].copy()
    if len(errors):
        # show the most confident wrong ones (often taxonomy ambiguity) and least confident ones (hard cases)
        errors["pred_confidence"] = errors["pred_confidence"].fillna(-1)
        sample = pd.concat([
            errors.sort_values("pred_confidence", ascending=False).head(50),
            errors.sort_values("pred_confidence", ascending=True).head(50),
        ]).drop_duplicates()

        keep_cols = [c for c in ["question", "department", truth_col, "pred_heading", "pred_confidence",
                                 "alt_heading_1", "alt_conf_1", "alt_heading_2", "alt_conf_2", "alt_heading_3", "alt_conf_3"] if c in df.columns]
        sample[keep_cols].to_csv(errors_path, index=False)
        print(f"\nWrote error sample for review: {errors_path}  ({len(sample)} rows)")
    else:
        print("\nNo errors found (unlikely!)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate holdout predictions against ground-truth headings.")
    ap.add_argument("--pred", required=True, help="CSV containing ground truth + predictions (output of predict_from_csv.py)")
    ap.add_argument("--truth-col", default="heading", help="Name of the ground-truth heading column (default: heading)")
    args = ap.parse_args()

    main(args.pred, truth_col=args.truth_col)
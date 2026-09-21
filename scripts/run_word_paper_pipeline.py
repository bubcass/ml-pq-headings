#!/usr/bin/env python3
"""Run the Word-question-paper to DistilBERT top-10 daily pipeline.

The final CSV begins with the established prediction schema and appends PQ
references and Word-paper provenance so it can be used immediately and later
reconciled with published open data.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from extract_question_paper import paper_date, read_docx_paragraphs


TOP_K = 10
FORMAL_COLUMNS = [
    "question",
    "heading",
    "department",
    "date",
    "pred_confidence",
    "pred_heading",
]
for rank in range(1, TOP_K + 1):
    FORMAL_COLUMNS.extend([f"alt_heading_{rank}", f"alt_conf_{rank}"])

PROVENANCE_COLUMNS = [
    "refNo",
    "question_number",
    "deputy",
    "deputy_source",
    "answer_type",
    "language",
    "raw_question",
    "question_clean",
    "department_paper",
    "source_question_paragraph",
    "source_pq_paragraph",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Extract a question-paper DOCX and assign DistilBERT top-10 headings.")
    parser.add_argument("--input", type=Path, required=True, help="Question-paper DOCX")
    parser.add_argument(
        "--ml-input-dir",
        type=Path,
        default=root / "outputs" / "word_ml_inputs",
        help="Directory for extracted ML-input CSVs",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=root / "outputs" / "word_ml_predictions",
        help="Directory for final DistilBERT prediction CSVs",
    )
    parser.add_argument("--date", help="Paper date override in YYYY-MM-DD format")
    parser.add_argument("--batch", type=int, default=32, help="DistilBERT batch size")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def validate_pair(inputs: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if len(inputs) != len(predictions):
        raise SystemExit(f"Prediction row count differs from extracted input: {len(inputs):,} != {len(predictions):,}")
    missing_predictions = set(FORMAL_COLUMNS) - set(predictions.columns)
    missing_inputs = set(PROVENANCE_COLUMNS) - set(inputs.columns)
    if missing_predictions:
        raise SystemExit(f"Prediction output is missing required columns: {sorted(missing_predictions)}")
    if missing_inputs:
        raise SystemExit(f"Extracted input is missing provenance columns: {sorted(missing_inputs)}")
    for column in ("question", "department", "date"):
        if not inputs[column].astype(str).equals(predictions[column].astype(str)):
            raise SystemExit(f"Input and prediction {column!r} columns are not aligned.")
    if not predictions["pred_heading"].astype(str).equals(predictions["alt_heading_1"].astype(str)):
        raise SystemExit("pred_heading does not match alt_heading_1.")
    if not predictions["pred_confidence"].between(0.0, 1.0).all():
        raise SystemExit("Prediction confidence is outside [0, 1].")
    if inputs["refNo"].astype(str).str.strip().eq("").any():
        raise SystemExit("Extracted input contains a blank PQ reference.")


def write_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=output_path.parent, delete=False, prefix=f".{output_path.name}."
    ) as handle:
        df.to_csv(handle, index=False)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, output_path)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    paper = args.input.expanduser().resolve()
    ml_input_dir = args.ml_input_dir.expanduser().resolve()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    ml_input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    source_date = paper_date(read_docx_paragraphs(paper), args.date)
    input_csv = ml_input_dir / f"{source_date}_questions_ml_input.csv"
    final_csv = prediction_dir / f"{source_date}_predictions_distilbert_top10.csv"

    print("Step 1/3: extracting the Word question paper", flush=True)
    extract_command = [
        sys.executable,
        str(root / "scripts" / "extract_question_paper.py"),
        "--input",
        str(paper),
        "--output",
        str(input_csv),
    ]
    if args.date:
        extract_command.extend(["--date", args.date])
    subprocess.run(extract_command, check=True)

    with tempfile.NamedTemporaryFile(
        suffix=".csv", dir=prediction_dir, delete=False, prefix=f".{source_date}_distilbert."
    ) as handle:
        prediction_temp = Path(handle.name)

    try:
        print("Step 2/3: assigning DistilBERT top-10 headings", flush=True)
        predict_command = [
            sys.executable,
            str(root / "standalone_distilbert_test" / "run_test.py"),
            "--input",
            str(input_csv),
            "--output",
            str(prediction_temp),
            "--repo",
            str(root),
            "--batch",
            str(args.batch),
            "--top-k",
            str(TOP_K),
            "--device",
            args.device,
        ]
        subprocess.run(predict_command, check=True)

        print("Step 3/3: validating and adding PQ provenance", flush=True)
        inputs = pd.read_csv(input_csv, keep_default_na=False, low_memory=False)
        predictions = pd.read_csv(prediction_temp, keep_default_na=False, low_memory=False)
        validate_pair(inputs, predictions)

        output = predictions[FORMAL_COLUMNS].copy()
        for column in PROVENANCE_COLUMNS:
            output[column] = inputs[column]
        write_csv_atomic(output, final_csv)
    finally:
        if prediction_temp.exists():
            prediction_temp.unlink()

    print(f"Completed {len(output):,} questions.")
    print(f"ML input: {input_csv}")
    print(f"Predictions with PQ references: {final_csv}")


if __name__ == "__main__":
    main()

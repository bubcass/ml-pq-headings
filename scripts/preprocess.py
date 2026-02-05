#!/usr/bin/env python3
"""
Clean parliamentary question text for ML heading prediction.

- Removes predictable procedural boilerplate (Deputy X asked..., statement on the matter, etc.)
- Reduces "department leakage" by stripping "Minister for ..." etc. from the question text
- Preserves meaningful tokens including acronyms and years (HSE, 2025, etc.)
- Writes a new CSV with a `question_clean` column

Usage:
  python clean_pqs.py --in questionsForML.csv --out questionsForML_clean.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# ----------------------------
# PQ-aware cleaning rules
# ----------------------------

# Procedural patterns: remove these wherever they appear (carefully)
PATTERNS = [
    # Leading numbering like "1." or "12."
    (r"^\s*\d+\.\s*", " "),

    # "Deputy <name> asked ..."
    (r"^\s*deputy\s+[a-záéíóú\-\'\.\s]+\s+asked\s+", " "),

    # "asked the Minister for X ..."
    (r"\basked\s+the\s+minister\s+for\s+[^,]+,?\s*", " "),

    # "asked the Minister of State at <department> ..."
    (r"\basked\s+the\s+minister\s+of\s+state\s+at\s+[^,]+,?\s*", " "),

    # References to officeholders used as departments in PQs
    (r"\bthe\s+taoiseach\b", " "),
    (r"\bthe\s+tánaiste\b", " "),
    (r"\btaoiseach\b", " "),
    (r"\btánaiste\b", " "),

    # Tail boilerplate (common endings)
    (r"\b(if|whether)\s+he\s+will\s+make\s+a\s+statement\s+on\s+the\s+matter\b", " "),
    (r"\b(if|whether)\s+she\s+will\s+make\s+a\s+statement\s+on\s+the\s+matter\b", " "),
    (r"\band\s+if\s+he\s+will\s+make\s+a\s+statement\b", " "),
    (r"\band\s+if\s+she\s+will\s+make\s+a\s+statement\b", " "),
    (r"\band\s+if\s+so\b", " "),
    (r"\band\s+if\s+not\b", " "),

    # Generic low-signal phrasing
    (r"\bto\s+outline\b", " "),
    (r"\bto\s+provide\s+details\b", " "),
    (r"\bin\s+relation\s+to\b", " "),
]

# Compile once for speed (case-insensitive)
COMPILED = [(re.compile(pat, flags=re.IGNORECASE), repl) for pat, repl in PATTERNS]


def clean_pq(text: str) -> str:
    """PQ-aware text cleaning. Keeps acronyms and digits; removes procedural boilerplate."""
    if text is None:
        return ""
    t = str(text).strip().lower()

    # Apply regex rewrites
    for rx, repl in COMPILED:
        t = rx.sub(repl, t)

    # Keep words, digits, underscore, hyphen. Remove other punctuation.
    # This preserves "hse", "covid-19", "2025", etc.
    t = re.sub(r"[^\w\s\-]", " ", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    ap.add_argument(
        "--question-col",
        default="question",
        help="Name of the column containing raw question text (default: question)",
    )
    ap.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Print N before/after examples for sanity check (default: 10)",
    )
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    df = pd.read_csv(in_path)

    if args.question_col not in df.columns:
        raise SystemExit(
            f"Column '{args.question_col}' not found. Available columns: {list(df.columns)}"
        )

    df["question_clean"] = df[args.question_col].map(clean_pq)

    # Optional quick checks
    if args.examples > 0:
        sample = df[[args.question_col, "question_clean"]].dropna().head(args.examples)
        print("\n--- BEFORE / AFTER EXAMPLES ---")
        for i, row in sample.iterrows():
            print(f"\n[{i}] BEFORE: {row[args.question_col]}")
            print(f"[{i}] AFTER : {row['question_clean']}")

        # Length stats
        raw_len = df[args.question_col].fillna("").astype(str).map(lambda s: len(s.split()))
        clean_len = df["question_clean"].fillna("").astype(str).map(lambda s: len(s.split()))
        print("\n--- TOKEN COUNT (word) STATS ---")
        print(f"raw   mean={raw_len.mean():.1f}, median={raw_len.median():.0f}, p95={raw_len.quantile(0.95):.0f}")
        print(f"clean mean={clean_len.mean():.1f}, median={clean_len.median():.0f}, p95={clean_len.quantile(0.95):.0f}")

    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
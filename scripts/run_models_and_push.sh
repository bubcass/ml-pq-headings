#!/usr/bin/env bash
set -e  # stop if any command fails

CANON_URL="https://raw.githubusercontent.com/bubcass/PQs/refs/heads/main/PQs_2026_paginated.json"

echo "Running DistilBERT..."
python scripts/build_metrics_from_canon.py \
  --canon-url "$CANON_URL" \
  --model artifacts/models_transformer/distilbert_pq_heading \
  --out-preds outputs/predictions_distilbert_latest.csv \
  --out-metrics outputs/metrics_daily_distilbert.csv \
  --max-k 10 \
  --report-ks 1,4,10

echo "Running DeBERTa..."
python scripts/build_metrics_from_canon.py \
  --canon-url "$CANON_URL" \
  --model artifacts/models_transformer/deberta_v3_small_pq_heading \
  --out-preds outputs/predictions_latest.csv \
  --out-metrics outputs/metrics_daily.csv \
  --max-k 10 \
  --report-ks 1,4,10

echo "Committing to Git..."
git add outputs/metrics_daily_distilbert.csv \
        outputs/metrics_daily.csv \
        outputs/predictions_distilbert_latest.csv \
        outputs/predictions_latest.csv

git commit -m "Update daily PQ heading metrics (DistilBERT + DeBERTa $(date +%Y-%m-%d))" || echo "Nothing to commit"

git push

echo "Done."
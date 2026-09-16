#!/usr/bin/env bash
set -euo pipefail

CANON_URL="https://raw.githubusercontent.com/bubcass/PQs/refs/heads/main/PQs_2026_paginated.json"
PUBLISH=true
if [[ "${1:-}" == "--no-push" ]]; then
  PUBLISH=false
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--no-push]" >&2
  exit 2
fi

STAGE_DIR="$(mktemp -d outputs/.metrics_run.XXXXXX)"
CANON_SNAPSHOT="$STAGE_DIR/PQs_2026_paginated.json"

cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo "Downloading one canonical snapshot for both models..."
curl --fail --location --silent --show-error "$CANON_URL" --output "$CANON_SNAPSHOT"

echo "Running DistilBERT..."
python scripts/build_metrics_from_canon.py \
  --canon-path "$CANON_SNAPSHOT" \
  --model artifacts/models_transformer/distilbert_pq_heading \
  --out-preds "$STAGE_DIR/predictions_distilbert_latest.csv" \
  --out-metrics "$STAGE_DIR/metrics_daily_distilbert.csv" \
  --max-k 10 \
  --report-ks 1,4,10

echo "Running DeBERTa..."
python scripts/build_metrics_from_canon.py \
  --canon-path "$CANON_SNAPSHOT" \
  --model artifacts/models_transformer/deberta_v3_small_pq_heading \
  --out-preds "$STAGE_DIR/predictions_latest.csv" \
  --out-metrics "$STAGE_DIR/metrics_daily.csv" \
  --max-k 10 \
  --report-ks 1,4,10

echo "Validating the paired annual outputs..."
python scripts/validate_model_outputs.py \
  --distil-preds "$STAGE_DIR/predictions_distilbert_latest.csv" \
  --distil-metrics "$STAGE_DIR/metrics_daily_distilbert.csv" \
  --deberta-preds "$STAGE_DIR/predictions_latest.csv" \
  --deberta-metrics "$STAGE_DIR/metrics_daily.csv"

echo "Publishing validated files locally..."
mv "$STAGE_DIR/predictions_distilbert_latest.csv" outputs/predictions_distilbert_latest.csv
mv "$STAGE_DIR/metrics_daily_distilbert.csv" outputs/metrics_daily_distilbert.csv
mv "$STAGE_DIR/predictions_latest.csv" outputs/predictions_latest.csv
mv "$STAGE_DIR/metrics_daily.csv" outputs/metrics_daily.csv

if [[ "$PUBLISH" == true ]]; then
  echo "Committing to Git..."
  git add outputs/metrics_daily_distilbert.csv \
          outputs/metrics_daily.csv \
          outputs/predictions_distilbert_latest.csv \
          outputs/predictions_latest.csv

  git commit -m "Update daily PQ heading metrics (DistilBERT + DeBERTa $(date +%Y-%m-%d))" || echo "Nothing to commit"
  git push
else
  echo "Local outputs updated; Git commit and push skipped (--no-push)."
fi

echo "Done."

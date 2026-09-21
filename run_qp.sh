#!/usr/bin/env bash
#
# Run the Word-document tandem test only.  This does not call, modify, or
# publish the established web/open-data pipeline.
#
# Usage:
#   ./run_qp.sh
#   ./run_qp.sh inputs/word_question_papers/2026-09-16_question_paper.docx
#
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [path/to/question_paper.docx]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIRECTORY="$SCRIPT_DIR/inputs/word_question_papers"
ML_INPUT_DIRECTORY="$SCRIPT_DIR/outputs/word_ml_inputs"
PREDICTION_DIRECTORY="$SCRIPT_DIR/outputs/word_ml_predictions"

if [[ $# -eq 1 ]]; then
  WORD_DOCUMENT="$1"
else
  shopt -s nullglob
  papers=("$SOURCE_DIRECTORY"/????-??-??_question_paper.docx)
  shopt -u nullglob
  if [[ ${#papers[@]} -eq 0 ]]; then
    echo "No dated Word papers found in: $SOURCE_DIRECTORY" >&2
    exit 2
  fi
  IFS=$'\n' papers=($(printf '%s\n' "${papers[@]}" | sort))
  unset IFS
  WORD_DOCUMENT="${papers[${#papers[@]} - 1]}"
fi

if [[ ! -f "$WORD_DOCUMENT" ]]; then
  echo "Word document not found: $WORD_DOCUMENT" >&2
  exit 2
fi

if [[ ! -x "$SCRIPT_DIR/.venv311/bin/python" ]]; then
  echo "Expected Python environment not found: $SCRIPT_DIR/.venv311/bin/python" >&2
  exit 2
fi

echo "Running Word-document tandem test."
echo "Word paper: $WORD_DOCUMENT"
echo "ML input: $ML_INPUT_DIRECTORY"
echo "Predictions: $PREDICTION_DIRECTORY"

exec "$SCRIPT_DIR/.venv311/bin/python" \
  "$SCRIPT_DIR/scripts/run_word_paper_pipeline.py" \
  --input "$WORD_DOCUMENT" \
  --ml-input-dir "$ML_INPUT_DIRECTORY" \
  --prediction-dir "$PREDICTION_DIRECTORY"

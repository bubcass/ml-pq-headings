# Automated Classification of Parliamentary Questions

This repository contains a proof-of-concept system for automatically assigning
official thematic labels (question headings) to parliamentary questions (PQs) using supervised
machine learning.

Performance for 2026 data: [Supervised machine learning applied to question headings](https://observablehq.com/@cassdavid/ml-performance)

Models at [huggingface](https://huggingface.co/bubcass)

## What’s included
- Training and evaluation scripts (TensorFlow, DistilBERT, DeBERTa)
- Evaluation tooling (top-1, top-k, confidence thresholds)
- A lightweight UI proof-of-concept for batch classification

## What’s not included
- Training data
- Trained model weights
- Some prediction outputs

These are intentionally excluded for size and governance reasons.

## Status
Research / Proof of concept

## Rebuild the 2026 evaluation

The annual runner downloads one canonical PQ snapshot, evaluates both DistilBERT
and DeBERTa against that same snapshot, validates the paired outputs, and only
then replaces the four CSV files in `outputs/`.

```bash
# Rebuild locally without committing or pushing
./run_metrics.sh --no-push

# Rebuild, validate, commit the output CSVs, and push
./run_metrics.sh
```

Inference settings come from each model's `metadata.json`, including cleaned
question text, maximum token length, and tokenizer type. The daily metrics retain
overall accuracy for the frozen-label test and also report model-vocabulary
coverage and accuracy among headings present in that frozen vocabulary.

## Create a daily Word question dataset

Store dated Word question papers in `inputs/word_question_papers/`, using this
filename pattern: `YYYY-MM-DD_question_paper.docx`. For example:

```text
inputs/word_question_papers/2026-09-16_question_paper.docx
```

To extract a paper only, run:

```bash
python scripts/extract_question_paper.py \
  --input /path/to/Question_Paper.docx
```

It writes `outputs/word_ml_inputs/<paper-date>_questions_ml_input.csv` by
default. Use `--output` to choose another location.

The CSV has the established prediction-input fields and adds PQ reference,
deputy, question-paper department, language, answer type and source paragraph
locations. Department mapping is explicit in
`config/word_paper_department_aliases.json`; a newly named department stops the
run until it is mapped to a model department label.

## Run the complete Word to DistilBERT pipeline

For the separate Word-document tandem test (the usual command), run:

```bash
./run_qp.sh
```

With no argument, it selects the latest dated paper in
`inputs/word_question_papers/`. To run a particular paper, supply its path:

```bash
./run_qp.sh inputs/word_question_papers/2026-09-16_question_paper.docx
```

It writes extracted CSVs to `outputs/word_ml_inputs/` and prediction CSVs to
`outputs/word_ml_predictions/`. It never invokes the established web/open-data
workflow.

The underlying Python pipeline can also be called directly:

```bash
python scripts/run_word_paper_pipeline.py \
  --input /path/to/Question_Paper.docx
```

This first creates the daily ML-input CSV, then assigns the established top 10
DistilBERT headings. Its final CSV keeps the established prediction columns and
appends `refNo` plus question-paper provenance fields.

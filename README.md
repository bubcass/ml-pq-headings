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

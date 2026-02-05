#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer

# -----------------------------
# Config
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # repo root if poc_ui/app/server.py
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "models_transformer" / "deberta_v3_small_pq_heading"
MODEL_DIR = Path(os.environ.get("PQ_MODEL_DIR", str(DEFAULT_MODEL_DIR))).resolve()

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Try to reuse your existing cleaner; fall back to minimal cleaning
try:
    from preprocess import clean_pq  # type: ignore
except Exception:
    def clean_pq(s: str) -> str:
        return " ".join(str(s).split()).strip()

# -----------------------------
# Load model artifacts once
# -----------------------------
if not MODEL_DIR.exists():
    raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

tokenizer = DebertaV2Tokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(DEVICE)
model.eval()

label_map_path = MODEL_DIR / "label_map.json"
if not label_map_path.exists():
    raise RuntimeError(f"Missing label_map.json in: {MODEL_DIR}")

id2label: Dict[int, str] = {int(k): v for k, v in json.loads(label_map_path.read_text(encoding="utf-8")).items()}

app = FastAPI(title="PQ Heading Classifier (DeBERTa PoC)")


# -----------------------------
# Helpers
# -----------------------------
def build_text(department: str, question: str) -> str:
    q_clean = clean_pq(question)
    dept = str(department or "").strip()
    return f"[DEPT] {dept} [SEP] {q_clean}"


@torch.inference_mode()
def predict_rows(rows: List[Tuple[str, str]], topk: int = 4, max_len: int = 256, batch_size: int = 16) -> List[Dict[str, Any]]:
    """
    rows: list of (department, question)
    returns: list of dicts with pred + alt headings/conf
    """
    outputs: List[Dict[str, Any]] = []

    # clamp topk
    topk = max(1, min(int(topk), 10))

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [build_text(d, q) for d, q in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)

        top_probs, top_idx = torch.topk(probs, k=topk, dim=1)

        for r in range(len(batch)):
            idxs = top_idx[r].tolist()
            ps = top_probs[r].tolist()

            pred_id = int(idxs[0])
            pred_heading = id2label.get(pred_id, str(pred_id))
            pred_conf = float(ps[0])

            row_out: Dict[str, Any] = {
                "pred_heading": pred_heading,
                "pred_confidence": pred_conf,
            }

            # alternates (include alt_heading_1 etc; keep same style as your evaluation script expects)
            for j in range(topk):
                hid = int(idxs[j])
                row_out[f"alt_heading_{j+1}"] = id2label.get(hid, str(hid))
                row_out[f"alt_conf_{j+1}"] = float(ps[j])

            outputs.append(row_out)

    return outputs


def dataframe_from_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if name.endswith(".json"):
        obj = json.loads(file_bytes.decode("utf-8"))
        # accept either list[dict] or {"data":[...]}
        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]
        if not isinstance(obj, list):
            raise ValueError("JSON must be a list of objects, or an object with a 'data' list.")
        return pd.DataFrame(obj)
    raise ValueError("Unsupported file type. Please upload a .csv or .json file.")


def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


# -----------------------------
# UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PQ Heading Classifier (DeBERTa PoC)</title>
  <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:980px;margin:40px auto;padding:0 16px;line-height:1.4}}
    .card{{border:1px solid #e5e5e5;border-radius:14px;padding:18px;margin:16px 0}}
    label{{display:block;margin:10px 0 6px;font-weight:600}}
    input[type="number"], input[type="text"]{{padding:10px;border:1px solid #ccc;border-radius:10px;width:220px}}
    input[type="file"]{{margin-top:6px}}
    button{{padding:10px 14px;border:0;border-radius:12px;background:#111;color:#fff;font-weight:700;cursor:pointer}}
    button:hover{{opacity:.9}}
    code{{background:#f6f6f6;padding:2px 6px;border-radius:6px}}
    .hint{{color:#555;font-size:14px}}
  </style>
</head>
<body>
  <h1>PQ Heading Classifier (DeBERTa PoC)</h1>
  <p class="hint">
    Model loaded from: <code>{MODEL_DIR}</code> (device: <code>{DEVICE}</code>)
  </p>

  <div class="card">
    <h2>Upload a CSV or JSON</h2>
    <p class="hint">
      Required columns: <code>question</code> and <code>department</code>.
      (If your file also contains <code>heading</code>, it will be preserved in the output.)
    </p>

    <form action="/predict-file" method="post" enctype="multipart/form-data">
      <label>File (.csv or .json)</label>
      <input type="file" name="file" accept=".csv,.json" required />

      <label>Top-k suggestions</label>
      <input type="number" name="topk" value="4" min="1" max="10" />

      <label>Max length</label>
      <input type="number" name="max_len" value="256" min="64" max="512" />

      <label>Batch size</label>
      <input type="number" name="batch" value="16" min="1" max="128" />

      <div style="margin-top:14px">
        <button type="submit">Run predictions and download CSV</button>
      </div>
    </form>
  </div>

  <div class="card">
    <h2>Notes</h2>
    <ul>
      <li>This is a local PoC: upload → model inference → download.</li>
      <li>Output columns include <code>pred_heading</code>, <code>pred_confidence</code>, and <code>alt_heading_*</code>/<code>alt_conf_*</code>.</li>
      <li>You can then run your existing <code>scripts/evaluate_holdout.py</code> against the downloaded CSV if it includes ground truth <code>heading</code>.</li>
    </ul>
  </div>
</body>
</html>
"""


@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    topk: int = Form(4),
    max_len: int = Form(256),
    batch: int = Form(16),
):
    file_bytes = await file.read()
    df = dataframe_from_upload(file_bytes, file.filename or "")
    require_cols(df, ["question", "department"])

    # Build prediction input list
    rows = list(zip(df["department"].astype(str).tolist(), df["question"].astype(str).tolist()))
    preds = predict_rows(rows, topk=topk, max_len=max_len, batch_size=batch)

    out = df.copy()
    pred_df = pd.DataFrame(preds)
    out = pd.concat([out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    # Stream back as CSV
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)

    out_name = (Path(file.filename or "input").stem + "_predictions.csv")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
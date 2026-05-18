"""Generate predictions using the v9 model -> predictions_v9.csv

Usage:
    python predict_v9.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from predict import run_predict

MODEL_V9   = ROOT / "models" / "model_v9.pkl"
OUT_V9     = ROOT / "predictions_v9.csv"
SUBMIT_V9  = ROOT / "predictions.csv"

if __name__ == "__main__":
    if not MODEL_V9.exists():
        print(f"ERROR: model not found at {MODEL_V9}")
        print("Wait for train_v9.py to finish first.")
        sys.exit(1)
    out_path = run_predict(version=9, model_path=MODEL_V9, out_path=OUT_V9, autoregressive=False)

    # Save submission file: "predict" header + values
    df = pd.read_csv(out_path)
    pd.DataFrame({"predict": df["predict"]}).to_csv(SUBMIT_V9, index=False)
    print(f"Submission file saved -> {SUBMIT_V9}")
    print(f"Done. Upload predictions.csv to the platform.")

"""Run v8 inference and save in correct submission format."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import run_predict

out = run_predict(
    version=8,
    model_path=Path("models/model_v8.pkl"),
    out_path=Path("predictions_v8_raw.csv"),
    autoregressive=False,
)
df = pd.read_csv(out)
df["predict"].to_csv("predictions_v8.csv", index=False, header=False)
vals = df["predict"].values
print(f"v8 clean -> predictions_v8.csv  ({len(df)} rows)")
print(f"  sample: {vals[:3]}")
print(f"  min={vals.min():.3f}  mean={vals.mean():.3f}  max={vals.max():.3f}")

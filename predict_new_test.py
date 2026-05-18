"""Generate predictions for the new test dataset.

Handles non-standard timestamps (e.g. 22:59:59.875 → rounds to 23:00:00)
and saves predictions in the correct submission format.

Usage:
    python predict_new_test.py
    python predict_new_test.py --input data/my_test.csv --model models/model_v8.pkl
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import DATETIME_COL, TARGET_COL, INSTALLED_CAPACITY_MW
from src.preprocessing import (
    build_features, clip_predictions,
    apply_empirical_curve,
)
from train_v9 import apply_conditional_means


def run(input_csv: Path, model_path: Path, out_path: Path) -> None:
    print(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)
    regressors  = model["regressors"]
    feat_names  = model["feature_names"]
    weights     = np.array(model["ensemble_weights"])
    pc_map      = model.get("pc_map")
    cond_means  = model.get("cond_means")
    corrector   = model.get("corrector")
    print(f"  members={len(regressors)}  features={len(feat_names)}")

    print(f"Loading test data from {input_csv} ...")
    df_orig = pd.read_csv(input_csv)
    df_orig[DATETIME_COL] = pd.to_datetime(df_orig[DATETIME_COL])

    # Round sub-minute timestamps to nearest hour (e.g. 22:59:59.875 → 23:00:00)
    rounded = df_orig[DATETIME_COL].dt.round("h")
    n_rounded = (rounded != df_orig[DATETIME_COL]).sum()
    if n_rounded > 0:
        print(f"  Rounding {n_rounded} timestamps to nearest hour")
        df_orig[DATETIME_COL] = rounded

    orig_datetimes = df_orig[DATETIME_COL].values
    print(f"  rows={len(df_orig)}  period: {df_orig[DATETIME_COL].min()} .. {df_orig[DATETIME_COL].max()}")

    # Sort chronologically for feature engineering, preserve original order
    df = df_orig.sort_values(DATETIME_COL).reset_index(drop=True)

    print("Building features ...")
    df = build_features(df, use_target_lags=False)

    if pc_map:
        print("Applying empirical power curve ...")
        df = apply_empirical_curve(df, pc_map)

    if cond_means:
        print("Applying conditional mean features ...")
        df = apply_conditional_means(df, cond_means)

    missing = [f for f in feat_names if f not in df.columns]
    if missing:
        print(f"  Filling {len(missing)} missing features with 0")
        for f in missing:
            df[f] = 0.0

    X = df[feat_names].values.astype(float)
    member_preds = np.array([reg.predict(X) for reg in regressors])
    raw = member_preds.T @ weights
    if corrector is not None:
        raw = raw + corrector.predict(X)
    preds_chrono = clip_predictions(raw, INSTALLED_CAPACITY_MW)

    # Map back to original row order
    chrono_map = dict(zip(df[DATETIME_COL].values, preds_chrono))
    preds = np.array([chrono_map[dt] for dt in orig_datetimes])

    # Save in submission format: header "predict" + values
    pd.DataFrame({"predict": preds}).to_csv(out_path, index=False)
    print(f"\nSaved {len(preds)} predictions -> {out_path}")
    print(f"  min={preds.min():.3f}  mean={preds.mean():.3f}  max={preds.max():.3f}")
    print(f"  sample (first 3): {preds[:3]}")

    # Also overwrite predictions.csv for quick submission
    submit = ROOT / "predictions.csv"
    pd.DataFrame({"predict": preds}).to_csv(submit, index=False)
    print(f"  -> also saved to {submit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--model", default="models/model_v8.pkl")
    parser.add_argument("--out",   default="predictions_new_test.csv")
    args = parser.parse_args()

    # Auto-detect new test file if not specified
    if args.input is None:
        data_dir = ROOT / "data"
        candidates = [f for f in data_dir.glob("*test_dataset*.csv")]
        if not candidates:
            print("ERROR: no test_dataset*.csv found in data/"); sys.exit(1)
        # Pick newest
        args.input = str(sorted(candidates, key=lambda f: f.stat().st_mtime)[-1])
        print(f"Auto-detected test file: {args.input}")

    run(Path(args.input), Path(args.model), Path(args.out))

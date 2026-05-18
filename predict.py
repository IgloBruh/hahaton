"""Generate predictions on valid_features.csv using a saved WindPowerModel.

Usage:
    python predict.py [--version V] [--model PATH] [--out PATH] [--no-autoregressive]

Saves predictions to predictions_vV.csv (default v=1).

When the saved model uses target-lag features, autoregressive mode is used by
default: the last TARGET_LAG_MAX hours of training actuals initialise the lag
buffer; then predictions roll forward one step at a time so each row sees the
predicted (not zero) lag values.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib

from config import (
    TRAIN_CSV, VALID_CSV, MODEL_PATH, DATETIME_COL, TARGET_COL,
    INSTALLED_CAPACITY_MW,
)
from src.data_loader import load_valid, load_train
from src.preprocessing import (
    build_features, clip_predictions, TARGET_LAG_OFFSETS,
    apply_empirical_curve,
)
from train_v9 import apply_conditional_means
from src.model import WindPowerModel

_TARGET_LAG_COLS = [f"target_lag{k}" for k in TARGET_LAG_OFFSETS]
_MAX_LAG = max(TARGET_LAG_OFFSETS)


def _load_model(path: Path):
    """Load either a WindPowerModel or a plain dict (v3 format)."""
    blob = joblib.load(path)
    if isinstance(blob, dict):
        return blob  # v3 dict with 'regressors' and 'feature_names'
    return blob  # WindPowerModel


def _model_attrs(model):
    if isinstance(model, dict):
        return model["regressors"], model["feature_names"]
    return model.regressors, model.feature_names


def _model_weights(model) -> np.ndarray | None:
    if isinstance(model, dict) and "ensemble_weights" in model:
        return np.array(model["ensemble_weights"])
    return None


def _model_uses_target_lags(model) -> bool:
    if isinstance(model, dict):
        return model.get("use_target_lags", True)
    return True


def _model_pc_map(model) -> dict | None:
    if isinstance(model, dict):
        return model.get("pc_map", None)
    return None


def _model_cond_means(model) -> dict | None:
    if isinstance(model, dict):
        return model.get("cond_means", None)
    return None


def _model_corrector(model):
    if isinstance(model, dict):
        return model.get("corrector", None)
    return None


def _has_target_lags(model) -> bool:
    _, feature_names = _model_attrs(model)
    return any(f in feature_names for f in _TARGET_LAG_COLS)


def _run_autoregressive(
    model,
    df: pd.DataFrame,
    init_buffer: np.ndarray,
) -> np.ndarray:
    """Predict row-by-row, feeding predicted values back as target lags.

    init_buffer: last _MAX_LAG actuals from training (shape [_MAX_LAG]).
    """
    regressors, feature_names = _model_attrs(model)
    n = len(df)
    X = df[feature_names].values.astype(float)
    col_idx = {f: i for i, f in enumerate(feature_names)}
    lag_indices = {
        lag: col_idx[f"target_lag{lag}"]
        for lag in TARGET_LAG_OFFSETS
        if f"target_lag{lag}" in col_idx
    }

    preds = np.empty(n, dtype=float)
    buf = list(init_buffer)

    for t in range(n):
        for lag, ci in lag_indices.items():
            idx = len(buf) - lag
            X[t, ci] = buf[idx] if idx >= 0 else 0.0
        p = float(np.mean([
            reg.predict(X[t:t+1])[0] for reg in regressors
        ]))
        p = float(np.clip(p, 0.0, INSTALLED_CAPACITY_MW))
        preds[t] = p
        buf.append(p)

    return preds


def run_predict(
    version: int = 1,
    model_path: Path | None = None,
    out_path: Path | None = None,
    autoregressive: bool = True,
) -> Path:
    model_path = Path(model_path) if model_path else MODEL_PATH
    out_path = Path(out_path) if out_path else ROOT / f"predictions_v{version}.csv"

    print(f"Loading model from {model_path} ...")
    model = _load_model(model_path)
    regressors, feature_names = _model_attrs(model)
    print(f"  members={len(regressors)}  features={len(feature_names)}")

    uses_lags  = _has_target_lags(model) and _model_uses_target_lags(model)
    weights    = _model_weights(model)
    pc_map     = _model_pc_map(model)
    cond_means = _model_cond_means(model)
    corrector  = _model_corrector(model)
    print(f"  model uses target lags: {uses_lags}")
    if weights is not None:
        print(f"  using optimised ensemble weights (non-uniform)")
    if pc_map is not None:
        print(f"  applying empirical power curve ({len(pc_map['bin_centers'])} bins)")
    if cond_means is not None:
        print(f"  applying conditional mean features")
    if corrector is not None:
        print(f"  applying residual corrector")

    # Load preserving original row order (for correct output ordering)
    print(f"Loading test data from {VALID_CSV} ...")
    df_orig_order = pd.read_csv(VALID_CSV)
    df_orig_order[DATETIME_COL] = pd.to_datetime(df_orig_order[DATETIME_COL])
    orig_datetimes = df_orig_order[DATETIME_COL].values  # original order

    # load_valid sorts chronologically — required for lag/lead feature engineering
    df = load_valid(VALID_CSV)
    print(f"  rows={len(df)}")

    print("Building features ...")
    df = build_features(df, use_target_lags=False)

    if pc_map is not None:
        print("Applying empirical power curve ...")
        df = apply_empirical_curve(df, pc_map)

    if cond_means is not None:
        print("Applying conditional mean features ...")
        df = apply_conditional_means(df, cond_means)

    # Add any model features missing from test set (target lags, etc.)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"  filling {len(missing)} missing features with 0: {missing[:6]}")
        for f in missing:
            df[f] = 0.0

    if uses_lags and autoregressive:
        print("Autoregressive mode: initialising lag buffer from training tail ...")
        train_df = load_train(TRAIN_CSV)
        init_buf = train_df[TARGET_COL].values[-_MAX_LAG:]
        print(f"  using last {len(init_buf)} training actuals as lag seed")
        preds_chrono = _run_autoregressive(model, df, init_buf)
    else:
        print("Batch mode ...")
        X = df[feature_names].values.astype(float)
        member_preds = np.array([reg.predict(X) for reg in regressors])
        if weights is not None:
            raw = member_preds.T @ weights
        else:
            raw = member_preds.mean(axis=0)
        if corrector is not None:
            raw = raw + corrector.predict(X)
        preds_chrono = clip_predictions(raw, INSTALLED_CAPACITY_MW)

    # Predictions are in chronological order; reorder to match original file order
    chrono_map = dict(zip(df[DATETIME_COL].values, preds_chrono))
    preds_orig_order = np.array([chrono_map[dt] for dt in orig_datetimes])

    # Save with datetime for traceability + matching the original row order
    out = pd.DataFrame({
        DATETIME_COL: orig_datetimes,
        "predict": preds_orig_order,
    })
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} predictions -> {out_path}")
    print(f"  first row: {out.iloc[0, 0]}  predict={out.iloc[0, 1]:.4f}")
    print(f"  last  row: {out.iloc[-1, 0]}  predict={out.iloc[-1, 1]:.4f}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-autoregressive", action="store_true")
    args = parser.parse_args()
    run_predict(
        version=args.version,
        model_path=args.model,
        out_path=args.out,
        autoregressive=not args.no_autoregressive,
    )

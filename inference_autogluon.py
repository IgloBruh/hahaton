"""Run AutoGluon inference: load trained predictor, predict for valid_features.csv,
write predictions_autogluon.csv (one column, no header, same row order as input).

Target lag features are filled autoregressively:
  - Initial values from real training-tail targets.
  - Each step uses the previous prediction as the lag value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (
    SEED, TRAIN_CSV, VALID_CSV, TARGET_COL, DATETIME_COL,
    INSTALLED_CAPACITY_MW, AG_MODEL_PATH, AG_PREDICTIONS_PATH,
)
from src.utils import set_global_seed
from src.data_loader import load_train, load_valid
from src.preprocessing import build_features, feature_columns, clip_predictions, TARGET_LAG_OFFSETS

HISTORY_HOURS = 48


def main() -> None:
    set_global_seed(SEED)

    print(f"-> Loading AutoGluon predictor from {AG_MODEL_PATH} ...")
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(str(AG_MODEL_PATH))

    print(f"-> Loading test features from {VALID_CSV} ...")
    raw_valid = load_valid(VALID_CSV)
    n_rows = len(raw_valid)
    print(f"   rows = {n_rows}")

    print("-> Loading training tail for lag/rolling context ...")
    train = load_train(TRAIN_CSV)
    max_lag = max(TARGET_LAG_OFFSETS) if TARGET_LAG_OFFSETS else 0
    history_len = max(HISTORY_HOURS, max_lag)
    train_tail = train.tail(history_len).copy()
    tail_targets = list(train_tail[TARGET_COL].values)
    train_tail = train_tail.drop(columns=[TARGET_COL], errors="ignore")

    combined = pd.concat([train_tail, raw_valid], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)

    print("-> Building features (with full history context, NO target lags) ...")
    feat_df = build_features(combined, use_target_lags=False)

    valid_mask = feat_df[DATETIME_COL].isin(raw_valid[DATETIME_COL])
    feat_valid = feat_df.loc[valid_mask].copy()
    feat_valid = feat_valid.sort_values(DATETIME_COL).reset_index(drop=True)
    assert len(feat_valid) == n_rows

    feats = feature_columns(feat_valid)

    # Determine which target lag columns are in the feature set
    target_lag_cols = [f"target_lag{lag}" for lag in TARGET_LAG_OFFSETS
                       if f"target_lag{lag}" in feats]

    if target_lag_cols:
        print(f"-> Autoregressive prediction ({len(target_lag_cols)} target lag cols) ...")
        for col in target_lag_cols:
            feat_valid[col] = 0.0

        target_history = list(tail_targets)
        all_preds = []

        for i in range(n_rows):
            for lag in TARGET_LAG_OFFSETS:
                col = f"target_lag{lag}"
                if col not in target_lag_cols:
                    continue
                idx = len(target_history) - lag
                feat_valid.at[feat_valid.index[i], col] = target_history[idx] if idx >= 0 else 0.0

            row = feat_valid.iloc[[i]][feats]
            pred = float(predictor.predict(row, as_pandas=False)[0])
            pred = float(clip_predictions(np.array([pred]), INSTALLED_CAPACITY_MW)[0])
            all_preds.append(pred)
            target_history.append(pred)

        preds_sorted = np.array(all_preds)
    else:
        print("-> Predicting (no target lags, single pass) ...")
        preds_raw = predictor.predict(feat_valid[feats], as_pandas=False)
        preds_sorted = clip_predictions(np.asarray(preds_raw, dtype=float), INSTALLED_CAPACITY_MW)

    pred_by_dt = dict(zip(feat_valid[DATETIME_COL].values, preds_sorted))

    original = pd.read_csv(VALID_CSV)
    original[DATETIME_COL] = pd.to_datetime(original[DATETIME_COL])
    out = original[DATETIME_COL].map(pred_by_dt).values

    assert len(out) == n_rows, "Row-count mismatch."
    assert not pd.isna(out).any(), "NaN in predictions."

    pd.DataFrame({"predict": out}).to_csv(AG_PREDICTIONS_PATH, index=False)
    print(f"OK Wrote {len(out)} predictions -> {AG_PREDICTIONS_PATH}")
    print(f"   stats: min={out.min():.3f}  mean={out.mean():.3f}  max={out.max():.3f}")
    print("To submit: copy predictions_autogluon.csv -> predictions.csv")


if __name__ == "__main__":
    main()

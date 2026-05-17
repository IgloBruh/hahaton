"""Run inference: load the trained ensemble, predict for valid_features.csv,
write predictions.csv (one column, no header, same row order as input).

Key detail
----------
Lag / lead / rolling features require chronological context. The 2 126 valid
rows alone don't have 24 hours of preceding history for the first day of
January 2026. We therefore concatenate the **training tail** to the head of
the validation slice before building features, then drop the training rows
again before predicting.

Target lag features (target_lag1 .. target_lag24) are filled autoregressively:
  - Initial values come from the real training-tail targets.
  - Each subsequent step uses the model's own prediction as the lag value.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from config import (
    SEED, TRAIN_CSV, VALID_CSV, MODEL_PATH, PREDICTIONS_PATH,
    DATETIME_COL, INSTALLED_CAPACITY_MW, TARGET_COL,
)
from src.utils import set_global_seed
from src.data_loader import load_train, load_valid
from src.preprocessing import build_features, clip_predictions, TARGET_LAG_OFFSETS
from src.model import WindPowerModel


HISTORY_HOURS = 48


def main() -> None:
    set_global_seed(SEED)

    print(f"-> Loading model from {MODEL_PATH} ...")
    model = WindPowerModel.load(MODEL_PATH)
    print(f"   ensemble members = {model.n_members}")

    print(f"-> Loading test features from {VALID_CSV} ...")
    raw_valid = load_valid(VALID_CSV)
    n_rows = len(raw_valid)
    print(f"   rows = {n_rows}")

    print(f"-> Loading training tail for lag/rolling context ...")
    train = load_train(TRAIN_CSV)
    # Extended tail: enough history for both weather lags and target lags
    max_lag = max(TARGET_LAG_OFFSETS) if TARGET_LAG_OFFSETS else 0
    history_len = max(HISTORY_HOURS, max_lag)
    train_tail = train.tail(history_len).copy()
    # Keep a copy of tail targets for the autoregressive loop
    tail_targets = list(train_tail[TARGET_COL].values)
    train_tail = train_tail.drop(columns=[TARGET_COL], errors="ignore")

    combined = pd.concat([train_tail, raw_valid], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)

    print("-> Building features (with full history context) ...")
    feat_df = build_features(combined)

    valid_mask = feat_df[DATETIME_COL].isin(raw_valid[DATETIME_COL])
    feat_valid = feat_df.loc[valid_mask].copy().reset_index(drop=True)
    feat_valid = feat_valid.sort_values(DATETIME_COL).reset_index(drop=True)
    assert len(feat_valid) == n_rows

    # Determine which target lag columns are actually used by the model
    target_lag_cols = [f"target_lag{lag}" for lag in TARGET_LAG_OFFSETS
                       if f"target_lag{lag}" in model.feature_names]

    if target_lag_cols:
        print(f"-> Autoregressive prediction ({len(target_lag_cols)} target lag cols) ...")
        # Initialise all target lag columns to 0
        for col in target_lag_cols:
            feat_valid[col] = 0.0

        target_history = list(tail_targets)  # grows as we predict
        all_preds = []

        for i in range(n_rows):
            # Fill target lags from history (real or predicted)
            for lag in TARGET_LAG_OFFSETS:
                col = f"target_lag{lag}"
                if col not in target_lag_cols:
                    continue
                idx = len(target_history) - lag
                feat_valid.at[i, col] = target_history[idx] if idx >= 0 else 0.0

            pred = float(clip_predictions(
                model.predict(feat_valid.iloc[[i]]), INSTALLED_CAPACITY_MW
            )[0])
            all_preds.append(pred)
            target_history.append(pred)

        preds_sorted = np.array(all_preds)
    else:
        print("-> Predicting (no target lags in model, single pass) ...")
        preds_sorted = clip_predictions(model.predict(feat_valid), INSTALLED_CAPACITY_MW)

    pred_by_dt = dict(zip(feat_valid[DATETIME_COL].values, preds_sorted))

    original = pd.read_csv(VALID_CSV)
    original[DATETIME_COL] = pd.to_datetime(original[DATETIME_COL])
    out = original[DATETIME_COL].map(pred_by_dt).values

    assert len(out) == n_rows, "Row-count mismatch."
    assert not pd.isna(out).any(), "NaN in predictions."

    pd.Series(out).to_csv(PREDICTIONS_PATH, index=False, header=False)
    print(f"OK Wrote {len(out)} predictions -> {PREDICTIONS_PATH}")
    print(f"   stats: min={out.min():.3f}  mean={out.mean():.3f}  max={out.max():.3f}")


if __name__ == "__main__":
    main()

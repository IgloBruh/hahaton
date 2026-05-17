"""Run inference: load the trained ensemble, predict for valid_features.csv,
write predictions.csv (one column, no header, same row order as input).

Key detail
----------
Lag / lead / rolling features require chronological context. The 2 126 valid
rows alone don't have 24 hours of preceding history for the first day of
January 2026. We therefore concatenate the **training tail** to the head of
the validation slice before building features, then drop the training rows
again before predicting.

The training set ends at 2025-12-31 23:00 and the valid set starts at
2026-01-01 00:00 — they are chronologically contiguous, so this is exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from config import (
    SEED, TRAIN_CSV, VALID_CSV, MODEL_PATH, PREDICTIONS_PATH,
    DATETIME_COL, INSTALLED_CAPACITY_MW, TARGET_COL,
)
from src.utils import set_global_seed
from src.data_loader import load_train, load_valid
from src.preprocessing import build_features, clip_predictions
from src.model import WindPowerModel


# How many hours of training tail to prepend. >24 is enough for the largest lag.
HISTORY_HOURS = 48


def main() -> None:
    set_global_seed(SEED)

    print(f"→ Loading model from {MODEL_PATH} ...")
    model = WindPowerModel.load(MODEL_PATH)
    print(f"   ensemble members = {model.n_members}")

    print(f"→ Loading test features from {VALID_CSV} ...")
    raw_valid = load_valid(VALID_CSV)            # sorted ascending by datetime
    n_rows = len(raw_valid)
    print(f"   rows = {n_rows}")

    print(f"→ Loading training tail for lag/rolling context ...")
    train = load_train(TRAIN_CSV)
    train_tail = train.tail(HISTORY_HOURS).copy()
    # Drop target so the schemas align (valid has no target column)
    train_tail = train_tail.drop(columns=[TARGET_COL], errors="ignore")

    # Concatenate: history first, then valid features (chronological order)
    combined = pd.concat([train_tail, raw_valid], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)

    print("→ Building features (with full history context) ...")
    feat_df = build_features(combined)

    # Keep only the rows belonging to the validation period
    valid_mask = feat_df[DATETIME_COL].isin(raw_valid[DATETIME_COL])
    feat_valid = feat_df.loc[valid_mask].copy()
    assert len(feat_valid) == n_rows, (
        f"Expected {n_rows} valid rows after feature build, got {len(feat_valid)}"
    )
    feat_valid = feat_valid.sort_values(DATETIME_COL).reset_index(drop=True)

    print("→ Predicting (ensemble mean) ...")
    preds_sorted = clip_predictions(model.predict(feat_valid), INSTALLED_CAPACITY_MW)
    pred_by_dt = dict(zip(feat_valid[DATETIME_COL].values, preds_sorted))

    # The submission needs predictions in the SAME ORDER as the original
    # valid_features.csv file. We therefore reload it without sorting,
    # then map predictions back by datetime.
    original = pd.read_csv(VALID_CSV)
    original[DATETIME_COL] = pd.to_datetime(original[DATETIME_COL])
    out = original[DATETIME_COL].map(pred_by_dt).values

    assert len(out) == n_rows, "Row-count mismatch — submission would be invalid."
    assert not pd.isna(out).any(), "NaN found in predictions — feature build skipped a row."

    pd.Series(out).to_csv(PREDICTIONS_PATH, index=False, header=False)
    print(f"✓ Wrote {len(out)} predictions → {PREDICTIONS_PATH}")
    print(f"   stats: min={out.min():.3f}  mean={out.mean():.3f}  max={out.max():.3f}")


if __name__ == "__main__":
    main()

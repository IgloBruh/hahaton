"""Run AutoGluon inference: load trained predictor, predict for valid_features.csv,
write predictions_autogluon.csv (one column, no header, same row order as input).

Same lag-context strategy as inference.py: prepend the last 48 hours of the
training set to the validation slice before building temporal features, then
keep only the validation rows for prediction.
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
from src.preprocessing import build_features, feature_columns, clip_predictions

HISTORY_HOURS = 48


def main() -> None:
    set_global_seed(SEED)

    print(f"→ Loading AutoGluon predictor from {AG_MODEL_PATH} ...")
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(str(AG_MODEL_PATH))

    print(f"→ Loading test features from {VALID_CSV} ...")
    raw_valid = load_valid(VALID_CSV)
    n_rows = len(raw_valid)
    print(f"   rows = {n_rows}")

    print("→ Loading training tail for lag/rolling context ...")
    train = load_train(TRAIN_CSV)
    train_tail = train.tail(HISTORY_HOURS).copy()
    train_tail = train_tail.drop(columns=[TARGET_COL], errors="ignore")

    combined = pd.concat([train_tail, raw_valid], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)

    print("→ Building features (with full history context) ...")
    feat_df = build_features(combined)

    valid_mask = feat_df[DATETIME_COL].isin(raw_valid[DATETIME_COL])
    feat_valid = feat_df.loc[valid_mask].copy()
    assert len(feat_valid) == n_rows, (
        f"Expected {n_rows} valid rows after feature build, got {len(feat_valid)}"
    )
    feat_valid = feat_valid.sort_values(DATETIME_COL).reset_index(drop=True)

    feats = feature_columns(feat_valid)

    print("→ Predicting with AutoGluon ...")
    preds_raw = predictor.predict(feat_valid[feats], as_pandas=False)
    preds_sorted = clip_predictions(np.asarray(preds_raw, dtype=float), INSTALLED_CAPACITY_MW)
    pred_by_dt = dict(zip(feat_valid[DATETIME_COL].values, preds_sorted))

    # Preserve original row order from valid_features.csv
    original = pd.read_csv(VALID_CSV)
    original[DATETIME_COL] = pd.to_datetime(original[DATETIME_COL])
    out = original[DATETIME_COL].map(pred_by_dt).values

    assert len(out) == n_rows, "Row-count mismatch — submission would be invalid."
    assert not pd.isna(out).any(), "NaN found in predictions — feature build skipped a row."

    pd.Series(out).to_csv(AG_PREDICTIONS_PATH, index=False, header=False)
    print(f"✓ Wrote {len(out)} predictions → {AG_PREDICTIONS_PATH}")
    print(f"   stats: min={out.min():.3f}  mean={out.mean():.3f}  max={out.max():.3f}")
    print()
    print("To submit: copy predictions_autogluon.csv → predictions.csv")


if __name__ == "__main__":
    main()

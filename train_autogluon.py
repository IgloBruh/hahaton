"""Train AutoGluon TabularPredictor on wind-farm power data.

Uses the same feature pipeline as train.py (build_features) but replaces
the LightGBM-only ensemble with AutoGluon's multi-model ensemble
(LightGBM + XGBoost + CatBoost + ExtraTrees + Neural Net + WeightedEnsemble).

Key design choices
------------------
* Time-based split: last VAL_FRACTION of rows = validation (same as train.py).
  Passed as `tuning_data` so AutoGluon uses our chronological holdout instead
  of random k-fold — avoids temporal data leakage.
* num_bag_folds=0 / num_stack_levels=0: disables random cross-validation
  inside AutoGluon, which is required for correctness on time-series data.
* KNN excluded: degrades badly with 190 features at this dataset size.

Usage
-----
  python train_autogluon.py                       # 1 hour, high_quality
  python train_autogluon.py --time_limit 7200     # 2 hours
  python train_autogluon.py --preset best_quality # longest, highest quality
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (
    SEED, TRAIN_CSV, TARGET_COL, DATETIME_COL,
    VAL_FRACTION, INSTALLED_CAPACITY_MW, AG_MODEL_PATH,
)
from src.utils import set_global_seed, regression_metrics, pretty_print_metrics
from src.data_loader import load_train
from src.preprocessing import build_features, feature_columns, clip_predictions


def main(time_limit: int = 3600, preset: str = "high_quality") -> None:
    set_global_seed(SEED)

    print("Loading training data...")
    df = load_train(TRAIN_CSV)
    print(f"  rows={len(df)},  range=[{df[DATETIME_COL].min()}, {df[DATETIME_COL].max()}]")

    print("Building features...")
    df = build_features(df)
    feats = feature_columns(df)
    print(f"  {len(feats)} features.")

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df   = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  train={len(train_df)}  val={len(val_df)}"
          f"  (val [{val_df[DATETIME_COL].min()}, {val_df[DATETIME_COL].max()}])")

    y_val = val_df[TARGET_COL].values

    # AutoGluon receives only feature columns + target (no datetime)
    keep_cols = feats + [TARGET_COL]
    train_ag = train_df[keep_cols]
    val_ag   = val_df[keep_cols]

    from autogluon.tabular import TabularPredictor

    # Remove existing predictor directory so fit() starts fresh
    import shutil
    if AG_MODEL_PATH.exists():
        shutil.rmtree(AG_MODEL_PATH)

    print(f"\nFitting AutoGluon  preset={preset!r}  time_limit={time_limit}s ...")
    predictor = TabularPredictor(
        label=TARGET_COL,
        eval_metric="mean_absolute_error",
        path=str(AG_MODEL_PATH),
        verbosity=2,
    ).fit(
        train_data=train_ag,
        tuning_data=val_ag,      # explicit time-based holdout, no random k-fold
        presets=preset,
        time_limit=time_limit,
        num_bag_folds=0,         # disable bagging (would use random CV internally)
        num_stack_levels=0,      # disable stacking (would leak future into past)
        excluded_model_types=["KNN"],
    )

    # Leaderboard on validation set
    print("\nLeaderboard (MAE on validation set, lower = better):")
    lb = predictor.leaderboard(val_ag, silent=True)
    print(lb[["model", "score_val", "fit_time"]].to_string(index=False))

    # Final metrics in our standard format
    preds_raw = predictor.predict(val_ag[feats], as_pandas=False)
    preds_clip = clip_predictions(np.asarray(preds_raw, dtype=float), INSTALLED_CAPACITY_MW)
    m = regression_metrics(y_val, preds_clip)
    pretty_print_metrics("AutoGluon best ensemble (val)", m)

    print(f"\nModel saved → {AG_MODEL_PATH}")
    print("Run  python inference_autogluon.py  to generate predictions_autogluon.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_limit", type=int, default=3600,
        help="Training time budget in seconds (default: 3600)",
    )
    parser.add_argument(
        "--preset", type=str, default="high_quality",
        choices=["medium_quality", "high_quality", "best_quality"],
        help="AutoGluon quality preset (default: high_quality)",
    )
    args = parser.parse_args()
    main(args.time_limit, args.preset)

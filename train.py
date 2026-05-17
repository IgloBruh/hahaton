"""Train the wind-farm hourly power model.

Pipeline
--------
1. Load and chronologically sort the training data.
2. Build features (raw + physics + calendar + lag/lead/rolling).
3. Time-based split: last VAL_FRACTION of rows = validation.
4. Pass-1: fit ensemble with LightGBM early stopping; compute Permutation
   Importance on the validation set; prune noise features (PI < threshold).
5. Pass-2 (if features were pruned): refit with the reduced feature set and
   report updated validation metrics.
6. Retrain every member on the FULL dataset for the final saved model.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (
    SEED, TRAIN_CSV, MODEL_PATH, TARGET_COL, DATETIME_COL,
    VAL_FRACTION, FINAL_BOOST_MULTIPLIER, MAX_ITER, PATIENCE,
    ENSEMBLE_CONFIGS, INSTALLED_CAPACITY_MW, PI_PRUNE_THRESHOLD,
)
from src.utils import set_global_seed, regression_metrics, pretty_print_metrics
from src.data_loader import load_train
from src.preprocessing import build_features, feature_columns, clip_predictions
from src.model import WindPowerModel


def _compute_permutation_importance(
    model: WindPowerModel,
    X_arr: np.ndarray,
    y_val: np.ndarray,
    feats: list[str],
    base_mae: float,
    n_repeats: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    """Fast permutation importance using numpy arrays directly."""
    rng = np.random.default_rng(seed)
    scores: dict[str, float] = {}
    for col_idx, col_name in enumerate(feats):
        deltas = []
        orig_col = X_arr[:, col_idx].copy()
        for _ in range(n_repeats):
            X_arr[:, col_idx] = rng.permutation(orig_col)
            member_preds = np.stack(
                [reg.predict(X_arr) for reg in model.regressors], axis=0
            )
            p_perm = clip_predictions(member_preds.mean(axis=0), INSTALLED_CAPACITY_MW)
            deltas.append(float(np.mean(np.abs(p_perm - y_val))) - base_mae)
        X_arr[:, col_idx] = orig_col
        scores[col_name] = float(np.mean(deltas))
    return scores


def _fit_and_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    feats: list[str],
    label: str = "PASS",
) -> tuple[WindPowerModel, np.ndarray]:
    """Fit ensemble and report per-member + ensemble val metrics."""
    model = WindPowerModel.fit_with_validation(
        X_train=train_df, y_train=y_train,
        X_val=val_df, y_val=y_val,
        feature_names=feats,
        ensemble_configs=ENSEMBLE_CONFIGS,
        max_iter=MAX_ITER, patience=PATIENCE,
    )
    member_preds = model.predict_members(val_df)
    for k, p in enumerate(member_preds):
        p_clip = clip_predictions(p, INSTALLED_CAPACITY_MW)
        m = regression_metrics(y_val, p_clip)
        pretty_print_metrics(f"[{label}] m{k+1}", m)

    val_pred = clip_predictions(model.predict(val_df), INSTALLED_CAPACITY_MW)
    m = regression_metrics(y_val, val_pred)
    pretty_print_metrics(f"[{label}] ENSEMBLE", m)
    return model, val_pred


def main() -> None:
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

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values

    # ------------------------------------------------------------------ #
    # Pass 1 — full feature set                                           #
    # ------------------------------------------------------------------ #
    print(f"\nPass 1: fitting {len(ENSEMBLE_CONFIGS)}-member ensemble ({len(feats)} feats)...")
    model_p1, val_pred_p1 = _fit_and_report(
        train_df, val_df, y_train, y_val, feats, label="P1"
    )

    # LightGBM gain importance
    print("\nTop-30 features by LightGBM gain importance:")
    fi = model_p1.feature_importance(importance_type="gain")
    for feat, score in fi.head(30).items():
        print(f"  {feat:<45s} {score:>12.1f}")

    # Permutation importance
    print(f"\nComputing Permutation Importance (n_repeats=3)...")
    Xv_arr = val_df[feats].values.astype(float)
    base_mae = float(np.mean(np.abs(val_pred_p1 - y_val)))
    pi_scores = _compute_permutation_importance(
        model_p1, Xv_arr, y_val, feats, base_mae, n_repeats=3, seed=SEED
    )
    pi_series = pd.Series(pi_scores).sort_values(ascending=False)
    print("Top-25 by Permutation Importance (MAE delta when feature is shuffled):")
    for feat, score in pi_series.head(25).items():
        print(f"  {feat:<45s} {score:>+8.4f}")

    # Save PI to file for inspection
    pi_out = ROOT / "pi_scores.json"
    pi_out.write_text(json.dumps(pi_scores, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------ #
    # Feature pruning                                                      #
    # ------------------------------------------------------------------ #
    bad_feats = [f for f, s in pi_scores.items() if s < PI_PRUNE_THRESHOLD]
    feats_final = [f for f in feats if f not in bad_feats]
    print(f"\nPruning threshold={PI_PRUNE_THRESHOLD}:  "
          f"removed {len(bad_feats)},  kept {len(feats_final)} features.")

    if bad_feats:
        print(f"  Removed: {bad_feats[:10]}{'...' if len(bad_feats) > 10 else ''}")

    # ------------------------------------------------------------------ #
    # Pass 2 — pruned features (only if anything was removed)             #
    # ------------------------------------------------------------------ #
    if bad_feats:
        print(f"\nPass 2: refitting with {len(feats_final)} features...")
        model_final_val, val_pred_p2 = _fit_and_report(
            train_df, val_df, y_train, y_val, feats_final, label="P2"
        )
        best_iters = model_final_val.best_iterations
    else:
        print("\nNo features pruned — using Pass 1 model.")
        model_final_val = model_p1
        best_iters = model_p1.best_iterations

    # ------------------------------------------------------------------ #
    # Final model — retrain on full dataset                               #
    # ------------------------------------------------------------------ #
    print("\nRetraining ensemble on full dataset...")
    final_model = WindPowerModel.fit_full(
        X=df, y=df[TARGET_COL].values,
        feature_names=feats_final,
        ensemble_configs=ENSEMBLE_CONFIGS,
        best_iterations=best_iters,
        boost_multiplier=FINAL_BOOST_MULTIPLIER,
    )
    final_model.save(MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"  members    : {final_model.n_members}")
    print(f"  features   : {len(feats_final)}")
    print(f"  best_iters : {final_model.best_iterations}")


if __name__ == "__main__":
    main()

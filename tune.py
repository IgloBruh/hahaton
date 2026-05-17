"""Optuna-based hyperparameter search for the LightGBM wind-power model.

Usage
-----
    python tune.py                  # 50 trials, saves tune_results.json
    python tune.py --trials 100     # more trials

After the run, copy the best params into config.py ENSEMBLE_CONFIGS or pass
them directly to the LGBMRegressor in src/model.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import SEED, TRAIN_CSV, TARGET_COL, VAL_FRACTION, INSTALLED_CAPACITY_MW
from src.data_loader import load_train
from src.preprocessing import build_features, feature_columns
from src.preprocessing import clip_predictions
from src.utils import set_global_seed

MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 100


def _objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "objective": trial.suggest_categorical("objective", ["regression", "regression_l1"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "n_estimators": MAX_ESTIMATORS,
        "random_state": SEED,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    preds = clip_predictions(model.predict(X_val), INSTALLED_CAPACITY_MW)
    mae = float(np.mean(np.abs(preds - y_val)))

    trial.set_user_attr("best_iteration", int(model.best_iteration_))
    return mae


def main(n_trials: int = 50) -> None:
    set_global_seed(SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("→ Loading and preparing data...")
    df = load_train(TRAIN_CSV)
    df = build_features(df)
    feats = feature_columns(df)
    print(f"   {len(feats)} features, {len(df)} rows")

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[feats].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feats].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values

    print(f"→ Starting Optuna study ({n_trials} trials)...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n=== Best result ===")
    print(f"  MAE (val) : {best.value:.4f}")
    print(f"  best_iter : {best.user_attrs.get('best_iteration', '?')}")
    print("  params    :")
    for k, v in best.params.items():
        print(f"    {k:<22s}: {v}")

    result = {
        "mae": best.value,
        "best_iteration": best.user_attrs.get("best_iteration"),
        "params": best.params,
    }
    out_path = ROOT / "tune_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Saved → {out_path}")

    # Print a ready-to-paste ENSEMBLE_CONFIGS snippet
    p = best.params
    print("\n--- Suggested config entry (paste into config.py ENSEMBLE_CONFIGS) ---")
    print(
        f"    dict(objective={p['objective']!r}, "
        f"learning_rate={p['learning_rate']:.5f}, "
        f"num_leaves={p['num_leaves']}, "
        f"min_child_samples={p['min_child_samples']}, "
        f"reg_lambda={p['reg_lambda']:.5f}, "
        f"reg_alpha={p['reg_alpha']:.2e}, "
        f"colsample_bytree={p['colsample_bytree']:.3f}, "
        f"subsample={p['subsample']:.3f}, "
        f"subsample_freq=1, seed=42),"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(n_trials=args.trials)

"""Train v8: Best ensemble — empirical power curve + ramp features + optimised weights.

What's new vs v5/v6
--------------------
* NO target-lag features (train/test mismatch eliminated, error_rate 8.38→8.06%).
* Empirical power curve: fitted from training data, maps hub wind speed →
  expected MW per turbine. Captures site-specific turbine behaviour that the
  parametric SG curve misses (control algorithm, blade contamination, wake
  calibration, etc.).  This is the highest-ROI single addition for wind power
  forecasting.
* Wind-ramp features: (v_lead - v_lag) at multiple horizons — capture whether
  wind is accelerating or decelerating around this timestep.
* Wider Optuna search (60 trials, new cache so params are re-tuned for the
  new feature set).
* 26-member ensemble: 5 LGBM×3 seeds + 3 DART + 3 XGB + 5 CatBoost.
* Scipy-optimised ensemble weights instead of simple average.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import joblib
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.optimize import minimize

from config import (
    SEED, TRAIN_CSV, INSTALLED_CAPACITY_MW,
    TARGET_COL, VAL_FRACTION,
    MAX_ITER, PATIENCE, FINAL_BOOST_MULTIPLIER,
)
from src.utils import set_global_seed, regression_metrics, pretty_print_metrics
from src.data_loader import load_train
from src.preprocessing import (
    build_features, feature_columns, clip_predictions,
    fit_empirical_curve, apply_empirical_curve,
)

MODEL_V8_PATH = ROOT / "models" / "model_v8.pkl"
OPTUNA_CACHE  = ROOT / "optuna_v8.json"

N_OPTUNA_TRIALS = 60
N_TOP_CONFIGS   = 5
LGBM_SEEDS      = [42, 7, 2024]

# Wider search than v5 — re-tune without target lags + new features
_LR_LO, _LR_HI   = 0.005, 0.03
_NL_LO, _NL_HI   = 50,   250
_MCS_LO, _MCS_HI = 30,   130
_RL_LO, _RL_HI   = 0.05, 10.0
_CBT_LO, _CBT_HI = 0.45, 0.95
_SS_LO,  _SS_HI  = 0.70, 1.00

_DART_CONFIGS = [
    dict(num_leaves=100, learning_rate=0.05, drop_rate=0.10, skip_drop=0.50, seed=42),
    dict(num_leaves=150, learning_rate=0.05, drop_rate=0.15, skip_drop=0.50, seed=7),
    dict(num_leaves=80,  learning_rate=0.05, drop_rate=0.20, skip_drop=0.40, seed=2024),
]
_XGB_CONFIGS = [
    dict(learning_rate=0.010, max_depth=7, subsample=0.80, colsample_bytree=0.55, seed=42),
    dict(learning_rate=0.008, max_depth=8, subsample=0.85, colsample_bytree=0.50, seed=7),
    dict(learning_rate=0.012, max_depth=6, subsample=0.75, colsample_bytree=0.60, seed=2024),
]
_CB_CONFIGS = [
    dict(learning_rate=0.010, depth=7,  l2_leaf_reg=5.0, random_seed=42),
    dict(learning_rate=0.008, depth=8,  l2_leaf_reg=3.0, random_seed=7),
    dict(learning_rate=0.012, depth=9,  l2_leaf_reg=2.0, random_seed=2024),
    dict(learning_rate=0.007, depth=10, l2_leaf_reg=1.0, random_seed=13),
    dict(learning_rate=0.006, depth=11, l2_leaf_reg=0.5, random_seed=99),
]


# ---- Model factories -------------------------------------------------------

def _make_lgbm_gbdt(params: dict, seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1", boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=0.0,
        colsample_bytree=params["colsample_bytree"],
        subsample=params["subsample"],
        subsample_freq=1,
        random_state=seed, verbose=-1,
    )


def _make_lgbm_dart(cfg: dict, n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1", boosting_type="dart",
        n_estimators=n_estimators,
        num_leaves=cfg["num_leaves"],
        learning_rate=cfg["learning_rate"],
        drop_rate=cfg["drop_rate"],
        skip_drop=cfg["skip_drop"],
        min_child_samples=40, reg_lambda=1.0,
        colsample_bytree=0.65, subsample=0.90, subsample_freq=1,
        random_state=cfg["seed"], verbose=-1,
    )


def _make_xgb(cfg: dict, n_estimators: int, early: bool = True) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:absoluteerror", n_estimators=n_estimators,
        learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"],
        subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
        reg_lambda=1.0, reg_alpha=0.0, random_state=cfg["seed"], verbosity=0,
        early_stopping_rounds=PATIENCE if early else None,
    )


def _make_cb(cfg: dict, iterations: int, early: bool = True) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE", iterations=iterations,
        learning_rate=cfg["learning_rate"], depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"], random_seed=cfg["random_seed"],
        verbose=0, early_stopping_rounds=PATIENCE if early else None,
        task_type="CPU",
    )


# ---- Weight optimisation ---------------------------------------------------

def _optimize_weights(member_preds: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Scipy SLSQP — optimal convex combination minimising MAE on val set."""
    n = member_preds.shape[0]
    w0 = np.ones(n) / n

    def mae(w):
        return float(np.mean(np.abs(member_preds.T @ w - y_val)))

    result = minimize(
        mae, w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 3000, "ftol": 1e-13},
    )
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


# ---- Optuna objective ------------------------------------------------------

def _lgbm_val_mae(X_tr, y_tr, X_v, y_v, params: dict) -> tuple[float, int]:
    reg = _make_lgbm_gbdt(params, seed=42, n_estimators=MAX_ITER)
    reg.fit(X_tr, y_tr, eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(0)])
    mae = float(np.mean(np.abs(reg.predict(X_v) - y_v)))
    return mae, int(reg.best_iteration_)


# ============================================================================

def main() -> None:
    set_global_seed(SEED)

    # ------------------------------------------------------------------ #
    # Load data + base features (no target lags, with ramp features)      #
    # ------------------------------------------------------------------ #
    print("Loading and preparing data...")
    df = load_train(TRAIN_CSV)
    df = build_features(df, use_target_lags=False)

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df   = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  train={len(train_df)}  val={len(val_df)}")

    # ------------------------------------------------------------------ #
    # Fit empirical power curve on training data only                     #
    # ------------------------------------------------------------------ #
    print("\nFitting empirical power curve from training data...")
    pc_map = fit_empirical_curve(train_df, TARGET_COL)
    train_df = apply_empirical_curve(train_df, pc_map)
    val_df   = apply_empirical_curve(val_df,   pc_map)
    # Also build full dataset version for final retrain
    df_full  = apply_empirical_curve(df,       pc_map)

    n_bins = len(pc_map["bin_centers"])
    pc_arr = np.array(pc_map["medians"])
    peak_mw = float(np.max(pc_arr))
    peak_v  = float(np.array(pc_map["bin_centers"])[np.argmax(pc_arr)])
    print(f"  {n_bins} bins, peak production/turbine={peak_mw:.3f} MW at v={peak_v:.1f} m/s")

    feats = feature_columns(train_df)
    print(f"  {len(feats)} features (incl. empirical curve + ramp features)")

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values
    X_train = train_df[feats].values.astype(float)
    X_val   = val_df[feats].values.astype(float)
    X_full  = df_full[feats].values.astype(float)
    y_full  = df_full[TARGET_COL].values

    # ------------------------------------------------------------------ #
    # Optuna — tune LightGBM on new feature set (new cache)               #
    # ------------------------------------------------------------------ #
    if OPTUNA_CACHE.exists():
        print(f"\nLoading cached Optuna results from {OPTUNA_CACHE} ...")
        cache = json.loads(OPTUNA_CACHE.read_text())
        top_params = cache["top_params"]
        print(f"  Best single-member MAE: {cache['best_mae']:.4f}")
    else:
        print(f"\nRunning Optuna ({N_OPTUNA_TRIALS} trials, empirical curve + ramp features)...")

        def objective(trial: optuna.Trial) -> float:
            p = {
                "learning_rate":     trial.suggest_float("lr",  _LR_LO,  _LR_HI,  log=True),
                "num_leaves":        trial.suggest_int("nl",    _NL_LO,  _NL_HI),
                "min_child_samples": trial.suggest_int("mcs",   _MCS_LO, _MCS_HI),
                "reg_lambda":        trial.suggest_float("rl",  _RL_LO,  _RL_HI,  log=True),
                "colsample_bytree":  trial.suggest_float("cbt", _CBT_LO, _CBT_HI),
                "subsample":         trial.suggest_float("ss",  _SS_LO,  _SS_HI),
            }
            mae, _ = _lgbm_val_mae(X_train, y_train, X_val, y_val, p)
            return mae

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        trials_sorted = sorted(study.trials, key=lambda t: t.value)
        top_params_raw = [t.params for t in trials_sorted[:N_TOP_CONFIGS]]

        def _remap(p: dict) -> dict:
            return {
                "learning_rate":     p["lr"],
                "num_leaves":        p["nl"],
                "min_child_samples": p["mcs"],
                "reg_lambda":        p["rl"],
                "colsample_bytree":  p["cbt"],
                "subsample":         p["ss"],
            }
        top_params = [_remap(p) for p in top_params_raw]

        OPTUNA_CACHE.write_text(json.dumps({
            "top_params": top_params,
            "best_mae": study.best_value,
        }, indent=2))
        print(f"  Best MAE: {study.best_value:.4f}")
        print(f"  Saved -> {OPTUNA_CACHE}")

    # ------------------------------------------------------------------ #
    # Build ensemble: 5×3 LGBM + 3 DART + 3 XGB + 5 CB = 26 members     #
    # ------------------------------------------------------------------ #
    all_members, all_best_iters, all_labels = [], [], []
    dart_start = xgb_start = cb_start = 0

    # --- LightGBM GBDT ---------------------------------------------------
    print(f"\nFitting top-{N_TOP_CONFIGS} Optuna LGBM x {len(LGBM_SEEDS)} seeds...")
    for ci, params in enumerate(top_params):
        for seed in LGBM_SEEDS:
            reg = _make_lgbm_gbdt(params, seed=seed, n_estimators=MAX_ITER)
            reg.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                                lgb.log_evaluation(0)])
            bi  = int(reg.best_iteration_)
            mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
            print(f"   lgbm c{ci+1} s{seed:<5d}  iter={bi:<4d}  MAE={mae:.4f}")
            all_members.append(reg); all_best_iters.append(bi)
            all_labels.append(f"lgbm_c{ci+1}_s{seed}")

    # --- LightGBM DART ---------------------------------------------------
    dart_start = len(all_members)
    print(f"\nFitting {len(_DART_CONFIGS)} DART members...")
    for k, cfg in enumerate(_DART_CONFIGS):
        dart_iter = min(MAX_ITER, 800)
        reg = _make_lgbm_dart(cfg, n_estimators=dart_iter)
        reg.fit(X_train, y_train)
        mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
        print(f"   dart {k+1}  nl={cfg['num_leaves']}  MAE={mae:.4f}")
        all_members.append(reg); all_best_iters.append(dart_iter)
        all_labels.append(f"dart_nl{cfg['num_leaves']}_s{cfg['seed']}")

    # --- XGBoost ---------------------------------------------------------
    xgb_start = len(all_members)
    print(f"\nFitting {len(_XGB_CONFIGS)} XGBoost members...")
    for k, cfg in enumerate(_XGB_CONFIGS):
        reg = _make_xgb(cfg, n_estimators=MAX_ITER)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        bi  = int(reg.best_iteration)
        mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
        print(f"   xgb  {k+1}  depth={cfg['max_depth']}  iter={bi:<4d}  MAE={mae:.4f}")
        all_members.append(reg); all_best_iters.append(bi)
        all_labels.append(f"xgb_d{cfg['max_depth']}_s{cfg['seed']}")

    # --- CatBoost --------------------------------------------------------
    cb_start = len(all_members)
    print(f"\nFitting {len(_CB_CONFIGS)} CatBoost members...")
    for k, cfg in enumerate(_CB_CONFIGS):
        reg = _make_cb(cfg, iterations=MAX_ITER)
        reg.fit(X_train, y_train, eval_set=(X_val, y_val))
        bi  = int(reg.best_iteration_)
        mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
        print(f"   cb   {k+1}  depth={cfg['depth']}  iter={bi:<4d}  MAE={mae:.4f}")
        all_members.append(reg); all_best_iters.append(bi)
        all_labels.append(f"cb_d{cfg['depth']}_s{cfg['random_seed']}")

    # ---- Ensemble evaluation -------------------------------------------
    member_val_preds = np.array([
        clip_predictions(m.predict(X_val), INSTALLED_CAPACITY_MW)
        for m in all_members
    ])

    ens_equal = clip_predictions(member_val_preds.mean(axis=0), INSTALLED_CAPACITY_MW)
    m_eq = regression_metrics(y_val, ens_equal)

    print(f"\nOptimising ensemble weights ({len(all_members)} members)...")
    opt_weights = _optimize_weights(member_val_preds, y_val)
    ens_opt = clip_predictions(member_val_preds.T @ opt_weights, INSTALLED_CAPACITY_MW)
    m_opt = regression_metrics(y_val, ens_opt)

    print(f"\n{'='*60}")
    pretty_print_metrics("V8 equal  ", m_eq)
    pretty_print_metrics("V8 optimised", m_opt)
    print(f"{'='*60}")

    # Top-10 weighted members
    top_idx = np.argsort(opt_weights)[::-1][:10]
    print("\nTop-10 members by weight:")
    for i in top_idx:
        if opt_weights[i] > 1e-4:
            print(f"  {all_labels[i]:<30s}  w={opt_weights[i]:.4f}")

    if m_opt["MAE"] <= 4.0:
        print("  *** TARGET MAE <= 4.0 REACHED ***")

    # ---- Retrain on full dataset ----------------------------------------
    print(f"\nRetraining {len(all_members)} members on full dataset...")
    final = []
    for i, (member, bi, label) in enumerate(zip(all_members, all_best_iters, all_labels)):
        rounds = max(200, int(round(bi * FINAL_BOOST_MULTIPLIER)))
        if label.startswith("dart"):
            di  = i - dart_start
            reg = _make_lgbm_dart(_DART_CONFIGS[di], n_estimators=rounds)
            reg.fit(X_full, y_full)
        elif label.startswith("lgbm"):
            ci  = i // len(LGBM_SEEDS)
            si  = i % len(LGBM_SEEDS)
            reg = _make_lgbm_gbdt(top_params[ci], seed=LGBM_SEEDS[si], n_estimators=rounds)
            reg.fit(X_full, y_full)
        elif label.startswith("xgb"):
            xi  = i - xgb_start
            reg = _make_xgb(_XGB_CONFIGS[xi], n_estimators=rounds, early=False)
            reg.fit(X_full, y_full)
        else:
            ci2 = i - cb_start
            reg = _make_cb(_CB_CONFIGS[ci2], iterations=rounds, early=False)
            reg.fit(X_full, y_full)
        print(f"   [{label}]  rounds={rounds}")
        final.append(reg)

    MODEL_V8_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "regressors":      final,
        "feature_names":   feats,
        "best_iterations": [max(200, int(round(b * FINAL_BOOST_MULTIPLIER)))
                             for b in all_best_iters],
        "member_labels":   all_labels,
        "ensemble_weights": opt_weights.tolist(),
        "pc_map":          pc_map,        # empirical power curve
        "use_target_lags": False,
        "val_metrics": {"equal": m_eq, "optimised": m_opt},
    }, MODEL_V8_PATH)

    print(f"\nSaved -> {MODEL_V8_PATH}")
    print(f"  {len(final)} members  |  {len(feats)} features")
    print(f"  val MAE equal={m_eq['MAE']:.4f}  optimised={m_opt['MAE']:.4f}")
    print(f"  val nMAE%  equal={m_eq['nMAE_%']:.2f}%  optimised={m_opt['nMAE_%']:.2f}%")


if __name__ == "__main__":
    main()

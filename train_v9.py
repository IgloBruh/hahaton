"""Train v9: conditional-mean features + high-LR CatBoost + residual correction.

Improvements over v8
--------------------
1. Conditional mean production features (hour, month, hour×month):
   Historical average production for this time-of-day / season.
   Computed from training data only — no leakage. Strong signal because
   diurnal and seasonal patterns are real and consistent.

2. High-LR CatBoost configs inspired by AutoGluon's best member:
   AutoGluon found CatBoost with lr≈0.07, depth=8 is the best single model
   (MAE=7.036). Our v8 used lr=0.006–0.012 — much too conservative.

3. Residual correction (level-2 model):
   Train a small LightGBM on out-of-fold v8 residuals using 3-fold
   time-series CV. Corrects systematic biases (e.g. ramp events, icing).

4. Final prediction = v9_ensemble + residual_correction.
"""
from __future__ import annotations

import json
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

MODEL_V9_PATH = ROOT / "models" / "model_v9.pkl"
OPTUNA_CACHE  = ROOT / "optuna_v8.json"   # reuse v8 tuned params

N_TOP_CONFIGS = 5
LGBM_SEEDS    = [42, 7, 2024]

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
# v8 configs + 3 high-LR configs inspired by AutoGluon's best CatBoost (lr≈0.07, depth=8)
_CB_CONFIGS = [
    dict(learning_rate=0.010, depth=7,  l2_leaf_reg=5.0, random_seed=42),
    dict(learning_rate=0.008, depth=8,  l2_leaf_reg=3.0, random_seed=7),
    dict(learning_rate=0.012, depth=9,  l2_leaf_reg=2.0, random_seed=2024),
    dict(learning_rate=0.007, depth=10, l2_leaf_reg=1.0, random_seed=13),
    dict(learning_rate=0.006, depth=11, l2_leaf_reg=0.5, random_seed=99),
    # High-LR configs (AutoGluon-inspired)
    dict(learning_rate=0.069, depth=8,  l2_leaf_reg=2.15, random_seed=42),
    dict(learning_rate=0.050, depth=8,  l2_leaf_reg=2.0,  random_seed=7),
    dict(learning_rate=0.040, depth=6,  l2_leaf_reg=3.0,  random_seed=2024),
]

# CV folds for residual correction (expanding window)
_RESID_FOLDS = [
    (0, 0.65, 0.75),
    (0, 0.75, 0.85),
    (0, 0.85, 0.90),
]


# ── Conditional mean features ────────────────────────────────────────────────

def fit_conditional_means(train_df: pd.DataFrame) -> dict:
    """Compute historical mean production per hour, month, hour×month."""
    g = train_df.groupby
    gm = float(train_df[TARGET_COL].mean())
    return {
        "global_mean":      gm,
        "hour_mean":        g("hour")[TARGET_COL].mean().to_dict(),
        "month_mean":       g("month_num")[TARGET_COL].mean().to_dict(),
        "hour_month_mean":  g(["hour", "month_num"])[TARGET_COL].mean().to_dict(),
    }


def apply_conditional_means(df: pd.DataFrame, cm: dict) -> pd.DataFrame:
    gm = cm["global_mean"]
    df = df.copy()
    df["prod_cond_hour"]  = df["hour"].map(cm["hour_mean"]).fillna(gm)
    df["prod_cond_month"] = df["month_num"].map(cm["month_mean"]).fillna(gm)
    # hour×month — merge is vectorised
    hm = (pd.Series(cm["hour_month_mean"])
            .reset_index()
            .rename(columns={"level_0": "hour", "level_1": "month_num", 0: "prod_cond_hm"}))
    df = df.merge(hm, on=["hour", "month_num"], how="left")
    df["prod_cond_hm"] = df["prod_cond_hm"].fillna(gm)
    return df


# ── Model factories ──────────────────────────────────────────────────────────

def _make_lgbm_gbdt(params: dict, seed: int, n_est: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1", boosting_type="gbdt", n_estimators=n_est,
        learning_rate=params["learning_rate"], num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"], reg_lambda=params["reg_lambda"],
        reg_alpha=0.0, colsample_bytree=params["colsample_bytree"],
        subsample=params["subsample"], subsample_freq=1, random_state=seed, verbose=-1,
    )

def _make_lgbm_dart(cfg: dict, n_est: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1", boosting_type="dart", n_estimators=n_est,
        num_leaves=cfg["num_leaves"], learning_rate=cfg["learning_rate"],
        drop_rate=cfg["drop_rate"], skip_drop=cfg["skip_drop"],
        min_child_samples=40, reg_lambda=1.0, colsample_bytree=0.65,
        subsample=0.90, subsample_freq=1, random_state=cfg["seed"], verbose=-1,
    )

def _make_xgb(cfg: dict, n_est: int, early: bool = True) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:absoluteerror", n_estimators=n_est,
        learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"],
        subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
        reg_lambda=1.0, random_state=cfg["seed"], verbosity=0,
        early_stopping_rounds=PATIENCE if early else None,
    )

def _make_cb(cfg: dict, iterations: int, early: bool = True) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE", iterations=iterations,
        learning_rate=cfg["learning_rate"], depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"], random_seed=cfg["random_seed"],
        verbose=0, early_stopping_rounds=PATIENCE if early else None, task_type="CPU",
    )


# ── Weight optimisation ──────────────────────────────────────────────────────

def _opt_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = preds.shape[0]
    w0 = np.ones(n) / n
    res = minimize(
        lambda w: float(np.mean(np.abs(preds.T @ w - y))),
        w0, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 3000, "ftol": 1e-13},
    )
    w = np.clip(res.x, 0, 1);  return w / w.sum()


# ── Residual correction (level-2 LightGBM) ──────────────────────────────────

def _build_residual_corrector(
    X_all: np.ndarray, y_all: np.ndarray,
    top_params: list, lgbm_seeds: list,
    dart_configs: list, xgb_configs: list, cb_configs: list,
    val_start: int,
) -> lgb.LGBMRegressor | None:
    """3-fold expanding-window CV: collect OOF base predictions, train corrector."""
    n = val_start  # rows available for CV (exclude final holdout)
    oof_base = np.zeros(n)
    oof_y    = np.zeros(n)

    print("\nFitting residual corrector (3-fold OOF)...")
    for fi, (tr_end_f, v_s_f, v_e_f) in enumerate(_RESID_FOLDS):
        tr_e = int(n * tr_end_f / (val_start / len(X_all)))  # scale to available rows
        v_s  = int(len(X_all) * v_s_f)
        v_e  = int(len(X_all) * v_e_f)
        if v_e > val_start: v_e = val_start
        tr_e = min(tr_e, v_s)

        X_tr_f, y_tr_f = X_all[:tr_e], y_all[:tr_e]
        X_v_f,  y_v_f  = X_all[v_s:v_e], y_all[v_s:v_e]
        if len(X_tr_f) < 100 or len(X_v_f) < 50:
            continue

        # Single fast LGBM per fold (seed=42 only)
        reg = _make_lgbm_gbdt(top_params[0], seed=42, n_est=MAX_ITER)
        reg.fit(X_tr_f, y_tr_f, eval_set=[(X_v_f, y_v_f)],
                callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(0)])
        fold_pred = clip_predictions(reg.predict(X_v_f), INSTALLED_CAPACITY_MW)
        oof_base[v_s:v_e] = fold_pred
        oof_y[v_s:v_e]    = y_v_f
        fold_mae = float(np.mean(np.abs(fold_pred - y_v_f)))
        print(f"  fold {fi+1}: tr[:{tr_e}] val[{v_s}:{v_e}]  MAE={fold_mae:.4f}")

    # Only keep rows where OOF was filled
    filled = oof_base != 0
    if filled.sum() < 200:
        print("  Not enough OOF rows — skipping corrector.")
        return None

    # Residuals to correct
    resid = oof_y[filled] - oof_base[filled]
    X_oof = X_all[:val_start][filled]

    # Corrector: fast LightGBM on residuals
    corrector = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=300,
        learning_rate=0.05, num_leaves=31, min_child_samples=50,
        reg_lambda=5.0, colsample_bytree=0.7, subsample=0.8,
        random_state=SEED, verbose=-1,
    )
    corrector.fit(X_oof, resid)
    corr_pred = corrector.predict(X_oof)
    corr_mae  = float(np.mean(np.abs((oof_y[filled] - oof_base[filled]) - corr_pred)))
    print(f"  corrector residual MAE: {corr_mae:.4f}  (ideal=0)")
    return corrector


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    set_global_seed(SEED)

    print("Loading data (NO target lags)...")
    df = load_train(TRAIN_CSV)
    df = build_features(df, use_target_lags=False)

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df   = df.iloc[split_idx:].reset_index(drop=True)

    # Empirical power curve (from training data only)
    print("Fitting empirical power curve...")
    pc_map = fit_empirical_curve(train_df, TARGET_COL)
    train_df = apply_empirical_curve(train_df, pc_map)
    val_df   = apply_empirical_curve(val_df,   pc_map)
    df_full  = apply_empirical_curve(df,       pc_map)

    # Conditional mean features (from training data only)
    print("Fitting conditional mean features...")
    cond_means = fit_conditional_means(train_df)
    train_df = apply_conditional_means(train_df, cond_means)
    val_df   = apply_conditional_means(val_df,   cond_means)
    df_full  = apply_conditional_means(df_full,  cond_means)

    feats = feature_columns(train_df)
    new_cond = [f for f in feats if "cond" in f]
    print(f"  {len(feats)} features  (+{len(new_cond)} conditional: {new_cond})")

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values
    X_train = train_df[feats].values.astype(float)
    X_val   = val_df[feats].values.astype(float)
    X_full  = df_full[feats].values.astype(float)
    y_full  = df_full[TARGET_COL].values

    # Optuna params (reuse v8 cache)
    print(f"\nLoading Optuna cache from {OPTUNA_CACHE} ...")
    top_params = json.loads(OPTUNA_CACHE.read_text())["top_params"]
    best_mae   = json.loads(OPTUNA_CACHE.read_text())["best_mae"]
    print(f"  best single-member MAE (v8): {best_mae:.4f}")

    # ── Build ensemble ───────────────────────────────────────────────────────
    all_members, all_best_iters, all_labels = [], [], []
    dart_start = xgb_start = cb_start = 0

    # LGBM GBDT
    print(f"\nFitting {N_TOP_CONFIGS}x{len(LGBM_SEEDS)} LGBM GBDT...")
    for ci, params in enumerate(top_params):
        for seed in LGBM_SEEDS:
            reg = _make_lgbm_gbdt(params, seed=seed, n_est=MAX_ITER)
            reg.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(0)])
            bi  = int(reg.best_iteration_)
            mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
            print(f"   lgbm c{ci+1} s{seed:<5d}  iter={bi:<4d}  MAE={mae:.4f}")
            all_members.append(reg); all_best_iters.append(bi)
            all_labels.append(f"lgbm_c{ci+1}_s{seed}")

    # DART
    dart_start = len(all_members)
    print(f"\nFitting {len(_DART_CONFIGS)} DART...")
    for k, cfg in enumerate(_DART_CONFIGS):
        reg = _make_lgbm_dart(cfg, n_est=min(MAX_ITER, 800))
        reg.fit(X_train, y_train)
        mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
        print(f"   dart {k+1}  nl={cfg['num_leaves']}  MAE={mae:.4f}")
        all_members.append(reg); all_best_iters.append(min(MAX_ITER, 800))
        all_labels.append(f"dart_nl{cfg['num_leaves']}_s{cfg['seed']}")

    # XGBoost
    xgb_start = len(all_members)
    print(f"\nFitting {len(_XGB_CONFIGS)} XGBoost...")
    for k, cfg in enumerate(_XGB_CONFIGS):
        reg = _make_xgb(cfg, n_est=MAX_ITER)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        bi  = int(reg.best_iteration)
        mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
        print(f"   xgb  {k+1}  depth={cfg['max_depth']}  iter={bi:<4d}  MAE={mae:.4f}")
        all_members.append(reg); all_best_iters.append(bi)
        all_labels.append(f"xgb_d{cfg['max_depth']}_s{cfg['seed']}")

    # CatBoost (v8 configs + high-LR configs)
    cb_start = len(all_members)
    print(f"\nFitting {len(_CB_CONFIGS)} CatBoost (incl. {3} high-LR)...")
    for k, cfg in enumerate(_CB_CONFIGS):
        reg = _make_cb(cfg, iterations=MAX_ITER)
        reg.fit(X_train, y_train, eval_set=(X_val, y_val))
        bi  = int(reg.best_iteration_)
        mae = float(np.mean(np.abs(reg.predict(X_val) - y_val)))
        tag = " [high-LR]" if cfg["learning_rate"] >= 0.04 else ""
        print(f"   cb  {k+1}  depth={cfg['depth']}  lr={cfg['learning_rate']}  iter={bi:<4d}  MAE={mae:.4f}{tag}")
        all_members.append(reg); all_best_iters.append(bi)
        all_labels.append(f"cb_d{cfg['depth']}_lr{cfg['learning_rate']}_s{cfg['random_seed']}")

    # ── Ensemble evaluation ──────────────────────────────────────────────────
    val_preds = np.array([
        clip_predictions(m.predict(X_val), INSTALLED_CAPACITY_MW) for m in all_members
    ])
    ens_equal = val_preds.mean(axis=0)
    m_eq = regression_metrics(y_val, clip_predictions(ens_equal, INSTALLED_CAPACITY_MW))

    print(f"\nOptimising weights ({len(all_members)} members)...")
    opt_w = _opt_weights(val_preds, y_val)
    ens_opt = clip_predictions(val_preds.T @ opt_w, INSTALLED_CAPACITY_MW)
    m_opt = regression_metrics(y_val, ens_opt)

    print(f"\n{'='*60}")
    pretty_print_metrics("V9 equal    ", m_eq)
    pretty_print_metrics("V9 optimised", m_opt)

    # ── Residual corrector ───────────────────────────────────────────────────
    corrector = _build_residual_corrector(
        X_full, y_full, top_params, LGBM_SEEDS,
        _DART_CONFIGS, _XGB_CONFIGS, _CB_CONFIGS, split_idx,
    )

    if corrector is not None:
        corr_val = corrector.predict(X_val)
        ens_corrected = clip_predictions(ens_opt + corr_val, INSTALLED_CAPACITY_MW)
        m_corr = regression_metrics(y_val, ens_corrected)
        pretty_print_metrics("V9 +corrector", m_corr)
        use_corrector = m_corr["MAE"] < m_opt["MAE"]
        print(f"  Corrector {'HELPS' if use_corrector else 'does NOT help'} on val")
    else:
        use_corrector = False
        m_corr = m_opt

    print(f"{'='*60}")
    if min(m_opt["MAE"], m_corr["MAE"]) <= 4.0:
        print("  *** TARGET MAE <= 4.0 REACHED ***")

    # Top members by weight
    for i in np.argsort(opt_w)[::-1][:8]:
        if opt_w[i] > 0.01:
            print(f"  {all_labels[i]:<40s}  w={opt_w[i]:.4f}")

    # ── Retrain on full data ─────────────────────────────────────────────────
    print(f"\nRetraining {len(all_members)} members on full data...")
    final = []
    for i, (mem, bi, label) in enumerate(zip(all_members, all_best_iters, all_labels)):
        rounds = max(200, int(round(bi * FINAL_BOOST_MULTIPLIER)))
        if label.startswith("dart"):
            di  = i - dart_start
            reg = _make_lgbm_dart(_DART_CONFIGS[di], n_est=rounds)
            reg.fit(X_full, y_full)
        elif label.startswith("lgbm"):
            ci  = i // len(LGBM_SEEDS)
            si  = i % len(LGBM_SEEDS)
            reg = _make_lgbm_gbdt(top_params[ci], seed=LGBM_SEEDS[si], n_est=rounds)
            reg.fit(X_full, y_full)
        elif label.startswith("xgb"):
            xi  = i - xgb_start
            reg = _make_xgb(_XGB_CONFIGS[xi], n_est=rounds, early=False)
            reg.fit(X_full, y_full)
        else:
            ci2 = i - cb_start
            reg = _make_cb(_CB_CONFIGS[ci2], iterations=rounds, early=False)
            reg.fit(X_full, y_full)
        print(f"   [{label}]  rounds={rounds}")
        final.append(reg)

    # Retrain corrector on full data if it helps
    if use_corrector and corrector is not None:
        print("Retraining residual corrector on full data...")
        full_base = clip_predictions(
            np.mean([m.predict(X_full) for m in final], axis=0), INSTALLED_CAPACITY_MW
        )
        corrector.fit(X_full, y_full - full_base)

    MODEL_V9_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "regressors":      final,
        "feature_names":   feats,
        "best_iterations": [max(200, int(round(b * FINAL_BOOST_MULTIPLIER))) for b in all_best_iters],
        "member_labels":   all_labels,
        "ensemble_weights": opt_w.tolist(),
        "pc_map":          pc_map,
        "cond_means":      cond_means,
        "corrector":       corrector if use_corrector else None,
        "use_target_lags": False,
        "val_metrics":     {"equal": m_eq, "optimised": m_opt, "corrected": m_corr},
    }, MODEL_V9_PATH)

    print(f"\nSaved -> {MODEL_V9_PATH}")
    print(f"  {len(final)} members  |  {len(feats)} features")
    print(f"  val MAE: equal={m_eq['MAE']:.4f}  optimised={m_opt['MAE']:.4f}  corrected={m_corr['MAE']:.4f}")
    print(f"  nMAE%:   equal={m_eq['nMAE_%']:.2f}%  optimised={m_opt['nMAE_%']:.2f}%  corrected={m_corr['nMAE_%']:.2f}%")


if __name__ == "__main__":
    main()

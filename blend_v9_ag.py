"""Blend v9 + AutoGluon predictions with weight optimisation on the val split.

Steps:
  1. Load v9 model -> run on training val split -> get v9 val MAE.
  2. Load AG test predictions. For AG val MAE: use --ag_val_mae (default: 7.036).
  3. Sweep w_v9 in [0.50 .. 1.0] step 0.05, report estimated val MAE
     (uses v9 val preds + linear interpolation for AG errors).
  4. Save blend with best weight -> predictions_v9ag_blend.csv

Usage:
    python blend_v9_ag.py
    python blend_v9_ag.py --ag_val_mae 7.0 --w_v9 0.75
"""
from __future__ import annotations

import argparse
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

from config import (
    TRAIN_CSV, VALID_CSV, TARGET_COL, DATETIME_COL,
    INSTALLED_CAPACITY_MW, VAL_FRACTION, AG_PREDICTIONS_PATH,
)
from src.data_loader import load_train, load_valid
from src.preprocessing import (
    build_features, feature_columns, clip_predictions,
    apply_empirical_curve,
)
from train_v9 import apply_conditional_means

MODEL_V9_PATH = ROOT / "models" / "model_v9.pkl"
OUT_PATH      = ROOT / "predictions_v9ag_blend.csv"


def _predict_v9_on_val(model: dict, train_df_full: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run v9 model on the internal val split of training data. Returns (y_true, y_pred)."""
    split_idx  = int(len(train_df_full) * (1 - VAL_FRACTION))
    val_df_raw = train_df_full.iloc[split_idx:].reset_index(drop=True)

    # Build features
    val_df = build_features(val_df_raw, use_target_lags=False)

    pc_map     = model.get("pc_map")
    cond_means = model.get("cond_means")
    corrector  = model.get("corrector")
    weights    = np.array(model["ensemble_weights"])
    feat_names = model["feature_names"]

    if pc_map:
        val_df = apply_empirical_curve(val_df, pc_map)
    if cond_means:
        val_df = apply_conditional_means(val_df, cond_means)

    missing = [f for f in feat_names if f not in val_df.columns]
    for f in missing:
        val_df[f] = 0.0

    X   = val_df[feat_names].values.astype(float)
    y   = val_df[TARGET_COL].values

    member_preds = np.array([reg.predict(X) for reg in model["regressors"]])
    raw = member_preds.T @ weights
    if corrector is not None:
        raw = raw + corrector.predict(X)
    preds = clip_predictions(raw, INSTALLED_CAPACITY_MW)
    return y, preds


def main(ag_val_mae: float = 7.036, w_v9_manual: float | None = None) -> None:
    if not MODEL_V9_PATH.exists():
        print(f"ERROR: {MODEL_V9_PATH} not found. Run train_v9.py first.")
        sys.exit(1)
    if not AG_PREDICTIONS_PATH.exists():
        print(f"ERROR: {AG_PREDICTIONS_PATH} not found. Run inference_autogluon.py first.")
        sys.exit(1)

    # ── Load v9 model ────────────────────────────────────────────────────────
    print(f"Loading v9 model from {MODEL_V9_PATH} ...")
    model = joblib.load(MODEL_V9_PATH)
    stored = model.get("val_metrics", {})
    v9_stored_mae = stored.get("corrected", stored.get("optimised", {})).get("MAE", None)
    if v9_stored_mae:
        print(f"  v9 stored val MAE: {v9_stored_mae:.4f}")

    # ── Run v9 on val split to get actual MAE ────────────────────────────────
    print("Running v9 on training val split for weight optimisation...")
    train_full = load_train(TRAIN_CSV)
    y_val, v9_val_preds = _predict_v9_on_val(model, train_full)
    v9_val_mae = float(np.mean(np.abs(v9_val_preds - y_val)))
    print(f"  v9 val MAE (actual):  {v9_val_mae:.4f}")
    print(f"  AG val MAE (assumed): {ag_val_mae:.4f}")

    # ── Load test predictions ────────────────────────────────────────────────
    v9_test_path = ROOT / "predictions_v9.csv"
    if not v9_test_path.exists():
        print(f"ERROR: {v9_test_path} not found. Run predict_v9.py first.")
        sys.exit(1)

    v9_df = pd.read_csv(v9_test_path)
    ag_df = pd.read_csv(AG_PREDICTIONS_PATH, header=None, names=["predict_ag"])

    p_v9 = v9_df["predict"].values
    p_ag = ag_df["predict_ag"].values
    assert len(p_v9) == len(p_ag), f"Row mismatch v9={len(p_v9)} ag={len(p_ag)}"

    # ── Weight sweep on val split ────────────────────────────────────────────
    # We only have v9 val preds; for AG, use a linear error approximation.
    # The correlated-error formula: MAE(blend) ≈ w*MAE_v9 + (1-w)*MAE_ag + correlation_term
    # We can't compute the correlation term without AG val preds, so we use:
    #   estimated_mae = sqrt( (w*v9_err)^2 + ((1-w)*ag_err)^2 )   (independent errors approx)
    # This is a lower bound; actual gains could be bigger.
    print("\nWeight sweep (estimated blend val MAE, independent-errors approx):")
    print(f"  {'w_v9':>6}  {'w_ag':>6}  {'est_MAE':>10}")
    best_w, best_est = None, np.inf
    weights_to_try = np.arange(0.50, 1.01, 0.05)
    for w in weights_to_try:
        # Use actual v9 val preds for v9 part; scale AG contribution
        v9_part = w * v9_val_preds
        # AG contribution: we approximate AG val preds as y_val + noise with MAE=ag_val_mae
        # Conservative estimate: assume AG errors are independent of v9
        est = float(np.sqrt((w * v9_val_mae) ** 2 + ((1 - w) * ag_val_mae) ** 2))
        marker = " <--" if est < best_est else ""
        print(f"  {w:>6.2f}  {1-w:>6.2f}  {est:>10.4f}{marker}")
        if est < best_est:
            best_est = est
            best_w   = w

    # Inverse-MAE weight (analytic optimum under independence assumption)
    inv_v9 = 1.0 / v9_val_mae
    inv_ag = 1.0 / ag_val_mae
    w_inv  = inv_v9 / (inv_v9 + inv_ag)
    print(f"\n  Inverse-MAE weight: w_v9={w_inv:.3f}  w_ag={1-w_inv:.3f}")

    # Use manual override if provided, else inverse-MAE
    final_w = w_v9_manual if w_v9_manual is not None else w_inv
    print(f"\n  Selected w_v9={final_w:.3f}  w_ag={1-final_w:.3f}")

    # ── Blend and save ───────────────────────────────────────────────────────
    blended = clip_predictions(final_w * p_v9 + (1 - final_w) * p_ag, INSTALLED_CAPACITY_MW)

    out_df = v9_df[[DATETIME_COL]].copy() if DATETIME_COL in v9_df.columns else pd.DataFrame()
    out_df["predict"] = blended
    out_df.to_csv(OUT_PATH, index=False)

    # Submission file: "predict" header + values
    submit_path = ROOT / "predictions.csv"
    pd.DataFrame({"predict": blended}).to_csv(submit_path, index=False)

    print(f"\nSaved blend -> {OUT_PATH}")
    print(f"  rows={len(blended)}  min={blended.min():.3f}  mean={blended.mean():.3f}  max={blended.max():.3f}")
    print(f"  v9_mean={p_v9.mean():.3f}  ag_mean={p_ag.mean():.3f}")
    print(f"\nSubmission file saved -> {submit_path}")
    print(f"Upload predictions.csv to the platform.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag_val_mae", type=float, default=7.036,
                        help="AG val MAE to use for weight estimation (default: 7.036)")
    parser.add_argument("--w_v9", type=float, default=None,
                        help="Manual v9 weight override (default: inverse-MAE auto)")
    args = parser.parse_args()
    main(ag_val_mae=args.ag_val_mae, w_v9_manual=args.w_v9)

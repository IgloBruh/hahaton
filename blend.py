"""Blend predictions_v8.csv and predictions_autogluon.csv.

Weights are optimised on validation MAE (v8 optimised=6.9935, AG estimated).
Simple weighted average: w_v8 * v8 + w_ag * ag.

Usage:
    python blend.py                        # equal weights 50/50
    python blend.py --w_v8 0.6 --w_ag 0.4 # manual weights
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DATETIME_COL, INSTALLED_CAPACITY_MW
from src.preprocessing import clip_predictions


def main(w_v8: float = 0.5, w_ag: float = 0.5) -> None:
    v8_path = ROOT / "predictions_v8.csv"
    ag_path  = ROOT / "predictions_autogluon.csv"

    if not v8_path.exists():
        raise FileNotFoundError(f"predictions_v8.csv not found — run predict.py first")
    if not ag_path.exists():
        raise FileNotFoundError(f"predictions_autogluon.csv not found — run inference_autogluon.py first")

    v8 = pd.read_csv(v8_path)
    ag = pd.read_csv(ag_path, header=None, names=["predict_ag"])

    if DATETIME_COL in v8.columns:
        out = pd.DataFrame({DATETIME_COL: v8[DATETIME_COL]})
        p_v8 = v8["predict"].values
    else:
        out = pd.DataFrame()
        p_v8 = v8.iloc[:, 0].values

    p_ag = ag["predict_ag"].values

    assert len(p_v8) == len(p_ag), f"Row count mismatch: v8={len(p_v8)} ag={len(p_ag)}"

    # Normalise weights
    total = w_v8 + w_ag
    w_v8 /= total
    w_ag  /= total

    blended = clip_predictions(w_v8 * p_v8 + w_ag * p_ag, INSTALLED_CAPACITY_MW)

    out["predict"] = blended
    out_path = ROOT / f"predictions_blend_v8{w_v8:.2f}_ag{w_ag:.2f}.csv"
    out.to_csv(out_path, index=False)

    print(f"Blended {len(blended)} predictions -> {out_path}")
    print(f"  weights: v8={w_v8:.2f}  ag={w_ag:.2f}")
    print(f"  stats: min={blended.min():.3f}  mean={blended.mean():.3f}  max={blended.max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_v8", type=float, default=0.5)
    parser.add_argument("--w_ag",  type=float, default=0.5)
    args = parser.parse_args()
    main(args.w_v8, args.w_ag)

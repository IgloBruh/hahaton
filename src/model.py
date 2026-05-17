"""Ensemble of LightGBM gradient-boosted regressors.

Each member is a :class:`lightgbm.LGBMRegressor`.
Diversity is generated through different random seeds, loss objectives
(``regression`` = RMSE and ``regression_l1`` = MAE), tree shapes,
learning rates, and column/row subsampling.
Final prediction is the simple mean of member predictions.

Why LightGBM?
- Faster training and often better accuracy than HistGradientBoosting on
  tabular data due to leaf-wise tree growth and histogram binning.
- Native categorical support (unused here) and missing-value handling.
- Built-in early stopping via callbacks avoids manual staged_predict loops.
- Feature importance (gain / split) available for post-training analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


# --------------------------- single-member helpers --------------------------
def _make_regressor(cfg: dict, n_estimators: int) -> lgb.LGBMRegressor:
    """Build one LGBMRegressor from a config dict."""
    return lgb.LGBMRegressor(
        objective=cfg["objective"],
        learning_rate=cfg["learning_rate"],
        n_estimators=n_estimators,
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        reg_lambda=cfg.get("reg_lambda", 1.0),
        reg_alpha=cfg.get("reg_alpha", 0.0),
        colsample_bytree=cfg["colsample_bytree"],
        subsample=cfg.get("subsample", 1.0),
        subsample_freq=cfg.get("subsample_freq", 1),
        random_state=cfg["seed"],
        verbose=-1,
    )


# --------------------------- ensemble class ---------------------------------
@dataclass
class WindPowerModel:
    """Ensemble of LightGBM regressors with a shared feature list."""
    regressors: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    best_iterations: list = field(default_factory=list)
    member_configs: list = field(default_factory=list)

    # ----- fit on train, pick best_iter via early stopping on validation -----
    @classmethod
    def fit_with_validation(
        cls,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        feature_names: Sequence[str],
        ensemble_configs: list[dict],
        *,
        max_iter: int = 2000,
        patience: int = 120,
        verbose: bool = True,
    ) -> "WindPowerModel":
        feature_names = list(feature_names)
        Xt = X_train[feature_names].values
        Xv = X_val[feature_names].values

        members, best_iters = [], []
        for k, cfg in enumerate(ensemble_configs):
            reg = _make_regressor(cfg, n_estimators=max_iter)
            reg.fit(
                Xt, y_train,
                eval_set=[(Xv, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=patience, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iter = int(reg.best_iteration_)
            members.append(reg)
            best_iters.append(best_iter)
            if verbose:
                p = reg.predict(Xv)
                rmse = float(np.sqrt(np.mean((p - y_val) ** 2)))
                mae = float(np.mean(np.abs(p - y_val)))
                print(f"   member {k+1}/{len(ensemble_configs)}  "
                      f"obj={cfg['objective']:<14s}  seed={cfg['seed']:<5d}  "
                      f"best_iter={best_iter:<4d}  RMSE_val={rmse:.4f}  MAE_val={mae:.4f}")

        return cls(
            regressors=members,
            feature_names=feature_names,
            best_iterations=best_iters,
            member_configs=list(ensemble_configs),
        )

    # ----- refit on the full dataset using each member's known best_iter ----
    @classmethod
    def fit_full(
        cls,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: Sequence[str],
        ensemble_configs: list[dict],
        best_iterations: list[int],
        boost_multiplier: float = 1.10,
        *,
        verbose: bool = True,
    ) -> "WindPowerModel":
        """Retrain every member on the full dataset without validation.

        Each member uses ``best_iter * boost_multiplier`` rounds (floored at 200).
        """
        feature_names = list(feature_names)
        X_full = X[feature_names].values

        members, used_iters = [], []
        for k, (cfg, best_iter) in enumerate(zip(ensemble_configs, best_iterations)):
            rounds = max(200, int(round(best_iter * boost_multiplier)))
            reg = _make_regressor(cfg, n_estimators=rounds)
            reg.fit(X_full, y)
            members.append(reg)
            used_iters.append(rounds)
            if verbose:
                print(f"   member {k+1}/{len(ensemble_configs)}  "
                      f"obj={cfg['objective']:<14s}  seed={cfg['seed']:<5d}  "
                      f"rounds={rounds}")
        return cls(
            regressors=members,
            feature_names=feature_names,
            best_iterations=used_iters,
            member_configs=list(ensemble_configs),
        )

    # ----- inference --------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average over ensemble members (deterministic)."""
        Xn = X[self.feature_names].values
        preds = np.stack([reg.predict(Xn) for reg in self.regressors], axis=0)
        return preds.mean(axis=0)

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-member predictions (shape = [n_members, n_rows])."""
        Xn = X[self.feature_names].values
        return np.stack([reg.predict(Xn) for reg in self.regressors], axis=0)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Average LightGBM feature importance across ensemble members.

        Parameters
        ----------
        importance_type : "gain" (default) or "split"
        """
        scores = np.zeros(len(self.feature_names), dtype=float)
        for reg in self.regressors:
            scores += reg.booster_.feature_importance(importance_type=importance_type)
        scores /= len(self.regressors)
        return pd.Series(scores, index=self.feature_names).sort_values(ascending=False)

    # ----- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"regressors": self.regressors,
             "feature_names": self.feature_names,
             "best_iterations": self.best_iterations,
             "member_configs": self.member_configs},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "WindPowerModel":
        blob = joblib.load(path)
        return cls(
            regressors=list(blob["regressors"]),
            feature_names=list(blob["feature_names"]),
            best_iterations=list(blob.get("best_iterations", [])),
            member_configs=list(blob.get("member_configs", [])),
        )

    # ----- introspection ----------------------------------------------------
    @property
    def n_members(self) -> int:
        return len(self.regressors)
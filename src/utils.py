"""Helper utilities: seeding and metrics."""
import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Seed every source of randomness we may touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Optional: torch/tf are not used here, but stub for future-proofing.
    try:
        import torch  # noqa: F401
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R², and capacity-normalised nMAE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    # Normalised MAE — MAE as % of installed capacity (industry-standard for wind)
    from config import INSTALLED_CAPACITY_MW
    nmae_pct = mae / INSTALLED_CAPACITY_MW * 100.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "nMAE_%": nmae_pct}


def pretty_print_metrics(name: str, metrics: dict) -> None:
    """Print a metrics dict in a readable line."""
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    print(f"[{name}] " + "  ".join(parts))

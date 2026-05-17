"""LSTM inference: load trained model, predict for valid_features.csv,
write predictions_lstm.csv (one column, no header, same row order as input).

Run with .venv_ag (Python 3.11) — same environment as train_lstm.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    DATETIME_COL, INSTALLED_CAPACITY_MW, LSTM_MODEL_PATH,
    LSTM_PREDICTIONS_PATH, LSTM_SCALER_PATH, SEED, TARGET_COL, TRAIN_CSV, VALID_CSV,
)
from src.data_loader import load_train, load_valid
from src.preprocessing import build_features, clip_predictions
from src.utils import set_global_seed

SEQ_LEN = 48
HISTORY_HOURS = 48


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def main() -> None:
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"-> Loading LSTM from {LSTM_MODEL_PATH} ...")
    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)
    model = LSTMRegressor(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    scaler = joblib.load(LSTM_SCALER_PATH)
    feats: list[str] = checkpoint["feature_names"]

    print(f"-> Loading test features from {VALID_CSV} ...")
    raw_valid = load_valid(VALID_CSV)
    n_rows = len(raw_valid)
    print(f"   rows = {n_rows}")

    print("-> Loading training tail ...")
    train = load_train(TRAIN_CSV)
    train_tail = train.tail(HISTORY_HOURS).copy()
    train_tail = train_tail.drop(columns=[TARGET_COL], errors="ignore")

    combined = pd.concat([train_tail, raw_valid], ignore_index=True)
    combined = combined.sort_values(DATETIME_COL).reset_index(drop=True)

    print("-> Building features ...")
    feat_df = build_features(combined)

    # Scale all features at once
    all_scaled = scaler.transform(feat_df[feats].values).astype(np.float32)

    valid_mask = feat_df[DATETIME_COL].isin(raw_valid[DATETIME_COL]).values
    valid_indices = np.where(valid_mask)[0]
    assert len(valid_indices) == n_rows

    print("-> Predicting ...")
    all_preds: list[float] = []
    with torch.no_grad():
        for vi in valid_indices:
            start = vi - SEQ_LEN
            if start < 0:
                pad = np.zeros((SEQ_LEN - vi, len(feats)), dtype=np.float32)
                seq = np.concatenate([pad, all_scaled[:vi]], axis=0)
            else:
                seq = all_scaled[start:vi]
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            pred_scaled = model(x).item()
            pred_mw = float(clip_predictions(
                np.array([pred_scaled * INSTALLED_CAPACITY_MW]), INSTALLED_CAPACITY_MW
            )[0])
            all_preds.append(pred_mw)

    feat_valid_dt = feat_df.loc[valid_mask, DATETIME_COL].reset_index(drop=True)
    feat_valid_sorted = feat_valid_dt.sort_values().reset_index(drop=True)
    preds_arr = np.array(all_preds)
    pred_by_dt = dict(zip(feat_valid_sorted.values, preds_arr))

    original = pd.read_csv(VALID_CSV)
    original[DATETIME_COL] = pd.to_datetime(original[DATETIME_COL])
    out = original[DATETIME_COL].map(pred_by_dt).values

    assert len(out) == n_rows, "Row-count mismatch."
    assert not pd.isna(out).any(), "NaN in predictions."

    pd.Series(out).to_csv(LSTM_PREDICTIONS_PATH, index=False, header=False)
    print(f"OK Wrote {len(out)} predictions -> {LSTM_PREDICTIONS_PATH}")
    print(f"   stats: min={out.min():.3f}  mean={out.mean():.3f}  max={out.max():.3f}")


if __name__ == "__main__":
    main()

"""Train a 2-layer LSTM for wind-farm power prediction.

Uses the same feature pipeline as the other models but adds temporal context
via a sliding-window approach (seq_len=48 hours).

Features are z-score normalised (StandardScaler). Target is scaled to [0, 1]
by dividing by installed capacity (90.09 MW).

Run with .venv_ag (Python 3.11) — PyTorch is installed there via AutoGluon.

Usage
-----
    .venv_ag\\Scripts\\python train_lstm.py
    .venv_ag\\Scripts\\python train_lstm.py --epochs 200 --hidden_size 512
"""
from __future__ import annotations

import argparse
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import (
    DATETIME_COL, INSTALLED_CAPACITY_MW, LSTM_MODEL_PATH,
    LSTM_SCALER_PATH, MODELS_DIR, SEED, TARGET_COL, TRAIN_CSV, VAL_FRACTION,
)
from src.data_loader import load_train
from src.preprocessing import build_features, clip_predictions, feature_columns
from src.utils import pretty_print_metrics, regression_metrics, set_global_seed

SEQ_LEN = 48


class WindDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx: idx + self.seq_len])
        y = torch.tensor(self.y[idx + self.seq_len])
        return x, y


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


def _train_epoch(model: nn.Module, loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y_b)
    return total / len(loader.dataset)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts = []
    for X_b, _ in loader:
        parts.append(model(X_b.to(device)).cpu().numpy())
    return np.concatenate(parts)


def main(epochs: int = 100, lr: float = 1e-3, batch_size: int = 64,
         hidden_size: int = 256, patience: int = 15) -> None:
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading and building features...")
    df = load_train(TRAIN_CSV)
    df = build_features(df)
    feats = feature_columns(df)
    print(f"  {len(feats)} features, {len(df)} rows")

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    y_val_mw = val_df[TARGET_COL].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feats].values)
    X_val = scaler.transform(val_df[feats].values)

    y_train_scaled = train_df[TARGET_COL].values / INSTALLED_CAPACITY_MW
    y_val_scaled = y_val_mw / INSTALLED_CAPACITY_MW

    train_ds = WindDataset(X_train, y_train_scaled)
    val_ds = WindDataset(X_val, y_val_scaled)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = LSTMRegressor(input_size=len(feats), hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5, verbose=False,
    )
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_state: dict | None = None
    no_improve = 0

    print(f"Training ({epochs} epochs, patience={patience}, hidden={hidden_size}) ...")
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        preds_scaled = _predict(model, val_loader, device)
        preds_mw = clip_predictions(
            preds_scaled * INSTALLED_CAPACITY_MW, INSTALLED_CAPACITY_MW
        )
        # val_ds skips first SEQ_LEN rows
        y_val_aligned = y_val_mw[SEQ_LEN:]
        mae = float(np.mean(np.abs(preds_mw - y_val_aligned)))
        scheduler.step(mae)

        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}"
                  f"  val_MAE={mae:.4f}  best={best_mae:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "input_size": len(feats),
         "hidden_size": hidden_size, "feature_names": feats},
        LSTM_MODEL_PATH,
    )
    joblib.dump(scaler, LSTM_SCALER_PATH)
    print(f"Saved -> {LSTM_MODEL_PATH}")

    preds_scaled = _predict(model, val_loader, device)
    preds_mw = clip_predictions(preds_scaled * INSTALLED_CAPACITY_MW, INSTALLED_CAPACITY_MW)
    m = regression_metrics(y_val_mw[SEQ_LEN:], preds_mw)
    pretty_print_metrics("LSTM val (excl. first 48h)", m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    main(args.epochs, args.lr, args.batch_size, args.hidden_size, args.patience)

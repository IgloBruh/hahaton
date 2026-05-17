"""Read raw CSV files into well-typed DataFrames."""
import pandas as pd
from pathlib import Path

from config import DATETIME_COL


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    return df


def load_train(path: Path) -> pd.DataFrame:
    """Load training CSV (features + target), sorted by datetime ascending."""
    return _read(path)


def load_valid(path: Path) -> pd.DataFrame:
    """Load test features CSV, sorted by datetime ascending."""
    return _read(path)

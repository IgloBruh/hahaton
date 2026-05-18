"""Cleaning + feature engineering for the wind-farm dataset.

Design notes
------------
* The target ``Выработка. Результирующий расчет`` is hourly active power (MW).
  We never use it as an input feature.
* Wind direction columns in the raw data are in the range ~[0, 0.36]; multiplied
  by 1000 they correspond to degrees [0, 360]. We decode them to sin/cos
  pairs and to (u, v) wind vector components.
* Aerodynamic power for a wind turbine is proportional to wind speed cubed;
  ``v**3`` and the physical ``ρ·v³`` predictor are exposed directly.
* ``Кол-во_ВЭУ_в_ремонте`` (turbines in repair) scales available capacity
  linearly — we expose ``available_fraction`` = (26 - in_repair) / 26.
* Lag, lead, and rolling-window aggregates of weather variables are added.
  This is legal because all weather variables are FORECASTS known in advance
  for the whole prediction period — no leakage.
* Parametric Siemens-Gamesa-style power curve is applied to wind speeds at
  hub heights — gives the model a strong physical prior for the v→P mapping.
* Missing ``wind_speed_180m`` / ``wind_direction_180m`` (early 2022) are filled
  from the 120 m level — they are vertically closest and tightly correlated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATETIME_COL, TARGET_COL, NUM_TURBINES_TOTAL, P_RATED_PER_TURBINE


# ----- Raw-column groups -------------------------------------------------
SPEED_COLS = ["wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"]
DIR_COLS = ["wind_direction_10m", "wind_direction_80m",
            "wind_direction_120m", "wind_direction_180m"]
EXTRA_NUMERIC = [
    "wind_gusts_10m", "temperature_80m", "temperature_120m", "pressure_msl",
    "rain", "showers", "snowfall", "cloud_cover_low", "Кол-во_ВЭУ_в_ремонте",
]

# Columns to compute lag/lead/rolling features over.
# wind_speed_hub and Кол-во_ВЭУ_в_ремонте are derived columns added later;
# _add_temporal_features skips missing columns so order is safe.
KEY_LAG_COLS = SPEED_COLS + [
    "wind_gusts_10m", "temperature_80m", "pressure_msl",
    "wind_speed_hub",   # hub-height proxy (added by _add_wind_speed_physics)
    "cloud_cover_low",  # cloud cover patterns
    # NOTE: Кол-во_ВЭУ_в_ремонте excluded — repair count changes slowly so its
    # lags/leads are near-duplicates of available_fraction and add noise.
]
LAG_OFFSETS = [1, 2, 3, 6, 12, 24, 48]   # 48-h lag for diurnal-cycle context
LEAD_OFFSETS = [1, 2, 3, 4, 5, 6]         # full 6-step forecast horizon
ROLLING_WINDOWS = [3, 6, 12, 24]


# ----- Cleaning ----------------------------------------------------------
def fill_missing_high_altitude(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing 180 m level columns from the 120 m level (closest height)."""
    df = df.copy()
    mask = df["wind_speed_180m"].isna()
    df.loc[mask, "wind_speed_180m"] = df.loc[mask, "wind_speed_120m"]
    mask = df["wind_direction_180m"].isna()
    df.loc[mask, "wind_direction_180m"] = df.loc[mask, "wind_direction_120m"]
    # Final safety net for any other numeric NaNs
    for c in SPEED_COLS + DIR_COLS + EXTRA_NUMERIC:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


# ----- Static feature engineering (per-row, no temporal context) --------
def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df[DATETIME_COL].dt.hour
    df["month_num"] = df[DATETIME_COL].dt.month
    df["day_of_year"] = df[DATETIME_COL].dt.dayofyear
    df["day_of_week"] = df[DATETIME_COL].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def _add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decode wind direction (deg/1000) to sin/cos + (u, v) vector components."""
    for c in DIR_COLS:
        deg = df[c] * 1000.0
        rad = np.deg2rad(deg)
        df[f"{c}_sin"] = np.sin(rad)
        df[f"{c}_cos"] = np.cos(rad)

    # (u, v) wind components — meteorological convention
    for h in ["10m", "80m", "120m", "180m"]:
        deg = df[f"wind_direction_{h}"] * 1000.0
        rad = np.deg2rad(deg)
        df[f"u_{h}"] = -df[f"wind_speed_{h}"] * np.sin(rad)
        df[f"v_{h}"] = -df[f"wind_speed_{h}"] * np.cos(rad)

    # Wind veer (circular direction differences between heights) — stability proxy
    for c1, c2 in [("80m", "10m"), ("120m", "80m"), ("180m", "120m")]:
        deg1 = df[f"wind_direction_{c1}"] * 1000.0
        deg2 = df[f"wind_direction_{c2}"] * 1000.0
        diff = (deg1 - deg2 + 180.0) % 360.0 - 180.0
        df[f"veer_{c1}_{c2}"] = diff
    return df


def _add_wind_speed_physics(df: pd.DataFrame) -> pd.DataFrame:
    """v², v³, vertical shears, multi-height aggregates."""
    for c in SPEED_COLS:
        df[f"{c}_cubed"] = df[c] ** 3
        df[f"{c}_squared"] = df[c] ** 2

    # Wind-shear features (difference between heights) — capture stability
    df["shear_80_10"] = df["wind_speed_80m"] - df["wind_speed_10m"]
    df["shear_120_80"] = df["wind_speed_120m"] - df["wind_speed_80m"]
    df["shear_180_120"] = df["wind_speed_180m"] - df["wind_speed_120m"]
    df["shear_180_10"] = df["wind_speed_180m"] - df["wind_speed_10m"]

    # Mean speed across heights — denoised wind speed
    df["wind_speed_mean"] = df[SPEED_COLS].mean(axis=1)
    df["wind_speed_mean_cubed"] = df["wind_speed_mean"] ** 3

    # Hub-height proxy: weighted avg of 80m and 120m (rotor swept area ≈ 50-130m)
    df["wind_speed_hub"] = 0.5 * (df["wind_speed_80m"] + df["wind_speed_120m"])
    df["wind_speed_hub_cubed"] = df["wind_speed_hub"] ** 3

    # Gust-to-mean ratio – turbulence proxy
    df["gust_ratio"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + 1e-3)

    # Atmospheric thermal gradient (stability indicator)
    df["temp_gradient_120_80"] = df["temperature_120m"] - df["temperature_80m"]
    return df


def _add_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    df["available_turbines"] = NUM_TURBINES_TOTAL - df["Кол-во_ВЭУ_в_ремонте"]
    df["available_fraction"] = df["available_turbines"] / NUM_TURBINES_TOTAL
    df["v3_x_available"] = df["wind_speed_mean_cubed"] * df["available_fraction"]
    df["v3_hub_x_available"] = df["wind_speed_hub_cubed"] * df["available_fraction"]
    return df


def _add_thermo(df: pd.DataFrame) -> pd.DataFrame:
    """Air density ρ = P/(R·T) and the canonical predictor ρ·v³."""
    T_K = df["temperature_80m"] + 273.15
    P_pa = df["pressure_msl"] * 100.0           # hPa → Pa
    df["air_density"] = P_pa / (287.058 * T_K)
    df["rho_v3"] = df["air_density"] * df["wind_speed_mean_cubed"]
    df["rho_v3_hub"] = df["air_density"] * df["wind_speed_hub_cubed"]
    # Most physically direct predictor: aerodynamic power at hub × availability
    df["rho_v3_hub_available"] = df["rho_v3_hub"] * df["available_fraction"]
    return df


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional derived features beyond the core physics."""
    # Vertical wind speed variability — turbulence intensity proxy
    df["wind_speed_std"] = df[SPEED_COLS].std(axis=1)

    # Total precipitation — icing / blade-contamination risk
    df["precip_total"] = (
        df["rain"].clip(lower=0)
        + df["showers"].clip(lower=0)
        + df["snowfall"].clip(lower=0)
    )

    # Wind speed range across heights — shear proxy
    df["wind_speed_range"] = df[SPEED_COLS].max(axis=1) - df[SPEED_COLS].min(axis=1)

    # Pressure tendency (weather-front / ramp-event indicator)
    p = df["pressure_msl"]
    df["pressure_change_1h"] = p - p.shift(1)
    df["pressure_change_3h"] = p - p.shift(3)

    # Icing-risk indicator: sub-zero temperature + active precipitation
    df["icing_risk"] = (
        (df["temperature_80m"] < 2.0).astype(float)
        * (df["precip_total"] > 0).astype(float)
    )

    # Effective turbulence: gust excess × vertical variability
    df["turbulence_index"] = df["gust_ratio"] * df["wind_speed_std"]

    return df


def _siemens_gamesa_power_curve(v: np.ndarray) -> np.ndarray:
    """Parametric power curve for Siemens Gamesa SG 3.4-132 (3.465 MW rated).

    Cubic ramp between cut-in (3 m/s) and rated speed (13 m/s), constant
    rated power 13–25 m/s, zero above cut-out.
    """
    cut_in, rated_v, cut_out = 3.0, 13.0, 25.0
    v = np.asarray(v, dtype=float)
    p = np.where(v < cut_in, 0.0,
        np.where(v < rated_v,
                 P_RATED_PER_TURBINE * ((v - cut_in) / (rated_v - cut_in)) ** 3,
        np.where(v < cut_out, P_RATED_PER_TURBINE, 0.0)))
    return p


def _add_power_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply manufacturer power curve at hub heights (× available turbines).

    Yields the theoretical farm output assuming every working turbine
    operated exactly on its nameplate curve. Strong physical baseline.
    """
    avail = df["available_turbines"]
    df["pc_80m"] = _siemens_gamesa_power_curve(df["wind_speed_80m"]) * avail
    df["pc_120m"] = _siemens_gamesa_power_curve(df["wind_speed_120m"]) * avail
    df["pc_hub"] = _siemens_gamesa_power_curve(df["wind_speed_hub"]) * avail
    df["pc_mean"] = _siemens_gamesa_power_curve(df["wind_speed_mean"]) * avail
    return df


def _add_post_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that require both temporal lead columns and capacity info.

    Called after _add_temporal_features so lead columns already exist.
    """
    avail = df["available_fraction"]
    avail_t = df["available_turbines"]

    for n in [1, 2, 3]:
        hub_col = f"wind_speed_hub_lead{n}"
        if hub_col not in df.columns:
            continue
        v = df[hub_col]
        # Physics-based power proxy at future step n (most direct signal)
        df[f"v3_hub_lead{n}_x_avail"] = v ** 3 * avail
        # Manufacturer power curve applied to forecasted hub wind speed
        df[f"pc_hub_lead{n}"] = _siemens_gamesa_power_curve(v) * avail_t

    return df


# ----- Temporal features (lags / leads / rolling windows) ---------------
def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shift/rolling features over key weather columns.

    Assumes df is chronologically sorted. Operates by row-position; gaps in
    the hourly index are short (median 1h, max a few hours) so this is fine.

    Implementation note: we collect every new column into a dict and concat
    once at the end, instead of assigning ~140 columns one-by-one. This is
    ~5–10× faster on a wide DataFrame and silences pandas' fragmentation
    PerformanceWarning.
    """
    new_cols: dict[str, pd.Series] = {}

    # Lags / leads
    for col in KEY_LAG_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        for lag in LAG_OFFSETS:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag)
        for lead in LEAD_OFFSETS:
            new_cols[f"{col}_lead{lead}"] = s.shift(-lead)

    # Rolling stats — past-only (no centering) to avoid look-ahead inside training
    for col in SPEED_COLS + ["wind_gusts_10m"]:
        s = df[col]
        for w in ROLLING_WINDOWS:
            roll = s.rolling(w, min_periods=1)
            new_cols[f"{col}_roll{w}_mean"] = roll.mean()
            new_cols[f"{col}_roll{w}_std"] = roll.std()

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


TARGET_LAG_OFFSETS = [1, 2, 3, 6, 12, 24]


# ----- Ramp features (wind acceleration / deceleration) ------------------
def _add_ramp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ramp = future wind speed minus past wind speed — captures ramping events.

    Called after _add_temporal_features so lag/lead columns already exist.
    Skips columns that are absent (e.g., on very short DataFrames).
    """
    new_cols: dict[str, pd.Series] = {}
    for n_past, n_future in [(1, 1), (1, 3), (3, 3)]:
        lag_col  = f"wind_speed_hub_lag{n_past}"
        lead_col = f"wind_speed_hub_lead{n_future}"
        if lag_col in df.columns and lead_col in df.columns:
            key = f"wind_ramp_{n_past}h_to_{n_future}h"
            new_cols[key] = df[lead_col] - df[lag_col]
            new_cols[f"{key}_cubed"] = new_cols[key] ** 3

    # Speed-normalised ramp rate — tells the *relative* acceleration
    if "wind_speed_hub" in df.columns and "wind_ramp_1h_to_1h" in new_cols:
        denom = df["wind_speed_hub"].clip(lower=1.0)
        new_cols["wind_ramp_rel"] = new_cols["wind_ramp_1h_to_1h"] / denom

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ----- Empirical power curve (fitted from training data) -----------------
def fit_empirical_curve(
    df: pd.DataFrame,
    target_col: str,
    bin_width: float = 0.5,
    min_samples: int = 8,
) -> dict:
    """Fit a smooth empirical power curve from training data.

    Maps hub-height wind speed → median production per turbine (MW).
    Returns a serialisable dict so it can be stored in model artefacts.
    """
    available = df["available_turbines"].clip(lower=1)
    ppt = df[target_col] / available          # production per turbine
    v = df["wind_speed_hub"].values

    bins = np.arange(0.0, 31.0, bin_width)
    centers = bins[:-1] + bin_width / 2.0
    medians = np.full(len(centers), np.nan)

    for i in range(len(centers)):
        mask = (v >= bins[i]) & (v < bins[i + 1])
        if mask.sum() >= min_samples:
            medians[i] = float(np.median(ppt.values[mask]))

    # Interpolate NaN bins linearly
    valid = ~np.isnan(medians)
    if valid.sum() >= 2:
        medians = np.interp(centers, centers[valid], medians[valid])
    else:
        medians = np.zeros(len(centers))

    # Enforce physical constraints: 0 below cut-in (3 m/s), 0 above cut-out (25 m/s)
    medians[centers < 3.0]  = np.minimum(medians[centers < 3.0], 0.0)
    medians[centers > 25.0] = 0.0

    return {"bin_centers": centers.tolist(), "medians": medians.tolist(), "bin_width": bin_width}


def apply_empirical_curve(df: pd.DataFrame, pc_map: dict) -> pd.DataFrame:
    """Add empirical power curve features to a DataFrame.

    pc_map must have been returned by fit_empirical_curve.
    """
    centers = np.array(pc_map["bin_centers"])
    medians = np.array(pc_map["medians"])
    v = df["wind_speed_hub"].values

    ppt = np.interp(v, centers, medians, left=0.0, right=0.0)
    df = df.copy()
    df["empirical_pc_per_turbine"] = ppt
    df["empirical_pc"] = ppt * df["available_turbines"]
    # Residual vs. manufacturer SG curve — captures calibration offset
    if "pc_hub" in df.columns:
        df["empirical_vs_sg"] = df["empirical_pc"] - df["pc_hub"]
    return df


def _add_target_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features of the target variable (past production).

    Only added when TARGET_COL is present in df (training mode).
    During inference these columns are filled externally by the autoregressive
    prediction loop using previously predicted values.
    """
    if TARGET_COL not in df.columns:
        return df
    new_cols = {}
    s = df[TARGET_COL]
    for lag in TARGET_LAG_OFFSETS:
        new_cols[f"target_lag{lag}"] = s.shift(lag)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ----- Public API -------------------------------------------------------
def build_features(df: pd.DataFrame, use_target_lags: bool = True) -> pd.DataFrame:
    """Build the full feature set on a chronologically-sorted DataFrame."""
    df = fill_missing_high_altitude(df)
    df = _add_calendar(df)
    df = _add_wind_direction_features(df)
    df = _add_wind_speed_physics(df)
    df = _add_capacity_features(df)
    df = _add_thermo(df)
    df = _add_extra_features(df)
    df = _add_power_curve_features(df)
    df = _add_temporal_features(df)
    df = _add_ramp_features(df)
    if use_target_lags:
        df = _add_target_lags(df)

    # Fill NaNs created by shift/rolling at the boundaries
    df = df.ffill().bfill()
    df = df.fillna(0.0)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns to feed the model: everything numeric except target/datetime/raw bookkeeping."""
    drop = {DATETIME_COL, TARGET_COL, "hour_of_day"}
    feats = [c for c in df.columns
             if c not in drop
             and pd.api.types.is_numeric_dtype(df[c])]
    return feats


def clip_predictions(pred: np.ndarray, capacity: float = 90.09) -> np.ndarray:
    """Predictions must be in [0, installed_capacity]."""
    return np.clip(pred, 0.0, capacity)

"""Project-wide configuration."""
from pathlib import Path

# --------- Reproducibility ---------
SEED = 42

# --------- Paths ---------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_CSV = DATA_DIR / "train_dataset.csv"
VALID_CSV = DATA_DIR / "valid_features.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

PREDICTIONS_PATH = ROOT_DIR / "predictions.csv"
AG_MODEL_PATH = MODELS_DIR / "autogluon_predictor"
AG_PREDICTIONS_PATH = ROOT_DIR / "predictions_autogluon.csv"

# --------- Target & schema ---------
TARGET_COL = "Выработка. Результирующий расчет"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# --------- Domain constants (from dataset README) ---------
INSTALLED_CAPACITY_MW = 90.09          # 26 turbines × 3.465 MW (Siemens Gamesa)
NUM_TURBINES_TOTAL = 26
P_RATED_PER_TURBINE = 3.465            # MW
ROTOR_HEIGHT_M = 80

# --------- Training ---------
# Reserve the last ~10% of training rows as an internal time-based hold-out
VAL_FRACTION = 0.10

MAX_ITER = 3000
PATIENCE = 150

# After validation, retrain on the full dataset using best_iteration scaled up slightly
FINAL_BOOST_MULTIPLIER = 1.15

# Feature pruning: remove features with permutation-importance below this threshold
# (negative = shuffling the feature actually improves MAE → it's pure noise).
PI_PRUNE_THRESHOLD = -0.005

# --------- Ensemble ---------
# Strategy: build ensemble around the Optuna-found optimum (MAE=7.050, regression_l1).
# Members share the best base config but vary in seed, slight param perturbations,
# and one MSE member for variance-reduction diversity.
# Optuna best: regression_l1, lr=0.01427, num_leaves=114, min_child_samples=43,
#              reg_lambda=0.00161, colsample_bytree=0.511, subsample=0.927
_LR  = 0.01427
_NL  = 114
_MCS = 43
_RL  = 0.00161
_CBT = 0.511
_SS  = 0.927

ENSEMBLE_CONFIGS = [
    # Core: 3× best Optuna config with different random seeds (subsample diversity)
    dict(objective="regression_l1", learning_rate=_LR,  num_leaves=_NL,  min_child_samples=_MCS, reg_lambda=_RL,   reg_alpha=0.0, colsample_bytree=_CBT, subsample=_SS,  subsample_freq=1, seed=42),
    dict(objective="regression_l1", learning_rate=_LR,  num_leaves=_NL,  min_child_samples=_MCS, reg_lambda=_RL,   reg_alpha=0.0, colsample_bytree=_CBT, subsample=_SS,  subsample_freq=1, seed=7),
    dict(objective="regression_l1", learning_rate=_LR,  num_leaves=_NL,  min_child_samples=_MCS, reg_lambda=_RL,   reg_alpha=0.0, colsample_bytree=_CBT, subsample=_SS,  subsample_freq=1, seed=2024),
    # Slight variations: fewer/more leaves, tighter regularisation
    dict(objective="regression_l1", learning_rate=_LR,  num_leaves=100,  min_child_samples=40,   reg_lambda=0.005, reg_alpha=0.0, colsample_bytree=0.55, subsample=0.90, subsample_freq=1, seed=13),
    dict(objective="regression_l1", learning_rate=_LR,  num_leaves=127,  min_child_samples=50,   reg_lambda=0.001, reg_alpha=0.0, colsample_bytree=0.50, subsample=0.95, subsample_freq=1, seed=99),
    # Additional L1 member with different seed (replaces MSE member that was weaker)
    dict(objective="regression_l1", learning_rate=_LR,  num_leaves=_NL,  min_child_samples=_MCS, reg_lambda=_RL,   reg_alpha=0.0, colsample_bytree=_CBT, subsample=_SS,  subsample_freq=1, seed=1337),
    # Slightly lower LR for even finer convergence
    dict(objective="regression_l1", learning_rate=0.010, num_leaves=150, min_child_samples=35,   reg_lambda=0.001, reg_alpha=0.0, colsample_bytree=0.50, subsample=0.90, subsample_freq=1, seed=777),
]

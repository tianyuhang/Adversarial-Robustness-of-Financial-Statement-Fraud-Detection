#!/usr/bin/env python3
"""Evaluate leverage-aware ensemble defenses for FSFD models.

The experiment compares clean training, unconstrained and accounting-consistent
adversarial training, structured bagging, and leverage-aware augmentation.
"""

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

# PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Intel.*")
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"


# =============================================================================
# 1. Experiment configuration
# User-adjustable settings are centralized here to avoid scattered constants.
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 File paths and output
# -----------------------------------------------------------------------------
DATA_PATH = "finance_data_smoothed.xlsx"
STOCK_NAME_COLUMN = "\u80a1\u7968\u7b80\u79f0"
OUTPUT_DIR = "results"
OUTPUT_FILE = "ensemble_defense_results.xlsx"

# -----------------------------------------------------------------------------
# 1.2 Runtime mode and model set
# -----------------------------------------------------------------------------
FAST_SCREEN_SNN_ONLY = False  # True runs only Tabular SNN for a quick screen.
TARGET_MODELS = (
    ["Tabular SNN"]
    if FAST_SCREEN_SNN_ONLY
    else ["XGBoost", "LightGBM", "Tabular SNN", "Tabular ResNet"]
)
SUBSTITUTE_MODEL_TYPE = "Tabular SNN"  # Attacker-side substitute model.
MINORITY_CLASS_WEIGHT = 10.0  # Positive-class training weight.

# -----------------------------------------------------------------------------
# 1.3 Data split and random seeds
# -----------------------------------------------------------------------------
TEST_SIZE = 0.40  # Test-set fraction.
SPLIT_RANDOM_STATE = 11  # Train/test split seed.
GLOBAL_RANDOM_STATE = 1  # Global seed for all other experiment stages.
SUBSTITUTE_RANDOM_STATE = 1  # Substitute-model training seed.

# Fixed seed offsets keep paired comparisons reproducible.
SEED_BAG_MULTIPLIER = 300
SEED_MODEL_MULTIPLIER = 500
SEED_AT_SAMPLE_MULTIPLIER = 7000
SEED_COLLEV_SAMPLE_MULTIPLIER = 9000
SEED_STRUCTURED_SUBSPACE_MULTIPLIER = 1300
SEED_UNIFORM_SUBSPACE_MULTIPLIER = 1700
SEED_COLLEV_POOL_OFFSET = 90011

# -----------------------------------------------------------------------------
# 1.4 Feature structure -- 77D cleanup
# -----------------------------------------------------------------------------
EXPECTED_RAW_N_FEATURES = 89
EXPECTED_N_FEATURES = 77

# Remove exactly 12 duplicate / redundant raw variables.
# 29 Ending Operating Revenue   -> raw 0 Operating Revenue
# 32 Book Debt / Total Assets   -> raw 68 Debt-to-Asset
# 33 Book Debt                  -> raw 51 Total Liabilities
# 41 Annual Sales               -> raw 0 Operating Revenue
# 43 Total Sales                -> raw 0 Operating Revenue
# 53 Ending Payables            -> raw 7 Accounts Payable
# 61 Year-end Receivables       -> raw 45 Ending Receivables
# 62 Beginning Receivables copy -> raw 46 Beginning Receivables
# 70 Ending Cash                -> raw 6 Cash
# 81 Ending Operating Profit    -> raw 12 Operating Profit
# 84 Ending Net Profit          -> raw 10 Net Profit
# 88 Duplicate ROA              -> raw 15 ROA
DROPPED_RAW_INDICES = np.array(
    [29, 32, 33, 41, 43, 53, 61, 62, 70, 81, 84, 88], dtype=int
)

RAW_KEEP_INDICES = np.array(
    [
        i
        for i in range(EXPECTED_RAW_N_FEATURES)
        if i not in set(DROPPED_RAW_INDICES.tolist())
    ],
    dtype=int,
)

RAW_TO_CLEAN = {
    int(raw_idx): int(clean_idx) for clean_idx, raw_idx in enumerate(RAW_KEEP_INDICES)
}
CLEAN_TO_RAW = {
    int(clean_idx): int(raw_idx) for raw_idx, clean_idx in RAW_TO_CLEAN.items()
}

# 16 free variables B, using original RAW index semantics.
FREE_RAW_INDICES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 22, 45, 48, 56, 58, 64, 74, 78], dtype=int
)

# 49 derived variables P, using original RAW index semantics.
DERIVED_RAW_INDICES = np.array(
    [
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        24,
        25,
        26,
        27,
        28,
        31,
        34,
        37,
        38,
        39,
        40,
        42,
        44,
        47,
        49,
        50,
        51,
        52,
        55,
        57,
        59,
        60,
        63,
        65,
        66,
        67,
        68,
        69,
        72,
        73,
        75,
        76,
        77,
        79,
        80,
        83,
    ],
    dtype=int,
)

# 12 fixed variables Z, using original RAW index semantics.
FIXED_RAW_INDICES = np.array([9, 23, 30, 35, 36, 46, 54, 71, 82, 85, 86, 87], dtype=int)

# Convert RAW indices into the actual 77D model coordinates.
FREE_INDICES = np.array([RAW_TO_CLEAN[int(i)] for i in FREE_RAW_INDICES], dtype=int)
DERIVED_INDICES = np.array(
    [RAW_TO_CLEAN[int(i)] for i in DERIVED_RAW_INDICES], dtype=int
)
FIXED_INDICES = np.array([RAW_TO_CLEAN[int(i)] for i in FIXED_RAW_INDICES], dtype=int)

# Unconstrained attack space: free variables B + derived variables P.
# Fixed variables Z remain immutable. The sparsity budget lambda is enforced
# over the union B \cup P, so at most lambda coordinates are modified in total.
UNCONSTRAINED_ATTACK_INDICES = np.sort(np.concatenate([FREE_INDICES, DERIVED_INDICES]))

assert len(RAW_KEEP_INDICES) == EXPECTED_N_FEATURES == 77
assert len(FIXED_INDICES) == 12
assert len(FREE_INDICES) == 16
assert len(DERIVED_INDICES) == 49
assert len(set(FIXED_INDICES) | set(FREE_INDICES) | set(DERIVED_INDICES)) == 77
assert set(FIXED_INDICES).isdisjoint(set(FREE_INDICES))
assert set(FIXED_INDICES).isdisjoint(set(DERIVED_INDICES))
assert set(FREE_INDICES).isdisjoint(set(DERIVED_INDICES))
assert (
    len(UNCONSTRAINED_ATTACK_INDICES) == len(FREE_INDICES) + len(DERIVED_INDICES) == 65
)
assert set(UNCONSTRAINED_ATTACK_INDICES).isdisjoint(set(FIXED_INDICES))

# -----------------------------------------------------------------------------
# 1.5 Neural-network architecture and training
# -----------------------------------------------------------------------------
NN_HIDDEN_DIM = 128  # Hidden-layer width.
NN_EPOCHS = 30  # Training epochs.
NN_LR = 1e-3  # AdamW learning rate.
NN_BATCH_SIZE = 256  # Training batch size.
NN_WEIGHT_DECAY = 1e-4  # AdamW weight decay
SNN_DROPOUT = 0.10  # Tabular SNN AlphaDropout
RESNET_DROPOUT = 0.20  # Tabular ResNet Dropout
PREDICT_BATCH_SIZE = 1024  # Minimum inference batch size.
PREDICTION_THRESHOLD = 0.50  # Binary classification threshold.
LOSS_EPS = 1e-7  # Prevent log(0) in the manual BCE loss.

# -----------------------------------------------------------------------------
# 1.6 Adversarial training and attack settings
# -----------------------------------------------------------------------------
AT_RHO = 0.05  # Maximum relative perturbation used during training.
AT_LAMBDA = 10  # Maximum modified free variables during training.

RHO_LIST = [0.0, 0.01, 0.02, 0.05]  # Test-time manipulation-ratio grid.
LAMBDA_LIST = [0, 5, 10]  # Test-time sparsity grid.

PGD_STEPS = 300  # Sample-specific PGD iterations.
PGD_ALPHA_MULTIPLIER = 1.5  # PGD step-size multiplier.

# -----------------------------------------------------------------------------
# 1.7 Ensemble and random-subspace settings
# -----------------------------------------------------------------------------
POOL_SIZE = 1  # Number of base classifiers in each ensemble.
BAGGING_FRACTION = 0.8  # Clean-training fraction used by each base learner.
DERIVED_SUBSPACE_RATIO = 0.8  # Retain all Z+B and a random fraction of P.

ENSEMBLE_METHOD_KEYS = ["Bag", "ColLevBag", "UniformBag", "UniformColLevBag"]
METHOD_DISPLAY = {
    "Bag": "Bag",
    "ColLevBag": "Bag+ColLev",
    "UniformBag": "Keep-P-Sample-ZB-Bag",
    "UniformColLevBag": "Keep-P-Sample-ZB-Bag+ColLev",
}
REPORT_METHODS = [
    "Clean",
    "Unconst-AT",
    "Const-AT",
    "Bag",
    "Bag+ColLev",
    "Keep-P-Sample-ZB-Bag",
    "Keep-P-Sample-ZB-Bag+ColLev",
]

# -----------------------------------------------------------------------------
# 1.8 Column-leverage augmentation settings
# -----------------------------------------------------------------------------
COLLEV_SHARE = 0.25  # ColLev share of the augmentation budget.
LEVERAGE_CALIB_SAMPLES = 48  # Samples used to estimate the reference Jacobian.
LEVERAGE_FD_REL_STEP = 1e-3  # Relative step for central finite differences.
LEV_AUG_RHO = 0.02  # Maximum relative ColLev perturbation.
LEV_AUG_EXTRA_RATIO = 0.20  # Extra samples divided by clean-bag size.
LEV_POOL_COPIES = 1  # ColLev copies generated per positive sample.
LEV_COL_MULT_CLIP = (0.50, 2)  # Bounds for normalized leverage weights.
LEV_RANDOM_STD = 0.25  # Standard deviation of ColLev proposals.
NUMERICAL_EPS = 1e-12  # Numerical-stability constant.

# -----------------------------------------------------------------------------
# 1.9 PyTorch / GPU
# -----------------------------------------------------------------------------
NN_NUM_WORKERS = 0  # Keep at zero for Windows/Spyder; increase on Linux if useful.

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")
PIN_MEMORY = CUDA_AVAILABLE

# Prefer stable numerical behavior without forcing deterministic algorithms.
if CUDA_AVAILABLE:
    torch.backends.cudnn.benchmark = False


def print_runtime_summary():
    """Print the feature partition and active compute device."""
    print(
        "77D feature partition: "
        f"Z={len(FIXED_INDICES)}, B={len(FREE_INDICES)}, "
        f"P={len(DERIVED_INDICES)}, total={EXPECTED_N_FEATURES}"
    )
    print("Dropped raw duplicate/redundant columns: " f"{DROPPED_RAW_INDICES.tolist()}")
    print(f"PyTorch device: {DEVICE}")
    if CUDA_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime (PyTorch): {torch.version.cuda}")


# =============================================================================
# 2. PyTorch model architectures and wrappers
# =============================================================================


class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=NN_HIDDEN_DIM):
        super().__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.res_lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.res_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res_lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.res_bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(RESNET_DROPOUT)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.first(x)))
        r = self.relu(self.res_bn1(self.res_lin1(x)))
        r = self.dropout(r)
        r = self.res_bn2(self.res_lin2(r))
        x = self.relu(x + r)
        return self.sigmoid(self.out(x))


class TabularSNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=NN_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(SNN_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(SNN_DROPOUT),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PyTorchTabularClassifier(BaseEstimator, ClassifierMixin):
    """Standard neural-network classifier used by all Bagging + augmentation methods.

    No Jacobian regularizer is applied in this refined screen.  This deliberately
    isolates the effect of leverage-aware accounting-consistent augmentation.
    """

    def __init__(
        self,
        model_type,
        input_dim,
        hidden_dim=NN_HIDDEN_DIM,
        epochs=NN_EPOCHS,
        lr=NN_LR,
        batch_size=NN_BATCH_SIZE,
        random_state=GLOBAL_RANDOM_STATE,
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.model = (
            TabularResNet(self.input_dim, self.hidden_dim).to(DEVICE)
            if self.model_type == "resnet"
            else TabularSNN(self.input_dim, self.hidden_dim).to(DEVICE)
        )
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=NN_WEIGHT_DECAY
        )

        X_cpu = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_cpu = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_cpu, y_cpu),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NN_NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        self.model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(DEVICE, non_blocking=PIN_MEMORY)
                batch_y = batch_y.to(DEVICE, non_blocking=PIN_MEMORY)
                optimizer.zero_grad(set_to_none=True)
                out = self.model(batch_X).view(-1)
                loss = -(
                    batch_y * torch.log(out + LOSS_EPS) * MINORITY_CLASS_WEIGHT
                    + (1 - batch_y) * torch.log(1 - out + LOSS_EPS)
                ).mean()
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_cpu = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_cpu),
            batch_size=max(self.batch_size, PREDICT_BATCH_SIZE),
            shuffle=False,
            num_workers=NN_NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        chunks = []
        with torch.inference_mode():
            for (batch_X,) in loader:
                batch_X = batch_X.to(DEVICE, non_blocking=PIN_MEMORY)
                chunks.append(self.model(batch_X).view(-1).detach().cpu())
        probs = (
            np.empty(0, dtype=np.float32) if not chunks else torch.cat(chunks).numpy()
        )
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > PREDICTION_THRESHOLD).astype(int)


def get_classifier(
    model_type, input_dim=EXPECTED_N_FEATURES, random_state=GLOBAL_RANDOM_STATE
):
    """Build one of the four target-model families used in this experiment."""
    class_weight_dict = {0: 1, 1: MINORITY_CLASS_WEIGHT}

    if model_type == "XGBoost":
        return XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=MINORITY_CLASS_WEIGHT,
            random_state=random_state,
        )
    if model_type == "LightGBM":
        return lgb.LGBMClassifier(
            random_state=random_state,
            class_weight=class_weight_dict,
            verbose=-1,
        )
    if model_type == "Tabular SNN":
        return PyTorchTabularClassifier(
            model_type="snn",
            input_dim=input_dim,
            random_state=random_state,
        )
    if model_type == "Tabular ResNet":
        return PyTorchTabularClassifier(
            model_type="resnet",
            input_dim=input_dim,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Average predicted probabilities from all base classifiers."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        if len(self.models) == 0:
            return np.zeros((X.shape[0], 2))
        avg_proba = np.zeros((X.shape[0], 2))
        for model in self.models:
            avg_proba += model.predict_proba(X)
        return avg_proba / len(self.models)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > PREDICTION_THRESHOLD).astype(int)


class FeatureSubsetClassifier(BaseEstimator, ClassifierMixin):
    """Train/predict a base classifier on a fixed feature subset."""

    def __init__(self, base_estimator, feature_indices):
        self.base_estimator = base_estimator
        self.feature_indices = np.asarray(feature_indices, dtype=int)

    def fit(self, X, y):
        self.base_estimator.fit(X[:, self.feature_indices], y)
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X[:, self.feature_indices])

    def predict(self, X):
        return self.base_estimator.predict(X[:, self.feature_indices])


def choose_uniform_derived_subspace(rng):
    """Original-paper mask: all Z+B, random psi proportion of P."""
    keep = max(1, int(round(len(DERIVED_INDICES) * DERIVED_SUBSPACE_RATIO)))
    selected_derived = np.sort(rng.choice(DERIVED_INDICES, size=keep, replace=False))
    subset_indices = np.sort(
        np.concatenate([FIXED_INDICES, FREE_INDICES, selected_derived])
    )
    return selected_derived, subset_indices


def choose_keep_p_sample_zb_subspace(rng, n_features):
    """Dimension-matched control: keep all P, randomly sample only Z+B.

    The total retained feature count is matched to the structured
    Z+B+psi*P design. Thus the only difference is WHERE the feature
    reduction occurs:
        structured Bag: keep all Z+B, sample P
        control Bag:    keep all P, sample Z+B
    """
    n_structured = (
        len(FIXED_INDICES)
        + len(FREE_INDICES)
        + max(1, int(round(len(DERIVED_INDICES) * DERIVED_SUBSPACE_RATIO)))
    )

    n_keep = min(int(n_features), int(n_structured))
    n_p = len(DERIVED_INDICES)
    n_zb_keep = n_keep - n_p

    if n_zb_keep < 0:
        raise ValueError(
            "Cannot keep all P while matching the structured feature count."
        )

    zb_indices = np.sort(np.concatenate([FIXED_INDICES, FREE_INDICES]))

    n_zb_keep = min(n_zb_keep, len(zb_indices))
    selected_zb = np.sort(rng.choice(zb_indices, size=n_zb_keep, replace=False))

    return np.sort(np.concatenate([DERIVED_INDICES, selected_zb]))


def compute_perturbed_features(X, r_free):
    """77D accounting propagation with original RAW-index semantics.

    Only the 12 duplicate/redundant raw variables are removed. Current formulas
    are otherwise preserved, including the existing p20/p21/p35 fixes.

    r_free keys are 77D CLEAN indices.
    Return value is a 77D CLEAN perturbation vector.
    """
    X = np.array(X, dtype=float, copy=True)

    if X.ndim != 1 or X.shape[0] != EXPECTED_N_FEATURES:
        raise ValueError(
            f"compute_perturbed_features expects {EXPECTED_N_FEATURES} features, "
            f"got shape {X.shape}."
        )

    r = np.zeros(EXPECTED_N_FEATURES, dtype=float)

    for idx, val in r_free.items():
        idx = int(idx)
        if idx not in FREE_INDICES:
            raise ValueError(f"Clean index {idx} is not a free variable.")
        r[idx] = float(val)

    def ci(raw_idx):
        raw_idx = int(raw_idx)
        if raw_idx not in RAW_TO_CLEAN:
            raise KeyError(f"Raw feature {raw_idx} was removed from the 77D set.")
        return RAW_TO_CLEAN[raw_idx]

    def x(raw_idx):
        return float(X[ci(raw_idx)])

    def d(raw_idx):
        return float(r[ci(raw_idx)])

    def put(raw_idx, value):
        r[ci(raw_idx)] = float(value)

    def safe_divide(a, b):
        if b == 0:
            raise ZeroDivisionError
        return a / b

    try:
        # raw41 Annual Sales removed -> canonical raw0 Revenue.
        # Preserve its old one-for-one role wherever downstream logic needs it.

        # Operating Profit
        put(12, d(0))

        # Net Profit
        put(10, d(0) + d(78) - d(1) - d(2) - d(3))

        # EBIT / EBITDA (unchanged logic)
        put(20, x(10) + d(10) - x(20))
        put(18, d(20))

        # p21 EPS -- preserve current fixed formula
        put(37, safe_divide(d(10) * 10000000.0, x(36)))

        # CFO-related legacy propagation
        put(74, d(0) + d(10))

        # raw33 Book Debt removed -> canonical raw51 Total Liabilities
        # raw63 is retained derived; old code first seeded it from AP then
        # later overwrote it with the ratio formula, so no extra action needed.

        # Total Liabilities
        put(51, d(7) + d(64))

        # Current Assets
        put(66, d(4) + d(6) + d(48))

        # Total Assets
        put(8, d(5) + d(56) + d(66))

        # Non-cash Working Capital
        put(31, d(66) - d(7) - d(6))

        # Shareholders' Equity
        put(14, x(8) + d(5) + d(56) + d(66) - x(51) - d(51) - x(14))

        # Enterprise Value (preserve uploaded formula)
        put(17, d(51) - d(6))

        # Net Profit Margin
        put(11, safe_divide(x(10) + d(10), x(0) + d(0)) - x(11))

        # ROA
        put(15, safe_divide(x(10) + d(10), x(8) + d(8)) - x(15))

        # Total Assets / Revenue
        put(24, safe_divide(x(8) + d(8), x(0) + d(0)) - x(24))

        # Cash / Revenue
        put(25, safe_divide(x(6) + d(6), x(0) + d(0)) - x(25))

        # NCWC / Revenue
        put(26, safe_divide(x(31) + d(31), x(0) + d(0)) - x(26))

        # Fixed Assets / Total Assets
        put(27, safe_divide(x(5) + d(5), x(8) + d(8)) - x(27))

        # EV / EBITDA
        put(16, safe_divide(x(17) + d(17), x(18) + d(18)) - x(16))

        # EV / EBIT
        put(19, safe_divide(x(17) + d(17), x(20) + d(20)) - x(19))

        # ROE
        put(13, safe_divide(x(10) + d(10), x(14) + d(14)) - x(13))

        # Working Capital Growth
        if x(23) != 0:
            put(21, (x(22) + d(22) - x(23)) / x(23) - x(21))

        # Revenue Growth:
        # raw29 duplicate ending revenue removed; canonical raw0 is used.
        if x(30) != 0:
            put(28, (x(0) + d(0) - x(30)) / x(30) - x(28))

        # p20 P/E -- preserve current fixed formula
        put(34, safe_divide(x(35), x(37) + d(37)) - x(34))

        # P/B -- preserve current unit convention
        if x(36) != 0 and (x(8) + d(8) - x(51) - d(51)) != 0:
            put(38, x(35) / 10000000 / ((x(8) + d(8) - x(51) - d(51)) / x(36)) - x(38))

        # P/S:
        # raw41 Annual Sales removed -> canonical raw0 Revenue.
        put(40, safe_divide(x(39) + d(39), x(0) + d(0)) - x(40))

        # Receivables / Sales:
        # raw43 Total Sales removed -> canonical raw0 Revenue.
        if (x(8) + d(8)) != 0:
            put(42, safe_divide(x(4) + d(4), x(8) + d(8)) - x(42))
        elif (x(0) + d(0)) != 0:
            put(42, safe_divide(x(4) + d(4), x(0) + d(0)) - x(42))

        # Receivables Growth
        if x(46) != 0:
            put(44, (x(45) + d(45) - x(46)) / x(46) - x(44))

        # Inventory / Total Assets
        put(47, safe_divide(x(48) + d(48), x(8) + d(8)) - x(47))

        # p28 Inventory Growth Rate:
        # Delta p28 = (1 + p28) * Delta Inventory / Inventory
        put(49, (1.0 + x(49)) * safe_divide(d(48), x(48)))

        # Payables / Total Liabilities
        put(50, safe_divide(x(7) + d(7), x(51) + d(51)) - x(50))

        # Payables Growth:
        # raw53 duplicate ending payables removed -> canonical raw7 AP.
        if x(54) != 0:
            put(52, (x(7) + d(7) - x(54)) / x(54) - x(52))

        # Soft Assets / Total Assets
        put(55, safe_divide(x(56) + d(56), x(8) + d(8)) - x(55))

        # Impairment Loss Ratio
        if (x(8) + d(8)) != 0:
            put(57, safe_divide(x(58) + d(58), x(8) + d(8)) - x(57))
        elif (x(0) + d(0)) != 0:
            put(57, safe_divide(x(58) + d(58), x(0) + d(0)) - x(57))

        # Inventory Turnover -- preserve uploaded legacy formula
        put(59, d(0) + d(48))

        # p35 Receivables Turnover -- preserve current fixed formula.
        # raw61/raw62 duplicate AR fields removed -> raw45/raw46 canonical fields.
        if (x(45) + d(45) + x(46)) != 0:
            put(60, safe_divide(x(0) + d(0), (x(45) + d(45) + x(46)) / 2.0) - x(60))

        # Interest-bearing Debt Ratio
        if (x(8) + d(8)) != 0:
            put(63, safe_divide(x(64) + d(64), x(8) + d(8)) - x(63))
        elif (x(51) + d(51)) != 0:
            put(63, safe_divide(x(64) + d(64), x(51) + d(51)) - x(63))

        # Current Liabilities:
        # raw67 is retained. Preserve uploaded old relation seeded from AP.
        put(67, d(7))

        # Current Ratio
        put(65, safe_divide(x(66) + d(66), x(67) + d(67)) - x(65))

        # Debt-to-Asset
        put(68, safe_divide(x(51) + d(51), x(8) + d(8)) - x(68))

        # Cash Growth:
        # raw70 duplicate ending cash removed -> canonical raw6 Cash.
        if x(71) != 0:
            put(69, (x(6) + d(6) - x(71)) / x(71) - x(69))

        # Cash Sales Ratio
        put(72, safe_divide(x(74) + d(74), x(0) + d(0)) - x(72))

        # CFO / Net Profit
        put(73, safe_divide(x(74) + d(74), x(10) + d(10)) - x(73))

        # Cash / Total Assets
        put(75, safe_divide(x(6) + d(6), x(8) + d(8)) - x(75))

        # EV / Revenue
        put(76, safe_divide(x(17) + d(17), x(0) + d(0)) - x(76))

        # Non-operating Income / Revenue
        put(77, safe_divide(x(78) + d(78), x(0) + d(0)) - x(77))

        # p47 Non-operating Income Growth Rate:
        # Delta p47 = (1 + p47) * Delta Non-operating Income / Non-operating Income
        put(79, (1.0 + x(79)) * safe_divide(d(78), x(78)))

        # Operating Profit Growth:
        # raw81 duplicate ending operating profit removed -> canonical raw12.
        if x(82) != 0:
            put(80, (x(12) + d(12) - x(82)) / x(82) - x(80))

        # Net Profit Growth:
        # raw84 duplicate ending net profit removed -> canonical raw10.
        if x(85) != 0:
            put(83, (x(10) + d(10) - x(85)) / x(85) - x(83))

        # raw88 duplicate ROA removed; no duplicate update needed.

    except KeyError as e:
        raise ValueError(f"77D raw/clean mapping error: {e}") from e
    except IndexError as e:
        raise ValueError(f"The input vector is missing required indices: {e}") from e
    except ZeroDivisionError as e:
        raise ValueError(f"A denominator in the accounting rules is zero: {e}") from e

    return r


# =============================================================================
# 3.1 Accounting-articulation leverage utilities
# =============================================================================


def estimate_reference_column_leverage(
    X_std,
    scaler,
    n_calib=LEVERAGE_CALIB_SAMPLES,
    rel_step=LEVERAGE_FD_REL_STEP,
    random_state=GLOBAL_RANDOM_STATE,
):
    """Estimate robust free-variable column leverage in standardized coordinates.

    For each calibration observation, J[p, b] approximates
    d P_std[p] / d B_std[b] by central finite differences. The element-wise
    median Jacobian is then used to compute ||J[:, b]||_2 for every free variable.
    """
    rng = np.random.default_rng(random_state)
    n = min(int(n_calib), len(X_std))
    if n <= 0:
        raise ValueError("Cannot estimate J_r from an empty calibration set.")

    calib_idx = rng.choice(len(X_std), size=n, replace=False)
    X_raw = scaler.inverse_transform(np.asarray(X_std)[calib_idx])

    free_scale = np.asarray(scaler.scale_)[FREE_INDICES]
    derived_scale = np.asarray(scaler.scale_)[DERIVED_INDICES]
    derived_scale = np.where(
        np.abs(derived_scale) < NUMERICAL_EPS,
        1.0,
        derived_scale,
    )

    jacobians = []
    for x_raw in X_raw:
        J = np.full(
            (len(DERIVED_INDICES), len(FREE_INDICES)),
            np.nan,
            dtype=float,
        )
        for col, global_idx in enumerate(FREE_INDICES):
            h = rel_step * max(
                abs(float(x_raw[global_idx])),
                abs(float(scaler.scale_[global_idx])),
                1.0,
            )
            try:
                r_plus = compute_perturbed_features(x_raw, {int(global_idx): +h})
                r_minus = compute_perturbed_features(x_raw, {int(global_idx): -h})
                d_raw = (r_plus[DERIVED_INDICES] - r_minus[DERIVED_INDICES]) / (2.0 * h)

                # d(P_raw / sigma_P) / d(B_raw / sigma_B)
                J[:, col] = d_raw * (free_scale[col] / derived_scale)
            except Exception:
                continue

        jacobians.append(J)

    J_stack = np.stack(jacobians, axis=0)
    with np.errstate(all="ignore"):
        J_ref = np.nanmedian(J_stack, axis=0)
    J_ref = np.nan_to_num(J_ref, nan=0.0, posinf=0.0, neginf=0.0)

    return np.linalg.norm(J_ref, axis=0)


def make_column_leverage_weights(col_leverage):
    """Mean-one free-variable leverage weights from column norms ||J_{:,q}||_2."""
    lev = np.asarray(col_leverage, dtype=float)
    w = lev / (np.mean(lev) + NUMERICAL_EPS)
    w = np.clip(w, LEV_COL_MULT_CLIP[0], LEV_COL_MULT_CLIP[1])
    return w / (np.mean(w) + NUMERICAL_EPS)


def exact_reconstruct_from_delta_b_std(x_std, delta_b_std, scaler, rho=LEV_AUG_RHO):
    """Apply a free-variable delta and reconstruct derived variables with r(.).

    ``delta_b_std`` is expressed in standardized B coordinates. Before calling
    ``r(.)``, it is
    converted to raw accounting units and clipped to |Delta b_j| <= rho |b_j|.
    Returns the original sample if the articulation mapping enters an invalid
    ratio region.
    """
    x_std = np.asarray(x_std, dtype=float)
    x_raw = scaler.inverse_transform(x_std.reshape(1, -1))[0]
    delta_b_std = np.asarray(delta_b_std, dtype=float)
    delta_b_raw = delta_b_std * np.asarray(scaler.scale_)[FREE_INDICES]

    bounds = rho * np.abs(x_raw[FREE_INDICES])
    delta_b_raw = np.clip(delta_b_raw, -bounds, bounds)
    if np.allclose(delta_b_raw, 0.0):
        return x_std.copy(), False

    r_free = {int(idx): float(delta_b_raw[j]) for j, idx in enumerate(FREE_INDICES)}
    try:
        r = compute_perturbed_features(x_raw, r_free)
        adv_raw = x_raw + r
        adv_std = scaler.transform(adv_raw.reshape(1, -1))[0]
        if not np.all(np.isfinite(adv_std)):
            return x_std.copy(), False
        return adv_std, True
    except Exception:
        return x_std.copy(), False


def _rho_scaled_random_delta_std(x_std, scaler, rng, multipliers=None, rho=LEV_AUG_RHO):
    """Draw a standardized B perturbation scaled by the rho bounds."""
    x_raw = scaler.inverse_transform(np.asarray(x_std).reshape(1, -1))[0]
    bounds_raw = rho * np.abs(x_raw[FREE_INDICES])
    z = rng.normal(0.0, LEV_RANDOM_STD, size=len(FREE_INDICES))
    if multipliers is not None:
        z = z * np.asarray(multipliers)
    delta_raw = np.clip(z * bounds_raw, -bounds_raw, bounds_raw)
    scale_b = np.asarray(scaler.scale_)[FREE_INDICES]
    scale_b = np.where(np.abs(scale_b) < NUMERICAL_EPS, 1.0, scale_b)
    return delta_raw / scale_b


def build_column_leverage_pool(X_pos, scaler, col_weights, rng):
    """Emphasize B variables whose changes propagate most strongly through J_r."""
    out, ok = [], 0
    mult = np.sqrt(np.maximum(np.asarray(col_weights), 0.0))
    for x in np.asarray(X_pos):
        dB = _rho_scaled_random_delta_std(x, scaler, rng, multipliers=mult)
        xa, success = exact_reconstruct_from_delta_b_std(x, dB, scaler)
        out.append(xa)
        ok += int(success)
    return np.asarray(out), ok / max(1, len(out))


def generate_specific_perturbation(
    clf_sub, sample, scaler, mu, lam_change, pgd_steps=PGD_STEPS, candidate_indices=None
):
    """Generate a sparse PGD perturbation over a specified feature set.

    candidate_indices controls which coordinates may be modified. If omitted,
    the original constrained-attack behavior is preserved and only B is
    optimized. For Unconst-AT, B and P are both directly optimizable.

    In either case, at most ``lam_change`` coordinates are selected and each
    selected coordinate satisfies |Delta x_j| <= mu * |x_j| in raw units.
    """
    if candidate_indices is None:
        candidate_indices = FREE_INDICES
    candidate_indices = np.asarray(candidate_indices, dtype=int)

    if lam_change == 0 or mu == 0.0:
        return np.zeros(len(candidate_indices))

    sample_inv = scaler.inverse_transform(sample.reshape(1, -1))[0]
    sample_up = sample_inv * (1 + mu)
    sample_low = sample_inv * (1 - mu)

    up_bounds = scaler.transform(sample_up.reshape(1, -1))[0]
    low_bounds = scaler.transform(sample_low.reshape(1, -1))[0]
    swap_mask = low_bounds > up_bounds
    low_bounds[swap_mask], up_bounds[swap_mask] = (
        up_bounds[swap_mask],
        low_bounds[swap_mask],
    )

    model = clf_sub.model
    model.eval()

    x_tensor = torch.tensor(
        sample, dtype=torch.float32, device=DEVICE, requires_grad=True
    )
    out = model(x_tensor.unsqueeze(0)).view(-1)
    grad_tensor = torch.autograd.grad(
        out.sum(), x_tensor, create_graph=False, retain_graph=False
    )[0]
    grad = grad_tensor.detach().cpu().numpy()

    k = min(int(lam_change), len(candidate_indices))
    if k >= len(candidate_indices):
        active_local_indices = np.arange(len(candidate_indices))
    else:
        candidate_grads = np.abs(grad[candidate_indices])
        active_local_indices = np.argsort(candidate_grads)[-k:]

    active_global_indices = candidate_indices[active_local_indices]

    x_adv = sample.copy()
    max_range = up_bounds - low_bounds
    alpha = (max_range / pgd_steps) * PGD_ALPHA_MULTIPLIER
    alpha[alpha == 0] = 1e-3

    for _ in range(pgd_steps):
        x_tensor = torch.tensor(
            x_adv, dtype=torch.float32, device=DEVICE, requires_grad=True
        )
        out = model(x_tensor.unsqueeze(0)).view(-1)
        grad_tensor = torch.autograd.grad(
            out.sum(), x_tensor, create_graph=False, retain_graph=False
        )[0]
        g = grad_tensor.detach().cpu().numpy()
        x_adv[active_global_indices] -= alpha[active_global_indices] * np.sign(
            g[active_global_indices]
        )
        x_adv = np.clip(x_adv, low_bounds, up_bounds)

        mask = np.ones_like(x_adv, dtype=bool)
        mask[active_global_indices] = False
        x_adv[mask] = sample[mask]

    perturbation = x_adv - sample
    return perturbation[candidate_indices]


def calculate_sms_metric(rho_list, lambda_list, metric_matrix):
    sms_sum = 0.0
    for i in range(1, len(rho_list)):
        for j in range(1, len(lambda_list)):
            d_rho = rho_list[i] - rho_list[i - 1]
            d_lambda = lambda_list[j] - lambda_list[j - 1]
            corner_sum = (
                metric_matrix[i, j]
                + metric_matrix[i - 1, j]
                + metric_matrix[i, j - 1]
                + metric_matrix[i - 1, j - 1]
            )
            V_ij = d_rho * d_lambda * (corner_sum / 4.0)
            sms_sum += V_ij
    SMS = (1.0 / (rho_list[-1] * lambda_list[-1])) * sms_sum
    return SMS


# =============================================================================
# 4. Data loading and preprocessing
# =============================================================================


def main():
    print_runtime_summary()
    print("Loading data and configuring parameters...")
    try:
        df = pd.read_excel(DATA_PATH).drop(columns=STOCK_NAME_COLUMN, errors="ignore")
    except FileNotFoundError as exc:
        raise SystemExit(f"Input file not found: {DATA_PATH}") from exc

    Data = np.array(df)
    X_raw_full, y = Data[:, :-1], Data[:, -1]

    if X_raw_full.shape[1] != EXPECTED_RAW_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_RAW_N_FEATURES} raw financial features, "
            f"but found {X_raw_full.shape[1]}."
        )

    # Physically remove exactly the 12 duplicate/redundant variables.
    X_ = X_raw_full[:, RAW_KEEP_INDICES]

    scaler = StandardScaler()
    X_ = scaler.fit_transform(X_)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )
    for train_index, test_index in sss.split(X_, y):
        X_tr, y_tr = X_[train_index], y[train_index]
        X_te, y_te = X_[test_index], y[test_index]

    X_tr_P = X_tr[y_tr == 1]
    X_te_P = X_te[y_te == 1]
    X_te_N = X_te[y_te == 0]

    input_dim = X_tr.shape[1]
    if input_dim != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} features, but found {input_dim}."
        )

    print(f"Train/Test = {len(X_tr)}/{len(X_te)}")

    print("\nEstimating reference accounting-articulation column leverage ...")
    FREE_COL_LEVERAGE = estimate_reference_column_leverage(X_tr, scaler)
    FREE_COL_LEVERAGE_WEIGHTS = make_column_leverage_weights(FREE_COL_LEVERAGE)

    top_cols = np.argsort(FREE_COL_LEVERAGE)[-10:][::-1]
    print("Top column-leverage free variables (global index : ||J_col||):")
    print(
        ", ".join(
            f"{int(FREE_INDICES[c])}:{FREE_COL_LEVERAGE[c]:.3f}" for c in top_cols
        )
    )

    # =============================================================================
    # 4.1 Accounting-consistent adversarial-training pool (common base for ALL methods)
    # =============================================================================
    print(f"\nTraining Substitute Model ({SUBSTITUTE_MODEL_TYPE}) for AT Generation...")
    clf_substitute = get_classifier(
        SUBSTITUTE_MODEL_TYPE, input_dim=input_dim, random_state=SUBSTITUTE_RANDOM_STATE
    )
    clf_substitute.fit(X_tr, y_tr)

    print("\nGenerating adversarial training pools (Unconst-AT & Const-AT)...")
    adv_tr_P_unconst_list = []
    adv_tr_P_const_list = []
    at_reconstruction_failures = 0

    for ori in X_tr_P:
        # Unconst-AT: optimize directly over B + P.
        # The total number of modified B/P coordinates is <= AT_LAMBDA and every
        # selected coordinate obeys the same AT_RHO relative-magnitude bound.
        # Fixed variables Z remain unchanged and no articulation repair is imposed.
        unconst_delta = generate_specific_perturbation(
            clf_substitute,
            ori,
            scaler,
            AT_RHO,
            AT_LAMBDA,
            candidate_indices=UNCONSTRAINED_ATTACK_INDICES,
        )
        unconst_perturbation = np.zeros(X_tr_P.shape[1])
        unconst_perturbation[UNCONSTRAINED_ATTACK_INDICES] = unconst_delta
        adv_unconst = ori + unconst_perturbation
        adv_tr_P_unconst_list.append(adv_unconst)

        # Const-AT: preserve the original accounting-consistent design.
        # Optimize only B under the same rho/lambda budgets, then reconstruct P
        # exactly through the articulation mapping r(.).
        const_delta_b = generate_specific_perturbation(
            clf_substitute,
            ori,
            scaler,
            AT_RHO,
            AT_LAMBDA,
            candidate_indices=FREE_INDICES,
        )
        const_free_perturbation = np.zeros(X_tr_P.shape[1])
        const_free_perturbation[FREE_INDICES] = const_delta_b
        adv_free_only = ori + const_free_perturbation

        ori_raw = scaler.inverse_transform(ori.reshape(1, -1))[0]
        random_vector = (
            scaler.inverse_transform(adv_free_only.reshape(1, -1))[0] - ori_raw
        )
        r_free = {int(idx): float(random_vector[int(idx)]) for idx in FREE_INDICES}

        try:
            r = compute_perturbed_features(ori_raw, r_free)
            adv_raw = ori_raw + r
            adv_std = scaler.transform(adv_raw.reshape(1, -1))[0]
            adv_tr_P_const_list.append(adv_std)
        except Exception:
            adv_tr_P_const_list.append(ori)
            at_reconstruction_failures += 1

    adv_tr_P_unconst = np.asarray(adv_tr_P_unconst_list)
    adv_tr_P_const = np.asarray(adv_tr_P_const_list)

    X_aug_unconst = np.vstack([X_tr, adv_tr_P_unconst])
    y_aug_unconst = np.concatenate([y_tr, np.ones(len(adv_tr_P_unconst))])

    X_aug_const = np.vstack([X_tr, adv_tr_P_const])
    y_aug_const = np.concatenate([y_tr, np.ones(len(adv_tr_P_const))])

    print(
        f"AT reconstruction failure rate: "
        f"{at_reconstruction_failures / max(1, len(X_tr_P)):.2%}"
    )

    # Pre-compute ColLev augmentation pool once.
    # Bag and Bag+ColLev use the same accounting-consistent AT pool.
    # Bag+ColLev replaces part of the fixed augmentation budget with ColLev data.
    print("\nPrecomputing column-leverage augmentation pool...")
    X_LEV_POOL_SOURCE = np.repeat(X_tr_P, LEV_POOL_COPIES, axis=0)
    COLLEV_AUG_POOL, col_ok = build_column_leverage_pool(
        X_LEV_POOL_SOURCE,
        scaler,
        FREE_COL_LEVERAGE_WEIGHTS,
        np.random.default_rng(GLOBAL_RANDOM_STATE + SEED_COLLEV_POOL_OFFSET),
    )
    print(f"Exact ColLev augmentation success rate: {col_ok:.2%}")

    # =============================================================================
    # 5. Train the seven comparison methods
    #    Clean / Unconst-AT / Const-AT / Bag / Bag+ColLev /
    #    Keep-P-Sample-ZB-Bag / Keep-P-Sample-ZB-Bag+ColLev
    # =============================================================================
    clf_targets = {}
    EVAL_MODELS = []

    print(
        f"\nMatched augmentation design: Bag = 100% AT; "
        f"Bag+ColLev = {(1-COLLEV_SHARE):.0%} AT + {COLLEV_SHARE:.0%} ColLev"
    )

    for name in TARGET_MODELS:
        print(
            f"\n[{name}] Training Clean / Unconst-AT / Const-AT / Bag / "
            "Bag+ColLev / Keep-P-Sample-ZB-Bag / "
            "Keep-P-Sample-ZB-Bag+ColLev ..."
        )

        # -------------------------------------------------------------------------
        # 1) Clean baseline
        # -------------------------------------------------------------------------
        model_name_clean = f"{name} (Clean)"
        clf_clean = get_classifier(
            name, input_dim=input_dim, random_state=GLOBAL_RANDOM_STATE
        )
        clf_clean.fit(X_tr, y_tr)
        clf_targets[model_name_clean] = clf_clean
        EVAL_MODELS.append(model_name_clean)

        # -------------------------------------------------------------------------
        # 2) Unconstrained adversarial training
        # -------------------------------------------------------------------------
        model_name_uat = f"{name} (Unconst-AT)"
        clf_uat = get_classifier(
            name, input_dim=input_dim, random_state=GLOBAL_RANDOM_STATE + 1
        )
        clf_uat.fit(X_aug_unconst, y_aug_unconst)
        clf_targets[model_name_uat] = clf_uat
        EVAL_MODELS.append(model_name_uat)

        # -------------------------------------------------------------------------
        # 3) Accounting-consistent adversarial training
        # -------------------------------------------------------------------------
        model_name_cat = f"{name} (Const-AT)"
        clf_cat = get_classifier(
            name, input_dim=input_dim, random_state=GLOBAL_RANDOM_STATE + 2
        )
        clf_cat.fit(X_aug_const, y_aug_const)
        clf_targets[model_name_cat] = clf_cat
        EVAL_MODELS.append(model_name_cat)

        # -------------------------------------------------------------------------
        # 4-5) Plain Bagging:
        #      Bag / Bag+ColLev
        # -------------------------------------------------------------------------
        candidate_pools = {key: [] for key in ENSEMBLE_METHOD_KEYS}

        for round_i in range(1, POOL_SIZE + 1):

            # Same clean/original bag for both candidate pools.
            clean_bag_size = int(len(X_tr) * BAGGING_FRACTION)
            bag_seed = GLOBAL_RANDOM_STATE + round_i * SEED_BAG_MULTIPLIER
            model_seed = GLOBAL_RANDOM_STATE + round_i * SEED_MODEL_MULTIPLIER

            rng_clean = np.random.default_rng(bag_seed)
            idx_clean = rng_clean.choice(len(X_tr), size=clean_bag_size, replace=False)
            X_clean_bag = X_tr[idx_clean]
            y_clean_bag = y_tr[idx_clean]

            # Same total augmentation budget.
            n_aug_total = max(1, int(round(LEV_AUG_EXTRA_RATIO * clean_bag_size)))
            n_col = int(round(COLLEV_SHARE * n_aug_total))
            n_at_col = n_aug_total - n_col

            # Bag candidate: 100% accounting-consistent AT.
            rng_at = np.random.default_rng(
                GLOBAL_RANDOM_STATE + round_i * SEED_AT_SAMPLE_MULTIPLIER
            )
            ids_at_se = rng_at.choice(
                len(adv_tr_P_const),
                size=n_aug_total,
                replace=(n_aug_total > len(adv_tr_P_const)),
            )

            X_se_fit = np.vstack([X_clean_bag, adv_tr_P_const[ids_at_se]])
            y_se_fit = np.concatenate(
                [y_clean_bag, np.ones(n_aug_total, dtype=y_tr.dtype)]
            )

            # Bag+ColLev candidate:
            # same total count = 75% matched AT + 25% ColLev.
            ids_at_col = ids_at_se[:n_at_col]

            rng_col = np.random.default_rng(
                GLOBAL_RANDOM_STATE + round_i * SEED_COLLEV_SAMPLE_MULTIPLIER
            )
            ids_col = (
                rng_col.choice(
                    len(COLLEV_AUG_POOL),
                    size=n_col,
                    replace=(n_col > len(COLLEV_AUG_POOL)),
                )
                if n_col > 0
                else np.empty(0, dtype=int)
            )

            X_col_parts = [X_clean_bag, adv_tr_P_const[ids_at_col]]
            if n_col > 0:
                X_col_parts.append(COLLEV_AUG_POOL[ids_col])

            X_col_fit = np.vstack(X_col_parts)
            y_col_fit = np.concatenate(
                [
                    y_clean_bag,
                    np.ones(n_at_col, dtype=y_tr.dtype),
                    np.ones(n_col, dtype=y_tr.dtype),
                ]
            )

            # Strict paired fairness checks.
            assert n_aug_total == n_at_col + n_col
            assert len(X_se_fit) == len(X_col_fit)
            assert len(X_se_fit) - len(X_clean_bag) == n_aug_total
            assert len(X_col_fit) - len(X_clean_bag) == n_aug_total

            # Original-paper random subspace:
            # retain all fixed Z + all free B + 70% random P.
            rng_subspace = np.random.default_rng(
                GLOBAL_RANDOM_STATE + round_i * SEED_STRUCTURED_SUBSPACE_MULTIPLIER
            )
            selected_derived, subset_indices = choose_uniform_derived_subspace(
                rng_subspace
            )

            if round_i == 1:
                print(
                    f"  matched candidate/base learner: "
                    f"clean={clean_bag_size}, aug={n_aug_total}, "
                    f"features={len(subset_indices)}/{input_dim} "
                    f"(all Z+B + {len(selected_derived)}/{len(DERIVED_INDICES)} P); "
                    f"Bag: AT={n_aug_total}; "
                    f"Bag+ColLev: AT={n_at_col}, ColLev={n_col}"
                )

            # Same subspace + same model seed for the paired candidates.
            clf_se_base = get_classifier(
                name, input_dim=len(subset_indices), random_state=model_seed
            )
            clf_se = FeatureSubsetClassifier(clf_se_base, subset_indices)
            clf_se.fit(X_se_fit, y_se_fit)
            candidate_pools["Bag"].append(clf_se)

            clf_col_base = get_classifier(
                name, input_dim=len(subset_indices), random_state=model_seed
            )
            clf_col = FeatureSubsetClassifier(clf_col_base, subset_indices)
            clf_col.fit(X_col_fit, y_col_fit)
            candidate_pools["ColLevBag"].append(clf_col)

            # -------------------------------------------------------------
            # Keep-P-Sample-ZB baseline.
            # Same retained feature count as the structured Z+B+psi*P mask,
            # but keep ALL derived variables P and randomly sample only Z+B.
            # This isolates whether robustness gains come specifically from
            # sampling P rather than merely reducing the total feature dimension.
            # The paired control Bag / control Bag+ColLev candidates reuse
            # the exact same clean bag, augmentation samples and model seed.
            # -------------------------------------------------------------
            rng_uniform = np.random.default_rng(
                GLOBAL_RANDOM_STATE + round_i * SEED_UNIFORM_SUBSPACE_MULTIPLIER
            )
            uniform_subset_indices = choose_keep_p_sample_zb_subspace(
                rng_uniform, input_dim
            )

            if round_i == 1:
                n_z = int(np.sum(np.isin(uniform_subset_indices, FIXED_INDICES)))
                n_b = int(np.sum(np.isin(uniform_subset_indices, FREE_INDICES)))
                n_p = int(np.sum(np.isin(uniform_subset_indices, DERIVED_INDICES)))
                print(
                    f"  keep-P/sample-ZB baseline: "
                    f"features={len(uniform_subset_indices)}/{input_dim}, "
                    f"realized composition Z/B/P={n_z}/{n_b}/{n_p}"
                )

            clf_uniform_bag_base = get_classifier(
                name, input_dim=len(uniform_subset_indices), random_state=model_seed
            )
            clf_uniform_bag = FeatureSubsetClassifier(
                clf_uniform_bag_base, uniform_subset_indices
            )
            clf_uniform_bag.fit(X_se_fit, y_se_fit)
            candidate_pools["UniformBag"].append(clf_uniform_bag)

            clf_uniform_col_base = get_classifier(
                name, input_dim=len(uniform_subset_indices), random_state=model_seed
            )
            clf_uniform_col = FeatureSubsetClassifier(
                clf_uniform_col_base, uniform_subset_indices
            )
            clf_uniform_col.fit(X_col_fit, y_col_fit)
            candidate_pools["UniformColLevBag"].append(clf_uniform_col)

        # -------------------------------------------------------------------------
        # Plain Bagging: remove selective-ensemble screening only.
        # No validation-AUPRC ranking and no Yule-Q filtering are applied here.
        # All POOL_SIZE candidate models are retained and combined by soft voting.
        # Candidate construction above is otherwise unchanged.
        # -------------------------------------------------------------------------
        model_name_bag = f"{name} ({METHOD_DISPLAY['Bag']})"
        clf_targets[model_name_bag] = SoftVotingEnsemble(candidate_pools["Bag"])
        EVAL_MODELS.append(model_name_bag)

        model_name_collev = f"{name} ({METHOD_DISPLAY['ColLevBag']})"
        clf_targets[model_name_collev] = SoftVotingEnsemble(
            candidate_pools["ColLevBag"]
        )
        EVAL_MODELS.append(model_name_collev)

        model_name_uniform_bag = f"{name} ({METHOD_DISPLAY['UniformBag']})"
        clf_targets[model_name_uniform_bag] = SoftVotingEnsemble(
            candidate_pools["UniformBag"]
        )
        EVAL_MODELS.append(model_name_uniform_bag)

        model_name_uniform_collev = f"{name} ({METHOD_DISPLAY['UniformColLevBag']})"
        clf_targets[model_name_uniform_collev] = SoftVotingEnsemble(
            candidate_pools["UniformColLevBag"]
        )
        EVAL_MODELS.append(model_name_uniform_collev)

        print(
            f"    [{name}] Plain Bagging: "
            f"Bag={len(candidate_pools['Bag'])}, "
            f"Bag+ColLev={len(candidate_pools['ColLevBag'])}, "
            f"Keep-P-Sample-ZB-Bag={len(candidate_pools['UniformBag'])}, "
            f"Keep-P-Sample-ZB-Bag+ColLev={len(candidate_pools['UniformColLevBag'])}"
        )

    # =============================================================================
    # 6. 3D Security Surface Evaluation
    # =============================================================================
    print("\nStarting 3D Security Evaluation Surface Scan (Metric: AUPRC)...")
    print(f"Evaluating {len(EVAL_MODELS)} models; Substitute: {SUBSTITUTE_MODEL_TYPE}")

    prc_const_matrices = {
        name: np.zeros((len(RHO_LIST), len(LAMBDA_LIST))) for name in EVAL_MODELS
    }
    y_te_combined = np.concatenate([np.zeros(len(X_te_N)), np.ones(len(X_te_P))])

    base_prcs = {}
    for model_name in EVAL_MODELS:
        clf = clf_targets[model_name]
        probs_P = clf.predict_proba(X_te_P)[:, 1]
        probs_N = clf.predict_proba(X_te_N)[:, 1]
        base_prcs[model_name] = average_precision_score(
            y_te_combined, np.concatenate([probs_N, probs_P])
        )

    test_reconstruction_failures = 0
    test_reconstruction_total = 0
    for i, rho in enumerate(RHO_LIST):
        for j, lam in enumerate(LAMBDA_LIST):
            if rho == 0.0 or lam == 0:
                for model_name in EVAL_MODELS:
                    prc_const_matrices[model_name][i, j] = base_prcs[model_name]
                continue

            print(f"\nRunning Grid (rho={rho:.2f}, lambda={lam:2d})...")
            constrained_X_P_list = []
            for ori in X_te_P:
                multiply1 = generate_specific_perturbation(
                    clf_substitute, ori, scaler, rho, lam
                )
                specific_perturbation = np.zeros(X_te_P.shape[1])
                specific_perturbation[FREE_INDICES] = multiply1
                adv_unconst = ori + specific_perturbation
                ori_raw = scaler.inverse_transform(ori.reshape(1, -1))[0]
                random_vector = (
                    scaler.inverse_transform(adv_unconst.reshape(1, -1))[0] - ori_raw
                )
                r_free = {
                    int(idx): float(random_vector[int(idx)]) for idx in FREE_INDICES
                }
                test_reconstruction_total += 1
                try:
                    r = compute_perturbed_features(ori_raw, r_free)
                    adv_raw = ori_raw + r
                    adv_std = scaler.transform(adv_raw.reshape(1, -1))[0]
                    constrained_X_P_list.append(adv_std)
                except Exception:
                    constrained_X_P_list.append(ori)
                    test_reconstruction_failures += 1

            constrained_X_P = np.asarray(constrained_X_P_list)
            for model_name in EVAL_MODELS:
                clf = clf_targets[model_name]
                probs_N = clf.predict_proba(X_te_N)[:, 1]
                probs_P = clf.predict_proba(constrained_X_P)[:, 1]
                score = average_precision_score(
                    y_te_combined, np.concatenate([probs_N, probs_P])
                )
                prc_const_matrices[model_name][i, j] = score
                print(f"  [{model_name:42s}] Const AUPRC: {score:.4f}")

    print(
        f"\nEvaluation reconstruction failure rate: "
        f"{test_reconstruction_failures / max(1, test_reconstruction_total):.2%}"
    )

    # =============================================================================
    # 7. SMS / Clean AUPRC Summary and Excel Output
    # =============================================================================
    print("\n" + "=" * 120)
    print(
        "CLEAN / UNCONST-AT / CONST-AT / BAG / BAG+COLLEV / "
        "KEEP-P-SAMPLE-ZB-BAG / KEEP-P-SAMPLE-ZB-BAG+COLLEV COMPARISON"
    )
    print("=" * 120)

    summary_sms = []
    summary_clean = []

    for name in TARGET_MODELS:
        row_sms = {"Target Architecture": name}
        row_clean = {"Target Architecture": name}

        # Clean
        clean_model = f"{name} (Clean)"
        clean_sms = calculate_sms_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[clean_model]
        )
        clean_auprc = base_prcs[clean_model]
        row_sms["Clean SMS"] = clean_sms
        row_clean["Clean AUPRC"] = clean_auprc

        # Unconst-AT
        uat_model = f"{name} (Unconst-AT)"
        uat_sms = calculate_sms_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[uat_model]
        )
        uat_clean = base_prcs[uat_model]
        row_sms["Unconst-AT SMS"] = uat_sms
        row_clean["Unconst-AT AUPRC"] = uat_clean

        # Const-AT
        cat_model = f"{name} (Const-AT)"
        cat_sms = calculate_sms_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[cat_model]
        )
        cat_clean = base_prcs[cat_model]
        row_sms["Const-AT SMS"] = cat_sms
        row_clean["Const-AT AUPRC"] = cat_clean

        # Plain Bagging
        bag_model = f"{name} ({METHOD_DISPLAY['Bag']})"
        bag_sms = calculate_sms_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[bag_model]
        )
        bag_clean = base_prcs[bag_model]
        row_sms["Bag SMS"] = bag_sms
        row_clean["Bag AUPRC"] = bag_clean

        # Column-leverage-guided Bagging
        col_model = f"{name} ({METHOD_DISPLAY['ColLevBag']})"
        col_sms = calculate_sms_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[col_model]
        )
        col_clean = base_prcs[col_model]
        row_sms["Bag+ColLev SMS"] = col_sms
        row_sms["SMS Difference: ColLev vs Bag"] = col_sms - bag_sms
        row_clean["Bag+ColLev AUPRC"] = col_clean
        row_clean["AUPRC Difference: ColLev vs Bag"] = col_clean - bag_clean

        # Uniform all-feature sampling Bagging
        uniform_bag_model = f"{name} ({METHOD_DISPLAY['UniformBag']})"
        uniform_bag_sms = calculate_sms_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[uniform_bag_model]
        )
        uniform_bag_clean = base_prcs[uniform_bag_model]
        row_sms["Keep-P-Sample-ZB-Bag SMS"] = uniform_bag_sms
        row_clean["Keep-P-Sample-ZB-Bag AUPRC"] = uniform_bag_clean

        # Uniform all-feature sampling Bagging + ColLev
        uniform_col_model = f"{name} ({METHOD_DISPLAY['UniformColLevBag']})"
        uniform_col_sms = calculate_sms_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[uniform_col_model]
        )
        uniform_col_clean = base_prcs[uniform_col_model]
        row_sms["Keep-P-Sample-ZB-Bag+ColLev SMS"] = uniform_col_sms
        row_sms["SMS Difference: Keep-P ColLev vs Keep-P Bag"] = (
            uniform_col_sms - uniform_bag_sms
        )
        row_clean["Keep-P-Sample-ZB-Bag+ColLev AUPRC"] = uniform_col_clean
        row_clean["AUPRC Difference: Keep-P ColLev vs Keep-P Bag"] = (
            uniform_col_clean - uniform_bag_clean
        )

        # Direct feature-sampling contrasts
        row_sms["SMS Difference: Structured Bag vs Keep-P Bag"] = (
            bag_sms - uniform_bag_sms
        )
        row_sms["SMS Difference: Structured ColLev vs Keep-P ColLev"] = (
            col_sms - uniform_col_sms
        )
        row_clean["AUPRC Difference: Structured Bag vs Keep-P Bag"] = (
            bag_clean - uniform_bag_clean
        )
        row_clean["AUPRC Difference: Structured ColLev vs Keep-P ColLev"] = (
            col_clean - uniform_col_clean
        )

        summary_sms.append(row_sms)
        summary_clean.append(row_clean)

    df_sms_summary = pd.DataFrame(summary_sms)
    df_clean_summary = pd.DataFrame(summary_clean)

    pd.options.display.float_format = "{:.4f}".format

    print(">>> SMS Summary:")
    sms_print_cols = [
        c for c in df_sms_summary.columns if not c.startswith("SMS Difference:")
    ]
    print(df_sms_summary[sms_print_cols].to_string(index=False))

    print("\n>>> Clean AUPRC Summary:")
    auprc_print_cols = [
        c for c in df_clean_summary.columns if not c.startswith("AUPRC Difference:")
    ]
    print(df_clean_summary[auprc_print_cols].to_string(index=False))

    # Architecture-averaged summary for quick comparison.
    method_names = REPORT_METHODS
    avg_rows = []

    for method in method_names:
        sms_vals = []
        clean_vals = []
        tree_sms_vals = []
        tree_clean_vals = []

        for name in TARGET_MODELS:
            if method == "Bag":
                model_name = f"{name} ({METHOD_DISPLAY['Bag']})"
            elif method == "Bag+ColLev":
                model_name = f"{name} ({METHOD_DISPLAY['ColLevBag']})"
            elif method == "Keep-P-Sample-ZB-Bag":
                model_name = f"{name} ({METHOD_DISPLAY['UniformBag']})"
            elif method == "Keep-P-Sample-ZB-Bag+ColLev":
                model_name = f"{name} ({METHOD_DISPLAY['UniformColLevBag']})"
            else:
                model_name = f"{name} ({method})"

            sms_val = calculate_sms_metric(
                RHO_LIST, LAMBDA_LIST, prc_const_matrices[model_name]
            )
            clean_val = base_prcs[model_name]

            sms_vals.append(sms_val)
            clean_vals.append(clean_val)

            if name in ("XGBoost", "LightGBM"):
                tree_sms_vals.append(sms_val)
                tree_clean_vals.append(clean_val)

        avg_rows.append(
            {
                "Method": method,
                "Mean SMS": np.mean(sms_vals),
                "Mean Clean AUPRC": np.mean(clean_vals),
                "Tree Mean SMS": np.mean(tree_sms_vals) if tree_sms_vals else np.nan,
                "Tree Mean Clean AUPRC": (
                    np.mean(tree_clean_vals) if tree_clean_vals else np.nan
                ),
            }
        )

    df_method_average = pd.DataFrame(avg_rows)

    print("\n>>> Method Average Summary:")
    print(df_method_average.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    print(f"\nSaving results to: {output_path} ... ", end="")

    try:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            df_sms_summary.to_excel(writer, sheet_name="SMS_Summary", index=False)
            df_clean_summary.to_excel(writer, sheet_name="Clean_AUPRC", index=False)
            df_method_average.to_excel(
                writer, sheet_name="Method_Averages", index=False
            )

            used = set()
            short_arch = {
                "XGBoost": "XGB",
                "LightGBM": "LGB",
                "Tabular SNN": "SNN",
                "Tabular ResNet": "RES",
            }
            short_method = {
                "Clean": "Clean",
                "Unconst-AT": "UAT",
                "Const-AT": "CAT",
                "Bag": "Bag",
                "Bag+ColLev": "ColLevBag",
                "Keep-P-Sample-ZB-Bag": "UniBag",
                "Keep-P-Sample-ZB-Bag+ColLev": "UniColBag",
            }

            for model_name in EVAL_MODELS:
                arch = next(a for a in TARGET_MODELS if model_name.startswith(a + " ("))
                display = model_name[len(arch) + 2 : -1]

                base_sheet = (
                    f"{short_arch.get(arch, arch[:4])}_" f"{short_method[display]}_PRC"
                )[:31]

                sheet = base_sheet
                k = 1
                while sheet in used:
                    suffix = f"_{k}"
                    sheet = base_sheet[: 31 - len(suffix)] + suffix
                    k += 1
                used.add(sheet)

                pd.DataFrame(
                    prc_const_matrices[model_name],
                    index=[f"rho={r}" for r in RHO_LIST],
                    columns=[f"lam={lam}" for lam in LAMBDA_LIST],
                ).to_excel(writer, sheet_name=sheet)

            workbook = writer.book
            fmt_header = workbook.add_format(
                {"bold": True, "bg_color": "#D3D3D3", "border": 1, "align": "center"}
            )
            fmt_cells = workbook.add_format(
                {"border": 1, "align": "center", "num_format": "0.0000"}
            )

            for ws_name, df_out in [
                ("SMS_Summary", df_sms_summary),
                ("Clean_AUPRC", df_clean_summary),
                ("Method_Averages", df_method_average),
            ]:
                ws = writer.sheets[ws_name]
                for col_num, value in enumerate(df_out.columns.values):
                    ws.write(0, col_num, value, fmt_header)
                ws.set_column(0, 0, 22, fmt_cells)
                ws.set_column(1, len(df_out.columns) - 1, 18, fmt_cells)

        print("Success!")

    except PermissionError:
        root, ext = os.path.splitext(output_path)
        fallback = root + "_new" + ext
        print(f"File is locked; saving to: {fallback} ... ", end="")

        with pd.ExcelWriter(fallback, engine="xlsxwriter") as writer:
            df_sms_summary.to_excel(writer, sheet_name="SMS_Summary", index=False)
            df_clean_summary.to_excel(writer, sheet_name="Clean_AUPRC", index=False)
            df_method_average.to_excel(
                writer, sheet_name="Method_Averages", index=False
            )

        print("Success!")

    except Exception as e:
        print(f"Failed! Error: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate instance-specific adversarial robustness for FSFD models.

The script implements an accounting-consistent, sample-specific PGD attack
with differentiable articulation rules and a hard top-k sparsity projection.
"""

import copy
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Intel.*")
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"


# =============================================================================
# 1. Hyperparameters and model configuration
# =============================================================================

# File paths
DATA_PATH = "finance_data_smoothed.xlsx"
STOCK_NAME_COLUMN = "\u80a1\u7968\u7b80\u79f0"
OUTPUT_DIR = "results"
OUTPUT_FILE = "specific_attack_results.xlsx"

# Models and training
TARGET_MODELS = ["XGBoost", "LightGBM", "Tabular SNN", "Tabular ResNet"]
SUBSTITUTE_MODEL_TYPE = "Tabular SNN"
MINORITY_CLASS_WEIGHT = 5.0  # Positive-class weight used by every model.

# Attack-constraint grid for the three-dimensional security surface
RHO_LIST = [0.0, 0.01, 0.02, 0.05, 0.10]
LAMBDA_LIST = [0, 5, 10, 15, 20, 25]

# PGD settings
PGD_STEPS = 300
PGD_ALPHA_MULTIPLIER = 1.5  # Scale factor for sign-based PGD steps.

# Neural-network training settings
NN_HIDDEN_DIM = 128
NN_EPOCHS = 30
NN_LR = 1e-3
NN_BATCH_SIZE = 256

# Data split and random seeds
SPLIT_1_TEST_SIZE = 0.4
SPLIT_1_RANDOM_STATE = 11
GLOBAL_RANDOM_STATE = 1

# Indices of manipulable free variables
FREE_VAL_INDICES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    22,
    23,
    29,
    30,
    39,
    45,
    46,
    48,
    56,
    58,
    61,
    62,
    64,
    70,
    71,
    78,
    81,
    82,
    84,
    85,
]
FREE_VALUE_INDICES = np.array(FREE_VAL_INDICES)
NUM_FREE_VALUES = len(FREE_VALUE_INDICES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. PyTorch model architectures and scikit-learn-compatible wrapper
# =============================================================================


class TabularNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=NN_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


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
        self.dropout = nn.Dropout(0.2)
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
            nn.AlphaDropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class PyTorchTabularClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_type="resnet",
        input_dim=89,
        hidden_dim=NN_HIDDEN_DIM,
        epochs=NN_EPOCHS,
        lr=NN_LR,
        batch_size=NN_BATCH_SIZE,
        random_state=42,
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

        if self.model_type == "resnet":
            self.model = TabularResNet(self.input_dim, self.hidden_dim).to(DEVICE)
        elif self.model_type == "snn":
            self.model = TabularSNN(self.input_dim, self.hidden_dim).to(DEVICE)
        elif self.model_type == "nn":
            self.model = TabularNN(self.input_dim, self.hidden_dim).to(DEVICE)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(batch_X).view(-1)
                loss = -(
                    batch_y * torch.log(out + 1e-7) * MINORITY_CLASS_WEIGHT
                    + (1 - batch_y) * torch.log(1 - out + 1e-7)
                ).mean()
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            probs = self.model(X_tensor).view(-1).cpu().numpy()
        return np.vstack([1 - probs, probs]).T


def get_classifier(model_type, input_dim=89, random_state=GLOBAL_RANDOM_STATE):
    class_weight_dict = {0: 1, 1: MINORITY_CLASS_WEIGHT}

    if model_type == "XGBoost":
        return XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=MINORITY_CLASS_WEIGHT,
            random_state=random_state,
        )
    elif model_type == "LightGBM":
        return lgb.LGBMClassifier(
            random_state=random_state, class_weight=class_weight_dict
        )
    elif model_type == "Tabular SNN":
        return PyTorchTabularClassifier(
            model_type="snn", input_dim=input_dim, random_state=random_state
        )
    elif model_type == "Tabular ResNet":
        return PyTorchTabularClassifier(
            model_type="resnet", input_dim=input_dim, random_state=random_state
        )
    elif model_type == "Tabular NN":
        return PyTorchTabularClassifier(
            model_type="nn", input_dim=input_dim, random_state=random_state
        )
    elif model_type == "DecisionTree":
        return DecisionTreeClassifier(
            random_state=random_state, class_weight=class_weight_dict
        )
    elif model_type == "LR":
        return LogisticRegression(
            random_state=random_state, class_weight=class_weight_dict, max_iter=1000
        )
    elif model_type == "SVC":
        return SVC(
            probability=True, random_state=random_state, class_weight=class_weight_dict
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# =============================================================================
# 3. Accounting rules and differentiable Jacobian propagation
# =============================================================================


# NumPy implementation used to validate the final adversarial samples.
def compute_perturbed_features(X, r_free):
    X = copy.deepcopy(X)
    max_index = 89
    r = np.zeros(max_index, dtype=float)
    for idx, val in r_free.items():
        r[idx] = val

    try:

        def safe_divide(a, b):
            return a / b if b != 0 else 0

        r[41] = r[0]
        r[12] = r[0]
        r[10] = r[0] + r[78] - r[1] - r[2] - r[3]
        r[20] = X[10] + r[10] - X[20]
        r[18] = r[20]
        r[37] = r[10]
        r[74] = r[0] + r[10]
        r[33] = r[7]
        r[63] = r[7]
        r[67] = r[7]
        r[64] = 0
        r[51] = r[7] + r[64]
        r[66] = r[4] + r[6] + r[48]
        r[8] = r[5] + r[56] + r[66]
        r[31] = r[66] - r[7] - r[6]
        r[14] = X[8] + r[5] + r[56] + r[66] - X[51] - r[51] - X[14]
        r[17] = r[51] - r[6]

        r[11] = safe_divide(X[10] + r[10], X[0] + r[0]) - X[11]
        r[15] = safe_divide(X[10] + r[10], X[8] + r[8]) - X[15]
        r[24] = safe_divide(X[8] + r[8], X[0] + r[0]) - X[24]
        r[25] = safe_divide(X[6] + r[6], X[0] + r[0]) - X[25]
        r[26] = safe_divide(X[31] + r[31], X[0] + r[0]) - X[26]
        r[27] = safe_divide(X[5] + r[5], X[8] + r[8]) - X[27]
        r[16] = safe_divide(X[17] + r[17], X[18] + r[18]) - X[16]
        r[19] = safe_divide(X[17] + r[17], X[20] + r[20]) - X[19]
        r[13] = safe_divide(X[10] + r[10], X[14] + r[14]) - X[13]

        if (X[23] + r[23]) != 0:
            r[21] = (X[22] + r[22] - X[23] - r[23]) / (X[23] + r[23]) - X[21]
        if (X[30] + r[30]) != 0:
            r[28] = (X[29] + r[29] - X[30] - r[30]) / (X[30] + r[30]) - X[28]

        r[32] = safe_divide(X[33] + r[33], X[8] + r[8]) - X[32]
        r[34] = safe_divide(X[35] + r[10], X[37] + r[37]) - X[34]
        if X[36] != 0 and (X[8] + r[8] - X[51] - r[51]) != 0:
            r[38] = X[35] / 10000000 / ((X[8] + r[8] - X[51] - r[51]) / X[36]) - X[38]
        r[40] = safe_divide(X[39] + r[39], X[41] + r[41]) - X[40]

        if (X[8] + r[8]) != 0:
            r[42] = safe_divide(X[4] + r[4], X[8] + r[8]) - X[42]
        elif (X[0] + r[0]) != 0:
            r[42] = safe_divide(X[4] + r[4], X[0] + r[0]) - X[42]

        if (X[46] + r[46]) != 0:
            r[44] = (X[45] + r[45] - X[46] - r[46]) / (X[46] + r[46]) - X[44]

        r[47] = safe_divide(X[48] + r[48], X[8] + r[8]) - X[47]
        r[50] = safe_divide(X[7] + r[7], X[51] + r[51]) - X[50]

        if (X[54] + r[54]) != 0:
            r[52] = (X[53] + r[53] - X[54] - r[54]) / (X[54] + r[54]) - X[52]

        r[55] = safe_divide(X[56] + r[56], X[8] + r[8]) - X[55]

        if (X[8] + r[8]) != 0:
            r[57] = safe_divide(X[58] + r[58], X[8] + r[8]) - X[57]
        elif (X[0] + r[0]) != 0:
            r[57] = safe_divide(X[58] + r[58], X[0] + r[0]) - X[57]

        r[59] = r[0] + r[48]
        if (X[61] + r[61] + r[6] + X[62] + r[62]) != 0:
            r[60] = (
                safe_divide(X[0] + r[0], (X[61] + r[61] + r[6] + X[62] + r[62]) / 2)
                - X[60]
            )

        if (X[8] + r[8]) != 0:
            r[63] = safe_divide(X[64] + r[64], X[8] + r[8]) - X[63]
        elif (X[51] + r[51]) != 0:
            r[63] = safe_divide(X[64] + r[64], X[51] + r[51]) - X[63]

        r[65] = safe_divide(X[66] + r[66], X[67] + r[67]) - X[65]
        r[68] = safe_divide(X[51] + r[51], X[8] + r[8]) - X[68]

        if (X[71] + r[71]) != 0:
            r[69] = (X[70] + r[70] - X[71] - r[71]) / (X[71] + r[71]) - X[69]

        r[72] = safe_divide(X[74] + r[74], X[0] + r[0]) - X[72]
        r[73] = safe_divide(X[74] + r[74], X[10] + r[10]) - X[73]
        r[75] = safe_divide(X[6] + r[6], X[8] + r[8]) - X[75]
        r[76] = safe_divide(X[17] + r[17], X[0] + r[0]) - X[76]
        r[77] = safe_divide(X[78] + r[78], X[0] + r[0]) - X[77]

        if (X[82] + r[82]) != 0:
            r[80] = (X[81] + r[81] - X[82] - r[82]) / (X[82] + r[82]) - X[80]
        if (X[85] + r[85]) != 0:
            r[83] = (X[84] + r[84] - X[85] - r[85]) / (X[85] + r[85]) - X[83]

        r[88] = safe_divide(X[10] + r[10], X[8] + r[8]) - X[88]

    except Exception as e:
        raise ValueError(f"Error in NumPy propagation: {e}") from e

    return r


# Differentiable PyTorch implementation for accounting-adjusted gradients.
def compute_perturbed_features_torch(X_real, r_free_tensor, free_val_indices):
    """
    X_real: (1, 89) Tensor, real scale data.
    r_free_tensor: (1, NUM_FREE_VALUES) tensor of free-variable perturbations.
    """
    r = torch.zeros_like(X_real)
    r[0, free_val_indices] = r_free_tensor[0]

    def safe_divide(a, b):
        return a / (b + 1e-8 * torch.sign(b + 1e-15))

    r[:, 41] = r[:, 0]
    r[:, 12] = r[:, 0]
    r[:, 10] = r[:, 0] + r[:, 78] - r[:, 1] - r[:, 2] - r[:, 3]
    r[:, 20] = X_real[:, 10] + r[:, 10] - X_real[:, 20]
    r[:, 18] = r[:, 20]
    r[:, 37] = r[:, 10]
    r[:, 74] = r[:, 0] + r[:, 10]
    r[:, 33] = r[:, 7]
    r[:, 63] = r[:, 7]
    r[:, 67] = r[:, 7]
    r[:, 64] = 0.0
    r[:, 51] = r[:, 7] + r[:, 64]
    r[:, 66] = r[:, 4] + r[:, 6] + r[:, 48]
    r[:, 8] = r[:, 5] + r[:, 56] + r[:, 66]
    r[:, 31] = r[:, 66] - r[:, 7] - r[:, 6]
    r[:, 14] = (
        X_real[:, 8]
        + r[:, 5]
        + r[:, 56]
        + r[:, 66]
        - X_real[:, 51]
        - r[:, 51]
        - X_real[:, 14]
    )
    r[:, 17] = r[:, 51] - r[:, 6]

    r[:, 11] = (
        safe_divide(X_real[:, 10] + r[:, 10], X_real[:, 0] + r[:, 0]) - X_real[:, 11]
    )
    r[:, 15] = (
        safe_divide(X_real[:, 10] + r[:, 10], X_real[:, 8] + r[:, 8]) - X_real[:, 15]
    )
    r[:, 24] = (
        safe_divide(X_real[:, 8] + r[:, 8], X_real[:, 0] + r[:, 0]) - X_real[:, 24]
    )
    r[:, 25] = (
        safe_divide(X_real[:, 6] + r[:, 6], X_real[:, 0] + r[:, 0]) - X_real[:, 25]
    )
    r[:, 26] = (
        safe_divide(X_real[:, 31] + r[:, 31], X_real[:, 0] + r[:, 0]) - X_real[:, 26]
    )
    r[:, 27] = (
        safe_divide(X_real[:, 5] + r[:, 5], X_real[:, 8] + r[:, 8]) - X_real[:, 27]
    )
    r[:, 16] = (
        safe_divide(X_real[:, 17] + r[:, 17], X_real[:, 18] + r[:, 18]) - X_real[:, 16]
    )
    r[:, 19] = (
        safe_divide(X_real[:, 17] + r[:, 17], X_real[:, 20] + r[:, 20]) - X_real[:, 19]
    )
    r[:, 13] = (
        safe_divide(X_real[:, 10] + r[:, 10], X_real[:, 14] + r[:, 14]) - X_real[:, 13]
    )

    cond23 = torch.abs(X_real[:, 23] + r[:, 23]) > 1e-8
    r[:, 21] = torch.where(
        cond23,
        (X_real[:, 22] + r[:, 22] - X_real[:, 23] - r[:, 23])
        / (X_real[:, 23] + r[:, 23])
        - X_real[:, 21],
        r[:, 21],
    )

    cond30 = torch.abs(X_real[:, 30] + r[:, 30]) > 1e-8
    r[:, 28] = torch.where(
        cond30,
        (X_real[:, 29] + r[:, 29] - X_real[:, 30] - r[:, 30])
        / (X_real[:, 30] + r[:, 30])
        - X_real[:, 28],
        r[:, 28],
    )

    r[:, 32] = (
        safe_divide(X_real[:, 33] + r[:, 33], X_real[:, 8] + r[:, 8]) - X_real[:, 32]
    )
    r[:, 34] = (
        safe_divide(X_real[:, 35] + r[:, 10], X_real[:, 37] + r[:, 37]) - X_real[:, 34]
    )

    cond38 = (torch.abs(X_real[:, 36]) > 1e-8) & (
        torch.abs(X_real[:, 8] + r[:, 8] - X_real[:, 51] - r[:, 51]) > 1e-8
    )
    r[:, 38] = torch.where(
        cond38,
        (X_real[:, 35] / 1e7)
        / ((X_real[:, 8] + r[:, 8] - X_real[:, 51] - r[:, 51]) / (X_real[:, 36] + 1e-8))
        - X_real[:, 38],
        r[:, 38],
    )

    r[:, 40] = (
        safe_divide(X_real[:, 39] + r[:, 39], X_real[:, 41] + r[:, 41]) - X_real[:, 40]
    )

    cond42_1 = torch.abs(X_real[:, 8] + r[:, 8]) > 1e-8
    cond42_2 = torch.abs(X_real[:, 0] + r[:, 0]) > 1e-8
    r[:, 42] = torch.where(
        cond42_1,
        safe_divide(X_real[:, 4] + r[:, 4], X_real[:, 8] + r[:, 8]) - X_real[:, 42],
        torch.where(
            cond42_2,
            safe_divide(X_real[:, 4] + r[:, 4], X_real[:, 0] + r[:, 0]) - X_real[:, 42],
            r[:, 42],
        ),
    )

    cond46 = torch.abs(X_real[:, 46] + r[:, 46]) > 1e-8
    r[:, 44] = torch.where(
        cond46,
        (X_real[:, 45] + r[:, 45] - X_real[:, 46] - r[:, 46])
        / (X_real[:, 46] + r[:, 46])
        - X_real[:, 44],
        r[:, 44],
    )

    r[:, 47] = (
        safe_divide(X_real[:, 48] + r[:, 48], X_real[:, 8] + r[:, 8]) - X_real[:, 47]
    )
    r[:, 50] = (
        safe_divide(X_real[:, 7] + r[:, 7], X_real[:, 51] + r[:, 51]) - X_real[:, 50]
    )

    cond54 = torch.abs(X_real[:, 54] + r[:, 54]) > 1e-8
    r[:, 52] = torch.where(
        cond54,
        (X_real[:, 53] + r[:, 53] - X_real[:, 54] - r[:, 54])
        / (X_real[:, 54] + r[:, 54])
        - X_real[:, 52],
        r[:, 52],
    )

    r[:, 55] = (
        safe_divide(X_real[:, 56] + r[:, 56], X_real[:, 8] + r[:, 8]) - X_real[:, 55]
    )

    cond57_1 = torch.abs(X_real[:, 8] + r[:, 8]) > 1e-8
    cond57_2 = torch.abs(X_real[:, 0] + r[:, 0]) > 1e-8
    r[:, 57] = torch.where(
        cond57_1,
        safe_divide(X_real[:, 58] + r[:, 58], X_real[:, 8] + r[:, 8]) - X_real[:, 57],
        torch.where(
            cond57_2,
            safe_divide(X_real[:, 58] + r[:, 58], X_real[:, 0] + r[:, 0])
            - X_real[:, 57],
            r[:, 57],
        ),
    )

    r[:, 59] = r[:, 0] + r[:, 48]

    cond60 = (
        torch.abs(X_real[:, 61] + r[:, 61] + r[:, 6] + X_real[:, 62] + r[:, 62]) > 1e-8
    )
    r[:, 60] = torch.where(
        cond60,
        safe_divide(
            X_real[:, 0] + r[:, 0],
            (X_real[:, 61] + r[:, 61] + r[:, 6] + X_real[:, 62] + r[:, 62]) / 2,
        )
        - X_real[:, 60],
        r[:, 60],
    )

    cond63_1 = torch.abs(X_real[:, 8] + r[:, 8]) > 1e-8
    cond63_2 = torch.abs(X_real[:, 51] + r[:, 51]) > 1e-8
    r[:, 63] = torch.where(
        cond63_1,
        safe_divide(X_real[:, 64] + r[:, 64], X_real[:, 8] + r[:, 8]) - X_real[:, 63],
        torch.where(
            cond63_2,
            safe_divide(X_real[:, 64] + r[:, 64], X_real[:, 51] + r[:, 51])
            - X_real[:, 63],
            r[:, 63],
        ),
    )

    r[:, 65] = (
        safe_divide(X_real[:, 66] + r[:, 66], X_real[:, 67] + r[:, 67]) - X_real[:, 65]
    )
    r[:, 68] = (
        safe_divide(X_real[:, 51] + r[:, 51], X_real[:, 8] + r[:, 8]) - X_real[:, 68]
    )

    cond71 = torch.abs(X_real[:, 71] + r[:, 71]) > 1e-8
    r[:, 69] = torch.where(
        cond71,
        (X_real[:, 70] + r[:, 70] - X_real[:, 71] - r[:, 71])
        / (X_real[:, 71] + r[:, 71])
        - X_real[:, 69],
        r[:, 69],
    )

    r[:, 72] = (
        safe_divide(X_real[:, 74] + r[:, 74], X_real[:, 0] + r[:, 0]) - X_real[:, 72]
    )
    r[:, 73] = (
        safe_divide(X_real[:, 74] + r[:, 74], X_real[:, 10] + r[:, 10]) - X_real[:, 73]
    )
    r[:, 75] = (
        safe_divide(X_real[:, 6] + r[:, 6], X_real[:, 8] + r[:, 8]) - X_real[:, 75]
    )
    r[:, 76] = (
        safe_divide(X_real[:, 17] + r[:, 17], X_real[:, 0] + r[:, 0]) - X_real[:, 76]
    )
    r[:, 77] = (
        safe_divide(X_real[:, 78] + r[:, 78], X_real[:, 0] + r[:, 0]) - X_real[:, 77]
    )

    cond82 = torch.abs(X_real[:, 82] + r[:, 82]) > 1e-8
    r[:, 80] = torch.where(
        cond82,
        (X_real[:, 81] + r[:, 81] - X_real[:, 82] - r[:, 82])
        / (X_real[:, 82] + r[:, 82])
        - X_real[:, 80],
        r[:, 80],
    )

    cond85 = torch.abs(X_real[:, 85] + r[:, 85]) > 1e-8
    r[:, 83] = torch.where(
        cond85,
        (X_real[:, 84] + r[:, 84] - X_real[:, 85] - r[:, 85])
        / (X_real[:, 85] + r[:, 85])
        - X_real[:, 83],
        r[:, 83],
    )

    r[:, 88] = (
        safe_divide(X_real[:, 10] + r[:, 10], X_real[:, 8] + r[:, 8]) - X_real[:, 88]
    )

    return X_real + r


# Accounting-consistent sample-specific attack (ACSAA) with sign updates.
def generate_specific_perturbation(
    clf_sub, sample, scaler, rho, lam_change, pgd_steps=PGD_STEPS
):
    if lam_change == 0 or rho == 0.0:
        return np.zeros(NUM_FREE_VALUES)

    sample_inv = scaler.inverse_transform(sample.reshape(1, -1))[0]
    sample_up = sample_inv * (1 + rho)
    sample_low = sample_inv * (1 - rho)

    up_bounds = scaler.transform(sample_up.reshape(1, -1))[0]
    low_bounds = scaler.transform(sample_low.reshape(1, -1))[0]

    swap_mask = low_bounds > up_bounds
    low_bounds[swap_mask], up_bounds[swap_mask] = (
        up_bounds[swap_mask],
        low_bounds[swap_mask],
    )

    model = clf_sub.model
    model.eval()

    # Prepare tensors for standardization and inverse standardization.
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=DEVICE)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=DEVICE)
    X_orig_real = torch.tensor(
        sample_inv, dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)

    # Optimize free-variable perturbations in standardized coordinates.
    delta_free = torch.zeros(
        NUM_FREE_VALUES, dtype=torch.float32, requires_grad=True, device=DEVICE
    )

    U_free = torch.tensor(
        up_bounds[FREE_VALUE_INDICES], dtype=torch.float32, device=DEVICE
    )
    L_free = torch.tensor(
        low_bounds[FREE_VALUE_INDICES], dtype=torch.float32, device=DEVICE
    )
    bound_range = U_free - L_free

    # Set the sign-step size as a fraction of the feasible interval.
    alpha = (bound_range / pgd_steps) * PGD_ALPHA_MULTIPLIER
    alpha = torch.clamp(alpha, min=1e-4)

    k = int(lam_change)

    for _ in range(pgd_steps):
        x_adv_scaled = torch.tensor(
            sample, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        # Build a differentiable forward pass through the free variables.
        x_adv_scaled_free_part = x_adv_scaled[0, FREE_VALUE_INDICES] + delta_free
        x_adv_scaled = x_adv_scaled.clone()
        x_adv_scaled[0, FREE_VALUE_INDICES] = x_adv_scaled_free_part

        # Convert standardized features back to their original units.
        x_adv_real = x_adv_scaled * scaler_scale + scaler_mean
        r_free_real = (
            x_adv_real[0, FREE_VALUE_INDICES] - X_orig_real[0, FREE_VALUE_INDICES]
        )

        # Propagate the perturbation through the accounting rules.
        x_adv_real_constrained = compute_perturbed_features_torch(
            X_orig_real, r_free_real.unsqueeze(0), FREE_VALUE_INDICES
        )

        # Re-standardize the accounting-consistent sample for the network.
        x_final_scaled = (x_adv_real_constrained - scaler_mean) / scaler_scale

        out = model(x_final_scaled).view(-1)
        model.zero_grad()
        if delta_free.grad is not None:
            delta_free.grad.zero_()

        out.backward()
        grad = delta_free.grad

        # Apply sign-based PGD and hard top-k projection.
        with torch.no_grad():
            if k < NUM_FREE_VALUES:
                # Retain the k largest accounting-adjusted gradients.
                _, topk_indices = torch.topk(torch.abs(grad), k)
                mask = torch.zeros_like(grad, dtype=torch.bool)
                mask[topk_indices] = True
                grad = torch.where(mask, grad, torch.zeros_like(grad))

            # Move in the direction that lowers the fraud score.
            delta_free -= alpha * torch.sign(grad)

            # Project onto the box constraints.
            current_free_scaled = (
                torch.tensor(
                    sample[FREE_VALUE_INDICES], dtype=torch.float32, device=DEVICE
                )
                + delta_free
            )
            current_free_scaled = torch.max(
                torch.min(current_free_scaled, U_free), L_free
            )
            delta_free.copy_(
                current_free_scaled
                - torch.tensor(
                    sample[FREE_VALUE_INDICES], dtype=torch.float32, device=DEVICE
                )
            )

            # Apply the dynamic L0 sparsity projection P_lambda.
            if k < NUM_FREE_VALUES:
                _, topk_dev_indices = torch.topk(torch.abs(delta_free), k)
                sparse_mask = torch.zeros_like(delta_free, dtype=torch.bool)
                sparse_mask[topk_dev_indices] = True
                delta_free.copy_(
                    torch.where(sparse_mask, delta_free, torch.zeros_like(delta_free))
                )

        delta_free.requires_grad_(True)

    return delta_free.detach().cpu().numpy()


def calculate_ari_metric(rho_list, lambda_list, metric_matrix):
    base_score = metric_matrix[0, 0]
    if base_score == 0:
        return 0.0, 0.0

    sms_sum = 0.0
    for i in range(1, len(rho_list)):
        for j in range(1, len(lambda_list)):
            d_rho = rho_list[i] - rho_list[i - 1]
            d_lambda = lambda_list[j] - lambda_list[j - 1]
            cell_volume = (
                d_rho * d_lambda * (metric_matrix[i, j] + metric_matrix[i - 1, j - 1])
            )
            sms_sum += cell_volume / 2.0

    sms = sms_sum / (rho_list[-1] * lambda_list[-1])
    ari = (sms**2) / base_score
    return sms, ari


# =============================================================================
# 4. Data loading and preprocessing
# =============================================================================


def main():
    print("Loading data and configuring parameters...")
    try:
        df = pd.read_excel(DATA_PATH).drop(columns=STOCK_NAME_COLUMN, errors="ignore")
    except FileNotFoundError as exc:
        raise SystemExit(f"Input file not found: {DATA_PATH}") from exc

    data = np.asarray(df)
    features, y = data[:, :-1], data[:, -1]
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=SPLIT_1_TEST_SIZE, random_state=SPLIT_1_RANDOM_STATE
    )
    for train_index, test_index in sss.split(features, y):
        X_tr, y_tr = features[train_index], y[train_index]
        X_te, y_te = features[test_index], y[test_index]

        X_te_P = X_te[y_te == 1]
        X_te_N = X_te[y_te == 0]

    # =============================================================================
    # 5. Train target and substitute models
    # =============================================================================

    print(f"\nTraining {len(TARGET_MODELS)} Target Models for evaluation...")
    input_dim = X_tr.shape[1]

    clf_targets = {}
    for name in TARGET_MODELS:
        print(f"  -> Training {name}...")
        clf = get_classifier(
            name, input_dim=input_dim, random_state=GLOBAL_RANDOM_STATE
        )
        clf.fit(X_tr, y_tr)
        clf_targets[name] = clf

    print(f"\nTraining Substitute Model ({SUBSTITUTE_MODEL_TYPE})...")
    clf_substitute = get_classifier(
        SUBSTITUTE_MODEL_TYPE, input_dim=input_dim, random_state=0
    )
    clf_substitute.fit(X_tr, y_tr)

    # =============================================================================
    # 6. Evaluate the three-dimensional security surface
    # =============================================================================

    print("\nStarting 3D Security Evaluation Surface Scan (Metric: AUPRC)...")
    print(f"Target Models: {TARGET_MODELS} \nSubstitute Model: {SUBSTITUTE_MODEL_TYPE}")

    prc_const_matrices = {
        name: np.zeros((len(RHO_LIST), len(LAMBDA_LIST))) for name in TARGET_MODELS
    }
    prc_unconst_matrices = {
        name: np.zeros((len(RHO_LIST), len(LAMBDA_LIST))) for name in TARGET_MODELS
    }
    grid_details_records = []

    y_te_combined = np.concatenate([np.zeros(len(X_te_N)), np.ones(len(X_te_P))])

    base_prcs = {}
    for name, clf in clf_targets.items():
        base_probs_P = clf.predict_proba(X_te_P)[:, 1]
        base_probs_N = clf.predict_proba(X_te_N)[:, 1]

        base_probs_combined = np.concatenate([base_probs_N, base_probs_P])
        base_prcs[name] = average_precision_score(y_te_combined, base_probs_combined)
        print(f">>> Base AUPRC for {name:15s}: {base_prcs[name]:.4f}")

    for i, rho in enumerate(RHO_LIST):
        for j, lam in enumerate(LAMBDA_LIST):

            if rho == 0.0 or lam == 0:
                for name in TARGET_MODELS:
                    prc_const_matrices[name][i, j] = base_prcs[name]
                    prc_unconst_matrices[name][i, j] = base_prcs[name]
                    grid_details_records.append(
                        {
                            "Rho": rho,
                            "Lambda": lam,
                            "Target Model": name,
                            "Constrained AUPRC": base_prcs[name],
                            "Unconstrained AUPRC": base_prcs[name],
                        }
                    )
                continue

            print(
                f"\nRunning Grid (rho={rho:.2f}, lambda={lam:2d}) with "
                "accounting-adjusted PGD attacks (sign operator)..."
            )

            adv_unconst_X_list = []
            constrained_X_P_list = []

            for k in range(len(X_te_P)):
                ori = X_te_P[k]

                # Generate free-variable perturbations with the differentiable engine.
                specific_perturbation_free = generate_specific_perturbation(
                    clf_substitute, ori, scaler, rho, lam
                )

                specific_perturbation_full = np.zeros(X_te_P.shape[1])
                specific_perturbation_full[FREE_VALUE_INDICES] = (
                    specific_perturbation_free
                )

                adv_unconst = ori + specific_perturbation_full
                adv_unconst_X_list.append(adv_unconst)

                # Reconstruct and validate the sample with the exact NumPy rules.
                ori_inverse = scaler.inverse_transform(ori.reshape(1, -1))[0]
                random_vector = (
                    scaler.inverse_transform(adv_unconst.reshape(1, -1))[0]
                    - ori_inverse
                )
                r_free = {
                    idx: random_vector[m] for m, idx in enumerate(FREE_VALUE_INDICES)
                }

                try:
                    r = compute_perturbed_features(ori_inverse, r_free)
                    adv_real = ori_inverse + r
                    adv_real = scaler.transform(adv_real.reshape(1, -1))[0]
                    constrained_X_P_list.append(adv_real)
                except Exception:
                    constrained_X_P_list.append(ori)

            adv_unconst_X = np.array(adv_unconst_X_list)
            constrained_X_P = np.array(constrained_X_P_list)

            for name, clf in clf_targets.items():
                probs_N = clf.predict_proba(X_te_N)[:, 1]

                unconst_probs_P = clf.predict_proba(adv_unconst_X)[:, 1]
                unconst_probs_combined = np.concatenate([probs_N, unconst_probs_P])
                prc_unconst_matrices[name][i, j] = average_precision_score(
                    y_te_combined, unconst_probs_combined
                )

                const_probs_P = clf.predict_proba(constrained_X_P)[:, 1]
                const_probs_combined = np.concatenate([probs_N, const_probs_P])
                prc_const_matrices[name][i, j] = average_precision_score(
                    y_te_combined, const_probs_combined
                )

                print(
                    f"  [{name:15s}] Const AUPRC: "
                    f"{prc_const_matrices[name][i, j]:.4f} | Unconst AUPRC: "
                    f"{prc_unconst_matrices[name][i, j]:.4f}"
                )

                grid_details_records.append(
                    {
                        "Rho": rho,
                        "Lambda": lam,
                        "Target Model": name,
                        "Constrained AUPRC": prc_const_matrices[name][i, j],
                        "Unconstrained AUPRC": prc_unconst_matrices[name][i, j],
                    }
                )

    # =============================================================================
    # 7. Compute summary metrics and save the results
    # =============================================================================

    print("\n" + "=" * 90)
    print("FINAL EVALUATION METRICS (Accounting-Adjusted AUPRC)")
    print("=" * 90)

    final_metrics = []
    for name in TARGET_MODELS:
        sms_const, _ = calculate_ari_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[name]
        )
        sms_unconst, _ = calculate_ari_metric(
            RHO_LIST, LAMBDA_LIST, prc_unconst_matrices[name]
        )

        final_metrics.append(
            {
                "Target Model": name,
                "Base AUPRC": base_prcs[name],
                "Constrained SMS": sms_const,
                "Unconstrained SMS": sms_unconst,
            }
        )

    df_summary = pd.DataFrame(final_metrics)
    pd.options.display.float_format = "{:.4f}".format
    print(df_summary.to_string(index=False))

    df_grid_details = pd.DataFrame(grid_details_records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    print(f"\nSaving results to: {output_path} ... ", end="")
    try:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            df_grid_details.to_excel(writer, sheet_name="Grid_Details", index=False)

            for name in TARGET_MODELS:
                safe_name = name.replace(" ", "_").replace(".", "")[:20]
                pd.DataFrame(
                    prc_const_matrices[name],
                    index=[f"rho={r}" for r in RHO_LIST],
                    columns=[f"lam={lam}" for lam in LAMBDA_LIST],
                ).to_excel(writer, sheet_name=f"{safe_name}_C_PRC")
                pd.DataFrame(
                    prc_unconst_matrices[name],
                    index=[f"rho={r}" for r in RHO_LIST],
                    columns=[f"lam={lam}" for lam in LAMBDA_LIST],
                ).to_excel(writer, sheet_name=f"{safe_name}_U_PRC")

            workbook = writer.book
            fmt_header = workbook.add_format(
                {"bold": True, "bg_color": "#D3D3D3", "border": 1, "align": "center"}
            )
            fmt_cells = workbook.add_format(
                {"border": 1, "align": "center", "num_format": "0.0000"}
            )

            ws = writer.sheets["Summary"]
            for col_num, value in enumerate(df_summary.columns.values):
                ws.write(0, col_num, value, fmt_header)
            ws.set_column("A:A", 20, fmt_cells)
            ws.set_column("B:G", 18, fmt_cells)

            ws_grid = writer.sheets["Grid_Details"]
            for col_num, value in enumerate(df_grid_details.columns.values):
                ws_grid.write(0, col_num, value, fmt_header)
            ws_grid.set_column("A:B", 10, fmt_cells)
            ws_grid.set_column("C:C", 20, fmt_cells)
            ws_grid.set_column("D:E", 20, fmt_cells)

        print("Success!")
    except Exception as e:
        print(f"Failed! Error: {e}")


if __name__ == "__main__":
    main()

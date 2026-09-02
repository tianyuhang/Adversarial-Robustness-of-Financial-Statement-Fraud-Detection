#!/usr/bin/env python3
"""Evaluate universal adversarial robustness for FSFD models.

The script learns one accounting-consistent universal perturbation with PGD
and evaluates its transferability across multiple target architectures.
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
DATA_PATH = "finance_data2.xlsx"
STOCK_NAME_COLUMN = "\u80a1\u7968\u7b80\u79f0"
OUTPUT_DIR = "results"
OUTPUT_FILE = "universal_attack_results.xlsx"

# Models and training
TARGET_MODELS = ["XGBoost", "LightGBM", "Tabular SNN", "Tabular ResNet"]
SUBSTITUTE_MODEL_TYPE = "Tabular SNN"
MINORITY_CLASS_WEIGHT = 5.0  # Positive-class weight used by every model.

# Attack-constraint grid for the three-dimensional security surface
RHO_LIST = [0.0, 0.01, 0.02, 0.05, 0.10]
LAMBDA_LIST = [0, 5, 10, 15, 20, 25]

# PGD settings
PGD_STEPS = 1000
PGD_ALPHA_MULTIPLIER = 1.5
UAP_POOL_SIZE = 500  # Maximum number of training samples used to learn the UAP.

# Neural-network training settings
NN_HIDDEN_DIM = 128
NN_EPOCHS = 30
NN_LR = 1e-3
NN_BATCH_SIZE = 256

# Data split and random seeds
SPLIT_1_TEST_SIZE = 0.3
SPLIT_1_RANDOM_STATE = 111
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

        self.model = (
            TabularResNet(self.input_dim, self.hidden_dim).to(DEVICE)
            if self.model_type == "resnet"
            else TabularSNN(self.input_dim, self.hidden_dim).to(DEVICE)
        )
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
                # Use the same positive-class weight across all architectures.
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


# =============================================================================
# 3. Core functions and model factory
# =============================================================================


def get_classifier(model_type, input_dim=89, random_state=GLOBAL_RANDOM_STATE):
    # Use a consistent class-weight definition across model families.
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


def compute_perturbed_features(X, r_free):
    X = copy.deepcopy(X)
    max_index = 89
    r = np.zeros(max_index, dtype=float)

    for idx, val in r_free.items():
        if idx >= len(r):
            raise ValueError(f"Free-variable index {idx} is out of range")
        r[idx] = val

    try:

        def safe_divide(a, b):
            if b == 0:
                raise ZeroDivisionError
            return a / b

        r[41] = r[0]
        r[12] = r[0]
        r[10] = r[0] + r[78] - r[1] - r[2] - r[3]
        r[20] = X[10] + r[10] - X[20] if len(X) > 20 else 0
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
        r[14] = (
            X[8] + r[5] + r[56] + r[66] - X[51] - r[51] - X[14] if len(X) > 14 else 0
        )
        r[17] = r[51] - r[6]

        r[11] = safe_divide(X[10] + r[10], X[0] + r[0]) - X[11] if len(X) > 11 else 0
        r[15] = safe_divide(X[10] + r[10], X[8] + r[8]) - X[15] if len(X) > 15 else 0
        r[24] = safe_divide(X[8] + r[8], X[0] + r[0]) - X[24] if len(X) > 24 else 0
        r[25] = safe_divide(X[6] + r[6], X[0] + r[0]) - X[25] if len(X) > 25 else 0
        r[26] = safe_divide(X[31] + r[31], X[0] + r[0]) - X[26] if len(X) > 26 else 0
        r[27] = safe_divide(X[5] + r[5], X[8] + r[8]) - X[27] if len(X) > 27 else 0
        r[16] = safe_divide(X[17] + r[17], X[18] + r[18]) - X[16] if len(X) > 16 else 0
        r[19] = safe_divide(X[17] + r[17], X[20] + r[20]) - X[19] if len(X) > 19 else 0
        r[13] = safe_divide(X[10] + r[10], X[14] + r[14]) - X[13] if len(X) > 13 else 0

        if len(X) > 23 and (X[23] + r[23]) != 0:
            r[21] = (X[22] + r[22] - X[23] - r[23]) / (X[23] + r[23]) - X[21]
        if len(X) > 30 and (X[30] + r[30]) != 0:
            r[28] = (X[29] + r[29] - X[30] - r[30]) / (X[30] + r[30]) - X[28]

        r[32] = safe_divide(X[33] + r[33], X[8] + r[8]) - X[32] if len(X) > 32 else 0
        r[34] = safe_divide(X[35] + r[10], X[37] + r[37]) - X[34] if len(X) > 34 else 0
        if len(X) > 38 and X[36] != 0 and (X[8] + r[8] - X[51] - r[51]) != 0:
            r[38] = X[35] / 10000000 / ((X[8] + r[8] - X[51] - r[51]) / X[36]) - X[38]
        r[40] = safe_divide(X[39] + r[39], X[41] + r[41]) - X[40] if len(X) > 40 else 0

        if len(X) > 42:
            if (X[8] + r[8]) != 0:
                r[42] = safe_divide(X[4] + r[4], X[8] + r[8]) - X[42]
            elif (X[0] + r[0]) != 0:
                r[42] = safe_divide(X[4] + r[4], X[0] + r[0]) - X[42]

        if len(X) > 46 and (X[46] + r[46]) != 0:
            r[44] = (X[45] + r[45] - X[46] - r[46]) / (X[46] + r[46]) - X[44]

        r[47] = safe_divide(X[48] + r[48], X[8] + r[8]) - X[47] if len(X) > 47 else 0
        r[50] = safe_divide(X[7] + r[7], X[51] + r[51]) - X[50] if len(X) > 50 else 0

        if len(X) > 54 and (X[54] + r[54]) != 0:
            r[52] = (X[53] + r[53] - X[54] - r[54]) / (X[54] + r[54]) - X[52]

        r[55] = safe_divide(X[56] + r[56], X[8] + r[8]) - X[55] if len(X) > 55 else 0

        if len(X) > 57:
            if (X[8] + r[8]) != 0:
                r[57] = safe_divide(X[58] + r[58], X[8] + r[8]) - X[57]
            elif (X[0] + r[0]) != 0:
                r[57] = safe_divide(X[58] + r[58], X[0] + r[0]) - X[57]

        r[59] = r[0] + r[48]
        if len(X) > 62 and (X[61] + r[61] + r[6] + X[62] + r[62]) != 0:
            r[60] = (
                safe_divide(X[0] + r[0], (X[61] + r[61] + r[6] + X[62] + r[62]) / 2)
                - X[60]
            )

        if len(X) > 63:
            if (X[8] + r[8]) != 0:
                r[63] = safe_divide(X[64] + r[64], X[8] + r[8]) - X[63]
            elif (X[51] + r[51]) != 0:
                r[63] = safe_divide(X[64] + r[64], X[51] + r[51]) - X[63]

        r[65] = safe_divide(X[66] + r[66], X[67] + r[67]) - X[65] if len(X) > 65 else 0
        r[68] = safe_divide(X[51] + r[51], X[8] + r[8]) - X[68] if len(X) > 68 else 0

        if len(X) > 71 and (X[71] + r[71]) != 0:
            r[69] = (X[70] + r[70] - X[71] - r[71]) / (X[71] + r[71]) - X[69]

        r[72] = safe_divide(X[74] + r[74], X[0] + r[0]) - X[72] if len(X) > 72 else 0
        r[73] = safe_divide(X[74] + r[74], X[10] + r[10]) - X[73] if len(X) > 73 else 0
        r[75] = safe_divide(X[6] + r[6], X[8] + r[8]) - X[75] if len(X) > 75 else 0
        r[76] = safe_divide(X[17] + r[17], X[0] + r[0]) - X[76] if len(X) > 76 else 0
        r[77] = safe_divide(X[78] + r[78], X[0] + r[0]) - X[77] if len(X) > 77 else 0

        if len(X) > 82 and (X[82] + r[82]) != 0:
            r[80] = (X[81] + r[81] - X[82] - r[82]) / (X[82] + r[82]) - X[80]
        if len(X) > 85 and (X[85] + r[85]) != 0:
            r[83] = (X[84] + r[84] - X[85] - r[85]) / (X[85] + r[85]) - X[83]

        r[88] = safe_divide(X[10] + r[10], X[8] + r[8]) - X[88] if len(X) > 88 else 0

    except IndexError as e:
        raise ValueError(f"The input vector is missing required indices: {e}") from e
    except ZeroDivisionError as e:
        raise ValueError(f"A denominator in the accounting rules is zero: {e}") from e

    return r


def generate_uap_pgd(clf_sub, samples, scaler, mu, lam_change, pgd_steps=PGD_STEPS):
    """Learn one PGD-based universal perturbation from a sample pool."""
    if lam_change == 0 or mu == 0.0:
        return np.zeros(NUM_FREE_VALUES)

    # Recover the original units and derive feasible per-sample bounds.
    samples_up = scaler.inverse_transform(samples) * (1 + mu)
    samples_low = scaler.inverse_transform(samples) * (1 - mu)

    # Convert the bounds back to standardized coordinates.
    up_bounds = scaler.transform(samples_up) - samples
    low_bounds = scaler.transform(samples_low) - samples

    # Use pool medians as global UAP bounds. Replace them with extrema across
    # samples if strict sample-wise feasibility is required.
    up_bound = torch.tensor(
        np.median(up_bounds[:, FREE_VALUE_INDICES], axis=0),
        dtype=torch.float32,
        device=DEVICE,
    )
    low_bound = torch.tensor(
        np.median(low_bounds[:, FREE_VALUE_INDICES], axis=0),
        dtype=torch.float32,
        device=DEVICE,
    )

    # Ensure that every lower bound is no greater than its upper bound.
    swap_mask = low_bound > up_bound
    low_bound[swap_mask], up_bound[swap_mask] = (
        up_bound[swap_mask],
        low_bound[swap_mask],
    )

    if not hasattr(clf_sub, "model"):
        raise ValueError(
            "PGD requires a PyTorch-based substitute model, such as Tabular SNN."
        )

    model = clf_sub.model
    model.eval()

    # Initialize the universal perturbation over free variables only.
    v = torch.zeros(
        NUM_FREE_VALUES,
        dtype=torch.float32,
        requires_grad=True,
        device=DEVICE,
    )
    X_tensor = torch.tensor(samples, dtype=torch.float32).to(DEVICE)

    # Scale each step to the feasible range of its coordinate.
    max_range = up_bound - low_bound
    alpha = (max_range / pgd_steps) * PGD_ALPHA_MULTIPLIER
    alpha[max_range == 0] = 0.0  # Freeze coordinates with identical bounds.

    # Optimize the UAP jointly over the complete sample pool.
    for _ in range(pgd_steps):
        v_full = torch.zeros(samples.shape[1], dtype=torch.float32, device=DEVICE)
        v_full[FREE_VALUE_INDICES] = v

        # Add the same perturbation to every sample in the pool.
        X_adv = X_tensor + v_full
        out = model(X_adv).view(-1)

        # Minimize the mean predicted fraud probability.
        loss = torch.mean(out)

        if v.grad is not None:
            v.grad.zero_()
        loss.backward()

        with torch.no_grad():
            # Take a sign-gradient step that lowers the fraud score.
            v -= alpha * v.grad.sign()

            # First projection: enforce the box constraints.
            v.clamp_(min=low_bound, max=up_bound)

            # Second projection: enforce the L0 sparsity budget.
            if lam_change < NUM_FREE_VALUES:
                v_abs = torch.abs(v)
                # Identify the top-k perturbation coordinates.
                _, topk_indices = torch.topk(v_abs, int(lam_change))
                # Zero every coordinate outside the top-k set.
                mask = torch.zeros_like(v, dtype=torch.bool)
                mask[topk_indices] = True
                v[~mask] = 0.0

        v.requires_grad_(True)

    return v.detach().cpu().numpy()


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

        X_tr_P = X_tr[y_tr == 1]
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
    # 6. Evaluate the three-dimensional security surface under the UAP
    # =============================================================================

    print(
        "\nStarting the 3D security-surface scan "
        "(metric: AUPRC; attack: universal PGD)..."
    )
    print(f"Target Models: {TARGET_MODELS} \nSubstitute Model: {SUBSTITUTE_MODEL_TYPE}")

    # Initialize one AUPRC surface per target architecture.
    prc_const_matrices = {
        name: np.zeros((len(RHO_LIST), len(LAMBDA_LIST))) for name in TARGET_MODELS
    }
    prc_unconst_matrices = {
        name: np.zeros((len(RHO_LIST), len(LAMBDA_LIST))) for name in TARGET_MODELS
    }

    # Retain detailed records for every point on the grid.
    grid_details_records = []

    # Draw the training pool used to learn the universal template.
    sample_indices = np.random.choice(
        len(X_tr_P), min(UAP_POOL_SIZE, len(X_tr_P)), replace=False
    )
    uap_samples_pool = X_tr_P[sample_indices]

    # Concatenate labels in the same order as the prediction vectors.
    y_te_combined = np.concatenate([np.zeros(len(X_te_N)), np.ones(len(X_te_P))])

    # =========================================================================
    # Compute the no-attack baseline AUPRC for every target model.
    # =========================================================================
    base_prcs = {}
    for name, clf in clf_targets.items():
        base_probs_P = clf.predict_proba(X_te_P)[:, 1]
        base_probs_N = clf.predict_proba(X_te_N)[:, 1]

        base_probs_combined = np.concatenate([base_probs_N, base_probs_P])
        base_prcs[name] = average_precision_score(y_te_combined, base_probs_combined)

        print(f">>> Base AUPRC for {name:15s}: {base_prcs[name]:.4f}")

    # =========================================================================
    # Scan the complete manipulation and sparsity grid.
    # =========================================================================
    for i, rho in enumerate(RHO_LIST):
        for j, lam in enumerate(LAMBDA_LIST):

            # A zero budget is identical to the no-attack baseline.
            if rho == 0.0 or lam == 0:
                for name in TARGET_MODELS:
                    prc_const_matrices[name][i, j] = base_prcs[name]
                    prc_unconst_matrices[name][i, j] = base_prcs[name]
                    # Add one detailed record for this grid point.
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
                "universal PGD attacks..."
            )

            # Learn one UAP for the current grid point.
            multiply1 = generate_uap_pgd(
                clf_substitute, uap_samples_pool, scaler, rho, lam
            )
            universal_perturbation = np.zeros(X_te_P.shape[1])
            universal_perturbation[FREE_VALUE_INDICES] = multiply1

            adv_unconst_X_list = []
            constrained_X_P_list = []

            # Apply the UAP to every positive test sample and reconstruct it.
            for k in range(len(X_te_P)):
                ori = X_te_P[k]

                # Add the learned universal template.
                adv_unconst = ori + universal_perturbation
                adv_unconst_X_list.append(adv_unconst)

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

            # Score every target architecture.
            for name, clf in clf_targets.items():
                probs_N = clf.predict_proba(X_te_N)[:, 1]

                # Unconstrained counterpart.
                unconst_probs_P = clf.predict_proba(adv_unconst_X)[:, 1]
                unconst_probs_combined = np.concatenate([probs_N, unconst_probs_P])
                prc_unconst_matrices[name][i, j] = average_precision_score(
                    y_te_combined, unconst_probs_combined
                )

                # Accounting-consistent counterpart.
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

                # Save the detailed result for this model and grid point.
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
    print("FINAL EVALUATION METRICS (AUPRC Based)")
    print("=" * 90)

    final_metrics = []
    for name in TARGET_MODELS:
        sms_const, ari_const = calculate_ari_metric(
            RHO_LIST, LAMBDA_LIST, prc_const_matrices[name]
        )
        sms_unconst, ari_unconst = calculate_ari_metric(
            RHO_LIST, LAMBDA_LIST, prc_unconst_matrices[name]
        )

        final_metrics.append(
            {
                "Target Model": name,
                "Base AUPRC": base_prcs[name],
                "Constrained SMS": sms_const,
                "Constrained ARI": ari_const,
                "Unconstrained SMS": sms_unconst,
                "Unconstrained ARI": ari_unconst,
            }
        )

    df_summary = pd.DataFrame(final_metrics)
    pd.options.display.float_format = "{:.4f}".format
    print(df_summary.to_string(index=False))

    # Convert detailed grid records to a DataFrame.
    df_grid_details = pd.DataFrame(grid_details_records)

    # Ensure that the output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # Save the results to a formatted Excel workbook.
    print(f"\nSaving results to: {output_path} ... ", end="")
    try:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            # Save detailed grid-level results.
            df_grid_details.to_excel(writer, sheet_name="Grid_Details", index=False)

            for name in TARGET_MODELS:
                safe_name = name.replace(" ", "_").replace(".", "")[:20]

                df_c_prc = pd.DataFrame(
                    prc_const_matrices[name],
                    index=[f"rho={r}" for r in RHO_LIST],
                    columns=[f"lam={lam}" for lam in LAMBDA_LIST],
                )
                df_u_prc = pd.DataFrame(
                    prc_unconst_matrices[name],
                    index=[f"rho={r}" for r in RHO_LIST],
                    columns=[f"lam={lam}" for lam in LAMBDA_LIST],
                )

                df_c_prc.to_excel(writer, sheet_name=f"{safe_name}_C_PRC")
                df_u_prc.to_excel(writer, sheet_name=f"{safe_name}_U_PRC")

            # Format the summary sheet.
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

            # Format the grid-details sheet.
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

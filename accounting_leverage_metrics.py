"""Estimate accounting-leverage metrics for an FSFD model."""

import random
import warnings

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration and feature indices
# ==========================================
DATA_PATH = "finance_data.xlsx"
STOCK_NAME_COLUMN = "\u80a1\u7968\u7b80\u79f0"
NUM_FEATURES = 89
NUM_SAMPLES = 20
RANDOM_SEED = 11

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
DERIVED_INDICES = [
    41,
    12,
    10,
    20,
    18,
    37,
    74,
    33,
    63,
    67,
    64,
    51,
    66,
    8,
    31,
    14,
    17,
    11,
    15,
    24,
    25,
    26,
    27,
    13,
    32,
    34,
    40,
    47,
    50,
    55,
    59,
    65,
    68,
    72,
    73,
    75,
    76,
    77,
    88,
    21,
]


# ==========================================
# 2. Demonstration fraud-detection network
# ==========================================
class SimpleFraudNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. Differentiable accounting-articulation mapping
# ==========================================
def apply_accounting_rules(raw_features, free_delta):
    """Apply a free-variable perturbation and reconstruct derived variables."""
    adjusted_features = [raw_features[i] for i in range(NUM_FEATURES)]
    residuals = [raw_features.new_zeros(()) for _ in range(NUM_FEATURES)]

    # Inject perturbations into the free variables.
    for i, idx in enumerate(FREE_VAL_INDICES):
        residuals[idx] = free_delta[i]

    # Propagate additive accounting identities.
    residuals[41] = residuals[0]
    residuals[12] = residuals[0]
    residuals[10] = (
        residuals[0] + residuals[78] - residuals[1] - residuals[2] - residuals[3]
    )
    residuals[20] = raw_features[10] + residuals[10] - raw_features[20]
    residuals[18] = residuals[20]
    residuals[37] = residuals[10]
    residuals[74] = residuals[0] + residuals[10]
    residuals[33] = residuals[7]
    residuals[63] = residuals[7]
    residuals[67] = residuals[7]
    residuals[64] = raw_features.new_zeros(())
    residuals[51] = residuals[7] + residuals[64]
    residuals[66] = residuals[4] + residuals[6] + residuals[48]
    residuals[8] = residuals[5] + residuals[56] + residuals[66]
    residuals[31] = residuals[66] - residuals[7] - residuals[6]
    residuals[14] = (
        raw_features[8]
        + residuals[5]
        + residuals[56]
        + residuals[66]
        - residuals[51]
        - raw_features[14]
    )
    residuals[17] = residuals[51] - residuals[6]

    # Recalculate ratio-based derived features with numerical stabilization.
    eps = 1e-9
    residuals[11] = (raw_features[10] + residuals[10]) / (
        raw_features[0] + residuals[0] + eps
    ) - raw_features[11]
    residuals[15] = (raw_features[10] + residuals[10]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[15]
    residuals[24] = (raw_features[8] + residuals[8]) / (
        raw_features[0] + residuals[0] + eps
    ) - raw_features[24]
    residuals[25] = (raw_features[6] + residuals[6]) / (
        raw_features[0] + residuals[0] + eps
    ) - raw_features[25]
    residuals[26] = (raw_features[31] + residuals[31]) / (
        raw_features[0] + residuals[0] + eps
    ) - raw_features[26]
    residuals[27] = (raw_features[5] + residuals[5]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[27]
    residuals[13] = (raw_features[10] + residuals[10]) / (
        raw_features[14] + residuals[14] + eps
    ) - raw_features[13]
    residuals[32] = (raw_features[33] + residuals[33]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[32]
    residuals[34] = (raw_features[35] + residuals[10]) / (
        raw_features[37] + residuals[37] + eps
    ) - raw_features[34]
    residuals[40] = (raw_features[39] + residuals[39]) / (
        raw_features[41] + residuals[41] + eps
    ) - raw_features[40]
    residuals[47] = (raw_features[48] + residuals[48]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[47]
    residuals[50] = (raw_features[7] + residuals[7]) / (
        raw_features[51] + residuals[51] + eps
    ) - raw_features[50]
    residuals[55] = (raw_features[56] + residuals[56]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[55]
    residuals[59] = residuals[0] + residuals[48]
    residuals[65] = (raw_features[66] + residuals[66]) / (
        raw_features[67] + residuals[67] + eps
    ) - raw_features[65]
    residuals[68] = (raw_features[51] + residuals[51]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[68]
    residuals[72] = (raw_features[74] + residuals[74]) / (
        raw_features[0] + residuals[0] + eps
    ) - raw_features[72]
    residuals[73] = (raw_features[74] + residuals[74]) / (
        raw_features[10] + residuals[10] + eps
    ) - raw_features[73]
    residuals[75] = (raw_features[6] + residuals[6]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[75]
    residuals[76] = (raw_features[17] + residuals[17]) / (
        raw_features[0] + residuals[0] + eps
    ) - raw_features[76]
    residuals[77] = (raw_features[78] + residuals[78]) / (
        raw_features[0] + residuals[0] + eps
    ) - raw_features[77]
    residuals[88] = (raw_features[10] + residuals[10]) / (
        raw_features[8] + residuals[8] + eps
    ) - raw_features[88]
    residuals[21] = (
        raw_features[22] + residuals[22] - raw_features[23] - residuals[23]
    ) / (raw_features[23] + residuals[23] + eps) - raw_features[21]

    # Merge free and derived residuals into the adjusted sample.
    for idx in FREE_VAL_INDICES + DERIVED_INDICES:
        adjusted_features[idx] = raw_features[idx] + residuals[idx]

    return torch.stack(adjusted_features)


# ==========================================
# 4. Accounting-leverage metrics
# ==========================================
def evaluate_leverage_metrics(model, raw_features, data_min, data_scale, sample_index):
    """Compute spectral, gradient-inflation, and derived-contribution metrics."""
    # Define the free-variable perturbation in normalized [0, 1] coordinates.
    delta_free_normalized = torch.zeros(len(FREE_VAL_INDICES), requires_grad=True)

    # Convert normalized perturbations back to the original feature units.
    delta_free_raw = delta_free_normalized / data_scale[FREE_VAL_INDICES]

    adjusted_raw = apply_accounting_rules(raw_features, delta_free_raw)

    # Normalize the reconstructed sample before passing it to the model.
    adjusted_normalized = (adjusted_raw - data_min) * data_scale

    # Obtain gradients with respect to free and derived features.
    detached_normalized = adjusted_normalized.detach().requires_grad_(True)
    model.zero_grad()
    score_detached = model(detached_normalized)
    score_detached.backward()

    grad_free = detached_normalized.grad[FREE_VAL_INDICES]
    grad_derived = detached_normalized.grad[DERIVED_INDICES]
    free_gradient_norm = torch.linalg.vector_norm(grad_free).item()

    # Compute the normalized-to-normalized accounting Jacobian.
    def normalized_derived_features(delta_normalized):
        delta_raw = delta_normalized / data_scale[FREE_VAL_INDICES]
        raw_output = apply_accounting_rules(raw_features, delta_raw)
        normalized_output = (raw_output - data_min) * data_scale
        return normalized_output[DERIVED_INDICES]

    jacobian_ml = torch.autograd.functional.jacobian(
        normalized_derived_features, delta_free_normalized
    )
    jacobian_ml_transpose = jacobian_ml.t()

    # Obtain the complete accounting-adjusted gradient.
    model.zero_grad()
    score = model(adjusted_normalized)
    score.backward()
    adjusted_gradient = delta_free_normalized.grad
    adjusted_gradient_norm = torch.linalg.vector_norm(adjusted_gradient).item()

    # Spectral multiplier: the Jacobian spectral norm.
    spectral_multiplier = torch.linalg.svdvals(jacobian_ml_transpose)[0].item()

    # Gradient inflation: the amplification of the complete attack gradient.
    gradient_inflation = adjusted_gradient_norm / (free_gradient_norm + 1e-12)

    # Derived contribution: the share attributable to derived-feature leverage.
    derived_contribution_vector = torch.matmul(jacobian_ml_transpose, grad_derived)
    derived_contribution_norm = torch.linalg.vector_norm(
        derived_contribution_vector
    ).item()
    derived_contribution = derived_contribution_norm / (adjusted_gradient_norm + 1e-12)

    print(f"Sample {sample_index}")
    print(f"   Spectral multiplier (L_spec): {spectral_multiplier:15.2f}")
    print(f"   Gradient inflation (I_grad):  {gradient_inflation:15.2f}")
    print(f"   Derived contribution (C_der): {derived_contribution:15.4f}\n")

    return spectral_multiplier, gradient_inflation, derived_contribution


# ==========================================
# 5. Load data and evaluate sampled observations
# ==========================================
if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_excel(DATA_PATH).drop(columns=STOCK_NAME_COLUMN, errors="ignore")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Unable to read the input file: {DATA_PATH}") from exc

    features = df.to_numpy()[:, :-1]

    # Fit the normalizer and convert its parameters to tensors.
    scaler = MinMaxScaler()
    scaler.fit(features)
    data_min = torch.tensor(scaler.data_min_, dtype=torch.float32)
    data_scale = torch.tensor(scaler.scale_, dtype=torch.float32)

    # Initialize the model and reproducible sampling generator.
    torch.manual_seed(RANDOM_SEED)
    sample_rng = random.Random(RANDOM_SEED)
    model = SimpleFraudNN()

    sample_count = min(NUM_SAMPLES, len(features))
    sample_indices = sample_rng.sample(range(len(features)), sample_count)

    print("=" * 60)
    print(f" Accounting-leverage metrics for {sample_count} observations")
    print("=" * 60)

    results = []
    for idx in sample_indices:
        raw_features = torch.tensor(features[idx], dtype=torch.float32)
        metrics = evaluate_leverage_metrics(
            model, raw_features, data_min, data_scale, sample_index=idx
        )
        results.append(metrics)

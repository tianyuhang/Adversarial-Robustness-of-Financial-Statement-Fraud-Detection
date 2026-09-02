# FSFD Adversarial Robustness

Research code for evaluating adversarial attacks and robustness-enhancement methods for financial statement fraud detection (FSFD) under accounting articulation constraints.

## Files

| File | Purpose |
| --- | --- |
| `specific_adversarial_attack.py` | Evaluates accounting-consistent sample-specific PGD attacks across multiple target models. |
| `universal_adversarial_attack.py` | Learns and evaluates a universal PGD perturbation. |
| `accounting_leverage_metrics.py` | Demonstrates three accounting-leverage metrics based on a differentiable articulation mapping. |
| `ensemble_defense_evaluation.py` | Compares clean training, adversarial training, bagging, feature subspaces, and leverage-aware augmentation. |
| `requirements.txt` | Lists the Python dependencies. |

## Usage

Use Python 3.10 or later and install the dependencies:

```bash
pip install -r requirements.txt
```

Place the required Excel data files in the project directory, or update `DATA_PATH` near the top of each script. The final column is treated as the target label. Run a script directly, for example:

```bash
python specific_adversarial_attack.py
```

Generated experiment workbooks are saved in `results/`. Input data are not included in this repository.

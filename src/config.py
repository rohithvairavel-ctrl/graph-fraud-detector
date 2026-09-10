"""Project paths and training defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Synthetic graph defaults
N_ACCOUNTS = 2000
N_LEGIT_EDGES = 8000
N_FRAUD_RINGS = 8
RING_SIZE = (6, 14)
SEED = 42

# Train/test
TEST_SIZE = 0.25
RANDOM_STATE = 42

TABULAR_FEATURES = [
    "tx_count",
    "tx_amount_sum",
    "tx_amount_mean",
    "tx_amount_std",
    "tx_amount_max",
    "unique_counterparties",
    "account_age_days",
    "avg_tx_hour",
    "night_tx_ratio",
]

GRAPH_FEATURES = [
    "degree",
    "in_degree",
    "out_degree",
    "pagerank",
    "clustering",
    "community_id",
    "community_size",
    "component_size",
    "shared_device_count",
    "shared_ip_count",
    "triangle_count",
    "cycle_flag",
    "neighbor_fraud_ratio_proxy",
]

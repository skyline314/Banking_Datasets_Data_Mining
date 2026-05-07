"""
Step 1 — Load the clean dataset and prepare it for clustering.
Reads the output of the ETL pipeline, selects clustering features,
scales them, and produces a random sample for expensive operations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

log = get_logger(__name__)

# ── Clustering features ──────────────────────────────────────────────────────
FEATURES = [
    "age_ordinal_normalized",
    "cust_account_balance_normalized",
    "transaction_amount_inr_normalized",
    "customer_freq_normalized",
]

# ── Random-sample size for computationally expensive tasks ────────────────────
SAMPLE_SIZE = 10_000


def load_and_prepare(input_path: Path):
    """
    Load the clean CSV produced by stage 1, select features, scale them,
    and draw a random sample for heavy operations (Silhouette / Hierarchical).

    Returns
    -------
    df_cluster : DataFrame   — rows used for clustering (features + no NaN)
    X_scaled   : ndarray     — StandardScaler-transformed feature matrix
    X_sample   : ndarray     — random subset of X_scaled (SAMPLE_SIZE rows)
    scaler     : StandardScaler — fitted scaler (needed for inverse_transform)
    """
    log.info("Loading clean dataset...")
    df = pd.read_csv(input_path)

    log.info(f"COLUMNS IN FILE: {df.columns.tolist()}")

    # Select features
    df_cluster = df[FEATURES].dropna()

    # Normalize the numeric features so large balances don't overpower age/frequency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

    # Take a random sample for computationally expensive tasks (Hierarchical & Silhouette)
    # Running these on 1,000,000+ rows will cause a MemoryError
    np.random.seed(42)
    sample_indices = np.random.choice(X_scaled.shape[0], SAMPLE_SIZE, replace=False)
    X_sample = X_scaled[sample_indices]

    log.info(f"Prepared {len(df_cluster)} rows, {len(FEATURES)} features, "
             f"sample size = {SAMPLE_SIZE}")

    return df_cluster, X_scaled, X_sample, scaler

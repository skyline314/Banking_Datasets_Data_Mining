"""
Step 1 -- Load the clean dataset and prepare it for clustering.
Reads the output of the ETL pipeline, selects clustering features,
scales them, assesses clustering tendency (Hopkins), and produces
a random sample for expensive operations.

Fixes applied from methodological audit:
  D3 -- Removed dead feature (customer_freq_normalized)
  D7 -- Added Hopkins statistic for clustering tendency assessment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from src.logger import get_logger

log = get_logger(__name__)

# ── Clustering features ──────────────────────────────────────────────────────
# [FIX D3] Removed customer_freq_normalized -- after KYC drop in Stage 1,
# 99.75% of customers have freq=1, making it near-constant (variance ≈ 0).
# Including it wastes 25% of the feature space and dilutes distance calculations.
FEATURES = [
    "age_ordinal_normalized",
    "cust_account_balance_normalized",
    "transaction_amount_inr_normalized",
]

# Mapping from normalized column names back to original-scale column names
# in clean_categorical.csv (used for true original-scale cluster profiling)
ORIGINAL_SCALE_MAP = {
    "age_ordinal_normalized":              "age",
    "cust_account_balance_normalized":     "cust_account_balance",
    "transaction_amount_inr_normalized":   "transaction_amount_inr",
}

# ── Random-sample size for computationally expensive tasks ────────────────────
SAMPLE_SIZE = 10_000
SEED = 42


# ── Hopkins statistic ────────────────────────────────────────────────────────
def hopkins_statistic(X: np.ndarray, sample_size: int = 500,
                      seed: int = 42) -> float:
    """
    Compute the Hopkins statistic to assess clustering tendency.

    Interpretation
    --------------
    H ≈ 0.5  ->  data is uniformly random (no clusters)
    H > 0.75 ->  data has significant cluster structure
    H -> 1.0  ->  data is highly clusterable
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    m = min(sample_size, n // 2)

    # Random sample of real data points
    idx = rng.choice(n, m, replace=False)
    X_sub = X[idx]

    # Nearest-neighbor distances for real points (2nd NN -- self is 1st)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    w_distances = nbrs.kneighbors(X_sub)[0][:, 1]

    # Distances for uniformly random points within data bounds
    X_random = rng.uniform(X.min(axis=0), X.max(axis=0), size=(m, d))
    u_distances = nbrs.kneighbors(X_random)[0][:, 0]

    H = u_distances.sum() / (u_distances.sum() + w_distances.sum())
    return H


def load_and_prepare(input_path: Path, categorical_path: Path):
    """
    Load the clean CSV produced by stage 1, select features, scale them,
    assess clustering tendency, and draw a random sample for heavy operations.

    Parameters
    ----------
    input_path       : Path -- clean.csv (Yeo-Johnson-normalized features)
    categorical_path : Path -- clean_categorical.csv (original-scale values)

    Returns
    -------
    df_cluster     : DataFrame   -- rows used for clustering (features + no NaN)
    X_scaled       : ndarray     -- StandardScaler-transformed feature matrix
    X_sample       : ndarray     -- random subset of X_scaled (SAMPLE_SIZE rows)
    scaler         : StandardScaler -- fitted scaler
    df_categorical : DataFrame   -- original-scale data for profile merging
    hopkins_score  : float       -- Hopkins statistic (clustering tendency)
    """
    log.info("Loading clean dataset...")
    df = pd.read_csv(input_path)
    log.info(f"COLUMNS IN FILE: {df.columns.tolist()}")

    # [FIX D3] Log the dropped feature's variance as evidence
    if "customer_freq_normalized" in df.columns:
        freq_var = df["customer_freq_normalized"].var()
        log.info(f"Dropped 'customer_freq_normalized' from clustering "
                 f"(variance={freq_var:.6f}, near-zero after KYC drop)")

    # Select features
    df_cluster = df[FEATURES].dropna()

    # StandardScaler on top of Yeo-Johnson to guarantee exact zero-mean / unit-var
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Reproducible random sample for expensive operations
    np.random.seed(SEED)
    sample_indices = np.random.choice(X_scaled.shape[0], SAMPLE_SIZE,
                                      replace=False)
    X_sample = X_scaled[sample_indices]

    # [FIX D7] Assess clustering tendency before running any algorithm
    log.info("Assessing clustering tendency (Hopkins statistic)...")
    hopkins_score = hopkins_statistic(X_scaled, sample_size=500, seed=SEED)
    log.info(f"  Hopkins statistic: {hopkins_score:.4f}")
    if hopkins_score > 0.75:
        log.info("  [OK] Data exhibits significant clustering tendency (H > 0.75)")
    elif hopkins_score > 0.5:
        log.warning(f"  [!] Weak clustering tendency (H = {hopkins_score:.4f}). "
                    f"Clusters may represent imposed structure.")
    else:
        log.warning(f"  [X] No significant clustering tendency (H = {hopkins_score:.4f}). "
                    f"Clustering may not be meaningful.")

    # Load original-scale data for profile merging (FIX D1 support)
    log.info(f"Loading original-scale data from {categorical_path}...")
    df_categorical = pd.read_csv(categorical_path)
    log.info(f"  Categorical shape: {df_categorical.shape}")

    log.info(f"Prepared {len(df_cluster)} rows, {len(FEATURES)} features, "
             f"sample size = {SAMPLE_SIZE}")

    return df_cluster, X_scaled, X_sample, scaler, df_categorical, hopkins_score

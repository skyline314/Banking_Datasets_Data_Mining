"""
Step 3 — Apply clustering algorithms.
  • K-Means on the full dataset
  • Hierarchical (Ward linkage) dendrogram on a small sample
  • DBSCAN outlier/noise detection on the 10 k sample
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN

from src.logger import get_logger

log = get_logger(__name__)


# ── 3a  K-Means ──────────────────────────────────────────────────────────────
def apply_kmeans(df_cluster: pd.DataFrame, X_scaled: np.ndarray,
                 optimal_k: int) -> pd.DataFrame:
    """
    Fit K-Means on the full scaled data and add a 'Cluster_Labels' column
    to *df_cluster*.  Returns the modified DataFrame.
    """
    log.info(f"Applying K-Means with K={optimal_k} on full dataset...")
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
    df_cluster = df_cluster.copy()
    df_cluster['Cluster_Labels'] = kmeans_final.fit_predict(X_scaled)
    return df_cluster


# ── 3b  Hierarchical Dendrogram ──────────────────────────────────────────────
def generate_dendrogram(X_scaled: np.ndarray, output_dir: Path,
                        dendro_sample_size: int = 3000) -> None:
    """
    Build a Ward-linkage dendrogram on a small random subset.
    Saves the plot to *output_dir*.
    """
    log.info("Generating Dendrograms (Ward Linkage)...")

    X_dendro_sample = X_scaled[
        np.random.choice(X_scaled.shape[0], dendro_sample_size, replace=False)
    ]

    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    shc.dendrogram(shc.linkage(X_dendro_sample, method='ward'))
    plt.axhline(y=15, color='r', linestyle='--')

    plot_path = output_dir / 'dendrogram_ward.png'
    plt.savefig(plot_path)
    plt.close()
    log.info(f"Saved '{plot_path}'.")


# ── 3c  DBSCAN outlier detection ─────────────────────────────────────────────
def detect_outliers_dbscan(X_sample: np.ndarray) -> int:
    """
    Run DBSCAN on the 10 k sample to identify noise / outlier profiles.
    Returns the number of outlier points found.
    """
    log.info("Applying DBSCAN to identify branch-level anomalies/noise...")

    # Eps and min_samples need tuning based on exact data distribution.
    # Here we use eps=0.5 and min_samples=10 as a standard starting point
    # for scaled data.
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    db_labels = dbscan.fit_predict(X_sample)

    outliers_count = list(db_labels).count(-1)
    log.info(f"DBSCAN found {outliers_count} noise/outlier points "
             f"out of {len(X_sample)} sampled records.")

    return outliers_count

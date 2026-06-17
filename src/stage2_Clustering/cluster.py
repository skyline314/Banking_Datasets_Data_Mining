"""
Step 3 -- Apply clustering algorithms.
  • K-Means on the full dataset
  • Hierarchical (Ward linkage) dendrogram on a small sample
  • DBSCAN outlier/noise detection with empirical eps via k-distance graph

Fixes applied from methodological audit:
  D2  -- DBSCAN eps determined empirically via k-distance graph + KneeLocator
  D5  -- Dendrogram random sample seeded for reproducibility
  D6  -- Dendrogram cut-line computed dynamically from linkage matrix
  D10 -- DBSCAN reports cluster count, sizes, and comparative Silhouette
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from src.stage2_Clustering.prepare import FEATURES, SEED
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
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++',
                          random_state=42, n_init=10)
    df_cluster = df_cluster.copy()
    df_cluster['Cluster_Labels'] = kmeans_final.fit_predict(X_scaled)
    return df_cluster


# ── 3b  Hierarchical Dendrogram ──────────────────────────────────────────────
def generate_dendrogram(X_scaled: np.ndarray, optimal_k: int,
                        output_dir: Path,
                        dendro_sample_size: int = 3000) -> None:
    """
    Build a Ward-linkage dendrogram on a small random subset.
    Saves the plot to *output_dir*.

    Fixes:
      D5 -- np.random.seed(SEED) before sampling for reproducibility
      D6 -- cut-line computed from linkage matrix to match optimal_k
    """
    log.info("Generating Dendrogram (Ward Linkage)...")

    # [FIX D5] Seed the sample for full reproducibility
    np.random.seed(SEED)
    X_dendro_sample = X_scaled[
        np.random.choice(X_scaled.shape[0], dendro_sample_size, replace=False)
    ]

    # Compute linkage
    Z = shc.linkage(X_dendro_sample, method='ward')

    # [FIX D6] Dynamically compute cut height from linkage matrix
    # For K clusters, cut between the (K-1)th and Kth largest merge distances
    n_merges = Z.shape[0]
    if 2 <= optimal_k <= n_merges:
        cut_height = (
            Z[-(optimal_k - 1), 2] + Z[-optimal_k, 2]
        ) / 2
    else:
        cut_height = Z[-2, 2] * 0.7  # fallback

    plt.figure(figsize=(12, 7))
    plt.title(f"Hierarchical Clustering Dendrogram "
              f"(Ward Linkage, n={dendro_sample_size})")
    shc.dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True)
    plt.axhline(y=cut_height, color='r', linestyle='--',
                label=f'Cut for K={optimal_k} (y={cut_height:.1f})')
    plt.legend(fontsize=11)
    plt.xlabel('Sample index (or cluster size)')
    plt.ylabel('Ward distance')

    plot_path = output_dir / 'dendrogram_ward.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Saved '{plot_path}'. Cut-line at y={cut_height:.1f} "
             f"for K={optimal_k}.")


# ── 3c  DBSCAN -- empirical eps selection + full reporting ────────────────────
def detect_outliers_dbscan(X_sample: np.ndarray,
                           output_dir: Path,
                           best_sil_kmeans: float = 0.0) -> int:
    """
    Run DBSCAN on the 10k sample with empirically determined eps.
    Returns the number of outlier points found.

    Fixes:
      D2  -- eps determined via k-distance graph knee detection
      D10 -- reports cluster count, cluster sizes, and comparative Silhouette
    """
    n_features = X_sample.shape[1]
    log.info("DBSCAN: computing k-distance graph for empirical eps...")

    # [FIX D2] k-distance plot to determine eps empirically
    # Heuristic: k = 2 × dimensionality
    k_neighbors = 2 * n_features
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)
    k_distances = np.sort(distances[:, -1])[::-1]  # sorted descending

    # Find the knee point of the k-distance curve
    kl_dbscan = KneeLocator(
        range(len(k_distances)), k_distances,
        curve='convex', direction='decreasing'
    )
    if kl_dbscan.knee is not None:
        optimal_eps = k_distances[kl_dbscan.knee]
        log.info(f"  k-distance knee at index {kl_dbscan.knee}, "
                 f"eps = {optimal_eps:.4f}")
    else:
        optimal_eps = 0.5
        log.warning(f"  No clear knee found -- falling back to eps = {optimal_eps}")

    # Plot k-distance graph
    plt.figure(figsize=(10, 5))
    plt.plot(k_distances, color='steelblue')
    plt.axhline(y=optimal_eps, color='red', linestyle='--',
                label=f'eps = {optimal_eps:.4f}')
    plt.title(f'k-Distance Graph (k={k_neighbors}) for DBSCAN eps Selection')
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'{k_neighbors}-th Nearest Neighbor Distance')
    plt.legend(fontsize=10)

    plot_path = output_dir / 'dbscan_k_distance_plot.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Saved '{plot_path}'.")

    # Run DBSCAN with empirically determined eps
    min_samples = k_neighbors  # rule of thumb: min_samples ≈ k
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_sample)

    # [FIX D10] Full reporting -- clusters AND noise
    n_clusters_found = len(set(db_labels) - {-1})
    n_noise = (db_labels == -1).sum()
    n_total = len(db_labels)

    log.info(f"DBSCAN Results (eps={optimal_eps:.4f}, "
             f"min_samples={min_samples}):")
    log.info(f"  Clusters found : {n_clusters_found}")
    log.info(f"  Noise points   : {n_noise} / {n_total} "
             f"({100 * n_noise / n_total:.1f}%)")

    if n_clusters_found > 0:
        # Cluster size distribution
        unique, counts = np.unique(db_labels[db_labels != -1],
                                   return_counts=True)
        for cid, cnt in zip(unique, counts):
            log.info(f"    Cluster {cid}: {cnt} points")

        # Comparative Silhouette (excluding noise)
        non_noise = db_labels != -1
        if non_noise.sum() > 1 and n_clusters_found > 1:
            db_sil = silhouette_score(X_sample[non_noise],
                                      db_labels[non_noise])
            log.info(f"  DBSCAN Silhouette (excl. noise): {db_sil:.4f}")
            if best_sil_kmeans > 0:
                log.info(f"  K-Means Silhouette:              "
                         f"{best_sil_kmeans:.4f}")
                winner = 'DBSCAN' if db_sil > best_sil_kmeans else 'K-Means'
                log.info(f"  -> {winner} produces tighter clusters.")
    else:
        log.warning("  DBSCAN found 0 clusters -- all points classified "
                    "as noise. Consider increasing eps.")

    return n_noise

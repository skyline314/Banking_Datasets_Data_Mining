"""
Step 2 -- Determine the optimal number of clusters (K).
Uses the Elbow method (WCSS) and Silhouette Score on a sample.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.logger import get_logger

log = get_logger(__name__)

# Full range evaluated and plotted
K_RANGE = range(2, 8)

# Practical subset for automatic selection (aligned with dendrogram analysis)
# K=2 is too coarse for meaningful business segmentation; dendrogram shows 3 branches
PRACTICAL_K_MIN = 3
PRACTICAL_K_MAX = 5


def evaluate_optimal_k(X_scaled: np.ndarray, X_sample: np.ndarray,
                       output_dir: Path) -> int:
    """
    Calculate Elbow (WCSS) and Silhouette scores for a range of K values.
    Saves the evaluation plot to *output_dir* and returns the chosen optimal K.

    Selection logic:
      1. Compute Silhouette for K in K_RANGE (2..7).
      2. Pick the K with the highest Silhouette within the practical range
         (2..PRACTICAL_K_MAX), which is aligned with hierarchical dendrogram
         analysis showing 3 main branches.

    Returns
    -------
    optimal_k : int -- selected number of clusters
    """
    log.info("Calculating Elbow Method and Silhouette Scores (on sample)...")

    wcss = []
    sil_scores = []

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)  # Fit on full data
        wcss.append(kmeans.inertia_)

        # Silhouette requires pairwise distances, so we use the sample
        sample_preds = kmeans.predict(X_sample)
        sil_scores.append(silhouette_score(X_sample, sample_preds))

    # Plot Elbow Method
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(K_RANGE, wcss, marker='o', linestyle='--')
    axes[0].set_title('Elbow Method for Optimal K')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('WCSS')

    # Plot Silhouette Score
    axes[1].plot(K_RANGE, sil_scores, marker='s', linestyle='--', color='orange')
    axes[1].set_title('Silhouette Score for Optimal K')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plot_path = output_dir / 'optimal_k_evaluation.png'
    plt.savefig(plot_path)
    plt.close()
    log.info(f"Saved '{plot_path}'.")

    # -- Full evaluation summary -----------------------------------------------
    log.info("K-evaluation summary:")
    for k, w, s in zip(K_RANGE, wcss, sil_scores):
        log.info(f"  K={k}  |  WCSS={w:,.1f}  |  Silhouette={s:.4f}")

    # -- Programmatic selection within practical range -------------------------
    practical_scores = {
        k: s for k, s in zip(K_RANGE, sil_scores)
        if PRACTICAL_K_MIN <= k <= PRACTICAL_K_MAX
    }
    optimal_k = max(practical_scores, key=practical_scores.get)
    log.info(
        f"Selected OPTIMAL_K = {optimal_k} "
        f"(highest Silhouette in practical range {PRACTICAL_K_MIN}..{PRACTICAL_K_MAX}: "
        f"{practical_scores[optimal_k]:.4f})"
    )

    return optimal_k


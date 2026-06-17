"""
Step 2 -- Determine the optimal number of clusters (K).
Uses the Elbow method (WCSS) and Silhouette Score on a sample.

Fixes applied from methodological audit:
  D4 -- Acknowledge and interpret low Silhouette scores
  D8 -- Objective K selection without artificial range restriction
  D9 -- Programmatic elbow detection with KneeLocator annotation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from src.logger import get_logger

log = get_logger(__name__)

# Full range evaluated and plotted
K_RANGE = range(2, 8)

# [FIX D8] Practical subset -- K=2 is NO LONGER excluded.
# Previous version set PRACTICAL_K_MIN=3 to force K=3, overriding the
# quantitative evidence (K=2 had higher Silhouette). Now the algorithm
# selects objectively from the full practical range.
PRACTICAL_K_MIN = 2
PRACTICAL_K_MAX = 5


def evaluate_optimal_k(X_scaled: np.ndarray, X_sample: np.ndarray,
                       output_dir: Path) -> tuple:
    """
    Calculate Elbow (WCSS) and Silhouette scores for a range of K values.
    Saves the evaluation plot to *output_dir* and returns the chosen optimal K
    together with its Silhouette score.

    Selection logic:
      1. Compute Silhouette for K in K_RANGE (2..7).
      2. Pick the K with the highest Silhouette within the practical range
         (PRACTICAL_K_MIN..PRACTICAL_K_MAX).
      3. Use KneeLocator to identify the WCSS elbow as supporting evidence.

    Returns
    -------
    optimal_k : int   -- selected number of clusters
    best_sil  : float -- Silhouette score for the selected K
    """
    log.info("Calculating Elbow Method and Silhouette Scores (on sample)...")

    wcss = []
    sil_scores = []

    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42,
                        n_init=10)
        kmeans.fit(X_scaled)  # Fit on full data
        wcss.append(kmeans.inertia_)

        # Silhouette requires pairwise distances, so we use the sample
        sample_preds = kmeans.predict(X_sample)
        sil_scores.append(silhouette_score(X_sample, sample_preds))

    # ── [FIX D9] Programmatic elbow detection ────────────────────────────
    kl = KneeLocator(list(K_RANGE), wcss, curve='convex',
                     direction='decreasing')
    elbow_k = kl.elbow if kl.elbow is not None else None
    log.info(f"Elbow detected at K={elbow_k}")

    # ── Full evaluation summary ──────────────────────────────────────────
    log.info("K-evaluation summary:")
    for k, w, s in zip(K_RANGE, wcss, sil_scores):
        markers = []
        if k == elbow_k:
            markers.append("elbow")
        log.info(f"  K={k}  |  WCSS={w:,.1f}  |  Silhouette={s:.4f}"
                 f"{'  <- ' + ', '.join(markers) if markers else ''}")

    # ── [FIX D8] Programmatic selection -- unrestricted practical range ────
    practical_scores = {
        k: s for k, s in zip(K_RANGE, sil_scores)
        if PRACTICAL_K_MIN <= k <= PRACTICAL_K_MAX
    }
    optimal_k = max(practical_scores, key=practical_scores.get)
    best_sil = practical_scores[optimal_k]

    log.info(
        f"Selected OPTIMAL_K = {optimal_k} "
        f"(highest Silhouette in range {PRACTICAL_K_MIN}..{PRACTICAL_K_MAX}: "
        f"{best_sil:.4f})"
    )

    # ── [FIX D4] Acknowledge and interpret Silhouette quality ─────────────
    if best_sil < 0.25:
        log.warning(
            f"[!] Silhouette = {best_sil:.4f} < 0.25 -- WEAK cluster structure. "
            f"The data may not have strong natural clusters along these features. "
            f"Segments should be interpreted as soft groupings, not hard boundaries."
        )
    elif best_sil < 0.50:
        log.warning(
            f"[!] Silhouette = {best_sil:.4f} < 0.50 -- MODERATE cluster overlap. "
            f"Segments are distinguishable but boundaries are not crisp."
        )
    else:
        log.info(f"[OK] Silhouette = {best_sil:.4f} -- good cluster separation.")

    # ── Plot with annotations ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    axes[0].plot(K_RANGE, wcss, marker='o', linestyle='--', color='steelblue')
    if elbow_k is not None:
        axes[0].axvline(x=elbow_k, color='red', linestyle=':',
                        alpha=0.7, label=f'Elbow at K={elbow_k}')
        axes[0].legend(fontsize=10)
    axes[0].set_title('Elbow Method for Optimal K')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('WCSS (Within-Cluster Sum of Squares)')

    # Silhouette plot
    axes[1].plot(K_RANGE, sil_scores, marker='s', linestyle='--',
                 color='darkorange')
    axes[1].axvline(x=optimal_k, color='red', linestyle=':',
                    alpha=0.7, label=f'Selected K={optimal_k}')
    axes[1].axhline(y=0.25, color='gray', linestyle=':',
                    alpha=0.5, label='Weak threshold (0.25)')
    axes[1].legend(fontsize=10)
    axes[1].set_title('Silhouette Score for Optimal K')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plot_path = output_dir / 'optimal_k_evaluation.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Saved '{plot_path}'.")

    return optimal_k, best_sil

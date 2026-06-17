"""
Step 4 -- Build cluster profiles and export the final CSV deliverable.
Uses the MERGE STRATEGY to produce profiles in true original-scale units,
completely bypassing the broken inverse-transform chain.

Fixes applied from methodological audit:
  D1 -- Double-scaling inverse-transform trap resolved via merge-to-original
       strategy instead of scaler.inverse_transform()

Previous approach (BROKEN):
  Stage 1: original -> PowerTransformer(yeo-johnson) -> *_normalized columns
  Stage 2: *_normalized -> StandardScaler -> X_scaled
  Profile: scaler.inverse_transform(cluster_means)
           -> reverses StandardScaler ONLY -> values still in Yeo-Johnson space
           -> e.g. cust_account_balance = 0.75 (a z-score, NOT INR 750)

Fixed approach (MERGE STRATEGY):
  1. Assign cluster labels to each row in df
  2. Merge with clean_categorical.csv (which has original-scale values)
  3. Compute per-cluster statistics on original-scale columns
  -> e.g. cust_account_balance_mean = INR 148,200 (TRUE rupee amount)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.stage2_Clustering.prepare import FEATURES, ORIGINAL_SCALE_MAP
from src.logger import get_logger

log = get_logger(__name__)


def build_and_export_profiles(df_cluster: pd.DataFrame,
                              df_categorical: pd.DataFrame,
                              hopkins_score: float,
                              best_sil: float,
                              output_dir: Path) -> pd.DataFrame:
    """
    Compute per-cluster means on TRUE original-scale values, attach customer
    counts, add named personas + business interpretation, and save to CSV
    + text report.

    Parameters
    ----------
    df_cluster     : DataFrame with Cluster_Labels column (index aligns with
                     the clean.csv row order)
    df_categorical : DataFrame from clean_categorical.csv (original-scale values)
    hopkins_score  : Hopkins statistic from prepare step
    best_sil       : Silhouette score for the selected K
    output_dir     : output directory for CSV and report files

    Returns
    -------
    cluster_profiles_real : DataFrame -- the final profile table
    """
    log.info("Building cluster profiles...")
    log.info("==========================================")
    log.info("FINAL CLUSTER PROFILES (BUSINESS INTERPRETATION)")
    log.info("==========================================")

    # ── [FIX D1] Merge strategy for true original-scale values ───────────
    # Map cluster labels back to the original-scale categorical data
    original_cols = [c for c in ORIGINAL_SCALE_MAP.values()
                     if c in df_categorical.columns]

    if len(df_categorical) == len(df_cluster) + df_cluster.index[0]:
        # If indices might not align perfectly, use positional alignment
        log.info("Using index-based merge with categorical data.")

    # Build a merged dataframe: cluster labels + original-scale columns
    df_merged = pd.DataFrame(index=df_cluster.index)
    df_merged['Cluster_Labels'] = df_cluster['Cluster_Labels'].values

    for orig_col in original_cols:
        if orig_col in df_categorical.columns:
            # Use .iloc with the index positions from df_cluster
            df_merged[orig_col] = df_categorical[orig_col].iloc[
                df_cluster.index
            ].values

    # ── Compute per-cluster statistics on ORIGINAL-SCALE values ──────────
    agg_funcs = ['mean', 'median', 'std', 'min', 'max']
    cluster_profiles_real = (
        df_merged
        .groupby('Cluster_Labels')[original_cols]
        .agg(agg_funcs)
        .round(2)
    )

    # Flatten MultiIndex columns: e.g. ('age', 'mean') -> 'age_mean'
    cluster_profiles_real.columns = [
        f"{col}_{stat}" for col, stat in cluster_profiles_real.columns
    ]

    # Add customer count and percentage
    cluster_sizes = df_merged['Cluster_Labels'].value_counts().sort_index()
    cluster_profiles_real['Customer_Count'] = cluster_sizes
    cluster_profiles_real['Pct_of_Total'] = (
        100 * cluster_sizes / cluster_sizes.sum()
    ).round(1)

    log.info(f"\n{cluster_profiles_real.to_string()}")

    # ── Also log normalized-space centroids for methodological reference ──
    cluster_centroids_norm = (
        df_cluster.groupby('Cluster_Labels')[FEATURES].mean().round(4)
    )
    log.info(f"\n(Normalized-space cluster centroids for reference:)")
    log.info(f"\n{cluster_centroids_norm.to_string()}")

    # ── Business interpretation text report ───────────────────────────────
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("CLUSTER BUSINESS PROFILES -- DETAILED REPORT")
    report_lines.append("(Values in TRUE original units)")
    report_lines.append("=" * 60)

    n_clusters = len(cluster_profiles_real)

    for cluster_id in sorted(cluster_profiles_real.index):
        row = cluster_profiles_real.loc[cluster_id]

        report_lines.append(f"\nCluster {cluster_id}")
        report_lines.append("-" * 60)
        report_lines.append(
            f"  Population  : {int(row['Customer_Count']):,} customers "
            f"({row['Pct_of_Total']}% of total)"
        )

        for orig_col in original_cols:
            unit = 'INR ' if 'balance' in orig_col or 'amount' in orig_col else ''
            mean_val = row.get(f'{orig_col}_mean', 'N/A')
            median_val = row.get(f'{orig_col}_median', 'N/A')
            std_val = row.get(f'{orig_col}_std', 'N/A')

            if isinstance(mean_val, (int, float)):
                report_lines.append(
                    f"  {orig_col:35s}: "
                    f"mean={unit}{mean_val:>12,.2f}  "
                    f"median={unit}{median_val:>12,.2f}  "
                    f"std={unit}{std_val:>10,.2f}"
                )
            else:
                report_lines.append(f"  {orig_col:35s}: {mean_val}")

    # ── Quality assessment section ────────────────────────────────────────
    report_lines.append("\n" + "=" * 60)
    report_lines.append("QUALITY ASSESSMENT")
    report_lines.append("=" * 60)

    report_lines.append(f"  Hopkins statistic : {hopkins_score:.4f}")
    if hopkins_score > 0.75:
        report_lines.append(
            "    -> Data has significant clustering tendency."
        )
    elif hopkins_score > 0.5:
        report_lines.append(
            "    -> Weak clustering tendency -- interpret segments as "
            "soft groupings, not hard boundaries."
        )
    else:
        report_lines.append(
            "    -> No significant clustering tendency detected."
        )

    report_lines.append(f"  Silhouette score  : {best_sil:.4f}")
    if best_sil < 0.25:
        report_lines.append(
            "    -> Weak cluster separation -- segments overlap substantially."
        )
    elif best_sil < 0.50:
        report_lines.append(
            "    -> Moderate cluster separation -- boundaries are fuzzy "
            "but distinguishable."
        )
    else:
        report_lines.append("    -> Good cluster separation.")

    report_lines.append("\n" + "=" * 60)
    report_lines.append(
        "Note: 'customer_freq' was EXCLUDED from clustering features "
        "because it is near-constant (99.75% = 1) after the KYC drop "
        "in Stage 1, contributing zero discriminative power."
    )
    report_lines.append(
        "Note: Profile values are in TRUE ORIGINAL SCALE (INR  for amounts, "
        "years for age) -- computed by merging cluster labels back to "
        "clean_categorical.csv, completely bypassing the StandardScaler / "
        "Yeo-Johnson inverse-transform chain."
    )
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    log.info(f"\n{report_text}")

    # ── Save outputs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'cluster_business_profiles_REAL.csv'
    cluster_profiles_real.to_csv(csv_path)
    log.info(f"Exported '{csv_path}' -- values in TRUE original units.")

    report_path = output_dir / 'cluster_business_interpretation.txt'
    report_path.write_text(report_text, encoding='utf-8')
    log.info(f"Exported '{report_path}'.")

    return cluster_profiles_real

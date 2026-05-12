"""
Step 4 — Build cluster profiles and export the final CSV deliverable.
Inverse-transforms the normalised means back to the original scale
so stakeholders can interpret the results directly.
Includes named personas and business interpretation for each cluster.
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.stage2_Clustering.prepare import FEATURES
from src.logger import get_logger

log = get_logger(__name__)

# ── Named personas & business interpretation (derived from cluster stats) ─────
CLUSTER_PERSONAS = {
    0: {
        "name": "High-Value Established Customers",
        "subtitle": "Established High-Balance Customers",
        "description": (
            "This segment consists of mature customers with above-average ages. "
            "They have very strong account balances and are accustomed to making "
            "transactions with the largest denominations."
        ),
        "actionable_insight": (
            "These are the bank's premium customers. The bank can prioritize offering "
            "Wealth Management products, investments, high-interest deposits, or premium "
            "credit cards to this segment to maximize Customer Lifetime Value (CLV)."
        ),
    },
    1: {
        "name": "Youth / Entry-Level Segment",
        "subtitle": "Gen-Z & Students",
        "description": (
            "This segment consists of very young customers, significantly below the "
            "average age. Consistent with their age, their savings balances are relatively "
            "low, and their daily transaction amounts are also small."
        ),
        "actionable_insight": (
            "A suitable business approach for this student/first-jobber segment is "
            "offering lifestyle merchant cashback promos (F&B, entertainment), "
            "e-wallet integration, or admin-fee-free savings accounts to acquire "
            "their loyalty early before they gain greater financial capabilities "
            "in the future."
        ),
    },
    2: {
        "name": "Low-Value Mass Market",
        "subtitle": "Passive Mass Market Customers",
        "description": (
            "The majority of the bank's customers fall into this cluster. Their age "
            "is standard or slightly above average, but their account balances tend "
            "to be very minimal and their transaction amounts are the lowest."
        ),
        "actionable_insight": (
            "This mass segment likely uses their accounts only as a transit for "
            "funds (e.g., receiving salary and then immediately withdrawing it all). "
            "The bank needs to educate this segment to start saving, hold lottery "
            "programs based on average balances, or offer cash loans / micro-paylater "
            "if they need quick liquidity."
        ),
    },
}


def build_and_export_profiles(df_cluster: pd.DataFrame,
                              scaler: StandardScaler,
                              output_dir: Path) -> pd.DataFrame:
    """
    Compute per-cluster means, inverse-transform to the real scale,
    attach customer counts, add named personas + business interpretation,
    and save to CSV + text report.

    Returns
    -------
    cluster_profiles_real : DataFrame — the final profile table
    """
    log.info("Building cluster profiles...")

    log.info("==========================================")
    log.info("FINAL CLUSTER PROFILES (BUSINESS INTERPRETATION)")
    log.info("==========================================")

    # Calculate the mean of the normalized features for each cluster
    cluster_profiles_norm = df_cluster.groupby('Cluster_Labels')[FEATURES].mean()

    # Return values to original scale using inverse_transform
    # for easier interpretation
    unscaled_means = scaler.inverse_transform(cluster_profiles_norm)
    unscaled_features = [f.replace('_normalized', '') for f in FEATURES]
    cluster_profiles_real = pd.DataFrame(
        unscaled_means,
        columns=unscaled_features,
        index=cluster_profiles_norm.index,
    ).round(2)

    # Add customer count per cluster
    cluster_sizes = df_cluster['Cluster_Labels'].value_counts()
    cluster_profiles_real['Customer_Count'] = cluster_sizes

    # Add persona names to the DataFrame
    cluster_profiles_real['Persona'] = cluster_profiles_real.index.map(
        lambda c: CLUSTER_PERSONAS.get(c, {}).get("name", f"Cluster {c}")
    )

    log.info(f"\n{cluster_profiles_real}")

    # -- Log business interpretation for each cluster -----------------------
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("CLUSTER BUSINESS PROFILES -- DETAILED REPORT")
    report_lines.append("=" * 60)

    # Fallback persona for clusters without a predefined interpretation
    _DEFAULT_PERSONA = {
        "name": "Unclassified Segment",
        "subtitle": "Unclassified Segment",
        "description": "This segment does not yet have a predefined business interpretation.",
        "actionable_insight": "Further analysis is required to determine the business strategy.",
    }

    for cluster_id in sorted(cluster_profiles_real.index):
        persona = CLUSTER_PERSONAS.get(cluster_id, _DEFAULT_PERSONA)
        row = cluster_profiles_real.loc[cluster_id]

        header = (f"\nCluster {cluster_id}: \"{persona['name']}\" "
                  f"({persona['subtitle']})")
        report_lines.append(header)
        report_lines.append("-" * 60)
        report_lines.append(f"  Population  : {int(row['Customer_Count']):,} customers")
        for feat in unscaled_features:
            report_lines.append(f"  {feat:40s}: {row[feat]:+.2f}")
        report_lines.append(f"\n  Description : {persona['description']}")
        report_lines.append(f"  Actionable  : {persona['actionable_insight']}")

    report_lines.append("\n" + "=" * 60)
    report_lines.append(
        "Note: customer_freq is uniformly 0.0 across all clusters because "
        "almost all customers recorded only 1 transaction in the dataset, "
        "so frequency does not differentiate clusters."
    )
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    log.info(f"\n{report_text}")

    # ── Save outputs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'cluster_business_profiles_REAL.csv'
    cluster_profiles_real.to_csv(csv_path)
    log.info(f"Exported '{csv_path}' for the final report.")

    report_path = output_dir / 'cluster_business_interpretation.txt'
    report_path.write_text(report_text, encoding='utf-8')
    log.info(f"Exported '{report_path}' for the final report.")

    return cluster_profiles_real


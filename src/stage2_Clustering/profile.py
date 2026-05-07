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
        "subtitle": "Nasabah Mapan Bersaldo Tinggi",
        "description": (
            "Segmen ini diisi oleh nasabah dewasa dengan umur di atas rata-rata. "
            "Mereka memiliki Saldo Akun yang sangat kuat dan terbiasa melakukan "
            "transaksi dengan nominal yang paling besar."
        ),
        "actionable_insight": (
            "Ini adalah nasabah premium bank. Bank bisa memprioritaskan penawaran "
            "produk Wealth Management, investasi, deposito bunga tinggi, atau kartu "
            "kredit premium kepada segmen ini untuk memaksimalkan Customer Lifetime "
            "Value (CLV)."
        ),
    },
    1: {
        "name": "Youth / Entry-Level Segment",
        "subtitle": "Gen-Z & Pelajar",
        "description": (
            "Segmen nasabah yang berumur sangat muda atau paling jauh di bawah "
            "rata-rata. Sesuai dengan usianya, saldo tabungan mereka tergolong "
            "rendah dan nominal transaksi harian mereka juga bernilai kecil."
        ),
        "actionable_insight": (
            "Pendekatan bisnis yang cocok untuk segmen mahasiswa/first-jobber ini "
            "adalah menawarkan promo cashback merchant lifestyle (F&B, hiburan), "
            "integrasi e-wallet, atau tabungan bebas biaya admin untuk mengakuisisi "
            "loyalitas mereka sejak dini sebelum mereka memiliki kapabilitas "
            "finansial yang lebih besar di masa depan."
        ),
    },
    2: {
        "name": "Low-Value Mass Market",
        "subtitle": "Nasabah Massal Pasif",
        "description": (
            "Mayoritas nasabah bank berada di klaster ini. Umur mereka standar/"
            "sedikit di atas rata-rata, namun saldo akun mereka cenderung sangat "
            "minim dan nominal transaksi mereka paling rendah."
        ),
        "actionable_insight": (
            "Segmen massal ini kemungkinan menggunakan rekening hanya sebagai jalur "
            "lewat dana (misal: menerima gaji lalu langsung ditarik habis). Bank "
            "perlu mengedukasi segmen ini untuk mulai menabung, mengadakan program "
            "undian berdasarkan saldo mengendap, atau menawarkan pinjaman tunai / "
            "paylater mikro jika mereka membutuhkan likuiditas cepat."
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

    # Kembalikan nilai ke skala asli menggunakan inverse_transform
    # agar mudah diinterpretasikan
    unscaled_means = scaler.inverse_transform(cluster_profiles_norm)
    unscaled_features = [f.replace('_normalized', '') for f in FEATURES]
    cluster_profiles_real = pd.DataFrame(
        unscaled_means,
        columns=unscaled_features,
        index=cluster_profiles_norm.index,
    ).round(2)

    # Menambahkan jumlah nasabah per klaster
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
        "subtitle": "Segmen Belum Terklasifikasi",
        "description": "Segmen ini belum memiliki interpretasi bisnis yang telah ditentukan.",
        "actionable_insight": "Diperlukan analisis lebih lanjut untuk menentukan strategi bisnis.",
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


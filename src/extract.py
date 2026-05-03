"""
Extract step — load raw CSV and compute customer-level stats on the FULL dataset.

Customer stats MUST be computed here, before any row drops occur in transform.
If computed after the KYC consistency drop, ~33% of rows are already removed
and multi-transaction customers are mostly gone — making customer_freq
near-constant and useless for transaction rate mining.
"""

import pandas as pd
from pathlib import Path

from src.logger import get_logger

log = get_logger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw CSV from disk."""
    log.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def compute_customer_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-customer aggregates on the FULL raw dataset.
    Returns a customer-level summary to be merged back after cleaning.
    """
    log.info("Computing customer stats on raw data (before any drops)")

    stats = df.groupby("CustomerID").agg(
        customer_freq    = ("TransactionID",        "count"),
        avg_txn_amount   = ("TransactionAmount (INR)", "mean"),
        total_txn_amount = ("TransactionAmount (INR)", "sum"),
    ).reset_index()

    stats = stats.rename(columns={"CustomerID": "customer_id"})

    log.info(f"  Unique customers: {stats.shape[0]:,}")
    log.info(f"  customer_freq — min: {stats['customer_freq'].min()}, "
             f"max: {stats['customer_freq'].max()}, "
             f"mean: {stats['customer_freq'].mean():.2f}")
    return stats


def extract(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full extract step.
    Returns (raw_df, customer_stats_df).
    """
    df = load_raw(raw_path)
    customer_stats = compute_customer_stats(df)
    return df, customer_stats
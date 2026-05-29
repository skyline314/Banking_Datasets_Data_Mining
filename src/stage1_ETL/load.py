"""
Load step — validate and export processed data to the output folder.
Includes pre-export assertions so corrupt files are never written silently.

Exports two files:
  clean.csv              — normalised features for clustering / distance-based ML
  clean_categorical.csv  — cleaned original-scale values for ARM / EDA
"""

import pandas as pd
from pathlib import Path

from src.logger import get_logger

log = get_logger(__name__)

# Columns kept in clean_categorical.csv — only what downstream stages need.
# Raw helper columns (transaction_time, hour, transaction_id, etc.) are excluded.
CATEGORICAL_KEEP_COLS = [
    "customer_id",
    "cust_gender",
    "cust_location",
    "cust_account_balance",
    "transaction_amount_inr",
    "age",
    "transaction_month",
    "is_weekend",
    "transaction_time_category",
    "customer_freq",
    "avg_txn_amount",
    "total_txn_amount",
]


def validate(df: pd.DataFrame) -> None:
    """Hard assertions before writing clean.csv — fails loudly rather than silently."""
    errors = []

    if "Unnamed: 0" in df.columns:
        errors.append("Ghost index column 'Unnamed: 0' found")

    if "customer_id" not in df.columns:
        errors.append("'customer_id' column is missing")

    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    if bool_cols:
        errors.append(f"Bool dtype OHE columns found: {bool_cols}")

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls):
        errors.append(f"Null values present:\n{cols_with_nulls.to_string()}")

    non_num = [
        c for c in df.columns
        if c != "customer_id" and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_num:
        errors.append(f"Non-numeric columns in output: {non_num}")

    if errors:
        for e in errors:
            log.error(f"[Validate] FAILED: {e}")
        raise ValueError(f"Pipeline output failed validation — {len(errors)} error(s). "
                         "Check logs for details.")

    log.info("[Validate] All checks passed ✓")


def validate_categorical(df: pd.DataFrame) -> None:
    """
    Assertions for clean_categorical.csv.

    Unlike clean.csv, non-numeric columns are EXPECTED here (gender, location,
    time category are kept as original strings for downstream discretization).
    """
    errors = []

    if "Unnamed: 0" in df.columns:
        errors.append("Ghost index column 'Unnamed: 0' found")

    if "customer_id" not in df.columns:
        errors.append("'customer_id' column is missing")

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls):
        errors.append(f"Null values present:\n{cols_with_nulls.to_string()}")

    # Verify expected columns are present
    missing = [c for c in CATEGORICAL_KEEP_COLS if c not in df.columns]
    if missing:
        errors.append(f"Expected columns missing: {missing}")

    if errors:
        for e in errors:
            log.error(f"[Validate-Cat] FAILED: {e}")
        raise ValueError(f"Categorical output failed validation — {len(errors)} error(s). "
                         "Check logs for details.")

    log.info("[Validate-Cat] All checks passed ✓")


def load(df: pd.DataFrame, output_path: Path) -> None:
    """Validate then write clean.csv — index=False prevents ghost column."""
    log.info(f"[Load] Output path: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate(df)

    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 ** 2)

    log.info(f"[Load] Exported {df.shape[0]:,} rows × {df.shape[1]} columns "
             f"({size_mb:.1f} MB)")
    log.info(f"[Load] Columns: {df.columns.tolist()}")


def load_categorical(df: pd.DataFrame, output_path: Path) -> None:
    """
    Select relevant columns, validate, and write clean_categorical.csv.

    This file preserves original-scale values (₹ amounts, string gender/location)
    so downstream stages like ARM can discretize them into domain-meaningful bins.
    """
    log.info(f"[Load-Cat] Output path: {output_path}")

    # Keep only the columns needed for downstream analysis
    present = [c for c in CATEGORICAL_KEEP_COLS if c in df.columns]
    df_out = df[present].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_categorical(df_out)

    df_out.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 ** 2)

    log.info(f"[Load-Cat] Exported {df_out.shape[0]:,} rows × {df_out.shape[1]} columns "
             f"({size_mb:.1f} MB)")
    log.info(f"[Load-Cat] Columns: {df_out.columns.tolist()}")
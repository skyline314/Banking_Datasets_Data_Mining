"""
Load step — validate and export clean.csv to the processed data folder.
Includes pre-export assertions so corrupt files are never written silently.
"""

import pandas as pd
from pathlib import Path

from src.logger import get_logger

log = get_logger(__name__)


def validate(df: pd.DataFrame) -> None:
    """Hard assertions before writing — pipeline fails loudly rather than silently."""
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
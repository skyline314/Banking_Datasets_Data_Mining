"""
Transform step — full cleaning and feature engineering pipeline.

Execution order matters:
  1.  Deduplicate
  2.  Standardize column names
  3.  Handle missing values  (before consistency check)
  4.  Type conversion
  5.  Check uniqueness
  6.  Drop inconsistent KYC records  (DOB / gender — no imputation)
  7.  Filter valid gender values
  8.  Merge location aliases  (BEFORE grouping and OHE)
  9.  Compute age  (elapsed days // 365)
  10. Extract temporal features  (BEFORE dropping transaction_date)
  11. Merge customer stats computed on raw data
  12. Build spending / demographic EDA helpers
  13. Encode categoricals  (binary, ordinal, OHE with drop_first)
  14. Drop raw / redundant columns
  15. Normalize numeric features with Yeo-Johnson
  16. Drop correlated features identified via heatmap
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

from src.logger import get_logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    KYC_FIELDS, LOCATION_ALIAS, TOP_N_LOCATIONS,
    AGE_MIN, AGE_MAX, AGE_BINS, AGE_LABELS, AGE_ORDINAL_MAP,
    TIME_PERIOD_CATEGORIES, VALID_GENDERS,
    COLS_TO_DROP, COLS_TO_NORMALIZE, COLS_CORRELATED_DROP,
)

log = get_logger(__name__)


# ── 1. Deduplicate ────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    log.info(f"[Dedup] Removed {dropped:,} duplicate rows — {len(df):,} remaining")
    return df


# ── 2. Standardize column names ───────────────────────────────────────────────
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def to_snake(col: str) -> str:
        col = re.sub(r"([a-z])([A-Z])", r"\1_\2", col)
        col = re.sub(r"\s+", "_", col)
        col = re.sub(r"[^\w]", "", col)
        return col.lower()

    mapping = {old: to_snake(old) for old in df.columns}
    df = df.rename(columns=mapping)
    renamed = {k: v for k, v in mapping.items() if k != v}
    log.info(f"[Columns] Renamed {len(renamed)} columns: {renamed}")
    return df


# ── 3. Handle missing values ──────────────────────────────────────────────────
def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    missing = df.isnull().sum()
    cols_with_nulls = missing[missing > 0]
    if len(cols_with_nulls):
        log.info(f"[Missing] Columns with nulls:\n{cols_with_nulls.to_string()}")
    df = df.dropna()
    log.info(f"[Missing] Dropped {before - len(df):,} rows with nulls — {len(df):,} remaining")
    return df


# ── 4. Type conversion ────────────────────────────────────────────────────────
def _time_period(hour: int) -> str:
    if   4  <= hour < 8:  return "Early Morning (04-08)"
    elif 8  <= hour < 12: return "Morning (08-12)"
    elif 12 <= hour < 16: return "Midday/Afternoon (12-16)"
    elif 16 <= hour < 20: return "Evening (16-20)"
    elif 20 <= hour < 24: return "Night (20-00)"
    else:                 return "Late Night (00-04)"


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    # IDs
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["customer_id"]    = df["customer_id"].astype(str)

    # Parse dates (dayfirst for DD/MM/YYYY format)
    for col in ["customer_dob", "transaction_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Fix 2-digit year: future dates parsed as 21xx → subtract 100 years
    future_mask = df["customer_dob"] > pd.Timestamp.today()
    df.loc[future_mask, "customer_dob"] -= pd.DateOffset(years=100)
    log.info(f"[Types] Fixed {future_mask.sum():,} future DOB values (2-digit year artifact)")

    # Validate and extract hour from HHMMSS integer
    col_str = df["transaction_time"].astype(str).str.zfill(6)
    invalid = col_str.apply(lambda x: x[2] in "6789" or x[4] in "6789")
    log.info(f"[Types] Invalid TransactionTime values: {invalid.sum():,}")

    df["hour"] = col_str.str[:2].astype(int)

    # Convert HHMMSS integer to clean time object (removes 1900-01-01 artifact)
    def to_time(x):
        s = f"{int(x):06d}"
        return pd.to_datetime(f"{s[:2]}:{s[2:4]}:{s[4:]}", format="%H:%M:%S").time()

    df["transaction_time"] = df["transaction_time"].apply(to_time)

    # Numeric
    df["cust_account_balance"]   = pd.to_numeric(df["cust_account_balance"],   errors="coerce")
    df["transaction_amount_inr"] = pd.to_numeric(df["transaction_amount_inr"], errors="coerce")

    log.info("[Types] Conversion complete")
    return df


# ── 5. Uniqueness check ───────────────────────────────────────────────────────
def check_uniqueness(df: pd.DataFrame) -> None:
    status = "unique" if df["transaction_id"].is_unique else "NOT unique"
    log.info(f"[Unique] transaction_id: {status}")
    log.info(f"[Unique] Unique customers: {df['customer_id'].nunique():,} / {len(df):,} rows")


# ── 6. Drop inconsistent KYC records ─────────────────────────────────────────
def drop_inconsistent_kyc(df: pd.DataFrame, id_col: str, kyc_fields: list) -> pd.DataFrame:
    """
    DOB and gender are KYC-verified fields.
    Statistical imputation is not permitted in banking — inconsistent records
    indicate identity corruption, system migration errors, or potential fraud.
    These customers are excluded entirely.
    """
    nunique = df.groupby(id_col)[kyc_fields].nunique()
    bad_ids = nunique[(nunique > 1).any(axis=1)].index
    bad_rows = df[df[id_col].isin(bad_ids)].shape[0]

    pct = 100 * bad_rows / len(df)
    log.warning(f"[KYC] {len(bad_ids):,} customers with inconsistent records "
                f"→ {bad_rows:,} rows removed ({pct:.1f}% of data)")
    log.warning("[KYC] Reason: DOB/gender are KYC fields — imputation not permitted")
    log.warning("[KYC] Mining conclusions reflect customers with verified identity only")

    df = df[~df[id_col].isin(bad_ids)]
    log.info(f"[KYC] Remaining: {len(df):,} rows")
    return df


# ── 7. Filter valid genders ───────────────────────────────────────────────────
def filter_gender(df: pd.DataFrame, valid: list) -> pd.DataFrame:
    before = len(df)
    df = df[df["cust_gender"].isin(valid)]
    log.info(f"[Gender] Filtered to {valid} — removed {before - len(df):,} rows")
    log.info(f"[Gender] Distribution:\n{df['cust_gender'].value_counts().to_string()}")
    return df


# ── 8. Merge location aliases ─────────────────────────────────────────────────
def merge_location_aliases(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    df["cust_location"] = df["cust_location"].replace(alias_map)
    for old, new in alias_map.items():
        log.info(f"[Location] Merged '{old}' → '{new}'")
    log.info(f"[Location] Top 5 after merge:\n{df['cust_location'].value_counts().head().to_string()}")
    return df


# ── 9. Compute age ────────────────────────────────────────────────────────────
def compute_age(df: pd.DataFrame, age_min: int, age_max: int) -> pd.DataFrame:
    df = df.dropna(subset=["customer_dob", "transaction_date"])

    # Elapsed days // 365 — avoids off-by-1 from year subtraction
    df["age"] = (df["transaction_date"] - df["customer_dob"]).dt.days // 365

    invalid = (df["age"] < age_min) | (df["age"] > age_max)
    log.info(f"[Age] Unrealistic ages (<{age_min} or >{age_max}): {invalid.sum():,} — removed")
    df = df[~invalid]
    log.info(f"[Age] Stats — min:{df['age'].min()}, max:{df['age'].max()}, "
             f"mean:{df['age'].mean():.1f}")
    return df


# ── 10. Extract temporal features ─────────────────────────────────────────────
def extract_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df["transaction_day_of_week"] = df["transaction_date"].dt.dayofweek  # 0=Mon
    df["transaction_month"]       = df["transaction_date"].dt.month
    df["is_weekend"]              = (df["transaction_date"].dt.dayofweek >= 5).astype(int)
    df["transaction_time_category"] = df["hour"].apply(_time_period)
    df["transaction_time_category"] = pd.Categorical(
        df["transaction_time_category"],
        categories=TIME_PERIOD_CATEGORIES,
        ordered=True,
    )
    log.info("[Temporal] Added: transaction_day_of_week, transaction_month, "
             "is_weekend, transaction_time_category")
    return df


# ── 11. Merge customer stats ───────────────────────────────────────────────────
def merge_customer_stats(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(stats, on="customer_id", how="left")
    log.info(f"[Stats] Merged customer stats — shape: {df.shape}")
    log.info(f"[Stats] customer_freq — mean: {df['customer_freq'].mean():.2f}, "
             f"max: {df['customer_freq'].max()}")
    return df


# ── 12. Age category and ordinal encoding ─────────────────────────────────────
def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    df["age_category"] = pd.cut(
        df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    df["age_ordinal"] = df["age_category"].map(AGE_ORDINAL_MAP).astype(int)
    log.info(f"[Encode] Age groups:\n{df['age_category'].value_counts().sort_index().to_string()}")
    return df


# ── 13. Encode categoricals ───────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Binary gender
    df["gender_encoded"] = df["cust_gender"].map({"M": 0, "F": 1})

    # Location grouping (Delhi already merged in step 8)
    top_locs = df["cust_location"].value_counts().nlargest(top_n).index
    df["location_grouped"] = df["cust_location"].apply(
        lambda x: x if x in top_locs else "OTHER"
    )
    log.info(f"[Encode] Location groups (top {top_n} + OTHER):\n"
             f"{df['location_grouped'].value_counts().to_string()}")

    return df


# ── 14. Drop raw columns & apply OHE ─────────────────────────────────────────
def apply_ohe_and_drop(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    # OHE with drop_first=True — avoids dummy variable trap
    df_final = pd.get_dummies(
        df,
        columns=["transaction_time_category", "location_grouped"],
        prefix=["time", "loc"],
        drop_first=True,
    )

    # Drop raw source columns
    existing = [c for c in cols_to_drop if c in df_final.columns]
    df_final = df_final.drop(columns=existing, errors="ignore")

    # Cast OHE bool → int (avoids dtype errors in sklearn / XGBoost)
    ohe_cols = [c for c in df_final.columns if c.startswith(("time_", "loc_"))]
    df_final[ohe_cols] = df_final[ohe_cols].astype(int)

    # Sanity check: no leftover non-numeric columns (except customer_id)
    non_num = [
        c for c in df_final.columns
        if c != "customer_id" and not pd.api.types.is_numeric_dtype(df_final[c])
    ]
    if non_num:
        log.warning(f"[OHE] Non-numeric columns still present — will be excluded from "
                    f"normalization: {non_num}")

    log.info(f"[OHE] Shape after encoding + drop: {df_final.shape}")
    log.info(f"[OHE] Columns: {df_final.columns.tolist()}")
    return df_final


# ── 15. Normalize numeric features ────────────────────────────────────────────
def normalize_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    missing_from_df = set(cols) - set(present)
    if missing_from_df:
        log.warning(f"[Norm] Columns not found, skipped: {missing_from_df}")

    pt = PowerTransformer(method="yeo-johnson")
    normalized = pt.fit_transform(df[present])
    norm_df = pd.DataFrame(
        normalized,
        columns=[c + "_normalized" for c in present],
        index=df.index,
    )

    df = df.drop(columns=present)
    df = pd.concat([df, norm_df], axis=1)
    log.info(f"[Norm] Yeo-Johnson applied to: {present}")
    return df


# ── 16. Drop correlated features ──────────────────────────────────────────────
def drop_correlated(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    log.info(f"[Corr] Dropped correlated features: {existing}")
    return df


# ── Master transform function ─────────────────────────────────────────────────
def transform(df: pd.DataFrame, customer_stats: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("START TRANSFORM")
    log.info("=" * 60)

    df = remove_duplicates(df)
    df = standardize_column_names(df)
    df = drop_missing(df)
    df = convert_types(df)
    check_uniqueness(df)
    df = drop_inconsistent_kyc(df, "customer_id", KYC_FIELDS)
    df = filter_gender(df, VALID_GENDERS)
    df = merge_location_aliases(df, LOCATION_ALIAS)
    df = compute_age(df, AGE_MIN, AGE_MAX)
    df = extract_temporal(df)
    df = merge_customer_stats(df, customer_stats)
    df = encode_age(df)
    df = encode_categoricals(df, TOP_N_LOCATIONS)
    df = apply_ohe_and_drop(df, COLS_TO_DROP)
    df = normalize_features(df, COLS_TO_NORMALIZE)
    df = drop_correlated(df, COLS_CORRELATED_DROP)

    log.info("=" * 60)
    log.info(f"TRANSFORM COMPLETE — final shape: {df.shape}")
    log.info("=" * 60)
    return df
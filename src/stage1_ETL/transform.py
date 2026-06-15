"""
Transform step — full cleaning and feature engineering pipeline.

Execution order matters:
  1.  Deduplicate
  2.  Standardize column names
  3.  Handle missing values  (before consistency check)
  4.  Type conversion
  5.  Check uniqueness
  6.  Outlier assessment  (report only — no removal)
  7.  Drop inconsistent KYC records  (DOB / gender — no imputation)
  8.  Filter valid gender values
  9.  Merge location aliases  (BEFORE grouping and OHE)
  10. Compute age  (elapsed days // 365)
  11. Extract temporal features  (BEFORE dropping transaction_date)
  12. Merge customer stats computed on raw data
  13. Build spending / demographic EDA helpers
  14. Encode categoricals  (binary, ordinal, OHE with drop_first)
  15. Cyclical month encoding  (sin/cos — replaces linear month normalization)
  16. Feature information assessment  (entropy + variance — report only)
  17. Drop raw / redundant columns
  18. Normalize numeric features with Yeo-Johnson
  19. Drop correlated features identified via heatmap
"""

import re
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
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
    """
    Drop all rows with any null values.

    Missing mechanism: MCAR (Missing Completely at Random), confirmed via:
      - Logistic regression pseudo-R² ≈ 0 (missingness unpredictable from other vars)
      - T-test: no significant difference in transaction_amount between missing /
        non-missing groups for cust_account_balance (the largest missing column)
      - Uniform missing rate across gender and location categories
    See scratch/missing_mechanism_analysis.py for full statistical evidence.

    Under MCAR, listwise deletion (dropna) is unbiased. Total loss < 0.7% of rows.
    KYC fields (DOB, gender) cannot be imputed regardless of mechanism — regulatory
    constraint. cust_account_balance (0.23% missing) is also MCAR so dropping is
    equally valid as median imputation with negligible impact.
    """
    before = len(df)
    missing = df.isnull().sum()
    cols_with_nulls = missing[missing > 0]
    if len(cols_with_nulls):
        log.info(f"[Missing] Columns with nulls:\n{cols_with_nulls.to_string()}")
    df = df.dropna()
    dropped = before - len(df)
    log.info(f"[Missing] Dropped {dropped:,} rows ({100*dropped/before:.2f}%) — "
             f"MCAR confirmed, listwise deletion is unbiased — {len(df):,} remaining")
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


# ── 6. Outlier assessment ─────────────────────────────────────────────────────
def assess_outliers(df: pd.DataFrame) -> None:
    """
    IQR-based outlier assessment — reports statistics, does NOT remove rows.

    Decision: outliers are RETAINED because:
      1. Unsupervised mining — no target variable to distort
      2. High-balance / high-value customers are a meaningful business segment
      3. Yeo-Johnson normalization compresses extreme values before clustering
    """
    cols = ["cust_account_balance", "transaction_amount_inr"]
    log.info("=" * 60)
    log.info("OUTLIER ASSESSMENT (IQR Method) — no rows removed")
    log.info("=" * 60)
    for col in cols:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_below = (df[col] < lower).sum()
        n_above = (df[col] > upper).sum()
        n_total = n_below + n_above
        pct = 100 * n_total / len(df)
        log.info(f"  {col}:")
        log.info(f"    Q1={q1:,.2f}  Q3={q3:,.2f}  IQR={iqr:,.2f}")
        log.info(f"    Fences: [{lower:,.2f}, {upper:,.2f}]")
        log.info(f"    Below fence: {n_below:,}  |  Above fence: {n_above:,}")
        log.info(f"    Total outliers: {n_total:,} ({pct:.1f}%)")
        log.info(f"    Range: [{df[col].min():,.2f}, {df[col].max():,.2f}]  "
                 f"Mean: {df[col].mean():,.2f}  Skew: {df[col].skew():.2f}")
    log.info("  DECISION: RETAIN outliers (see DECISIONS.MD)")
    log.info("=" * 60)


# ── 7. Drop inconsistent KYC records ─────────────────────────────────────────
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


# ── 15. Cyclical month encoding ───────────────────────────────────────────────
def encode_month_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace transaction_month (linear 1–12) with sin/cos pair.

    Rationale: month is cyclical — December (12) and January (1) are adjacent,
    not maximally distant. Yeo-Johnson on a linear month number would place
    December as far as possible from January in Euclidean space, distorting
    K-Means cluster boundaries along the temporal dimension.

    sin/cos encoding guarantees: distance(Dec, Jan) == distance(Jan, Feb).
    Values already lie in [−1, 1] — compatible with Yeo-Johnson-normalized
    features without additional scaling.
    """
    df["month_sin"] = np.sin(2 * np.pi * df["transaction_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["transaction_month"] / 12)
    log.info("[Encode] transaction_month → month_sin / month_cos (cyclical)")
    log.info(f"  month_sin range: [{df['month_sin'].min():.3f}, {df['month_sin'].max():.3f}]")
    log.info(f"  month_cos range: [{df['month_cos'].min():.3f}, {df['month_cos'].max():.3f}]")
    return df


# ── 16. Feature information assessment ───────────────────────────────────────
def assess_feature_information(df: pd.DataFrame) -> None:
    """
    Shannon entropy and variance per feature — report only, no rows removed.

    Decision: all features retained. In unsupervised mining each feature
    represents a distinct analytical dimension; removing a dimension (e.g.
    location, time) eliminates an entire analytical lens, not just noise.
    Entropy/variance inform relative contribution, not a drop threshold.
    See DECISIONS.MD for full justification.
    """
    exclude = {"customer_id"}
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    log.info("=" * 60)
    log.info("FEATURE INFORMATION ASSESSMENT (Entropy & Variance)")
    log.info("=" * 60)
    for col in numeric_cols:
        data = df[col].dropna()
        var  = data.var()
        if data.nunique() <= 20:
            probs = data.value_counts(normalize=True).values
        else:
            counts, _ = np.histogram(data, bins=20)
            counts = counts[counts > 0]
            probs  = counts / counts.sum()
        ent = scipy_entropy(probs, base=2)
        log.info(f"  {col:<45s} var={var:>12.4f}  entropy={ent:.4f} bits  "
                 f"unique={data.nunique()}")
    log.info("  DECISION: all features RETAINED (see DECISIONS.MD)")
    log.info("=" * 60)


# ── 17. Drop raw columns & apply OHE ─────────────────────────────────────────
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


# ── Base transform (cleaning only — no encoding / normalization) ──────────────
def transform_base(df: pd.DataFrame, customer_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning-only transform — preserves original column values.

    Applies: dedup → column standardization → null drop → type conversion →
    uniqueness check → outlier assessment → KYC consistency drop → gender
    filter → location alias merge → age computation → temporal extraction →
    customer stats merge.

    Returns a DataFrame with original-scale values suitable for discretization
    (Association Rule Mining) or exploratory analysis.
    """
    log.info("=" * 60)
    log.info("START TRANSFORM (base cleaning)")
    log.info("=" * 60)

    df = remove_duplicates(df)
    df = standardize_column_names(df)
    df = drop_missing(df)
    df = convert_types(df)
    check_uniqueness(df)
    assess_outliers(df)
    df = drop_inconsistent_kyc(df, "customer_id", KYC_FIELDS)
    df = filter_gender(df, VALID_GENDERS)
    df = merge_location_aliases(df, LOCATION_ALIAS)
    df = compute_age(df, AGE_MIN, AGE_MAX)
    df = extract_temporal(df)
    df = merge_customer_stats(df, customer_stats)

    log.info("=" * 60)
    log.info(f"BASE TRANSFORM COMPLETE — shape: {df.shape}")
    log.info("=" * 60)
    return df


# ── Full transform (encoding + normalization — for clustering) ────────────────
def transform(df: pd.DataFrame, customer_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Full transform pipeline — cleaning + encoding + normalization.

    Calls transform_base() for cleaning, then applies ordinal/OHE encoding,
    cyclical month encoding, feature information assessment, Yeo-Johnson
    normalization, and correlation-based feature drops.
    Output is suitable for distance-based algorithms (K-Means, DBSCAN).
    """
    df = transform_base(df, customer_stats)

    log.info("Continuing with encoding + normalization...")
    df = encode_age(df)
    df = encode_categoricals(df, TOP_N_LOCATIONS)
    df = encode_month_cyclical(df)
    df = apply_ohe_and_drop(df, COLS_TO_DROP)
    assess_feature_information(df)
    df = normalize_features(df, COLS_TO_NORMALIZE)
    df = drop_correlated(df, COLS_CORRELATED_DROP)

    log.info("=" * 60)
    log.info(f"TRANSFORM COMPLETE — final shape: {df.shape}")
    log.info("=" * 60)
    return df
"""
Pipeline configuration — all paths, constants, and feature settings in one place.
Edit this file to change behaviour without touching pipeline code.
"""

from pathlib import Path

# ── Project root (2 levels up from config/) ──────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = ROOT_DIR / "data" / "raw"       / "bank_transactions.csv"
PROCESSED_DATA_PATH    = ROOT_DIR / "data" / "stage1_ETL" / "clean.csv"
CATEGORICAL_DATA_PATH  = ROOT_DIR / "data" / "stage1_ETL" / "clean_categorical.csv"
LOG_DIR             = ROOT_DIR / "logs"

# ── Stage 2: Clustering ──────────────────────────────────────────────────────
CLUSTERING_INPUT_PATH  = PROCESSED_DATA_PATH          # reads clean.csv from stage 1
CLUSTERING_OUTPUT_DIR  = ROOT_DIR / "data" / "stage2_Clustering"

# ── Stage 3: Association Rule Mining ─────────────────────────────────────────
ASSOC_INPUT_PATH   = CATEGORICAL_DATA_PATH             # reads clean_categorical.csv (original-scale cleaned data)
ASSOC_OUTPUT_DIR   = ROOT_DIR / "data" / "stage3_ARM"

# Discretization bins — domain-justified for Indian banking context
BALANCE_BINS   = [0, 5_000, 25_000, 100_000, 500_000, float('inf')]
BALANCE_LABELS = ["Very Low (<5K)", "Low (5K-25K)", "Medium (25K-100K)",
                  "High (100K-500K)", "Very High (>500K)"]

TXN_AMOUNT_BINS   = [0, 100, 500, 2_000, 10_000, float('inf')]
TXN_AMOUNT_LABELS = ["Micro (<100)", "Small (100-500)", "Medium (500-2K)",
                     "Large (2K-10K)", "Very Large (>10K)"]

FREQ_BINS   = [0, 1, 3, float('inf')]
FREQ_LABELS = ["Single (1)", "Occasional (2-3)", "Frequent (4+)"]

MONTH_SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
    5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter",
}

# Apriori thresholds
APRIORI_MIN_SUPPORT    = 0.05     # 5% — ~35K transactions minimum
APRIORI_MIN_CONFIDENCE = 0.50     # 50%
APRIORI_MIN_LIFT       = 1.05     # Must exceed baseline co-occurrence
APRIORI_MAX_LEN        = 3        # Max items per itemset

# ── KYC fields — never imputed, only dropped if inconsistent ─────────────────
KYC_FIELDS = ["customer_dob", "cust_gender"]

# ── Location settings ─────────────────────────────────────────────────────────
LOCATION_ALIAS  = {"NEW DELHI": "DELHI"}   # merged before OHE
TOP_N_LOCATIONS = 15                       # rest → 'OTHER'

# ── Age settings ──────────────────────────────────────────────────────────────
AGE_MIN    = 0
AGE_MAX    = 100
AGE_BINS   = [0, 18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_ORDINAL_MAP = {
    "<18": 0, "18-24": 1, "25-34": 2, "35-44": 3,
    "45-54": 4, "55-64": 5, "65+": 6,
}

# ── Time period buckets ───────────────────────────────────────────────────────
TIME_PERIOD_CATEGORIES = [
    "Late Night (00-04)", "Early Morning (04-08)", "Morning (08-12)",
    "Midday/Afternoon (12-16)", "Evening (16-20)", "Night (20-00)",
]

# ── Valid gender values ───────────────────────────────────────────────────────
VALID_GENDERS = ["M", "F"]

# ── Columns dropped before export ─────────────────────────────────────────────
# customer_id intentionally NOT here — kept as joinable key
# transaction_month dropped here because cyclical encoding replaces it with
# month_sin / month_cos (applied in encode_month_cyclical before this drop)
COLS_TO_DROP = [
    "transaction_id", "customer_dob", "transaction_date",
    "transaction_time", "cust_location", "cust_gender",
    "age_category", "age", "hour",
    "transaction_month",   # replaced by month_sin / month_cos (cyclical encoding)
]

# ── Features normalized with Yeo-Johnson ─────────────────────────────────────
# transaction_month excluded — handled by cyclical sin/cos encoding instead
# age_ordinal included: Yeo-Johnson preserves ordinal ordering (monotonic
# transform) and is used solely to equalize scale with continuous features.
# With ~700K rows across 7 levels the lambda estimate is stable.
COLS_TO_NORMALIZE = [
    "cust_account_balance",
    "transaction_amount_inr",
    "customer_freq",
    "age_ordinal",
]

# ── Redundant columns removed after correlation analysis ─────────────────────
# Dropped separately from COLS_TO_DROP to make the architectural intent clear:
# these columns exist in the pipeline but are excluded due to high correlation,
# not because they are raw/helper columns.
#   avg_txn_amount      — r ≈ 1.00 with transaction_amount_inr (same transaction)
#   total_txn_amount    — r ≈ 1.00 with transaction_amount_inr (same transaction)
#   transaction_day_of_week — r = 0.78 with is_weekend (redundant encoding)
COLS_CORRELATED_DROP = [
    "avg_txn_amount",
    "total_txn_amount",
    "transaction_day_of_week",
]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE        = LOG_DIR / "pipeline.log"
LOG_FORMAT      = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
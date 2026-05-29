"""
Step 1 — Load clean_categorical.csv (already cleaned in Stage 1) and discretize
continuous variables into meaningful categorical groups for Apriori mining.

Justification for using clean_categorical.csv:
  This file retains the original ₹ values (no normalisation) after all Stage 1
  cleaning steps have been applied.  Discretizing these values produces
  domain-meaningful bins (e.g. "Balance: ₹25K–100K") without needing to
  re-run the entire ETL pipeline.

Columns available in clean_categorical.csv:
  customer_id, cust_gender, cust_location, cust_account_balance,
  transaction_amount_inr, age, transaction_month, is_weekend,
  transaction_time_category, customer_freq, avg_txn_amount, total_txn_amount
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.logger import get_logger

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent))
from config.config import (
    BALANCE_BINS, BALANCE_LABELS,
    TXN_AMOUNT_BINS, TXN_AMOUNT_LABELS,
    AGE_BINS, AGE_LABELS,
    FREQ_BINS, FREQ_LABELS,
    MONTH_SEASON_MAP, TOP_N_LOCATIONS,
)

log = get_logger(__name__)


# ── Discretization functions ──────────────────────────────────────────────────

def discretize_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin account balance into domain-relevant tiers based on Indian banking context.

    Rationale for bin boundaries:
      ₹0–5K      Very Low  — minimum-balance accounts, dormant/new customers
      ₹5K–25K    Low       — typical basic savings (avg Indian savings ~₹15K)
      ₹25K–100K  Medium    — active savers, salaried professionals
      ₹100K–500K High      — premium segment, fixed-deposit holders
      ₹500K+     Very High — high-net-worth / corporate accounts
    """
    df["balance_group"] = pd.cut(
        df["cust_account_balance"],
        bins=BALANCE_BINS,
        labels=BALANCE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)
    log.info(f"[Discretize] Balance groups:\n"
             f"{df['balance_group'].value_counts().sort_index().to_string()}")
    return df


def discretize_txn_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin transaction amount into spending categories.

    Rationale for bin boundaries:
      ₹0–100     Micro      — UPI micro-payments, chai/snack purchases
      ₹100–500   Small      — daily essentials, transport, meals
      ₹500–2K    Medium     — utility bills, online shopping
      ₹2K–10K    Large      — electronics, apparel, medical bills
      ₹10K+      Very Large — rent, EMI, large purchases, transfers
    """
    df["txn_amount_group"] = pd.cut(
        df["transaction_amount_inr"],
        bins=TXN_AMOUNT_BINS,
        labels=TXN_AMOUNT_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)
    log.info(f"[Discretize] Transaction amount groups:\n"
             f"{df['txn_amount_group'].value_counts().sort_index().to_string()}")
    return df


def discretize_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin age into life-stage groups using the same bins from Stage 1 config.

    Rationale: Same AGE_BINS used in ETL ensures consistency across stages.
      <18   — Minor / student
      18-24 — Young adult / college / first job
      25-34 — Early career / young professional
      35-44 — Mid-career / family building
      45-54 — Senior professional / peak earning
      55-64 — Pre-retirement
      65+   — Retired / senior citizen
    """
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
    ).astype(str)
    log.info(f"[Discretize] Age groups:\n"
             f"{df['age_group'].value_counts().sort_index().to_string()}")
    return df


def discretize_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin transaction frequency into engagement tiers.

    Rationale:
      1      Single     — one-time customer (majority of dataset)
      2-3    Occasional — returning customer
      4+     Frequent   — loyal / high-engagement customer
    """
    df["freq_group"] = pd.cut(
        df["customer_freq"],
        bins=FREQ_BINS,
        labels=FREQ_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)
    log.info(f"[Discretize] Frequency groups:\n"
             f"{df['freq_group'].value_counts().sort_index().to_string()}")
    return df


def discretize_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map transaction month to meteorological season.

    Rationale (Indian context):
      Winter (Dec–Feb) — festive aftermath, year-end spending
      Spring (Mar–May) — financial year-end, tax planning
      Summer (Jun–Aug) — vacation spending, school fees
      Autumn (Sep–Nov) — festive season (Diwali, Dussehra), peak spending
    """
    df["season"] = df["transaction_month"].map(MONTH_SEASON_MAP)
    log.info(f"[Discretize] Seasons:\n"
             f"{df['season'].value_counts().sort_index().to_string()}")
    return df


def discretize_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel 0/1 weekend flag to human-readable string for Apriori items."""
    df["day_type"] = df["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    log.info(f"[Discretize] Day type:\n"
             f"{df['day_type'].value_counts().to_string()}")
    return df


def discretize_location(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Group locations into top N + OTHER (same logic as Stage 1)."""
    top_locs = df["cust_location"].value_counts().nlargest(top_n).index
    df["location"] = df["cust_location"].apply(
        lambda x: x if x in top_locs else "OTHER"
    )
    log.info(f"[Discretize] Location groups (top {top_n} + OTHER):\n"
             f"{df['location'].value_counts().to_string()}")
    return df


def discretize_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Rename gender column for clarity in rules."""
    df["gender"] = df["cust_gender"]
    log.info(f"[Discretize] Gender:\n"
             f"{df['gender'].value_counts().to_string()}")
    return df


def discretize_time(df: pd.DataFrame) -> pd.DataFrame:
    """Use the already-computed time category from Stage 1 ETL."""
    df["time_period"] = df["transaction_time_category"].astype(str)
    log.info(f"[Discretize] Time period:\n"
             f"{df['time_period'].value_counts().to_string()}")
    return df


# ── Item-column definitions ───────────────────────────────────────────────────
# Each column listed here will be prefixed with its name and used as an Apriori item.
ITEM_COLUMNS = [
    "balance_group",
    "txn_amount_group",
    "age_group",
    "freq_group",
    "season",
    "day_type",
    "location",
    "gender",
    "time_period",
]


def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    """
    Convert the discretized DataFrame into a list of transaction baskets
    for the Apriori algorithm.

    Each transaction is a list of items like:
      ["balance=Medium (25K-100K)", "gender=F", "time_period=Evening (16-20)", ...]
    """
    transactions = []
    for _, row in df[ITEM_COLUMNS].iterrows():
        basket = [f"{col}={row[col]}" for col in ITEM_COLUMNS]
        transactions.append(basket)

    log.info(f"[Transactions] Built {len(transactions):,} transaction baskets "
             f"with {len(ITEM_COLUMNS)} item dimensions")
    log.info(f"[Transactions] Sample basket: {transactions[0]}")
    return transactions


# ── Master discretize function ────────────────────────────────────────────────

def load_and_discretize(input_path: Path) -> tuple[pd.DataFrame, list[list[str]]]:
    """
    Full Step 1 — load clean_categorical.csv (pre-cleaned by Stage 1 ETL)
    and discretize all continuous variables into categorical items.

    Parameters
    ----------
    input_path : Path — path to clean_categorical.csv

    Returns
    -------
    df_discretized : DataFrame — cleaned data with discretized columns
    transactions   : list[list[str]] — Apriori-ready transaction baskets
    """
    log.info("=" * 60)
    log.info("STEP 1 : LOAD & DISCRETIZE")
    log.info("=" * 60)

    # ── Load pre-cleaned data from Stage 1 ────────────────────────────────
    log.info(f"Loading pre-cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    log.info(f"Columns: {list(df.columns)}")

    # ── Discretize continuous variables ───────────────────────────────────
    log.info("Discretizing continuous variables into categorical bins...")
    df = discretize_balance(df)
    df = discretize_txn_amount(df)
    df = discretize_age(df)
    df = discretize_frequency(df)
    df = discretize_season(df)
    df = discretize_weekend(df)
    df = discretize_location(df, TOP_N_LOCATIONS)
    df = discretize_gender(df)
    df = discretize_time(df)

    # ── Build transaction baskets ─────────────────────────────────────────
    transactions = build_transactions(df)

    log.info("=" * 60)
    log.info(f"DISCRETIZATION COMPLETE — {len(transactions):,} transactions ready")
    log.info("=" * 60)

    return df, transactions

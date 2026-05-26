"""
Step 2 — Apply the Apriori algorithm to discover frequent itemsets,
generate association rules, and compute Support, Confidence, and Lift.

DM Concepts applied:
  • Frequent pattern mining (Apriori algorithm)
  • Rule interestingness measures: Support, Confidence, Lift
  • Threshold-based filtering to retain only non-trivial findings
"""

import pandas as pd
# pyrefly: ignore [missing-import]
from mlxtend.preprocessing import TransactionEncoder
# pyrefly: ignore [missing-import]
from mlxtend.frequent_patterns import apriori, association_rules

from src.logger import get_logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    APRIORI_MIN_SUPPORT,
    APRIORI_MIN_CONFIDENCE,
    APRIORI_MIN_LIFT,
    APRIORI_MAX_LEN,
)

log = get_logger(__name__)


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    """
    Convert list-of-lists transactions into a one-hot boolean DataFrame
    suitable for mlxtend's Apriori implementation.

    Each column represents a unique item (e.g. "balance=High (100K-500K)"),
    each row is True/False for that item's presence in the transaction.
    """
    log.info("Encoding transaction baskets into boolean matrix...")
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    log.info(f"  Transaction matrix shape: {df_encoded.shape}")
    log.info(f"  Total unique items: {len(te.columns_)}")
    return df_encoded


def find_frequent_itemsets(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """
    Run the Apriori algorithm to discover frequent itemsets.

    Threshold justification:
      min_support = 0.05 (5%) ensures each itemset appears in at least ~35K
      transactions out of ~700K — frequent enough to be statistically meaningful
      while keeping memory usage manageable for the Apriori algorithm.
      max_len = 3 limits combinatorial explosion while allowing multi-attribute rules.
    """
    log.info(f"Running Apriori (min_support={APRIORI_MIN_SUPPORT}, "
             f"max_len={APRIORI_MAX_LEN})...")

    frequent_itemsets = apriori(
        df_encoded,
        min_support=APRIORI_MIN_SUPPORT,
        use_colnames=True,
        max_len=APRIORI_MAX_LEN,
        low_memory=True,
    )

    log.info(f"  Frequent itemsets found: {len(frequent_itemsets):,}")

    if len(frequent_itemsets) == 0:
        log.warning("  No frequent itemsets found! Try lowering min_support.")
        return frequent_itemsets

    # Summary by itemset length
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    for k in sorted(frequent_itemsets["length"].unique()):
        count = (frequent_itemsets["length"] == k).sum()
        log.info(f"    {k}-itemsets: {count:,}")

    return frequent_itemsets


def generate_rules(frequent_itemsets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Generate association rules from frequent itemsets and compute
    Support, Confidence, and Lift for each rule.

    Filtering strategy:
      1. Confidence >= 0.50 — the rule must be right at least half the time
      2. Lift >= 1.05 — the association must be meaningfully stronger than
         random co-occurrence (lift=1.0 means independent)
      3. Rules sorted by Lift (desc) then Confidence (desc) — most
         surprising rules surface first

    Why these thresholds:
      • Confidence 50% avoids weak rules that hold less than half the time
      • Lift 1.05 ensures rules are at least 5% more likely than baseline
        — a lift of 1.0 means A and B are statistically independent
      • Together they filter out trivially obvious patterns while retaining
        genuinely interesting co-occurrences

    Returns
    -------
    rules_filtered : DataFrame — filtered and sorted association rules
    raw_rule_count : int — number of rules before lift filter (conf ≥ threshold)
    """
    if len(frequent_itemsets) == 0:
        log.warning("No frequent itemsets — skipping rule generation.")
        return pd.DataFrame(), 0

    log.info(f"Generating association rules (min_confidence={APRIORI_MIN_CONFIDENCE})...")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=APRIORI_MIN_CONFIDENCE,
        num_itemsets=len(frequent_itemsets),
    )

    raw_rule_count = len(rules)
    log.info(f"  Raw rules (conf ≥ {APRIORI_MIN_CONFIDENCE}): {raw_rule_count:,}")

    if len(rules) == 0:
        log.warning("  No rules meet confidence threshold.")
        return rules, 0

    # ── Apply lift filter ─────────────────────────────────────────────────
    rules_filtered = rules[rules["lift"] >= APRIORI_MIN_LIFT].copy()
    log.info(f"  After lift filter (≥ {APRIORI_MIN_LIFT}): {len(rules_filtered):,} rules")
    log.info(f"  Filtering removed {raw_rule_count - len(rules_filtered):,} trivial rules")

    if len(rules_filtered) == 0:
        log.warning("  No rules meet lift threshold. Returning unfiltered rules.")
        rules_filtered = rules.copy()

    # ── Sort by Lift (desc), then Confidence (desc) ───────────────────────
    rules_filtered = rules_filtered.sort_values(
        by=["lift", "confidence"], ascending=[False, False]
    ).reset_index(drop=True)

    # ── Clean up frozenset display ────────────────────────────────────────
    rules_filtered["antecedents_str"] = rules_filtered["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules_filtered["consequents_str"] = rules_filtered["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    # ── Log top 15 rules ─────────────────────────────────────────────────
    log.info("\n  TOP 15 ASSOCIATION RULES (sorted by Lift):")
    log.info(f"  {'#':<4} {'Lift':>6} {'Conf':>6} {'Supp':>6}  Rule")
    log.info("  " + "-" * 80)

    for i, row in rules_filtered.head(15).iterrows():
        log.info(
            f"  {i+1:<4} {row['lift']:6.2f} {row['confidence']:6.3f} "
            f"{row['support']:6.4f}  "
            f"{row['antecedents_str']} → {row['consequents_str']}"
        )

    return rules_filtered, raw_rule_count


def run_mining(transactions: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Full Step 2 — encode → Apriori → generate rules.

    Returns
    -------
    frequent_itemsets : DataFrame — all frequent itemsets with support
    rules            : DataFrame — filtered association rules with metrics
    raw_rule_count   : int — number of rules before lift filtering
    """
    log.info("=" * 60)
    log.info("STEP 2 : APRIORI MINING & RULE GENERATION")
    log.info("=" * 60)

    df_encoded = encode_transactions(transactions)
    frequent_itemsets = find_frequent_itemsets(df_encoded)
    rules, raw_rule_count = generate_rules(frequent_itemsets)

    log.info("=" * 60)
    log.info(f"MINING COMPLETE — {len(rules):,} filtered rules (from {raw_rule_count:,} candidates)")
    log.info("=" * 60)

    return frequent_itemsets, rules, raw_rule_count

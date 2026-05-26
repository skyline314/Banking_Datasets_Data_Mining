# Stage 3: Association Rule Mining — Decision Log

> This document records every decision made during the Association Rule Mining
> phase, including discretization rationale, threshold choices, and filtering
> strategy.

---

## 1. Data Source — Raw CSV Instead of clean.csv

**Decision:** Read the original `bank_transactions.csv` instead of the Stage 1
output `clean.csv`.

**Justification:**
`clean.csv` contains Yeo-Johnson normalised floats — the balance column has
values like `-0.787` instead of `₹2,270`. Discretizing normalised values into
bins like "₹25K–100K" is meaningless because the normalisation is a non-linear
power transform.

To create domain-meaningful categories, we need the original rupee values.
The same cleaning steps from Stage 1 (dedup, null drop, KYC, gender filter,
age computation) are reused by importing the functions directly — no code
duplication occurs.

---

## 2. Balance Discretization Bins

**Decision:** Use 5 bins: Very Low (<₹5K), Low (₹5K–25K), Medium (₹25K–100K),
High (₹100K–500K), Very High (>₹500K).

**Justification:**
- ₹5K is the typical minimum balance requirement for Indian savings accounts
- ₹25K represents the average savings balance for salaried individuals
- ₹100K marks the transition to premium banking segments
- ₹500K is the threshold for high-net-worth (HNI) classification in most
  Indian banks

These bins align with how Indian banks internally segment their customers
for product targeting and relationship management.

---

## 3. Transaction Amount Bins

**Decision:** Use 5 bins: Micro (<₹100), Small (₹100–500), Medium (₹500–2K),
Large (₹2K–10K), Very Large (>₹10K).

**Justification:**
- ₹100 is the typical UPI micro-payment ceiling (chai, snacks, auto-rickshaw)
- ₹500 covers daily essentials and transport
- ₹2K represents bill payments and online shopping
- ₹10K marks significant purchases (electronics, medical)
- Above ₹10K includes rent, EMI, and large fund transfers

---

## 4. Age Bins — Reuse from Stage 1

**Decision:** Use the same AGE_BINS defined in config.py that Stage 1 uses.

**Justification:** Ensures consistency across all pipeline stages. The same
customer aged 32 is classified as "25-34" in both clustering and association
mining, allowing cross-stage comparisons.

---

## 5. Frequency Bins

**Decision:** Use 3 bins: Single (1), Occasional (2-3), Frequent (4+).

**Justification:** After the KYC consistency drop in Stage 1, 99.75% of
remaining customers have exactly 1 transaction. The bins are designed to
capture the rare but analytically interesting multi-transaction customers.

---

## 6. Season Mapping

**Decision:** Map months to meteorological seasons with Indian context.

**Justification:** Indian consumer spending is strongly seasonal — Autumn
(Sep–Nov) is the festive season with peak spending. Quarterly grouping
captures this better than individual months, which would create 12 sparse
item categories.

---

## 7. Apriori Min Support = 0.05 (5%)

**Decision:** Set minimum support to 5%.

**Justification:** With ~700K transactions, 5% support means an itemset
must appear in at least ~35K transactions. This is:
- High enough to ensure statistical significance and prevent memory issues
- Low enough to capture major behavioral patterns
- Standard for large transaction databases in literature

Lower values (0.02) caused memory exhaustion during Apriori with 700K rows
and 47 unique items, producing an allocation request of ~19.5 GiB.

---

## 8. Confidence Threshold = 0.50 (50%)

**Decision:** Require at least 50% confidence for all exported rules.

**Justification:** A rule with <50% confidence is wrong more often than it
is right — it would be misleading to present as a "finding". The 50%
threshold ensures every rule holds at least half the time.

---

## 9. Lift Threshold = 1.05

**Decision:** Require lift ≥ 1.05 for filtered rules.

**Justification:** Lift = 1.0 means the antecedent and consequent are
statistically independent — no real association exists. Lift = 1.05 means
the combination is 5% more likely than baseline co-occurrence.

Initial threshold of 1.2 was too aggressive — with 5% support, 834 rules
passed confidence ≥ 0.50 but only 1 had lift ≥ 1.2. Lowering to 1.05
captures enough rules for a meaningful report while still excluding
statistically independent patterns.

---

## 10. Item Prefix Naming Convention

**Decision:** Prefix each item with its dimension name (e.g.,
`balance_group=High (100K-500K)`).

**Justification:** Without prefixes, a rule like `{High, M, Weekend}` is
ambiguous — "High" what? Prefixed items make rules self-documenting and
prevent cross-dimension confusion.

---

## 11. Diversity-Aware Rule Selection (Top Rules)

**Decision:** Select top rules using a round-robin diversity strategy across
consequent dimensions instead of simple top-N by lift.

**Justification:** Sorting purely by lift caused all top-10 rules to predict
`gender=M` (due to ~72% class prevalence in the dataset). While each rule was
statistically valid, presenting 10 variants of the same finding adds no
analytical value. The diversity-aware selection ensures the documented rules
cover different dimensions (demographic, temporal, geographic, spending),
providing genuinely distinct insights.

---

*Document version: 2.0 — updated with diversity-aware rule selection decision.*

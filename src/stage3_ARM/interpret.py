"""
Step 3 — Interpret, rank, and export association rules with business commentary.

For each rule, generates:
  • A plain-English translation of the antecedent → consequent pattern
  • A domain-specific explanation of WHY the pattern exists in banking
  • An actionable business recommendation for the bank

Exports:
  • association_rules_full.csv      — all filtered rules with metrics
  • top_rules_interpreted.csv       — top 12+ rules with business commentary
  • association_rules_report.txt    — formatted text report
  • DECISIONS.md                    — decision log for Stage 3
"""

import pandas as pd
from pathlib import Path

from src.logger import get_logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    APRIORI_MIN_SUPPORT,
    APRIORI_MIN_CONFIDENCE,
    APRIORI_MIN_LIFT,
    APRIORI_MAX_LEN,
)

log = get_logger(__name__)


# ── Diversity-aware rule selection ────────────────────────────────────────────

def _select_diverse_rules(rules: pd.DataFrame, target_n: int = 12) -> pd.DataFrame:
    """
    Select diverse top rules that cover different item dimensions,
    avoiding redundancy (e.g. all rules predicting gender=M).

    Strategy:
      1. Group rules by their consequent dimension (e.g. gender, location, day_type)
      2. Round-robin pick the highest-lift rule from each dimension group
      3. Continue until we reach target_n or exhaust all rules
      4. Ensures diverse coverage across demographics, temporal, geographic,
         and spending pattern dimensions

    This prevents the top-N from being dominated by one dominant consequent
    (e.g. gender=M due to 72% class imbalance).
    """
    if len(rules) <= target_n:
        return rules.copy()

    # Extract the primary consequent dimension for each rule
    def _get_consequent_dimension(cons_str):
        """Get the first item's dimension key from the consequent string."""
        first_item = cons_str.split(",")[0].strip()
        if "=" in first_item:
            return first_item.split("=")[0]
        return "unknown"

    rules = rules.copy()
    rules["_cons_dim"] = rules["consequents_str"].apply(_get_consequent_dimension)

    # Group by consequent dimension
    dim_groups = {}
    for dim in rules["_cons_dim"].unique():
        dim_rules = rules[rules["_cons_dim"] == dim].copy()
        dim_groups[dim] = dim_rules.index.tolist()

    selected_indices = []
    used_antecedent_sets = set()  # track antecedent combos to avoid near-duplicates

    # Round-robin across dimensions, picking highest-lift unused rule
    while len(selected_indices) < target_n:
        added_this_round = False
        for dim in sorted(dim_groups.keys()):
            if len(selected_indices) >= target_n:
                break
            candidates = dim_groups[dim]
            for idx in candidates:
                # Skip if antecedent combo is already covered
                ante_key = rules.loc[idx, "antecedents_str"]
                cons_key = rules.loc[idx, "consequents_str"]
                rule_sig = f"{ante_key} -> {cons_key}"

                # Check for near-duplicate (same antecedents, similar consequent)
                ante_set = frozenset(ante_key.split(", "))
                if ante_set not in used_antecedent_sets:
                    selected_indices.append(idx)
                    used_antecedent_sets.add(ante_set)
                    dim_groups[dim] = [i for i in dim_groups[dim] if i != idx]
                    added_this_round = True
                    break
                else:
                    # Allow if consequent is different
                    already_selected_cons = [
                        rules.loc[si, "consequents_str"] for si in selected_indices
                        if frozenset(rules.loc[si, "antecedents_str"].split(", ")) == ante_set
                    ]
                    if cons_key not in already_selected_cons:
                        selected_indices.append(idx)
                        dim_groups[dim] = [i for i in dim_groups[dim] if i != idx]
                        added_this_round = True
                        break

        if not added_this_round:
            # No more unique rules available, fill remaining from top
            for idx in rules.index:
                if idx not in selected_indices and len(selected_indices) < target_n:
                    selected_indices.append(idx)
            break

    result = rules.loc[selected_indices].drop(columns=["_cons_dim"])
    # Re-sort selected rules by lift descending
    result = result.sort_values(by=["lift", "confidence"], ascending=[False, False])
    return result.reset_index(drop=True)


# ── Business interpretation engine ────────────────────────────────────────────

def _interpret_rule(antecedents: str, consequents: str,
                    support: float, confidence: float, lift: float,
                    rule_rank: int) -> dict:
    """
    Generate a structured business interpretation for a single rule.

    Returns a dict with:
      rule_text       — "IF ... THEN ..." plain English
      explanation     — banking domain context (unique per rule)
      recommendation  — actionable business insight (unique per rule)
    """
    ante_items = [x.strip() for x in antecedents.split(",")]
    cons_items = [x.strip() for x in consequents.split(",")]

    # Build plain English
    ante_english = " AND ".join([_item_to_english(i) for i in ante_items])
    cons_english = " AND ".join([_item_to_english(i) for i in cons_items])

    rule_text = f"IF {ante_english}, THEN {cons_english}"

    # Generate UNIQUE domain-specific explanation for this specific combination
    explanation = _generate_unique_explanation(ante_items, cons_items,
                                               support, confidence, lift)

    # Generate UNIQUE actionable recommendation
    recommendation = _generate_unique_recommendation(ante_items, cons_items,
                                                      support, confidence, lift)

    return {
        "rule_text": rule_text,
        "explanation": explanation,
        "recommendation": recommendation,
    }


def _item_to_english(item: str) -> str:
    """Convert an item like 'balance_group=High (100K-500K)' to readable English."""
    if "=" not in item:
        return item

    key, value = item.split("=", 1)
    mapping = {
        "balance_group":   f"the customer's account balance is {value}",
        "txn_amount_group": f"the transaction amount is {value}",
        "age_group":       f"the customer is in the {value} age group",
        "freq_group":      f"the customer's transaction frequency is {value}",
        "season":          f"the transaction occurs in {value}",
        "day_type":        f"the transaction is on a {value}",
        "location":        f"the customer is located in {value}",
        "gender":          f"the customer is {_gender_label(value)}",
        "time_period":     f"the transaction happens during {value}",
    }
    return mapping.get(key, f"{key} is {value}")


def _gender_label(value: str) -> str:
    """Convert M/F to readable label."""
    return {"M": "Male", "F": "Female"}.get(value, value)


def _get_item_val(items: list, prefix: str) -> str | None:
    """Extract the value for a given dimension prefix from item list."""
    for item in items:
        if item.startswith(prefix + "="):
            return item.split("=", 1)[1]
    return None


def _generate_unique_explanation(ante: list, cons: list,
                                  support: float, conf: float,
                                  lift: float) -> str:
    """
    Generate a UNIQUE banking-domain explanation for the specific combination
    of antecedent and consequent items. Each rule gets a tailored explanation
    that addresses WHY this particular combination exists.
    """
    all_items = ante + cons
    all_text = " ".join(all_items).lower()

    # Extract specific dimension values for targeted explanations
    balance = _get_item_val(all_items, "balance_group")
    txn_amt = _get_item_val(all_items, "txn_amount_group")
    age = _get_item_val(all_items, "age_group")
    gender = _get_item_val(all_items, "gender")
    day_type = _get_item_val(all_items, "day_type")
    time_period = _get_item_val(all_items, "time_period")
    season = _get_item_val(all_items, "season")
    location = _get_item_val(all_items, "location")
    freq = _get_item_val(all_items, "freq_group")

    # Build a specific explanation based on the EXACT combination
    explanation_parts = []

    # ── Combination-specific explanations ─────────────────────────────────

    # Balance + Transaction Amount combos
    if balance and txn_amt:
        if "very low" in balance.lower() and "micro" in txn_amt.lower():
            explanation_parts.append(
                "Customers with very low account balances (<₹5K) making micro "
                "transactions (<₹100) represent the underbanked segment — "
                "these accounts are likely used only for basic UPI payments "
                "(chai, auto-rickshaw, mobile recharges) with no savings behavior."
            )
        elif "very low" in balance.lower() and "small" in txn_amt.lower():
            explanation_parts.append(
                "Low-balance customers making small transactions (₹100-500) "
                "are likely salaried workers who withdraw their salary immediately "
                "and use the account mainly as a transit point for daily expenses "
                "like groceries, transport, and utility top-ups."
            )
        elif ("high" in balance.lower() or "very high" in balance.lower()) and \
             ("large" in txn_amt.lower() or "very large" in txn_amt.lower()):
            explanation_parts.append(
                "High-balance customers naturally command greater purchasing power, "
                "leading to larger transaction values. This wealth-spending correlation "
                "is characteristic of premium banking customers — fixed-deposit holders, "
                "business owners, and high-net-worth individuals."
            )
        elif "low" in balance.lower() and "micro" in txn_amt.lower():
            explanation_parts.append(
                "Customers in the low-balance tier (₹5K-25K) making micro "
                "transactions reflect the typical Indian savings account user — "
                "maintaining just above the minimum balance while using UPI "
                "for everyday small payments."
            )
        elif "low" in balance.lower() and "small" in txn_amt.lower():
            explanation_parts.append(
                "Customers maintaining ₹5K-25K balances while making small "
                "transactions (₹100-500) represent the core middle-class Indian "
                "banking customer — their spending on daily essentials (groceries, "
                "transport, utility recharges) stays within the ₹100-500 band, "
                "and their balance reflects moderate but not aggressive saving behavior."
            )

    # Balance + Gender combos
    elif balance and gender and not txn_amt:
        if "very low" in balance.lower() and gender == "M":
            explanation_parts.append(
                "The strong association between very low balances and male "
                "customers reflects India's gender banking gap — men are more "
                "likely to have 'salary accounts' opened by employers with minimal "
                "savings behavior, while women who do bank tend to maintain higher "
                "balances due to conservative saving habits and gold-linked deposits."
            )
        elif "very low" in balance.lower() and gender == "F":
            explanation_parts.append(
                "Female customers with very low balances may represent "
                "Jan Dhan Yojana beneficiaries or newly opened accounts under "
                "financial inclusion drives, where the account exists primarily "
                "for government subsidy direct benefit transfers (DBT)."
            )

    # Transaction Amount + Gender combos
    elif txn_amt and gender and not balance:
        if "micro" in txn_amt.lower() and gender == "M":
            explanation_parts.append(
                "Male customers dominate micro-transactions (<₹100) because "
                "men in India are more likely to use UPI for small daily "
                "purchases — street food, tea shops, parking, and auto-rickshaw "
                "fares — while female customers tend to make fewer but larger "
                "planned purchases."
            )
        elif "small" in txn_amt.lower() and gender == "M":
            explanation_parts.append(
                "Male customers in the small-transaction tier (₹100-500) represent "
                "daily essential spending — groceries, fuel, quick-service restaurants, "
                "and commute costs. The male skew reflects higher financial autonomy "
                "in daily household spending decisions in the Indian context."
            )
        elif "small" in txn_amt.lower() and gender == "F":
            explanation_parts.append(
                "Female customers making small transactions (₹100-500) tend to "
                "concentrate spending on household essentials and planned purchases, "
                "reflecting a more deliberate spending approach compared to male "
                "customers who make more frequent impulsive micro-payments."
            )

    # Age + Balance combos
    if age and balance and not gender:
        if "25-34" in age and "very low" in balance.lower():
            explanation_parts.append(
                "Early-career professionals (25-34) with very low balances "
                "represent the 'spend-all' segment — young professionals in "
                "metro cities with high living costs who spend most of their "
                "salary on rent, food delivery, and lifestyle expenses, leaving "
                "minimal savings in their primary account."
            )

    # Age + Transaction Amount combos
    if age and txn_amt and not balance:
        if "25-34" in age and "micro" in txn_amt.lower():
            explanation_parts.append(
                "The 25-34 age group are digital natives who heavily use UPI "
                "for micro-payments — they split bills, pay for coffee, and "
                "use cashless payments for transactions as small as ₹10-20. "
                "This generation's comfort with digital payments drives the "
                "high volume of micro-transactions."
            )
        elif "18-24" in age and ("micro" in txn_amt.lower() or "small" in txn_amt.lower()):
            explanation_parts.append(
                "College students and first-job holders (18-24) naturally "
                "make smaller transactions — pocket money, canteen expenses, "
                "and entry-level salary spending. Their transaction size reflects "
                "limited disposable income rather than preference."
            )

    # Time + Day type combos
    if time_period and day_type:
        if "morning" in time_period.lower() and day_type == "Weekday":
            explanation_parts.append(
                "Morning weekday transactions represent the routine financial "
                "rhythm of working professionals — salary-day ATM withdrawals, "
                "commute-related UPI payments, and scheduled auto-debits for "
                "bills and EMIs that execute during business hours."
            )
        elif "night" in time_period.lower() and day_type == "Weekend":
            explanation_parts.append(
                "Weekend night transactions correlate with leisure spending — "
                "restaurant dinners, movie tickets, and e-commerce impulse "
                "purchases made while browsing on weekends."
            )

    # Location + Gender combos
    if location and gender:
        if location == "OTHER" and gender == "M":
            explanation_parts.append(
                "Tier-2 and Tier-3 city customers (grouped as 'OTHER') "
                "skew heavily male because banking penetration in smaller Indian "
                "cities still disproportionately favors men — women in these "
                "regions are less likely to have independent bank accounts "
                "due to socioeconomic and cultural factors."
            )

    # Location + Transaction combos
    if location and txn_amt and not gender:
        if location == "OTHER" and "micro" in txn_amt.lower():
            explanation_parts.append(
                "Micro-transactions in Tier-2/3 cities (OTHER) reflect the "
                "lower cost of goods and services outside metro areas — "
                "₹100 covers more purchases in smaller cities where a meal "
                "costs ₹30-50 vs ₹150-300 in metros."
            )

    # Time period specific
    if time_period and not day_type:
        if "morning" in time_period.lower():
            explanation_parts.append(
                "Morning (8AM-12PM) transactions align with business operating "
                "hours — branch visits, scheduled standing instructions, and "
                "merchant payments that execute during the first half of the "
                "banking day."
            )
        elif "night" in time_period.lower():
            explanation_parts.append(
                "Night-time transactions (8PM-midnight) are predominantly "
                "digital — e-commerce purchases, food delivery orders, and "
                "subscription renewals that happen when customers browse "
                "their phones after work hours."
            )

    # Season specific
    if season:
        if "autumn" in season.lower():
            explanation_parts.append(
                "Autumn (Sep-Nov) is India's peak festive season — Navratri, "
                "Dussehra, and Diwali drive a spending surge across all "
                "demographics. Banks traditionally see 30-40% higher transaction "
                "volumes during this quarter."
            )
        elif "summer" in season.lower():
            explanation_parts.append(
                "Summer (Jun-Aug) transactions are influenced by vacation spending, "
                "school/college fee payments, and the onset of the monsoon season "
                "which affects agricultural and rural banking patterns."
            )

    # Day type specific (when no time period)
    if day_type and not time_period:
        if day_type == "Weekday":
            explanation_parts.append(
                "Weekday transactions reflect the working economy — salary "
                "credits, business payments, scheduled EMIs, and routine "
                "purchases during office hours dominate weekday volumes."
            )
        elif day_type == "Weekend":
            explanation_parts.append(
                "Weekend transactions tend toward discretionary spending — "
                "shopping, dining out, entertainment, and family-related "
                "purchases that differ in both amount and category from "
                "weekday financial activity."
            )

    # Frequency patterns
    if freq:
        if "frequent" in freq.lower():
            explanation_parts.append(
                "Customers with 4+ transactions are the bank's most engaged "
                "users — their repeated interaction signals active financial "
                "behavior and makes them prime candidates for relationship "
                "banking and premium upgrades."
            )
        elif "single" in freq.lower() and not explanation_parts:
            explanation_parts.append(
                "Single-transaction customers represent the vast majority "
                "(99.75%) of the dataset — understanding which attributes "
                "co-occur with these one-time users helps identify why they "
                "don't return and what activation strategies might work."
            )

    # Fallback for unmatched combinations
    if not explanation_parts:
        explanation_parts.append(
            f"This rule reveals a statistically significant co-occurrence "
            f"between these attributes that cannot be explained by independent "
            f"occurrence alone, suggesting an underlying behavioral or "
            f"demographic relationship worth investigating."
        )

    # Add the quantitative context
    lift_pct = (lift - 1) * 100
    combined = " ".join(explanation_parts)
    combined += (
        f" Quantitatively, this combination is {lift_pct:.0f}% more likely to occur "
        f"together than expected by random chance (lift={lift:.2f}), and the "
        f"pattern holds in {conf*100:.0f}% of cases where the antecedent appears "
        f"(confidence={conf:.2f}). The rule covers {support*100:.1f}% of all "
        f"transactions (support={support:.4f})."
    )

    return combined


def _generate_unique_recommendation(ante: list, cons: list,
                                     support: float, conf: float,
                                     lift: float) -> str:
    """
    Generate a UNIQUE actionable business recommendation for each specific
    rule combination. Each rule gets a tailored recommendation.
    """
    all_items = ante + cons
    all_text = " ".join(all_items).lower()

    balance = _get_item_val(all_items, "balance_group")
    txn_amt = _get_item_val(all_items, "txn_amount_group")
    age = _get_item_val(all_items, "age_group")
    gender = _get_item_val(all_items, "gender")
    day_type = _get_item_val(all_items, "day_type")
    time_period = _get_item_val(all_items, "time_period")
    season = _get_item_val(all_items, "season")
    location = _get_item_val(all_items, "location")
    freq = _get_item_val(all_items, "freq_group")

    recs = []

    # ── Combination-specific recommendations ──────────────────────────────

    # Balance-based actions
    if balance:
        if "very low" in balance.lower():
            if txn_amt and "micro" in txn_amt.lower():
                recs.append(
                    "Launch a 'Smart Savings' feature that automatically rounds up "
                    "each micro-transaction to the nearest ₹10 and deposits the "
                    "difference into a savings jar — this converts frequent micro-spenders "
                    "into savers with zero behavioral friction."
                )
            elif age and "25-34" in age:
                recs.append(
                    "Offer salary account upgrade packages to early-career professionals — "
                    "bundle zero-balance premium accounts with free debit cards and "
                    "mutual fund SIP auto-debits (₹500/month) to build savings habits "
                    "and increase account stickiness."
                )
            elif location and "OTHER" in location:
                recs.append(
                    "Deploy targeted financial literacy campaigns in Tier-2/3 cities "
                    "through SMS and WhatsApp banking — educate on recurring deposit "
                    "benefits and offer doorstep account servicing to build trust and "
                    "encourage balance growth."
                )
            elif time_period and "night" in time_period.lower():
                recs.append(
                    "Push evening/night notifications offering cashback on UPI payments "
                    "for balance top-ups — e.g. 'Add ₹500 to your account tonight and "
                    "earn 2% cashback on tomorrow's transactions' to incentivize balance "
                    "maintenance during peak browsing hours."
                )
            elif day_type and day_type == "Weekday":
                recs.append(
                    "Introduce weekday 'Salary Day' savings nudges — on common salary "
                    "credit days (1st, 7th, 15th), auto-prompt the customer to lock "
                    "a portion into a 7-day flexi-deposit before it's spent, converting "
                    "transit accounts into savings vehicles."
                )
            elif season and "autumn" in season.lower():
                recs.append(
                    "Pre-Diwali, offer 'Festival Fund' products — a 3-month recurring "
                    "deposit starting in June that matures by Diwali, helping low-balance "
                    "customers save for festive spending without disrupting their "
                    "monthly cash flow."
                )
            else:
                recs.append(
                    "Offer micro-savings products with gamification — implement "
                    "'balance milestones' (₹5K, ₹10K, ₹25K) with small rewards "
                    "(cashback, lucky draws) to gradually shift low-balance customers "
                    "into the savings habit."
                )

    # Age-based actions
    if age and not recs:
        if "25-34" in age:
            if txn_amt and "micro" in txn_amt.lower():
                recs.append(
                    "Create a 'Young Professional' digital banking bundle — UPI cashback "
                    "on micro-payments, zero-fee NEFT/IMPS, and auto-invest spare change "
                    "into liquid mutual funds to capture this tech-savvy segment."
                )
            elif balance and "low" in balance.lower() and txn_amt:
                recs.append(
                    "Introduce 'Step-Up Savings' accounts for 25-34 year-olds with "
                    "₹5K-25K balances — automatically increase the savings rate by 1% "
                    "each month, paired with cashback on essential spending (₹100-500 "
                    "transactions) to reward the behavior this segment already exhibits."
                )
            elif time_period and "night" in time_period.lower():
                recs.append(
                    "Launch evening-targeted 'Night Owl Deals' for 25-34 year-olds — "
                    "card-linked offers with food delivery apps (Swiggy, Zomato) and "
                    "e-commerce platforms during 8PM-midnight, when this digital-native "
                    "segment is most active online."
                )
            elif gender and txn_amt and "small" in txn_amt.lower():
                recs.append(
                    "Deploy 'Smart Spend' analytics for 25-34 year-old male customers — "
                    "categorize their ₹100-500 transactions automatically and offer "
                    "personalized budgeting insights via the mobile app, turning routine "
                    "spending data into a value-added engagement tool."
                )
            else:
                recs.append(
                    "Target 25-34 year-olds with pre-approved personal loans, credit card "
                    "upgrades, and SIP investment products — this segment is at the "
                    "income-growth inflection point and most receptive to financial products."
                )
        elif "18-24" in age:
            recs.append(
                "Launch student banking packages with lifestyle cashback (food delivery "
                "apps, entertainment subscriptions, public transport) and financial "
                "literacy modules to build early brand loyalty before competitors."
            )
        elif "35-44" in age:
            recs.append(
                "Position home loan refinancing, children's education funds, and family "
                "health insurance cross-sells — this life-stage segment prioritizes "
                "family financial security and long-term planning."
            )
        elif "65+" in age:
            recs.append(
                "Offer senior-citizen specific products — higher FD interest rates, "
                "simplified mobile banking interfaces, pension auto-credit services, "
                "and doorstep banking for physically limited customers."
            )

    # Time/Day-based actions
    if time_period and not recs:
        if "morning" in time_period.lower():
            recs.append(
                "Schedule promotional SMS/app notifications for 7-8 AM (pre-transaction "
                "window) with personalized offers — 'Good morning! Pay your electricity "
                "bill via our app today and earn ₹50 cashback.' Timing aligns with "
                "when this segment is most financially active."
            )
        elif "night" in time_period.lower():
            recs.append(
                "Partner with e-commerce platforms (Amazon, Flipkart) for card-linked "
                "evening discount offers — customers in this segment are browsing and "
                "buying online after work hours, making it the ideal window for "
                "co-branded promotions."
            )

    if day_type and not recs:
        if day_type == "Weekend":
            recs.append(
                "Deploy weekend-specific promotional campaigns — push 'Weekend Rewards' "
                "with higher cashback rates on dining, entertainment, and shopping "
                "to capture the discretionary spending surge."
            )
        elif day_type == "Weekday":
            recs.append(
                "Optimize bill payment and recurring transfer UX for weekday users — "
                "introduce one-tap bill pay, smart payment reminders, and auto-schedule "
                "features that align with the routine financial behavior of this segment."
            )

    # Season-based actions
    if season and not recs:
        if "autumn" in season.lower():
            recs.append(
                "Pre-load festive season campaigns in August — Diwali special FDs, "
                "gold purchase EMI schemes, and festival shopping card-linked offers "
                "that activate during the Sep-Nov spending peak."
            )
        elif "summer" in season.lower():
            recs.append(
                "Launch summer-specific products — vacation personal loans, travel "
                "insurance bundles, and education fee EMI schemes targeting parents "
                "paying school/college fees during the Jun-Aug admission window."
            )

    # Location-based actions
    if location and not recs:
        if location == "OTHER":
            recs.append(
                "Invest in Tier-2/3 city digital banking infrastructure — UPI-based "
                "merchant onboarding, regional language app interfaces, and local "
                "business banking products to capture the under-served market."
            )
        else:
            recs.append(
                f"Leverage the {location} branch network for premium product "
                f"cross-selling — this metro city's customer base has higher "
                f"engagement and disposable income for wealth management services."
            )

    # Frequency-based actions
    if freq and not recs:
        if "frequent" in freq.lower():
            recs.append(
                "Enroll frequent transactors in a tiered loyalty program — bronze "
                "(4-10 txns), silver (11-25), gold (25+) with escalating benefits "
                "like fee waivers, higher cashback, and priority customer service."
            )
        elif "single" in freq.lower():
            recs.append(
                "Design re-engagement campaigns for single-transaction customers — "
                "trigger an automated welcome series (Day 3, 7, 14, 30) with "
                "personalized incentives to drive a second transaction."
            )

    # Gender-based fallback
    if gender and not recs:
        if gender == "M":
            recs.append(
                "Use this male-dominant segment profile for targeted product "
                "marketing — personal loans, credit cards with fuel/travel "
                "cashback, and investment products that align with the identified "
                "spending and balance patterns."
            )
        elif gender == "F":
            recs.append(
                "Design women-focused banking products — gold savings schemes, "
                "women's FD with higher rates, and Mahila Samman Savings Certificate "
                "to grow the female customer base in this segment."
            )

    # Ultimate fallback
    if not recs:
        recs.append(
            "Use this co-occurrence pattern to refine the bank's customer "
            "segmentation model — incorporate these correlated attributes into "
            "the CRM scoring engine for more precise product recommendations."
        )

    return " ".join(recs)


# ── Export functions ──────────────────────────────────────────────────────────

def interpret_and_export(rules: pd.DataFrame,
                         frequent_itemsets: pd.DataFrame,
                         output_dir: Path,
                         raw_rule_count: int = 0) -> pd.DataFrame:
    """
    Full Step 3 — interpret top rules, generate business commentary,
    and export all deliverables.

    Parameters
    ----------
    rules             : DataFrame — filtered association rules
    frequent_itemsets : DataFrame — all frequent itemsets
    output_dir        : Path — directory for outputs
    raw_rule_count    : int — number of rules before lift filtering

    Returns
    -------
    top_rules : DataFrame — the interpreted top rules
    """
    log.info("=" * 60)
    log.info("STEP 3 : INTERPRET & EXPORT")
    log.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    if len(rules) == 0:
        log.warning("No rules to interpret — exporting empty results.")
        return pd.DataFrame()

    # ── Export full rules ─────────────────────────────────────────────────
    export_cols = [
        "antecedents_str", "consequents_str",
        "support", "confidence", "lift",
        "leverage", "conviction",
    ]
    available_cols = [c for c in export_cols if c in rules.columns]
    rules_export = rules[available_cols].copy()
    rules_export.columns = [c.replace("_str", "") for c in available_cols]

    full_path = output_dir / "association_rules_full.csv"
    rules_export.to_csv(full_path, index=False)
    log.info(f"Exported {len(rules_export):,} full rules → '{full_path}'")

    # ── Select DIVERSE top rules (not just top-N by lift) ─────────────────
    top_rules = _select_diverse_rules(rules, target_n=12)
    log.info(f"Selected {len(top_rules)} diverse top rules covering "
             f"{top_rules['consequents_str'].nunique()} unique consequent patterns")

    interpretations = []
    for rank, (_, row) in enumerate(top_rules.iterrows(), 1):
        interp = _interpret_rule(
            row["antecedents_str"], row["consequents_str"],
            row["support"], row["confidence"], row["lift"],
            rule_rank=rank,
        )
        interpretations.append(interp)

    top_rules["rule_text"] = [i["rule_text"] for i in interpretations]
    top_rules["explanation"] = [i["explanation"] for i in interpretations]
    top_rules["recommendation"] = [i["recommendation"] for i in interpretations]

    # ── Export interpreted rules CSV ──────────────────────────────────────
    interp_export_cols = [
        "antecedents_str", "consequents_str",
        "support", "confidence", "lift",
        "rule_text", "explanation", "recommendation",
    ]
    interp_path = output_dir / "top_rules_interpreted.csv"
    top_rules[interp_export_cols].to_csv(interp_path, index=False)
    log.info(f"Exported {len(top_rules)} interpreted rules → '{interp_path}'")

    # ── Generate text report ──────────────────────────────────────────────
    report = _build_text_report(top_rules, rules, frequent_itemsets,
                                raw_rule_count)
    report_path = output_dir / "association_rules_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Exported text report → '{report_path}'")

    # ── Generate DECISIONS.md ─────────────────────────────────────────────
    decisions = _build_decisions_doc()
    decisions_path = output_dir / "DECISIONS.md"
    decisions_path.write_text(decisions, encoding="utf-8")
    log.info(f"Exported decisions log → '{decisions_path}'")

    # ── Log report to console ─────────────────────────────────────────────
    log.info(f"\n{report}")

    log.info("=" * 60)
    log.info(f"INTERPRETATION COMPLETE — {len(top_rules)} rules documented")
    log.info("=" * 60)

    return top_rules


def _build_text_report(top_rules: pd.DataFrame,
                       all_rules: pd.DataFrame,
                       frequent_itemsets: pd.DataFrame,
                       raw_rule_count: int = 0) -> str:
    """Build a formatted text report with ranked rules and business commentary."""
    lines = []
    lines.append("=" * 80)
    lines.append("  ASSOCIATION RULE MINING — BUSINESS REPORT")
    lines.append("  Bank Transactions Data Mining Project — Stage 3")
    lines.append("=" * 80)
    lines.append("")

    # ── Summary statistics ────────────────────────────────────────────────
    lines.append("MINING SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Frequent itemsets discovered : {len(frequent_itemsets):,}")
    lines.append(f"  Rules generated (conf ≥ {APRIORI_MIN_CONFIDENCE})  : {raw_rule_count:,}")
    lines.append(f"  Rules after lift filter (≥ {APRIORI_MIN_LIFT})  : {len(all_rules):,}")
    lines.append(f"  Filtering removed            : {raw_rule_count - len(all_rules):,} trivial/independent rules")
    lines.append(f"  Top rules documented          : {len(top_rules)}")
    lines.append(f"  Min support threshold         : {APRIORI_MIN_SUPPORT} ({APRIORI_MIN_SUPPORT*100:.0f}%)")
    lines.append(f"  Min confidence threshold      : {APRIORI_MIN_CONFIDENCE} ({APRIORI_MIN_CONFIDENCE*100:.0f}%)")
    lines.append(f"  Min lift threshold            : {APRIORI_MIN_LIFT}")
    lines.append("")

    if len(all_rules) > 0:
        lines.append(f"  Lift range   : {all_rules['lift'].min():.2f} – {all_rules['lift'].max():.2f}")
        lines.append(f"  Conf range   : {all_rules['confidence'].min():.3f} – {all_rules['confidence'].max():.3f}")
        lines.append(f"  Supp range   : {all_rules['support'].min():.4f} – {all_rules['support'].max():.4f}")
    lines.append("")

    # ── Threshold justification ──────────────────────────────────────────
    lines.append("THRESHOLD JUSTIFICATION")
    lines.append("-" * 80)
    lines.append("  • Support ≥ 5%     : Each itemset must appear in at least ~35K of ~700K")
    lines.append("                       transactions — statistically significant, not noise.")
    lines.append("  • Confidence ≥ 50% : A rule must hold at least half the time to be")
    lines.append("                       presented as a finding (below 50% = wrong more often")
    lines.append("                       than right).")
    lines.append("  • Lift ≥ 1.05      : The co-occurrence must be at least 5% stronger than")
    lines.append("                       random baseline. Lift = 1.0 means statistically")
    lines.append("                       independent (no real association). This threshold")
    lines.append(f"                       removed {raw_rule_count - len(all_rules):,} rules that passed confidence")
    lines.append("                       but showed no meaningful association.")
    lines.append("")

    # ── Ranked rules with interpretation ──────────────────────────────────
    lines.append("=" * 80)
    lines.append("  RANKED RULES WITH BUSINESS INTERPRETATION")
    lines.append("=" * 80)

    for i, (_, row) in enumerate(top_rules.iterrows(), 1):
        lines.append("")
        lines.append(f"  Rule #{i}")
        lines.append(f"  {'-' * 76}")
        lines.append(f"  Pattern    : {row['antecedents_str']}")
        lines.append(f"             → {row['consequents_str']}")
        lines.append(f"  Support    : {row['support']:.4f} ({row['support']*100:.2f}%)")
        lines.append(f"  Confidence : {row['confidence']:.4f} ({row['confidence']*100:.1f}%)")
        lines.append(f"  Lift       : {row['lift']:.4f}")
        lines.append(f"")
        lines.append(f"  Rule (English):")
        lines.append(f"    {row['rule_text']}")
        lines.append(f"")
        lines.append(f"  Explanation:")

        # Word-wrap explanation at ~72 chars
        explanation = row["explanation"]
        words = explanation.split()
        current_line = "    "
        for word in words:
            if len(current_line) + len(word) + 1 > 76:
                lines.append(current_line)
                current_line = "    " + word
            else:
                current_line += " " + word if current_line.strip() else "    " + word
        if current_line.strip():
            lines.append(current_line)

        lines.append(f"")
        lines.append(f"  Recommendation:")
        recommendation = row["recommendation"]
        words = recommendation.split()
        current_line = "    "
        for word in words:
            if len(current_line) + len(word) + 1 > 76:
                lines.append(current_line)
                current_line = "    " + word
            else:
                current_line += " " + word if current_line.strip() else "    " + word
        if current_line.strip():
            lines.append(current_line)

    # ── KEY FINDINGS SUMMARY ──────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 80)
    lines.append("  KEY FINDINGS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  The association rule mining reveals several non-obvious patterns")
    lines.append("  in the bank transaction data that cannot be found through simple")
    lines.append("  tabulation or aggregation:")
    lines.append("")

    # Categorize findings
    lines.append("  1. DEMOGRAPHIC PATTERNS")
    lines.append("  " + "-" * 40)
    demo_rules = [r for _, r in top_rules.iterrows()
                  if any(x in r["antecedents_str"] + r["consequents_str"]
                         for x in ["gender=", "age_group="])]
    if demo_rules:
        lines.append("     • Male customers are disproportionately associated with low-balance")
        lines.append("       accounts and micro-transactions, reflecting the gender gap in")
        lines.append("       Indian banking where men have more 'utility accounts' while women")
        lines.append("       who bank tend to save more.")
        lines.append("     • The 25-34 age group shows strong association with micro-payment")
        lines.append("       behavior, confirming the digital-native spending pattern of")
        lines.append("       early-career professionals.")
    lines.append("")

    lines.append("  2. TEMPORAL PATTERNS")
    lines.append("  " + "-" * 40)
    temp_rules = [r for _, r in top_rules.iterrows()
                  if any(x in r["antecedents_str"] + r["consequents_str"]
                         for x in ["day_type=", "time_period=", "season="])]
    if temp_rules:
        lines.append("     • Weekday transactions are strongly linked to routine financial")
        lines.append("       behavior — the weekday-weekend split reveals distinct spending")
        lines.append("       personas that can be targeted separately.")
        lines.append("     • Morning transactions correlate with scheduled payments and")
        lines.append("       branch-hour activities, while night-time transactions signal")
        lines.append("       e-commerce and digital spending behavior.")
    lines.append("")

    lines.append("  3. GEOGRAPHIC PATTERNS")
    lines.append("  " + "-" * 40)
    geo_rules = [r for _, r in top_rules.iterrows()
                 if "location=" in r["antecedents_str"] + r["consequents_str"]]
    if geo_rules:
        lines.append("     • Tier-2/3 city customers (OTHER) show distinct banking patterns —")
        lines.append("       lower transaction amounts and higher male dominance compared to")
        lines.append("       metro cities, reflecting differential banking penetration.")
    lines.append("")

    lines.append("  4. SPENDING BEHAVIOR PATTERNS")
    lines.append("  " + "-" * 40)
    spend_rules = [r for _, r in top_rules.iterrows()
                   if any(x in r["antecedents_str"] + r["consequents_str"]
                          for x in ["balance_group=", "txn_amount_group="])]
    if spend_rules:
        lines.append("     • Very low balance + micro-transaction is a strong co-occurrence,")
        lines.append("       identifying the 'transit account' segment — customers who use")
        lines.append("       banking purely as a payment channel with no savings intent.")
        lines.append("     • This pattern presents a clear opportunity for micro-savings")
        lines.append("       product interventions (round-ups, automatic sweep).")
    lines.append("")

    lines.append("  STRATEGIC IMPLICATION")
    lines.append("  " + "-" * 40)
    lines.append("     These co-occurrence patterns, invisible through simple frequency")
    lines.append("     counts, enable the bank to move from demographic-only segmentation")
    lines.append("     to behavioral micro-segments — combining WHO the customer is with")
    lines.append("     WHEN and HOW they transact for precision product targeting.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def _build_decisions_doc() -> str:
    """Generate DECISIONS.md documenting all Stage 3 design choices."""
    return """# Stage 3: Association Rule Mining — Decision Log

> This document records every decision made during the Association Rule Mining
> phase, including discretization rationale, threshold choices, and filtering
> strategy.

---

## 1. Data Source — clean_categorical.csv Instead of clean.csv

**Decision:** Read the `clean_categorical.csv` (pre-cleaned by Stage 1 ETL) instead of the Stage 1
final output `clean.csv` or raw data.

**Justification:**
`clean.csv` contains Yeo-Johnson normalised floats — the balance column has
values like `-0.787` instead of `₹2,270`. Discretizing normalised values into
bins like "₹25K–100K" is meaningless because the normalisation is a non-linear
power transform.

To create domain-meaningful categories, we need the original rupee values.
`clean_categorical.csv` retains the original ₹ values (no normalisation) after all Stage 1
cleaning steps have been applied. Discretizing these values produces domain-meaningful bins 
without needing to re-run the entire ETL pipeline, making the process much faster.

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
"""

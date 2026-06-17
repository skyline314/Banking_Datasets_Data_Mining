"""
Pipeline orchestrator -- runs the full Stage 3 Association Rule Mining workflow:
  discretize  ->  mine  ->  interpret & export
Import and call run() from run_association.py (project root entry point).
"""

import time
import traceback
import warnings
from pathlib import Path

from src.stage3_ARM.discretize import load_and_discretize
from src.stage3_ARM.mine       import run_mining
from src.stage3_ARM.interpret  import interpret_and_export
from src.logger import get_logger

log = get_logger(__name__)


def run(input_path: Path, output_dir: Path) -> bool:
    """
    Execute the full Association Rule Mining pipeline.
    Returns True on success, False on failure.

    Parameters
    ----------
    input_path : Path -- path to the raw CSV (same as Stage 1 input)
    output_dir : Path -- directory for all Stage 3 outputs
    """
    warnings.filterwarnings('ignore')
    start = time.time()

    log.info("==========================================================")
    log.info("       BANK TRANSACTIONS -- ASSOCIATION RULE MINING        ")
    log.info("==========================================================")
    log.info(f"Input  : {input_path}")
    log.info(f"Output : {output_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # -- STEP 1 / 3 : DISCRETIZE ------------------------------------------
        log.info("-- STEP 1 / 3 : LOAD & DISCRETIZE ---------------------------")
        df_discretized, transactions = load_and_discretize(input_path)

        # -- STEP 2 / 3 : MINE ------------------------------------------------
        log.info("-- STEP 2 / 3 : APRIORI MINING & RULE GENERATION ------------")
        frequent_itemsets, rules, raw_rule_count = run_mining(transactions)

        # -- STEP 3 / 3 : INTERPRET & EXPORT -----------------------------------
        log.info("-- STEP 3 / 3 : INTERPRET & EXPORT --------------------------")
        top_rules = interpret_and_export(rules, frequent_itemsets, output_dir,
                                         raw_rule_count=raw_rule_count)

        elapsed = time.time() - start
        log.info("==========================================================")
        log.info(f"  PIPELINE COMPLETE -- {elapsed:.1f}s")
        log.info("==========================================================")
        return True

    except Exception:
        elapsed = time.time() - start
        log.error("==========================================================")
        log.error("  PIPELINE FAILED")
        log.error("==========================================================")
        log.error(traceback.format_exc())
        log.error(f"Failed after {elapsed:.1f}s")
        return False

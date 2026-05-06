"""
Pipeline orchestrator — runs extract → transform → load in sequence.
Import and call run() from run_pipeline.py (project root entry point).
"""

import time
import traceback
from pathlib import Path

from src.stage1_ETL.extract   import extract
from src.stage1_ETL.transform import transform
from src.stage1_ETL.load      import load
from src.logger    import get_logger

log = get_logger(__name__)


def run(raw_path: Path, output_path: Path) -> bool:
    """
    Execute the full ETL pipeline.
    Returns True on success, False on failure.
    """
    start = time.time()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║          BANK TRANSACTIONS — ETL PIPELINE                ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Input  : {raw_path}")
    log.info(f"Output : {output_path}")

    try:
        # ── EXTRACT ──────────────────────────────────────────────────────────
        log.info("── STEP 1 / 3 : EXTRACT ────────────────────────────────────")
        raw_df, customer_stats = extract(raw_path)

        # ── TRANSFORM ────────────────────────────────────────────────────────
        log.info("── STEP 2 / 3 : TRANSFORM ──────────────────────────────────")
        clean_df = transform(raw_df, customer_stats)

        # ── LOAD ─────────────────────────────────────────────────────────────
        log.info("── STEP 3 / 3 : LOAD ───────────────────────────────────────")
        load(clean_df, output_path)

        elapsed = time.time() - start
        log.info("╔══════════════════════════════════════════════════════════╗")
        log.info(f"║  PIPELINE COMPLETE — {elapsed:.1f}s                              ║")
        log.info("╚══════════════════════════════════════════════════════════╝")
        return True

    except Exception:
        elapsed = time.time() - start
        log.error("╔══════════════════════════════════════════════════════════╗")
        log.error("║  PIPELINE FAILED                                          ║")
        log.error("╚══════════════════════════════════════════════════════════╝")
        log.error(traceback.format_exc())
        log.error(f"Failed after {elapsed:.1f}s")
        return False
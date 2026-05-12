#!/usr/bin/env python3
"""
Main entry point for the Data Mining Project.
Orchestrates all pipeline stages in sequence.
"""

import sys
import time
import traceback
from config import config
from src.stage1_ETL.pipeline import run as run_etl
from src.stage2_Clustering.pipeline import run as run_clustering
from src.logger import get_logger

log = get_logger(__name__)

def run_project():
    """
    Run all stages of the project.
    Currently: Stage 1 (ETL) and Stage 2 (Clustering).
    """
    start_time = time.time()
    
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║          BANK TRANSACTIONS DATA MINING PROJECT           ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    try:
        # ── STAGE 1 : ETL ──────────────────────────────────────────────────
        log.info("🚀 STAGE 1: Starting ETL Pipeline...")
        etl_success = run_etl(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
        
        if not etl_success:
            log.error("❌ STAGE 1 FAILED. Aborting project run.")
            return False

        # ── STAGE 2 : CLUSTERING ───────────────────────────────────────────
        log.info("🚀 STAGE 2: Starting Clustering Pipeline...")
        clustering_success = run_clustering(config.CLUSTERING_INPUT_PATH, config.CLUSTERING_OUTPUT_DIR)
        
        if not clustering_success:
            log.error("❌ STAGE 2 FAILED. Aborting project run.")
            return False

        # ── STAGE 3 : (Future Placeholder) ──────────────────────────────────
        # log.info("🚀 STAGE 3: (To be added soon)...")

        elapsed = time.time() - start_time
        log.info("╔══════════════════════════════════════════════════════════╗")
        log.info(f"║  ✅ ALL STAGES COMPLETE — Total Time: {elapsed:.1f}s              ║")
        log.info("╚══════════════════════════════════════════════════════════╝")
        return True

    except Exception:
        log.error("💥 CRITICAL ERROR during project execution:")
        log.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = run_project()
    if not success:
        sys.exit(1)

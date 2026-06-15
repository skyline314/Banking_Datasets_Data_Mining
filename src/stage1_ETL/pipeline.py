"""
Pipeline orchestrator — runs extract → transform → load in sequence.
Import and call run() from run_pipeline.py (project root entry point).

Produces two output files:
  clean.csv              — normalised features for clustering / distance-based ML
  clean_categorical.csv  — cleaned original-scale values for ARM / EDA
"""

import time
import traceback
from pathlib import Path

from src.stage1_ETL.extract    import extract
from src.stage1_ETL.transform  import (
    transform_base,
    encode_age, encode_categoricals, encode_month_cyclical,
    apply_ohe_and_drop, assess_feature_information,
    normalize_features, drop_correlated,
)
from src.stage1_ETL.load       import load, load_categorical, export_descriptive_stats
from src.logger     import get_logger
from config.config  import (
    TOP_N_LOCATIONS, COLS_TO_DROP, COLS_TO_NORMALIZE, COLS_CORRELATED_DROP,
)

log = get_logger(__name__)


def run(raw_path: Path, output_path: Path, categorical_path: Path) -> bool:
    """
    Execute the full ETL pipeline.
    Returns True on success, False on failure.
    """
    start = time.time()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║          BANK TRANSACTIONS — ETL PIPELINE                ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"Input       : {raw_path}")
    log.info(f"Output      : {output_path}")
    log.info(f"Output (cat): {categorical_path}")

    try:
        # ── EXTRACT ──────────────────────────────────────────────────────────
        log.info("── STEP 1 / 5 : EXTRACT ────────────────────────────────────")
        raw_df, customer_stats = extract(raw_path)

        # ── TRANSFORM (base — cleaning only) ─────────────────────────────────
        log.info("── STEP 2 / 5 : TRANSFORM (base cleaning) ───────────────────")
        base_df = transform_base(raw_df, customer_stats)

        # ── EDA — descriptive statistics on original-scale cleaned data ───────
        log.info("── STEP 3 / 5 : EDA (descriptive statistics) ────────────────")
        export_descriptive_stats(base_df, output_path.parent)

        # ── LOAD categorical (before encoding mutates the DataFrame) ─────────
        log.info("── STEP 4 / 5 : LOAD (clean_categorical.csv) ────────────────")
        load_categorical(base_df, categorical_path)

        # ── TRANSFORM (full — encoding + normalization) ──────────────────────
        log.info("── STEP 4b/ 5 : TRANSFORM (encoding + normalization) ────────")
        clean_df = base_df.copy()
        clean_df = encode_age(clean_df)
        clean_df = encode_categoricals(clean_df, TOP_N_LOCATIONS)
        clean_df = encode_month_cyclical(clean_df)
        clean_df = apply_ohe_and_drop(clean_df, COLS_TO_DROP)
        assess_feature_information(clean_df)
        clean_df = normalize_features(clean_df, COLS_TO_NORMALIZE)
        clean_df = drop_correlated(clean_df, COLS_CORRELATED_DROP)

        # ── LOAD normalised ──────────────────────────────────────────────────
        log.info("── STEP 5 / 5 : LOAD (clean.csv) ────────────────────────────")
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
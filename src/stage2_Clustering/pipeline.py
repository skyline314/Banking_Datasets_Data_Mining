"""
Pipeline orchestrator -- runs the full Stage 2 clustering workflow:
  prepare  ->  evaluate  ->  cluster  ->  profile
Import and call run() from run_clustering.py (project root entry point).

Updated to support all D1-D10 audit fixes:
  - Passes categorical data path for merge-to-original profiling (D1)
  - Passes optimal_k to dendrogram for dynamic cut-line (D6)
  - Passes output_dir and best_sil to DBSCAN for k-distance plot + comparison (D2, D10)
  - Passes hopkins_score and best_sil to profiling for quality reporting (D4, D7)
"""

import time
import traceback
import warnings
from pathlib import Path

from src.stage2_Clustering.prepare  import load_and_prepare
from src.stage2_Clustering.evaluate import evaluate_optimal_k
from src.stage2_Clustering.cluster  import (apply_kmeans,
                                            generate_dendrogram,
                                            detect_outliers_dbscan)
from src.stage2_Clustering.profile  import build_and_export_profiles
from src.logger import get_logger

log = get_logger(__name__)


def run(input_path: Path, categorical_path: Path, output_dir: Path) -> bool:
    """
    Execute the full Clustering pipeline.
    Returns True on success, False on failure.

    Parameters
    ----------
    input_path       : Path -- path to the clean CSV (output of Stage 1 ETL)
    categorical_path : Path -- path to clean_categorical.csv (original-scale)
    output_dir       : Path -- directory for all Stage 2 outputs (CSVs + plots)
    """
    warnings.filterwarnings('ignore')
    start = time.time()

    log.info("==========================================================")
    log.info("       BANK TRANSACTIONS -- CLUSTERING PIPELINE           ")
    log.info("==========================================================")
    log.info(f"Input       : {input_path}")
    log.info(f"Categorical : {categorical_path}")
    log.info(f"Output      : {output_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # -- STEP 1 / 4 : PREPARE ---------------------------------------------
        log.info("-- STEP 1 / 4 : LOAD & PREPARE DATA -----------------------")
        (df_cluster, X_scaled, X_sample,
         scaler, df_categorical, hopkins_score) = load_and_prepare(
            input_path, categorical_path
        )

        # -- STEP 2 / 4 : EVALUATE OPTIMAL K ----------------------------------
        log.info("-- STEP 2 / 4 : EVALUATE OPTIMAL K ------------------------")
        optimal_k, best_sil = evaluate_optimal_k(X_scaled, X_sample,
                                                  output_dir)

        # -- STEP 3 / 4 : CLUSTER ---------------------------------------------
        log.info("-- STEP 3 / 4 : APPLY CLUSTERING --------------------------")
        df_cluster = apply_kmeans(df_cluster, X_scaled, optimal_k)
        generate_dendrogram(X_scaled, optimal_k, output_dir)
        detect_outliers_dbscan(X_sample, output_dir,
                               best_sil_kmeans=best_sil)

        # -- STEP 4 / 4 : PROFILE & EXPORT ------------------------------------
        log.info("-- STEP 4 / 4 : PROFILE & EXPORT --------------------------")
        build_and_export_profiles(
            df_cluster, df_categorical,
            hopkins_score, best_sil, output_dir
        )

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

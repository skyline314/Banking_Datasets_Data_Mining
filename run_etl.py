"""
Entry point — run from the project root:
    python run_pipeline.py

Optional flags:
    --input  path/to/custom_raw.csv
    --output path/to/custom_clean.csv
"""

import argparse
import sys
from pathlib import Path

# ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.stage1_ETL.pipeline  import run


def parse_args():
    parser = argparse.ArgumentParser(description="Bank Transactions ETL Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help=f"Path to raw CSV (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help=f"Path for clean CSV output (default: {PROCESSED_DATA_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run(raw_path=args.input, output_path=args.output)
    sys.exit(0 if success else 1)
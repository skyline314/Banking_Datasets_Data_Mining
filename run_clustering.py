"""
Entry point — run from the project root:
    python run_clustering.py

Optional flags:
    --input  path/to/clean.csv
    --output path/to/output_dir
"""

import argparse
import sys
from pathlib import Path

# ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import CLUSTERING_INPUT_PATH, CLUSTERING_OUTPUT_DIR
from src.stage2_Clustering.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description="Bank Transactions Clustering Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=CLUSTERING_INPUT_PATH,
        help=f"Path to clean CSV from Stage 1 (default: {CLUSTERING_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLUSTERING_OUTPUT_DIR,
        help=f"Directory for clustering outputs (default: {CLUSTERING_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run(input_path=args.input, output_dir=args.output)
    sys.exit(0 if success else 1)

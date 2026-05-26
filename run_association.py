"""
Entry point — run from the project root:
    python run_association.py

Optional flags:
    --input  path/to/raw_bank_transactions.csv
    --output path/to/output_dir
"""

import argparse
import sys
from pathlib import Path

# ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import ASSOC_INPUT_PATH, ASSOC_OUTPUT_DIR
from src.stage3_ARM.pipeline import run


def parse_args():
    parser = argparse.ArgumentParser(description="Bank Transactions Association Rule Mining")
    parser.add_argument(
        "--input",
        type=Path,
        default=ASSOC_INPUT_PATH,
        help=f"Path to raw CSV (default: {ASSOC_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSOC_OUTPUT_DIR,
        help=f"Directory for ARM outputs (default: {ASSOC_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run(input_path=args.input, output_dir=args.output)
    sys.exit(0 if success else 1)

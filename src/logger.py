"""
Centralised logger — import get_logger() in any module.
Writes to both console and logs/pipeline.log simultaneously.
"""

import logging
import sys
from pathlib import Path

# resolve config without circular import
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "pipeline.log"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not _configured:
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8"),
        ]
        logging.basicConfig(
            level=logging.INFO,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            handlers=handlers,
        )
        _configured = True

    return logging.getLogger(name)
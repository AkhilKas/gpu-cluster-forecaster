"""Centralized logging setup."""
import logging
import sys


def setup_logging(level: str = "INFO"):
    """Configure structured logging for the project."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
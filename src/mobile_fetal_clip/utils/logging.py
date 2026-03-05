"""Logging setup for MobileFetalCLIP training."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rank: int = 0,
) -> logging.Logger:
    """Configure logging for training.

    Only rank 0 logs to console and file. Other ranks log warnings+ only.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
        rank: Process rank for distributed training.

    Returns:
        Root logger.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    if rank == 0:
        console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    else:
        console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File handler (rank 0 only)
    if log_file and rank == 0:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode="a")
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

    return root_logger

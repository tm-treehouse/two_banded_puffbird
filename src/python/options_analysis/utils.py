#!/usr/bin/env python3
"""
Utility functions for the options analysis package.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    log_level=logging.INFO, log_file="options_analysis.log", max_file_size_mb=1, backup_count=3
):
    """
    Configure logging with both file and console handlers.

    Parameters:
        log_level (int): Logging level (default: logging.INFO)
        log_file (str): Path to log file
        max_file_size_mb (int): Maximum log file size in MB before rotating
        backup_count (int): Number of backup logs to keep

    Returns:
        logger: Configured logger instance
    """
    # Initialize logger
    logger = logging.getLogger("options_analysis")

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / log_file

    # Configure logger
    logger.setLevel(log_level)

    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler - rotating log files
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=max_file_size_mb * 1024 * 1024, backupCount=backup_count
    )
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Rich console handler
    console_handler = RichHandler(console=Console())
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"Logging configured. Log file: {log_file_path}")
    return logger


def ensure_output_directory(directory="out"):
    """
    Create output directory if it doesn't exist.

    Parameters:
        directory (str): Directory name to create

    Returns:
        Path: Path object representing the created directory
    """
    out_dir = Path(directory)
    out_dir.mkdir(exist_ok=True)
    return out_dir

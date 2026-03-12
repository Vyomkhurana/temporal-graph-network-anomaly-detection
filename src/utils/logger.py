"""
Logging utility for the TGN project.

Provides consistent logging across all modules with file and console output.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Configure the logger for the project.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific log file name. If None, uses 'tgn.log'
    
    Example:
        >>> from src.utils.logger import setup_logger, logger
        >>> setup_logger(log_level="DEBUG")
        >>> logger.info("Training started")
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Determine log file name
    if log_file is None:
        log_file = "tgn.log"
    
    log_file_path = log_path / log_file
    
    # Add file handler
    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB",  # Rotate after 10 MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip"  # Compress rotated logs
    )
    
    logger.info(f"Logger initialized - Level: {log_level}, Log file: {log_file_path}")


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        Loguru logger instance
    """
    return logger

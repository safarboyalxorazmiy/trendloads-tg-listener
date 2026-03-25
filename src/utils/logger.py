"""Configure loguru with JSON format, file rotation, and stdout handler."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """Configure loguru logger for production use.

    - JSON-formatted file logs with 10 MB rotation and 7-day retention.
    - Human-readable stdout logs.
    """
    # Remove default handler
    logger.remove()

    # Stdout handler — human-readable
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler — JSON format, rotated at 10 MB
    logger.add(
        "logs/telegram_listener.log",
        level="DEBUG",
        format="{message}",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # thread-safe
    )

    logger.info("Logger initialised", level=log_level)

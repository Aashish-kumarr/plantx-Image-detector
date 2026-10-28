"""Centralized Python logging configuration."""

import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    """Configure root logger with log level and safe formatters."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Retrieve a named module logger under the 'plantdx' logger namespace."""
    return logging.getLogger(f"plantdx.{name}")

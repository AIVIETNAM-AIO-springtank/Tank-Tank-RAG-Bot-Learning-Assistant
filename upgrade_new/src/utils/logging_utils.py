"""Logging helpers for local development and tests."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a conservative default configuration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger(name)


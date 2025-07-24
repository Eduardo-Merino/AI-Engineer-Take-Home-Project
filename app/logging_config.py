# app/logging_config.py
"""
Central logging configuration.

Call `configure_logging()` once at application startup.
Use `logger = logging.getLogger(__name__)` inside modules.
"""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root logger with a standard formatter.

    Parameters
    ----------
    level : str
        Minimum log level (e.g. "DEBUG", "INFO", "WARNING").
    """
    root = logging.getLogger()
    if root.handlers:
        # Avoid configuring twice (e.g. during reload)
        return

    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(handler)

    # Optional: reduce noise from external libs
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

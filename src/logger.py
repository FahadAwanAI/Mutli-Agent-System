import logging
import json
import sys
import os
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Emits each log record as a single-line JSON object.
    Extra fields passed via extra={...} are merged into the top-level object.
    """

    _RESERVED = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge any extra={...} fields
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                entry[key] = value

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


def setup_logging(
    log_level: str | None = None,
    log_format: str | None = None,
) -> None:
    """
    Configure the root logger. Call once at application startup (main.py).

    Args:
        log_level:  Override LOG_LEVEL env var. Defaults to "INFO".
        log_format: Override LOG_FORMAT env var. "json" or "text". Defaults to "text".
    """
    level_str = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt_str = (log_format or os.getenv("LOG_FORMAT", "text")).lower()

    level = getattr(logging, level_str, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter() if fmt_str == "json" else TextFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger. Always pass __name__ as the argument.

    Example:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)

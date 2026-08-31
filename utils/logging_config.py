"""
ProofPilot — Structured JSON Logging Configuration
---------------------------------------------------
Outputs production-grade structured JSON logs with ISO timestamps, log levels,
module origin, and request ID context for Splunk / ELK / CloudWatch ingestion.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONLogFormatter(logging.Formatter):
    """Formats log records into single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = getattr(record, "request_id")
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


_logging_initialized = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configures global root logging with JSON formatting."""
    global _logging_initialized
    if _logging_initialized:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONLogFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing handlers to prevent duplicates
    root_logger.handlers = [handler]

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Returns a module logger configured with structured formatting."""
    setup_logging()
    return logging.getLogger(name)

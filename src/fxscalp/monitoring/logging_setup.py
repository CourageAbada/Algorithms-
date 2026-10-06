"""Structured JSON logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any

_SENSITIVE_KEYS = re.compile(r"(password|passwd|secret|token|api[_-]?key|login|account)", re.I)
_SENSITIVE_VALUE = re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key|login)\s*[=:]\s*\S+")
REDACTED = "***REDACTED***"


def redact(obj: Any) -> Any:
    """Recursively redact values under sensitive keys and key=value secrets in strings."""
    if isinstance(obj, dict):
        return {k: REDACTED if _SENSITIVE_KEYS.search(str(k)) else redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _SENSITIVE_VALUE.sub(lambda m: f"{m.group(1)}={REDACTED}", obj)
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload["ctx"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install a single JSON handler on the root logger (UTC timestamps)."""
    import time

    JsonFormatter.converter = time.gmtime  # type: ignore[assignment]
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())

"""Structured logging: every log record can carry a request_id (one HTTP
request) and/or a run_id (one agent run), set via contextvars rather than
threaded through every function signature — a tool three calls deep inside
the agent loop can log with the right run_id without its caller chain
knowing anything about logging.
"""

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

_RESERVED_LOG_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class _ContextFilter(logging.Filter):
    """Injects the current request_id/run_id (if any) into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.run_id = run_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra fields passed via `logger.info(msg,
    extra={...})` are included as-is, so callers can attach structured
    context (tool_name, duration_seconds, exit_code, ...) without it being
    mashed into the message string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.request_id:
            payload["request_id"] = record.request_id
        if record.run_id:
            payload["run_id"] = record.run_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_ATTRS and key not in ("request_id", "run_id"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable console format, with correlation IDs appended when present."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        tags = []
        if record.request_id:
            tags.append(f"request_id={record.request_id}")
        if record.run_id:
            tags.append(f"run_id={record.run_id}")
        return f"{base} [{', '.join(tags)}]" if tags else base


def configure_logging(level: str, log_format: str) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)

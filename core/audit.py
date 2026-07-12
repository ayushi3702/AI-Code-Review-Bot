"""Centralized logging configuration.

Call ``setup_logging()`` once at every application entry point (server startup,
CLI start).  After that, every ``logging.getLogger(__name__)`` in the project
automatically writes to both the console and ``logs/audit.log``.

Log format:
    2026-06-21 12:41:09,292 INFO     api.server: Scan queued scan_id=abc
"""
from __future__ import annotations
import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "audit.log")

_FMT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _MsFormatter(logging.Formatter):
    """Custom log formatter that renders timestamps with three-digit milliseconds.

    The built-in :class:`logging.Formatter` uses ``%f`` (microseconds, 6 digits)
    for sub-second precision.  This subclass overrides :meth:`formatTime` to
    produce the more readable ``2026-06-21 12:41:09,292`` style instead.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """Return the log timestamp formatted with three-digit milliseconds.

        Args:
            record:  The :class:`logging.LogRecord` being formatted.
            datefmt: Optional strftime format string; falls back to ``_DATEFMT``.

        Returns:
            Timestamp string of the form ``YYYY-MM-DD HH:MM:SS,mmm``.
        """
        import datetime
        ct = datetime.datetime.fromtimestamp(record.created)
        base = ct.strftime(datefmt or _DATEFMT)
        return f"{base},{record.msecs:03.0f}"


_formatter = _MsFormatter(fmt=_FMT)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to write to both the console and ``logs/audit.log``.

    Attaches two handlers to the root logger:

    - A :class:`~logging.StreamHandler` for live console output.
    - A :class:`~logging.handlers.RotatingFileHandler` that persists every
      log entry to ``logs/audit.log`` (10 MB per file, 5 backups retained).

    Both handlers share the same format::

        2026-06-21 12:41:09,292 INFO     agents.repo_orchestrator: Scan started ...

    Safe to call multiple times — subsequent calls are no-ops once handlers
    are already attached to the root logger.

    Args:
        level: Minimum log level for the root logger (default: ``INFO``).
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return  # already configured
    root.setLevel(level)

    # Silence noisy third-party loggers that flood the audit log with
    # internal housekeeping messages unrelated to application behaviour.
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    # ── Console ──────────────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setFormatter(_formatter)
    root.addHandler(console)

    # ── Rotating file → logs/audit.log ───────────────────────────────────────
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(_formatter)
    root.addHandler(file_handler)

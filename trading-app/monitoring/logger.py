"""
Structured logging setup for the trading application.

Provides:
- JSON renderer in production (any LOG_LEVEL other than DEBUG)
- Rich console renderer in development (LOG_LEVEL == DEBUG)
- Pre-configured structlog processors: timestamp, log level, logger name, event_type
- ``get_logger(name)`` factory
- Event-type constants for every significant engine event

Usage
-----
    from monitoring.logger import get_logger, TRADE_OPEN, SYSTEM_START

    log = get_logger(__name__)
    log.info(TRADE_OPEN, symbol="BTCUSDT", side="BUY", qty=0.001)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

# ---------------------------------------------------------------------------
# Event-type constants
# ---------------------------------------------------------------------------

TRADE_OPEN: str = "trade.open"
TRADE_CLOSE: str = "trade.close"
SIGNAL_GENERATED: str = "signal.generated"
RISK_CHECK: str = "risk.check"
ORDER_PLACED: str = "order.placed"
ORDER_FILLED: str = "order.filled"
PHASE_TRANSITION: str = "phase.transition"
CIRCUIT_BREAKER: str = "circuit_breaker"
SYSTEM_START: str = "system.start"
SYSTEM_STOP: str = "system.stop"


# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------


def _add_event_type(
    logger: Any,  # noqa: ANN401
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Ensure every log record has an ``event_type`` field.

    If the caller already included ``event_type`` it is left as-is; otherwise
    the field defaults to the ``event`` string value so dashboards can always
    filter on ``event_type``.
    """
    event_dict.setdefault("event_type", event_dict.get("event", ""))
    return event_dict


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _is_debug_mode() -> bool:
    """Return True when LOG_LEVEL environment variable equals 'DEBUG'."""
    import os

    return os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG"


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog and the stdlib logging bridge.

    Call this **once** at application startup before any loggers are acquired.

    Parameters
    ----------
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
        Defaults to ``INFO``.  When ``DEBUG``, a Rich console renderer is used
        instead of the JSON renderer.
    """
    debug_mode = log_level.upper() == "DEBUG"

    # ── Shared processors (run for both stdlib and structlog) ────────────────
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_event_type,
        structlog.processors.StackInfoRenderer(),
    ]

    # ── Choose final renderer ────────────────────────────────────────────────
    if debug_mode:
        # Rich console — colour, aligned columns, great for development
        renderer: Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        # JSON — machine-parseable, suitable for log aggregators
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ── Wire structlog into the stdlib logging system ────────────────────────
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Silence noisy third-party libraries
    for noisy in ("asyncpg", "websockets", "aiohttp", "urllib3", "telegram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module *name*.

    Example
    -------
    ::

        from monitoring.logger import get_logger

        log = get_logger(__name__)
        log.info("system.start", component="engine")
    """
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Auto-configure with sensible defaults when imported
# ---------------------------------------------------------------------------
# This allows modules to call ``get_logger`` immediately without needing to
# call ``configure_logging`` explicitly.  The engine entry-point should call
# ``configure_logging(settings.log_level)`` to override after settings load.

import os as _os

configure_logging(_os.environ.get("LOG_LEVEL", "INFO"))

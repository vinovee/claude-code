"""
CircuitBreaker — Redis-backed trading halt mechanism.

The circuit breaker can be in one of three states:

  CLOSED    Normal operation; both live and paper trading are permitted.
  OPEN      Hard halt; no new orders of any kind are accepted.
  HALF_OPEN Degraded mode; only paper (simulated) trades are permitted.

State is persisted to Redis under the key ``risk:circuit_breaker:state`` so
that a process restart does not accidentally re-enable trading after a trip.
Recovery from OPEN requires an explicit human call to ``reset()``.

Typical usage
-------------
    from risk.circuit_breaker import CircuitBreaker, CircuitBreakerState

    cb = CircuitBreaker()
    state = await cb.check(portfolio_state)

    if not cb.allow_live_trading():
        ...  # route to paper execution

    # Manual recovery (ops team decision):
    await cb.reset()

Telegram alerts
---------------
``trip()`` sends a Telegram message when ``telegram_bot_token`` and
``telegram_chat_id`` are configured in settings.  Failures are logged but
never propagate — the circuit-breaker logic must not block on external I/O.

Redis key schema
----------------
  risk:circuit_breaker:state   → "CLOSED" | "OPEN" | "HALF_OPEN"
  risk:circuit_breaker:reason  → last trip reason (human-readable string)
  risk:circuit_breaker:tripped_at → ISO-8601 timestamp of last trip
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import redis.asyncio as aioredis
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Redis key constants
# ---------------------------------------------------------------------------

_KEY_STATE: str = "risk:circuit_breaker:state"
_KEY_REASON: str = "risk:circuit_breaker:reason"
_KEY_TRIPPED_AT: str = "risk:circuit_breaker:tripped_at"

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class CircuitBreakerState(str, Enum):
    """Possible states of the circuit breaker."""

    CLOSED = "CLOSED"       # Normal — live trading allowed
    OPEN = "OPEN"           # Hard halt — no trading
    HALF_OPEN = "HALF_OPEN" # Degraded — paper trading only


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Redis-backed circuit breaker for the autonomous trading application.

    All state is held in Redis so that multiple processes (e.g. strategy
    workers + the execution engine) share a single consistent view.  An
    in-process cache is maintained for fast ``allow_live_trading()`` /
    ``allow_paper_trading()`` calls, refreshed whenever ``check()`` is
    called.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._settings = get_settings()
        self._redis_url: str = redis_url or self._settings.redis_url
        self._redis: aioredis.Redis | None = None
        # In-process state cache — authoritative source is Redis
        self._state: CircuitBreakerState = CircuitBreakerState.CLOSED

    # ------------------------------------------------------------------
    # Redis lifecycle helpers
    # ------------------------------------------------------------------

    async def _get_redis(self) -> aioredis.Redis:
        """Return (or lazily create) the async Redis client."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
        return self._redis

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    async def _load_state(self) -> CircuitBreakerState:
        """Read the current state from Redis, defaulting to CLOSED."""
        try:
            client = await self._get_redis()
            raw: str | None = await client.get(_KEY_STATE)
            if raw is None:
                return CircuitBreakerState.CLOSED
            return CircuitBreakerState(raw)
        except Exception:  # noqa: BLE001
            logger.exception(
                "circuit_breaker.redis_read_error",
                msg="Could not read state from Redis; defaulting to CLOSED.",
            )
            return CircuitBreakerState.CLOSED

    async def _save_state(
        self,
        state: CircuitBreakerState,
        reason: str = "",
    ) -> None:
        """Persist state (and optional reason / timestamp) to Redis."""
        try:
            client = await self._get_redis()
            pipe = client.pipeline()
            pipe.set(_KEY_STATE, state.value)
            if reason:
                pipe.set(_KEY_REASON, reason)
                pipe.set(
                    _KEY_TRIPPED_AT,
                    datetime.now(tz=timezone.utc).isoformat(),
                )
            await pipe.execute()
        except Exception:  # noqa: BLE001
            logger.exception(
                "circuit_breaker.redis_write_error",
                state=state.value,
                reason=reason,
            )

    # ------------------------------------------------------------------
    # Telegram notification helper
    # ------------------------------------------------------------------

    async def _send_telegram_alert(self, message: str) -> None:
        """Fire-and-forget Telegram alert.

        Silently swallows all errors so the circuit-breaker path is never
        blocked by notification failures.
        """
        token: str = self._settings.telegram_bot_token
        chat_id: str = self._settings.telegram_chat_id

        if not token or not chat_id:
            logger.debug(
                "circuit_breaker.telegram_not_configured",
                msg="Skipping Telegram alert — token or chat_id not set.",
            )
            return

        try:
            import httpx  # lazy import — not all environments have httpx at import time

            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            logger.info("circuit_breaker.telegram_alert_sent")
        except Exception:  # noqa: BLE001
            logger.exception(
                "circuit_breaker.telegram_alert_failed",
                msg="Failed to send Telegram alert; continuing.",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(self, portfolio_state: dict) -> CircuitBreakerState:
        """Evaluate the portfolio state and auto-trip the breaker if needed.

        The method compares current drawdown and daily P&L against the
        configured hard limits and transitions to OPEN or HALF_OPEN as
        appropriate.  If the breaker is already OPEN it is left unchanged —
        recovery requires an explicit call to ``reset()``.

        Parameters
        ----------
        portfolio_state:
            Dictionary that may contain:
            - ``equity`` (float, GBP)
            - ``peak_equity`` (float, GBP) — used to derive drawdown
            - ``drawdown`` (float, 0-1) — used directly if present
            - ``daily_pnl_pct`` (float, negative = loss)

        Returns
        -------
        CircuitBreakerState
            The state after evaluating the portfolio.
        """
        # Always refresh from Redis first
        self._state = await self._load_state()

        if self._state == CircuitBreakerState.OPEN:
            logger.warning(
                "circuit_breaker.already_open",
                msg="Circuit breaker is OPEN; refusing new trades.",
            )
            return self._state

        # --- Compute drawdown ---
        if "drawdown" in portfolio_state:
            drawdown: float = float(portfolio_state["drawdown"])
        else:
            equity: float = float(portfolio_state.get("equity", 0.0))
            peak_equity: float = float(portfolio_state.get("peak_equity", equity))
            drawdown = (
                0.0
                if peak_equity <= 0
                else max(0.0, 1.0 - equity / peak_equity)
            )

        daily_pnl_pct: float = float(portfolio_state.get("daily_pnl_pct", 0.0))
        max_drawdown: float = self._settings.max_drawdown_pct
        max_daily_loss: float = self._settings.max_daily_loss_pct

        # Hard trip conditions → OPEN
        if drawdown > max_drawdown:
            reason = (
                f"Drawdown {drawdown:.2%} exceeded hard limit {max_drawdown:.2%}."
            )
            await self.trip(reason)
            return self._state

        if daily_pnl_pct < -max_daily_loss:
            reason = (
                f"Daily loss {daily_pnl_pct:.2%} exceeded hard limit "
                f"-{max_daily_loss:.2%}."
            )
            await self.trip(reason)
            return self._state

        # Soft-warn conditions → HALF_OPEN (only if currently CLOSED)
        warn_drawdown: float = max_drawdown * 0.67
        warn_daily_loss: float = max_daily_loss * 0.75

        if self._state == CircuitBreakerState.CLOSED:
            if drawdown > warn_drawdown or daily_pnl_pct < -warn_daily_loss:
                reason = (
                    f"Approaching risk limits: drawdown={drawdown:.2%}, "
                    f"daily_pnl={daily_pnl_pct:.2%}. Switching to HALF_OPEN."
                )
                logger.warning(
                    "circuit_breaker.half_open",
                    drawdown=drawdown,
                    daily_pnl_pct=daily_pnl_pct,
                    reason=reason,
                )
                self._state = CircuitBreakerState.HALF_OPEN
                await self._save_state(self._state, reason)

        return self._state

    async def trip(self, reason: str) -> None:
        """Trip the circuit breaker to OPEN state.

        Sets state = OPEN in Redis, emits a structured log entry, and fires a
        Telegram alert.  Safe to call multiple times — idempotent once OPEN.

        Parameters
        ----------
        reason:
            Human-readable description of why the breaker was tripped.
        """
        self._state = CircuitBreakerState.OPEN
        await self._save_state(CircuitBreakerState.OPEN, reason)

        logger.error(
            "circuit_breaker.tripped",
            state="OPEN",
            reason=reason,
            tripped_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        alert_message = (
            "\U0001f6a8 *CIRCUIT BREAKER TRIPPED*\n\n"
            f"*Reason:* {reason}\n"
            f"*Time:* {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            "All live trading has been halted. Manual review required before resuming."
        )
        # Fire and forget — never await in a way that blocks the caller
        asyncio.ensure_future(self._send_telegram_alert(alert_message))

    async def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state.

        This is the only path back to CLOSED.  It must be called by a human
        operator (or an operations script) after reviewing the reason for the
        trip.  Automatic recovery is intentionally not implemented.
        """
        previous_state: CircuitBreakerState = self._state
        self._state = CircuitBreakerState.CLOSED

        try:
            client = await self._get_redis()
            pipe = client.pipeline()
            pipe.set(_KEY_STATE, CircuitBreakerState.CLOSED.value)
            pipe.delete(_KEY_REASON)
            pipe.delete(_KEY_TRIPPED_AT)
            await pipe.execute()
        except Exception:  # noqa: BLE001
            logger.exception("circuit_breaker.redis_reset_error")

        logger.info(
            "circuit_breaker.reset",
            previous_state=previous_state.value,
            new_state="CLOSED",
            reset_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def allow_live_trading(self) -> bool:
        """Return ``True`` only when the circuit breaker is CLOSED.

        Uses the in-process state cache.  Call ``check()`` first to ensure
        the cache reflects the latest Redis state.
        """
        return self._state == CircuitBreakerState.CLOSED

    def allow_paper_trading(self) -> bool:
        """Return ``True`` when the circuit breaker is CLOSED or HALF_OPEN.

        Uses the in-process state cache.  Call ``check()`` first to ensure
        the cache reflects the latest Redis state.
        """
        return self._state in (
            CircuitBreakerState.CLOSED,
            CircuitBreakerState.HALF_OPEN,
        )

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        """Return a status dict suitable for health-check endpoints.

        Reads the current state, reason, and trip timestamp from Redis.

        Returns
        -------
        dict with keys: state, reason, tripped_at, allow_live, allow_paper
        """
        try:
            client = await self._get_redis()
            state_raw, reason, tripped_at = await asyncio.gather(
                client.get(_KEY_STATE),
                client.get(_KEY_REASON),
                client.get(_KEY_TRIPPED_AT),
            )
            state = (
                CircuitBreakerState(state_raw)
                if state_raw
                else CircuitBreakerState.CLOSED
            )
        except Exception:  # noqa: BLE001
            logger.exception("circuit_breaker.get_status_error")
            state = self._state
            reason = None
            tripped_at = None

        # Keep in-process cache in sync
        self._state = state

        return {
            "state": state.value,
            "reason": reason or "",
            "tripped_at": tripped_at or "",
            "allow_live": self.allow_live_trading(),
            "allow_paper": self.allow_paper_trading(),
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CircuitBreaker":
        # Eagerly load state from Redis on enter
        self._state = await self._load_state()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

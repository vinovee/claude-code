"""
KrakenFeed — live OHLC WebSocket feed using the Kraken WebSocket API v2.

Endpoint: ``wss://ws.kraken.com/v2``

Subscription message format (v2)
---------------------------------
.. code-block:: json

    {
      "method": "subscribe",
      "params": {
        "channel": "ohlc",
        "symbol": ["BTC/USD", "ETH/USD"],
        "interval": 15
      }
    }

OHLC update message format (v2)
---------------------------------
.. code-block:: json

    {
      "channel": "ohlc",
      "type": "update",
      "data": [{
        "symbol": "BTC/USD",
        "open": "65000.0",
        "high": "65500.0",
        "low": "64800.0",
        "close": "65200.0",
        "volume": "12.5",
        "timestamp": "2024-01-01T12:15:00Z",
        "interval_begin": "2024-01-01T12:00:00Z"
      }]
    }

Symbol conversion (input → Kraken WS v2 format)
-------------------------------------------------
    BTCUSDT → BTC/USD   (strip USDT suffix, insert /USD)
    ETHUSDT → ETH/USD
    SOLUSDT → SOL/USD

Timeframe mapping (our strings → Kraken interval integers in minutes)
----------------------------------------------------------------------
    "1m"  → 1
    "5m"  → 5
    "15m" → 15
    "30m" → 30
    "1h"  → 60
    "4h"  → 240
    "1d"  → 1440

Reconnect strategy
------------------
Exponential back-off starting at 2 s, doubling up to 60 s, max 5 attempts
per symbol.  The attempt counter resets after a successful connection that
lasts longer than 30 s.

Prometheus metrics (``feed="kraken"`` label on all)
----------------------------------------------------
    kraken_candles_received_total{feed, symbol, timeframe}
    kraken_feed_reconnects_total{feed, symbol}
    kraken_feed_errors_total{feed, symbol}

Structured logging via Python stdlib ``logging``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from prometheus_client import Counter

if TYPE_CHECKING:
    from data.storage.timescale import AsyncTimescaleDB
    from data.storage.redis_cache import RedisCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_FEED_LABEL = "kraken"

candles_received_total = Counter(
    "kraken_candles_received_total",
    "Total closed OHLC candles received from the Kraken WebSocket feed",
    ["feed", "symbol", "timeframe"],
)

feed_reconnects_total = Counter(
    "kraken_feed_reconnects_total",
    "Total WebSocket reconnection attempts for the Kraken feed",
    ["feed", "symbol"],
)

feed_errors_total = Counter(
    "kraken_feed_errors_total",
    "Total errors encountered in the Kraken WebSocket feed",
    ["feed", "symbol"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WS_URL = "wss://ws.kraken.com/v2"

_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_BASE_DELAY = 2.0    # seconds
_RECONNECT_MAX_DELAY = 60.0    # seconds
_STABLE_CONNECTION_SECS = 30.0  # reset attempt counter after this many seconds up

# Mapping from our timeframe strings to Kraken interval integers (minutes)
_TIMEFRAME_TO_INTERVAL: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_kraken_symbol(symbol: str) -> str:
    """Convert our canonical pair string to Kraken WS v2 symbol format.

    Kraken WS v2 uses ``"BASE/QUOTE"`` notation with ``/USD`` as the quote.

    Examples
    --------
    ::

        "BTCUSDT" → "BTC/USD"
        "ETHUSDT" → "ETH/USD"
        "SOLUSDT" → "SOL/USD"

    If the symbol does not end in ``USDT`` or ``USD``, it is returned
    unchanged to let Kraken handle it (e.g. ``"BTC/GBP"`` already formatted).
    """
    upper = symbol.upper()

    # Already in Kraken format
    if "/" in upper:
        return upper

    # Special-case: BTC → BTC (Kraken WS v2 accepts BTC/USD, not XBTUSDT)
    if upper.endswith("USDT"):
        base = upper[:-4]   # strip "USDT"
        return f"{base}/USD"
    if upper.endswith("USD"):
        base = upper[:-3]   # strip "USD"
        return f"{base}/USD"

    # Unknown format — return as-is; Kraken will reject gracefully
    return upper


def _from_kraken_symbol(kraken_sym: str) -> str:
    """Convert a Kraken WS v2 symbol back to our canonical format.

    Examples
    --------
    ::

        "BTC/USD" → "BTCUSDT"
        "ETH/USD" → "ETHUSDT"
    """
    if "/" not in kraken_sym:
        return kraken_sym
    base, quote = kraken_sym.split("/", 1)
    if quote in ("USD", "USDT"):
        return f"{base}USDT"
    return f"{base}{quote}"


def _parse_interval_begin(ts_str: str) -> int:
    """Parse an ISO-8601 timestamp string to a Unix millisecond integer.

    Kraken sends ``"interval_begin": "2024-01-01T12:00:00Z"``.
    """
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000)
    except (ValueError, AttributeError):
        return 0


def _interval_to_timeframe(interval: int) -> str:
    """Convert a Kraken interval integer back to our timeframe string."""
    reverse: dict[int, str] = {v: k for k, v in _TIMEFRAME_TO_INTERVAL.items()}
    return reverse.get(interval, f"{interval}m")


def _normalise_ohlc(data: dict, timeframe: str) -> dict:
    """Normalise a single Kraken OHLC data entry to our candle dict.

    Parameters
    ----------
    data:
        One element from the ``"data"`` array of an OHLC update message.
    timeframe:
        The timeframe string (e.g. ``"15m"``) that this candle belongs to.

    Returns
    -------
    dict
        Keys: ``symbol``, ``timeframe``, ``ts`` (ms), ``open``, ``high``,
        ``low``, ``close``, ``volume``.
    """
    kraken_sym: str = data.get("symbol", "")
    canonical_symbol = _from_kraken_symbol(kraken_sym)

    interval_begin: str = data.get("interval_begin", data.get("timestamp", ""))
    ts_ms = _parse_interval_begin(interval_begin)

    return {
        "symbol": canonical_symbol,
        "timeframe": timeframe,
        "ts": ts_ms,
        "open": float(data.get("open", 0)),
        "high": float(data.get("high", 0)),
        "low": float(data.get("low", 0)),
        "close": float(data.get("close", 0)),
        "volume": float(data.get("volume", 0)),
    }


# ---------------------------------------------------------------------------
# KrakenFeed
# ---------------------------------------------------------------------------


class KrakenFeed:
    """Subscribe to Kraken WS v2 OHLC streams and fan out closed candles.

    Parameters
    ----------
    db:
        :class:`~data.storage.timescale.AsyncTimescaleDB` instance (must be
        already connected before calling :meth:`start`).
    cache:
        :class:`~data.storage.redis_cache.RedisCache` instance.
    """

    def __init__(
        self,
        db: "AsyncTimescaleDB",
        cache: "RedisCache",
    ) -> None:
        self._db = db
        self._cache = cache

        self._symbols: list[str] = []
        self._timeframes: list[str] = []
        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, symbols: list[str], timeframes: list[str]) -> None:
        """Start streaming OHLC candles for every (symbol, timeframe) pair.

        Spawns one background task per *symbol*.  Each task opens a single WS
        connection and subscribes to all requested *timeframes* in one message.
        Call :meth:`stop` to shut everything down cleanly.

        Parameters
        ----------
        symbols:
            Our canonical pair strings, e.g. ``["BTCUSDT", "ETHUSDT"]``.
        timeframes:
            Our timeframe strings, e.g. ``["15m", "1h"]``.
        """
        self._symbols = [s.upper() for s in symbols]
        self._timeframes = timeframes
        self._stop_event.clear()

        for symbol in self._symbols:
            task = asyncio.create_task(
                self._symbol_loop(symbol),
                name=f"kraken-feed:{symbol}",
            )
            self._tasks.append(task)

        logger.info(
            "KrakenFeed started: symbols=%s timeframes=%s",
            self._symbols,
            self._timeframes,
        )

    async def stop(self) -> None:
        """Signal all feed tasks to stop and await their completion."""
        logger.info("Stopping KrakenFeed...")
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                logger.warning("Feed task raised on shutdown: %s", r)
        self._tasks.clear()
        logger.info("KrakenFeed stopped")

    # ------------------------------------------------------------------
    # Internal: per-symbol reconnect loop
    # ------------------------------------------------------------------

    async def _symbol_loop(self, symbol: str) -> None:
        """Outer reconnect loop for a single symbol.

        Uses exponential back-off (2 s → 4 s → 8 s … up to 60 s).
        Resets the attempt counter when a connection is stable for ≥30 s.
        """
        attempts = 0
        delay = _RECONNECT_BASE_DELAY

        while not self._stop_event.is_set():
            t_connect = asyncio.get_event_loop().time()
            try:
                await self._connect_and_stream(symbol)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                feed_errors_total.labels(feed=_FEED_LABEL, symbol=symbol).inc()

                # Reset attempt counter if the previous connection was stable
                uptime = asyncio.get_event_loop().time() - t_connect
                if uptime >= _STABLE_CONNECTION_SECS:
                    attempts = 0
                    delay = _RECONNECT_BASE_DELAY

                attempts += 1
                if attempts > _MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Symbol %s: exceeded max reconnect attempts (%d), giving up. "
                        "Last error: %s",
                        symbol,
                        _MAX_RECONNECT_ATTEMPTS,
                        exc,
                    )
                    return

                feed_reconnects_total.labels(feed=_FEED_LABEL, symbol=symbol).inc()
                logger.warning(
                    "Symbol %s: WebSocket error (%s), reconnecting in %.1fs "
                    "(attempt %d/%d)...",
                    symbol,
                    exc,
                    delay,
                    attempts,
                    _MAX_RECONNECT_ATTEMPTS,
                )

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=delay,
                    )
                    # stop_event was set — exit cleanly
                    return
                except asyncio.TimeoutError:
                    pass  # normal case: delay elapsed, try to reconnect

                # Exponential back-off
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)
            else:
                # Clean exit (stop_event fired inside _connect_and_stream)
                break

    # ------------------------------------------------------------------
    # Internal: connect, subscribe, consume
    # ------------------------------------------------------------------

    async def _connect_and_stream(self, symbol: str) -> None:
        """Open a WS connection to Kraken v2, subscribe, and process messages.

        A single connection handles all timeframes for *symbol*.
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "websockets is required: pip install websockets>=12.0"
            ) from exc

        kraken_sym = _to_kraken_symbol(symbol)
        logger.info(
            "Connecting to Kraken WS v2 for %s → %s, timeframes=%s",
            symbol,
            kraken_sym,
            self._timeframes,
        )

        async with websockets.connect(
            _WS_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # Subscribe to each requested timeframe
            for tf in self._timeframes:
                interval = _TIMEFRAME_TO_INTERVAL.get(tf)
                if interval is None:
                    logger.warning(
                        "Unknown timeframe %r for symbol %s — skipping", tf, symbol
                    )
                    continue

                sub_msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "ohlc",
                        "symbol": [kraken_sym],
                        "interval": interval,
                    },
                }
                await ws.send(json.dumps(sub_msg))
                logger.debug(
                    "Sent subscription: symbol=%s interval=%d", kraken_sym, interval
                )

            # Main message loop
            async for raw in ws:
                if self._stop_event.is_set():
                    return

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to decode WS message: %s | raw=%r", exc, raw)
                    feed_errors_total.labels(
                        feed=_FEED_LABEL, symbol=symbol
                    ).inc()
                    continue

                await self._dispatch_message(msg, symbol)

    # ------------------------------------------------------------------
    # Internal: message dispatcher
    # ------------------------------------------------------------------

    async def _dispatch_message(self, msg: dict, symbol: str) -> None:
        """Route an inbound WS message to the appropriate handler.

        We handle:
        - ``channel="ohlc"`` + ``type="update"`` — closed-candle update
        - ``channel="ohlc"`` + ``type="snapshot"`` — same structure; treat
          identically (useful for initial state)
        - Subscription status messages (``method="subscribe"`` responses) —
          logged at DEBUG level
        - Heartbeat / status messages — ignored
        """
        channel = msg.get("channel", "")
        msg_type = msg.get("type", "")

        # Subscription confirmation or error
        if msg.get("method") in ("subscribe", "unsubscribe"):
            if msg.get("success") is False:
                logger.warning(
                    "Subscription failed for %s: %s",
                    symbol,
                    msg.get("error", msg),
                )
                feed_errors_total.labels(feed=_FEED_LABEL, symbol=symbol).inc()
            else:
                logger.debug("Subscription confirmed: %s", msg)
            return

        # Heartbeat
        if channel == "heartbeat" or msg.get("event") == "heartbeat":
            return

        # Status
        if channel == "status":
            return

        # OHLC candle update / snapshot
        if channel == "ohlc" and msg_type in ("update", "snapshot"):
            data_list: list[dict] = msg.get("data", [])
            # Infer the interval from the subscription params echo in the message
            # (Kraken WS v2 includes "interval" at the top level of OHLC messages)
            interval: int | None = msg.get("interval") or msg.get(
                "params", {}
            ).get("interval")

            if interval is not None:
                timeframe = _interval_to_timeframe(interval)
            else:
                # Fall back: use the first subscribed timeframe
                timeframe = self._timeframes[0] if self._timeframes else "15m"

            for entry in data_list:
                await self._handle_closed_candle(entry, timeframe, symbol)
            return

        # Unhandled — log at DEBUG to avoid noise
        logger.debug("Unhandled WS message channel=%r type=%r", channel, msg_type)

    # ------------------------------------------------------------------
    # Internal: persist and publish a single candle
    # ------------------------------------------------------------------

    async def _handle_closed_candle(
        self, data: dict, timeframe: str, symbol_hint: str
    ) -> None:
        """Normalise, persist to TimescaleDB, and publish to Redis.

        Parameters
        ----------
        data:
            One entry from the ``"data"`` array of an OHLC update message.
        timeframe:
            Resolved timeframe string (e.g. ``"15m"``).
        symbol_hint:
            Our canonical symbol used as a fallback if the message does not
            carry one.
        """
        try:
            candle = _normalise_ohlc(data, timeframe)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to normalise OHLC data: %s | data=%r", exc, data
            )
            feed_errors_total.labels(
                feed=_FEED_LABEL, symbol=symbol_hint
            ).inc()
            return

        sym = candle["symbol"]
        tf = candle["timeframe"]

        # Prometheus counter
        candles_received_total.labels(
            feed=_FEED_LABEL, symbol=sym, timeframe=tf
        ).inc()

        # Persist to TimescaleDB
        try:
            await self._db.save_candle(
                symbol=sym,
                timeframe=tf,
                ts=candle["ts"],
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
            )
        except Exception:
            feed_errors_total.labels(feed=_FEED_LABEL, symbol=sym).inc()
            logger.exception("DB error saving candle for %s/%s", sym, tf)

        # Update Redis cache
        try:
            await self._cache.set_candle(sym, tf, candle)
        except Exception:
            logger.warning("Redis error caching candle for %s/%s", sym, tf)

        # Publish to 'candles' pub/sub channel
        try:
            await self._cache.publish("candles", candle)
        except Exception:
            logger.warning("Redis error publishing candle for %s/%s", sym, tf)

        logger.debug(
            "Closed candle: %s/%s close=%.8f vol=%.4f ts=%d",
            sym,
            tf,
            candle["close"],
            candle["volume"],
            candle["ts"],
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "KrakenFeed":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

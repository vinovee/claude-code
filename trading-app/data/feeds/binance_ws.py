"""
BinanceFeed — live kline (candlestick) WebSocket feed for Binance.

Uses python-binance (AsyncClient + BinanceSocketManager).
Closed candles are persisted to TimescaleDB and published to Redis.

Environment variables
---------------------
  BINANCE_API_KEY        — Binance API key (optional for market data)
  BINANCE_API_SECRET     — Binance secret  (optional for market data)
  BINANCE_TESTNET        — set to any truthy value to use testnet endpoints

Prometheus metrics (exposed via prometheus_client default registry)
-------------------------------------------------------------------
  candles_received_total{symbol, timeframe}
  feed_reconnects_total{symbol}
  feed_errors_total{symbol}
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from prometheus_client import Counter

if TYPE_CHECKING:
    from data.storage.timescale import AsyncTimescaleDB
    from data.storage.redis_cache import RedisCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------

candles_received_total = Counter(
    "candles_received_total",
    "Total number of closed candles received from Binance feed",
    ["symbol", "timeframe"],
)

feed_reconnects_total = Counter(
    "feed_reconnects_total",
    "Total number of WebSocket reconnection attempts per symbol",
    ["symbol"],
)

feed_errors_total = Counter(
    "feed_errors_total",
    "Total number of errors encountered in the Binance feed",
    ["symbol"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_DELAY_SECONDS = 5

_TESTNET_WSS_URL = "wss://testnet.binance.vision/ws"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_testnet() -> bool:
    val = os.environ.get("BINANCE_TESTNET", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _normalise_kline(msg: dict) -> dict:
    """Normalise a raw Binance kline payload to a flat candle dict."""
    k = msg["k"]
    return {
        "symbol": msg["s"],
        "timeframe": k["i"],
        "ts": int(k["t"]),  # open time in ms
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
    }


# ---------------------------------------------------------------------------
# BinanceFeed
# ---------------------------------------------------------------------------


class BinanceFeed:
    """Subscribe to Binance kline streams and fan out closed candles."""

    def __init__(
        self,
        db: "AsyncTimescaleDB",
        cache: "RedisCache",
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self._db = db
        self._cache = cache
        self._api_key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("BINANCE_API_SECRET", "")
        self._testnet = _is_testnet()

        self._symbols: list[str] = []
        self._timeframes: list[str] = []
        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, symbols: list[str], timeframes: list[str]) -> None:
        """Start streaming klines for every (symbol, timeframe) combination.

        This method launches a background task per symbol and returns
        immediately.  Call ``await feed.stop()`` to shut down cleanly.
        """
        self._symbols = [s.upper() for s in symbols]
        self._timeframes = timeframes
        self._stop_event.clear()

        for symbol in self._symbols:
            task = asyncio.create_task(
                self._symbol_loop(symbol),
                name=f"binance-feed:{symbol}",
            )
            self._tasks.append(task)

        logger.info(
            "BinanceFeed started: symbols=%s timeframes=%s testnet=%s",
            self._symbols,
            self._timeframes,
            self._testnet,
        )

    async def stop(self) -> None:
        """Signal all feed tasks to stop and await their completion."""
        logger.info("Stopping BinanceFeed…")
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                logger.warning("Feed task raised on shutdown: %s", r)
        self._tasks.clear()
        logger.info("BinanceFeed stopped")

    # ------------------------------------------------------------------
    # Internal: per-symbol reconnect loop
    # ------------------------------------------------------------------

    async def _symbol_loop(self, symbol: str) -> None:
        """Outer reconnect loop for a single symbol."""
        attempts = 0
        while not self._stop_event.is_set():
            try:
                await self._connect_and_stream(symbol)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                feed_errors_total.labels(symbol=symbol).inc()
                attempts += 1
                if attempts > _MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Symbol %s: exceeded max reconnect attempts (%d), giving up. Last error: %s",
                        symbol,
                        _MAX_RECONNECT_ATTEMPTS,
                        exc,
                    )
                    return
                feed_reconnects_total.labels(symbol=symbol).inc()
                logger.warning(
                    "Symbol %s: WebSocket error (%s), reconnecting in %ds (attempt %d/%d)…",
                    symbol,
                    exc,
                    _RECONNECT_DELAY_SECONDS,
                    attempts,
                    _MAX_RECONNECT_ATTEMPTS,
                )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=_RECONNECT_DELAY_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass  # stop_event not set — proceed with reconnect
            else:
                # Clean exit (stop_event fired inside _connect_and_stream)
                break

    async def _connect_and_stream(self, symbol: str) -> None:
        """Create an AsyncClient, open multiplex socket, and consume messages."""
        # Deferred import so the module is importable even if python-binance
        # is not installed (allows unit-testing the module structure).
        try:
            from binance import AsyncClient, BinanceSocketManager  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "python-binance is required: pip install python-binance"
            ) from exc

        client = await AsyncClient.create(
            api_key=self._api_key or None,
            api_secret=self._api_secret or None,
            testnet=self._testnet,
        )
        try:
            bsm = BinanceSocketManager(client)

            # Build stream names:  <symbol>@kline_<interval>
            streams = [
                f"{symbol.lower()}@kline_{tf}"
                for tf in self._timeframes
            ]
            logger.info("Opening multiplex socket for %s → %s", symbol, streams)

            # BinanceSocketManager.multiplex_socket accepts a list of streams
            async with bsm.multiplex_socket(streams) as mux:
                async for msg in mux:
                    if self._stop_event.is_set():
                        return
                    if msg is None:
                        continue
                    # Unwrap the outer envelope from the multiplex socket
                    data = msg.get("data", msg)
                    if data.get("e") != "kline":
                        continue
                    kline = data.get("k", {})
                    if not kline.get("x"):
                        # Candle not yet closed — skip
                        continue
                    await self._handle_closed_candle(data)
        finally:
            await client.close_connection()

    # ------------------------------------------------------------------
    # Internal: handle a closed candle
    # ------------------------------------------------------------------

    async def _handle_closed_candle(self, msg: dict) -> None:
        """Persist and publish a single closed candle."""
        try:
            candle = _normalise_kline(msg)
        except (KeyError, ValueError) as exc:
            logger.warning("Failed to normalise kline message: %s | msg=%r", exc, msg)
            return

        symbol = candle["symbol"]
        timeframe = candle["timeframe"]

        # Increment Prometheus counter
        candles_received_total.labels(symbol=symbol, timeframe=timeframe).inc()

        # Persist to TimescaleDB
        try:
            await self._db.save_candle(
                symbol=symbol,
                timeframe=timeframe,
                ts=candle["ts"],
                open=candle["open"],
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                volume=candle["volume"],
            )
        except Exception:  # noqa: BLE001
            feed_errors_total.labels(symbol=symbol).inc()
            logger.exception("DB error saving candle for %s/%s", symbol, timeframe)

        # Cache in Redis
        try:
            await self._cache.set_candle(symbol, timeframe, candle)
        except Exception:  # noqa: BLE001
            logger.warning("Redis error caching candle for %s/%s", symbol, timeframe)

        # Publish to 'candles' channel
        try:
            await self._cache.publish("candles", candle)
        except Exception:  # noqa: BLE001
            logger.warning("Redis error publishing candle for %s/%s", symbol, timeframe)

        logger.debug(
            "Closed candle: %s/%s close=%.8f vol=%.2f",
            symbol,
            timeframe,
            candle["close"],
            candle["volume"],
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BinanceFeed":
        return self

    async def __aexit__(self, *_) -> None:  # type: ignore[override]
        await self.stop()

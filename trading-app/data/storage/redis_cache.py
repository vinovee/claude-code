"""
RedisCache — async Redis client for the trading application.

Uses redis[hiredis] (redis-py >= 4.2) async API.

Key naming conventions
----------------------
  candle:<symbol>:<timeframe>            → latest OHLCV dict (hash)
  indicator:<symbol>:<timeframe>:<name>  → float (string)
  signal:<strategy>:<symbol>             → float (string)

TTL policy
----------
  Candles / indicators: 2× the timeframe expressed in seconds.
  Signals: 60 seconds.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable

import redis.asyncio as aioredis
from redis.asyncio.client import PubSub

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe → seconds lookup
# ---------------------------------------------------------------------------

_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}


def _timeframe_ttl(timeframe: str) -> int:
    """Return TTL = 2× timeframe in seconds.  Defaults to 120 s if unknown."""
    return _TIMEFRAME_SECONDS.get(timeframe, 60) * 2


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


def _candle_key(symbol: str, timeframe: str) -> str:
    return f"candle:{symbol}:{timeframe}"


def _indicator_key(symbol: str, timeframe: str, indicator_name: str) -> str:
    return f"indicator:{symbol}:{timeframe}:{indicator_name}"


def _signal_key(strategy: str, symbol: str) -> str:
    return f"signal:{strategy}:{symbol}"


# ---------------------------------------------------------------------------
# RedisCache
# ---------------------------------------------------------------------------


class RedisCache:
    """Async Redis cache for candles, indicators, signals, and pub/sub."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._client: aioredis.Redis | None = None
        self._pubsub: PubSub | None = None
        self._subscriber_tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the Redis connection (lazy — called automatically on first use)."""
        if self._client is not None:
            return
        logger.info("Connecting to Redis at %s", self._url)
        self._client = aioredis.from_url(
            self._url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        # Verify connectivity
        await self._client.ping()
        logger.info("Redis connection established")

    async def close(self) -> None:
        """Cancel subscriber tasks and close the connection."""
        for task in self._subscriber_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._subscriber_tasks.clear()

        if self._pubsub is not None:
            await self._pubsub.close()
            self._pubsub = None

        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("Redis connection closed")

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            await self.connect()
        return self._client  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    async def set_candle(self, symbol: str, timeframe: str, candle_dict: dict) -> None:
        """Store the latest OHLCV candle as a Redis hash with TTL = 2× timeframe."""
        client = await self._get_client()
        key = _candle_key(symbol, timeframe)
        ttl = _timeframe_ttl(timeframe)

        # Serialise all values to strings for hash storage
        mapping = {k: str(v) for k, v in candle_dict.items()}

        pipe = client.pipeline()
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, ttl)
        await pipe.execute()

    async def get_candle(self, symbol: str, timeframe: str) -> dict | None:
        """Retrieve the latest candle dict, or *None* if not cached."""
        client = await self._get_client()
        key = _candle_key(symbol, timeframe)
        data = await client.hgetall(key)
        if not data:
            return None
        # Attempt numeric coercion for known float fields
        result: dict[str, Any] = {}
        float_fields = {"open", "high", "low", "close", "volume"}
        int_fields = {"ts"}
        for k, v in data.items():
            if k in float_fields:
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    result[k] = v
            elif k in int_fields:
                try:
                    result[k] = int(v)
                except (ValueError, TypeError):
                    result[k] = v
            else:
                result[k] = v
        return result

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    async def set_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        value: float,
    ) -> None:
        """Cache a single indicator value with TTL = 2× timeframe."""
        client = await self._get_client()
        key = _indicator_key(symbol, timeframe, indicator_name)
        ttl = _timeframe_ttl(timeframe)
        await client.set(key, str(value), ex=ttl)

    async def get_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
    ) -> float | None:
        """Retrieve a cached indicator value, or *None* if absent / expired."""
        client = await self._get_client()
        key = _indicator_key(symbol, timeframe, indicator_name)
        raw = await client.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            logger.warning("Invalid indicator value in cache for key %s: %r", key, raw)
            return None

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def set_signal(self, strategy: str, symbol: str, score: float) -> None:
        """Cache a strategy signal score with a 60-second TTL."""
        client = await self._get_client()
        key = _signal_key(strategy, symbol)
        await client.set(key, str(score), ex=60)

    async def get_signal(self, strategy: str, symbol: str) -> float | None:
        """Retrieve a cached signal score, or *None* if absent / expired."""
        client = await self._get_client()
        key = _signal_key(strategy, symbol)
        raw = await client.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            logger.warning("Invalid signal value in cache for key %s: %r", key, raw)
            return None

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def publish(self, channel: str, message: dict) -> None:
        """JSON-serialise *message* and publish it to *channel*."""
        client = await self._get_client()
        payload = json.dumps(message, default=str)
        await client.publish(channel, payload)

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[dict], Awaitable[None] | None],
    ) -> None:
        """Subscribe to *channel* and call *callback* for each message.

        The subscription runs in a background asyncio task.  Errors in
        *callback* are logged but do not stop the subscriber.
        """
        client = await self._get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        logger.info("Subscribed to Redis channel '%s'", channel)

        async def _listener() -> None:
            try:
                async for raw_message in pubsub.listen():
                    if raw_message["type"] != "message":
                        continue
                    try:
                        data = json.loads(raw_message["data"])
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning("Could not decode message on '%s': %s", channel, exc)
                        continue
                    try:
                        result = callback(data)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:  # noqa: BLE001
                        logger.exception("Error in subscriber callback for channel '%s'", channel)
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.close()

        task = asyncio.create_task(_listener(), name=f"redis-sub:{channel}")
        self._subscriber_tasks.append(task)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "RedisCache":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

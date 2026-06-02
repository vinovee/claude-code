"""
AsyncTimescaleDB — asyncpg-based client for TimescaleDB.

Tables expected (DDL not created here, managed by migrations):
  ohlcv          (symbol, timeframe, ts, open, high, low, close, volume)
  trades         (id serial PK, …arbitrary columns from trade dict…)
  positions      (symbol PK, …arbitrary columns from position dict…)
  portfolio_snapshots (id serial PK, …arbitrary columns from snapshot dict…)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import asyncpg
import pandas as pd
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETRY_KWARGS: dict[str, Any] = dict(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(
        (asyncpg.PostgresConnectionError, asyncpg.TooManyConnectionsError, OSError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _dsn() -> str:
    """Build DSN from env vars or fall back to a sensible default."""
    return os.environ.get(
        "TIMESCALE_DSN",
        "postgresql://postgres:postgres@localhost:5432/trading",
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AsyncTimescaleDB:
    """Async interface to TimescaleDB using an asyncpg connection pool."""

    def __init__(self, dsn: str | None = None, min_size: int = 2, max_size: int = 10) -> None:
        self._dsn = dsn or _dsn()
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the asyncpg connection pool (idempotent)."""
        if self._pool is not None:
            return

        async for attempt in AsyncRetrying(**_RETRY_KWARGS):
            with attempt:
                logger.info("Creating asyncpg pool → %s", self._dsn.split("@")[-1])
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    command_timeout=30,
                    server_settings={"application_name": "trading-app"},
                )
        logger.info("asyncpg pool ready (min=%d max=%d)", self._min_size, self._max_size)

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("asyncpg pool closed")

    async def _pool_acquire(self) -> asyncpg.Connection:
        if self._pool is None:
            raise RuntimeError("Pool not initialised — call connect() first")
        return self._pool.acquire()

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    async def save_candle(
        self,
        symbol: str,
        timeframe: str,
        ts: datetime | int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Upsert a single OHLCV candle into the *ohlcv* hypertable.

        *ts* may be a ``datetime`` (with or without tzinfo) or a Unix
        millisecond integer — it is normalised to a tz-aware ``datetime``
        before insertion.
        """
        if isinstance(ts, (int, float)):
            ts = datetime.utcfromtimestamp(int(ts) / 1000).replace(
                tzinfo=__import__("datetime").timezone.utc
            )

        sql = """
            INSERT INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (symbol, timeframe, ts)
            DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        async for attempt in AsyncRetrying(**_RETRY_KWARGS):
            with attempt:
                async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                    await conn.execute(sql, symbol, timeframe, ts, open, high, low, close, volume)

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
        end_ts: datetime | int | None = None,
    ) -> pd.DataFrame:
        """Return the most recent *limit* closed candles as a DataFrame.

        Columns: ts, open, high, low, close, volume  (ts is the index).
        """
        if isinstance(end_ts, (int, float)):
            end_ts = datetime.utcfromtimestamp(int(end_ts) / 1000).replace(
                tzinfo=__import__("datetime").timezone.utc
            )

        if end_ts is not None:
            sql = """
                SELECT ts, open, high, low, close, volume
                FROM   ohlcv
                WHERE  symbol = $1 AND timeframe = $2 AND ts <= $3
                ORDER  BY ts DESC
                LIMIT  $4
            """
            params = (symbol, timeframe, end_ts, limit)
        else:
            sql = """
                SELECT ts, open, high, low, close, volume
                FROM   ohlcv
                WHERE  symbol = $1 AND timeframe = $2
                ORDER  BY ts DESC
                LIMIT  $3
            """
            params = (symbol, timeframe, limit)

        async for attempt in AsyncRetrying(**_RETRY_KWARGS):
            with attempt:
                async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                    rows = await conn.fetch(sql, *params)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").set_index("ts")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return df

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def save_trade(self, trade: dict) -> None:
        """Insert a trade record.

        The *trade* dict keys must match the column names of the *trades*
        table.  A ``created_at`` column is populated automatically if not
        present.
        """
        if not trade:
            raise ValueError("trade dict must not be empty")

        trade = dict(trade)  # shallow copy — don't mutate caller's dict
        trade.setdefault("created_at", datetime.utcnow())

        columns = list(trade.keys())
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        col_list = ", ".join(f'"{c}"' for c in columns)
        sql = f"INSERT INTO trades ({col_list}) VALUES ({placeholders})"

        async for attempt in AsyncRetrying(**_RETRY_KWARGS):
            with attempt:
                async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                    await conn.execute(sql, *trade.values())

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def save_position(self, position: dict) -> None:
        """Upsert a position record keyed on *symbol*.

        Assumes a *positions* table with at least a ``symbol`` primary key
        column plus whatever columns appear in the *position* dict.
        """
        if "symbol" not in position:
            raise ValueError("position dict must contain 'symbol'")

        position = dict(position)
        position.setdefault("updated_at", datetime.utcnow())

        columns = list(position.keys())
        col_list = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))

        # Build the SET clause excluding the PK
        update_parts = [
            f'"{c}" = EXCLUDED."{c}"' for c in columns if c != "symbol"
        ]
        update_clause = ", ".join(update_parts)

        sql = f"""
            INSERT INTO positions ({col_list}) VALUES ({placeholders})
            ON CONFLICT (symbol)
            DO UPDATE SET {update_clause}
        """

        async for attempt in AsyncRetrying(**_RETRY_KWARGS):
            with attempt:
                async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                    await conn.execute(sql, *position.values())

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    async def save_portfolio_snapshot(self, snapshot: dict) -> None:
        """Insert a portfolio snapshot (append-only).

        A ``ts`` key defaults to ``datetime.utcnow()`` if absent.
        """
        if not snapshot:
            raise ValueError("snapshot dict must not be empty")

        snapshot = dict(snapshot)
        snapshot.setdefault("ts", datetime.utcnow())

        columns = list(snapshot.keys())
        col_list = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        sql = f"INSERT INTO portfolio_snapshots ({col_list}) VALUES ({placeholders})"

        async for attempt in AsyncRetrying(**_RETRY_KWARGS):
            with attempt:
                async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                    await conn.execute(sql, *snapshot.values())

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncTimescaleDB":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

#!/usr/bin/env python3
"""
Download historical OHLCV data from the Kraken REST API into TimescaleDB.

Usage
-----
.. code-block:: bash

    python scripts/download_historical.py --symbol BTCUSDT --interval 15m --start 2022-01-01
    python scripts/download_historical.py --symbol ETHUSDT --interval 1h  --start 2023-06-01 --end 2024-01-01
    python scripts/download_historical.py --symbol SOLUSDT --interval 1d  --start 2021-01-01 --dsn postgresql://trader:pass@localhost:5432/trading

Kraken REST notes
-----------------
- ``get_ohlc_data`` returns up to **720 candles** per call.
- Pagination is done via the ``since`` parameter (Unix second timestamp).
- Kraken returns candle data in *ascending* time order.
- The last element of the ``last`` field in the response is the timestamp to
  use as ``since`` for the next page.
- Kraken's internal BTC pair name is ``XBTUSDT``; we try that first, then
  fall back to the caller-supplied symbol.

Supported interval strings → Kraken interval integers (minutes)
----------------------------------------------------------------
    1m  →  1       5m  →  5       15m → 15
    30m → 30       1h  → 60       4h  → 240
    1d  → 1440

Dependencies
------------
    krakenex, asyncpg, rich, click  (all in requirements.txt)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import click
import krakenex
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Ensure project root is on the path so local package imports work when
# running this script directly (python scripts/download_historical.py …).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.storage.timescale import AsyncTimescaleDB  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANDLES_PER_PAGE = 720   # Kraken's hard limit per get_ohlc_data call

_INTERVAL_MAP: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# Kraken uses XBT internally for Bitcoin
_SYMBOL_REMAP: dict[str, list[str]] = {
    "BTCUSDT": ["XBTUSDT", "BTCUSDT"],
    "BTCUSD": ["XBTUSD", "BTCUSD"],
    "BTCGBP": ["XBTGBP", "BTCGBP"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_kraken_pairs(symbol: str) -> list[str]:
    """Return a list of Kraken pair names to try for a given symbol.

    For BTC-based pairs we try XBTUSDT first (Kraken native), then fall back
    to BTCUSDT.  For all other symbols the original name is used.
    """
    upper = symbol.upper()
    return _SYMBOL_REMAP.get(upper, [upper])


def _parse_date(date_str: str) -> datetime:
    """Parse an ISO date string (``YYYY-MM-DD``) into a tz-aware datetime."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        # Try full ISO format
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            raise click.BadParameter(
                f"Cannot parse date {date_str!r}. Use YYYY-MM-DD format."
            )


def _dt_to_unix(dt: datetime) -> int:
    """Convert a tz-aware datetime to a Unix second timestamp."""
    return int(dt.timestamp())


def _unix_to_dt(ts: int | float) -> datetime:
    """Convert a Unix second timestamp to a tz-aware datetime."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def _interval_to_timeframe(interval_int: int) -> str:
    """Convert a Kraken interval integer back to our timeframe string."""
    reverse = {v: k for k, v in _INTERVAL_MAP.items()}
    return reverse.get(interval_int, f"{interval_int}m")


# ---------------------------------------------------------------------------
# Kraken data fetcher
# ---------------------------------------------------------------------------


def _fetch_ohlc_page(
    api: krakenex.API,
    pair: str,
    interval: int,
    since: int,
) -> tuple[list[list], int | None]:
    """Fetch one page of OHLC data from Kraken.

    Parameters
    ----------
    api:
        Initialised ``krakenex.API`` instance (no credentials needed for
        public market data).
    pair:
        Kraken pair name, e.g. ``"XBTUSDT"`` or ``"ETHUSDT"``.
    interval:
        Candle interval in minutes.
    since:
        Unix second timestamp; fetch candles with open-time ≥ *since*.

    Returns
    -------
    (candles, last_ts)
        ``candles`` is a list of raw OHLC rows in Kraken format:
        ``[time, open, high, low, close, vwap, volume, count]``
        ``last_ts`` is the ``"last"`` field from the response (used as the
        ``since`` value for the next page), or ``None`` on the final page.

    Raises
    ------
    RuntimeError
        If Kraken returns a non-empty error list.
    """
    resp = api.query_public(
        "OHLC",
        {
            "pair": pair,
            "interval": interval,
            "since": since,
        },
    )

    errors: list[str] = resp.get("error", [])
    if errors:
        raise RuntimeError(f"Kraken API error: {'; '.join(errors)}")

    result: dict = resp.get("result", {})
    # The pair data is stored under either the exact pair name or Kraken's
    # internal name (e.g. "XXBTZUSDT" for XBTUSDT).  We pick the first
    # non-"last" key.
    candle_data: list[list] = []
    for key, val in result.items():
        if key == "last":
            continue
        if isinstance(val, list):
            candle_data = val
            break

    last_ts: int | None = result.get("last")
    return candle_data, last_ts


def _row_to_candle(row: list, timeframe: str, canonical_symbol: str) -> dict:
    """Convert a raw Kraken OHLC row to our candle dict.

    Kraken row format:
        [time, open, high, low, close, vwap, volume, count]

    We discard ``vwap`` and ``count``.
    """
    return {
        "symbol": canonical_symbol,
        "timeframe": timeframe,
        "ts": int(row[0]) * 1_000,  # seconds → milliseconds
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[6]),
    }


# ---------------------------------------------------------------------------
# Main download coroutine
# ---------------------------------------------------------------------------


async def _download(
    symbol: str,
    interval_str: str,
    start_dt: datetime,
    end_dt: datetime,
    dsn: str,
    dry_run: bool,
) -> None:
    """Core download loop.

    1. Resolves the Kraken pair name (tries XBT alias for BTC).
    2. Paginates backwards in time from *now* (or *end_dt*) to *start_dt*.
    3. Saves each batch to TimescaleDB via :meth:`AsyncTimescaleDB.save_candle`.
    """
    interval_int = _INTERVAL_MAP.get(interval_str)
    if interval_int is None:
        raise click.BadParameter(
            f"Unknown interval {interval_str!r}. "
            f"Supported: {', '.join(_INTERVAL_MAP)}"
        )
    timeframe = _interval_to_timeframe(interval_int)

    start_ts = _dt_to_unix(start_dt)
    end_ts = _dt_to_unix(end_dt)

    api = krakenex.API()  # public endpoints — no credentials needed

    # Resolve pair: try BTC-aware aliases first
    kraken_pair: str | None = None
    for candidate in _to_kraken_pairs(symbol):
        try:
            test_resp = api.query_public(
                "OHLC",
                {"pair": candidate, "interval": interval_int, "since": end_ts - interval_int * 60},
            )
            if not test_resp.get("error"):
                kraken_pair = candidate
                break
        except Exception:
            continue

    if kraken_pair is None:
        console.print(
            f"[red]Could not resolve Kraken pair for {symbol!r}.[/red]\n"
            "Tried: " + ", ".join(_to_kraken_pairs(symbol))
        )
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold cyan]Kraken Historical Downloader[/bold cyan]\n"
            f"Symbol: [yellow]{symbol}[/yellow]  Pair: [yellow]{kraken_pair}[/yellow]\n"
            f"Interval: [yellow]{interval_str}[/yellow]  "
            f"Start: [yellow]{start_dt.date()}[/yellow]  "
            f"End: [yellow]{end_dt.date()}[/yellow]",
            border_style="cyan",
        )
    )

    if dry_run:
        console.print("[yellow]DRY RUN — no data will be written to the database.[/yellow]")

    # We paginate forward from start_ts.  Kraken paginates via the "last"
    # field, which is the timestamp of the last candle returned.  We stop
    # when we either reach end_ts or the page returns fewer than the max
    # candles (i.e. we've hit the present).
    since_ts = start_ts
    total_saved = 0
    total_skipped = 0
    batch_count = 0

    db: AsyncTimescaleDB | None = None
    if not dry_run:
        db = AsyncTimescaleDB(dsn=dsn)
        await db.connect()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # We don't know the total candle count upfront; use an indeterminate
            # task that we update as we go.
            task = progress.add_task(
                f"Downloading [bold]{symbol}[/bold] {interval_str} candles...",
                total=None,
            )

            while since_ts < end_ts:
                batch_count += 1
                progress.update(
                    task,
                    description=(
                        f"Downloading [bold]{symbol}[/bold] {interval_str}  "
                        f"(from {_unix_to_dt(since_ts).strftime('%Y-%m-%d')})"
                    ),
                )

                try:
                    raw_candles, next_since = _fetch_ohlc_page(
                        api, kraken_pair, interval_int, since_ts
                    )
                except RuntimeError as exc:
                    console.print(f"\n[red]API error on batch {batch_count}: {exc}[/red]")
                    console.print("[yellow]Waiting 5 s before retrying...[/yellow]")
                    await asyncio.sleep(5)
                    continue
                except Exception as exc:
                    console.print(
                        f"\n[red]Unexpected error on batch {batch_count}: {exc}[/red]"
                    )
                    raise

                if not raw_candles:
                    # No more data
                    break

                # Filter to the requested time range
                filtered = [
                    r for r in raw_candles
                    if start_ts <= int(r[0]) <= end_ts
                ]

                # Save to DB
                for row in filtered:
                    candle = _row_to_candle(row, timeframe, symbol.upper())
                    if not dry_run and db is not None:
                        try:
                            await db.save_candle(
                                symbol=candle["symbol"],
                                timeframe=candle["timeframe"],
                                ts=candle["ts"],
                                open=candle["open"],
                                high=candle["high"],
                                low=candle["low"],
                                close=candle["close"],
                                volume=candle["volume"],
                            )
                            total_saved += 1
                        except Exception as exc:
                            console.print(
                                f"\n[yellow]Warning: failed to save candle "
                                f"ts={candle['ts']}: {exc}[/yellow]"
                            )
                            total_skipped += 1
                    else:
                        total_saved += 1  # count in dry-run mode too

                progress.update(task, advance=len(filtered))

                # Advance pagination cursor
                if next_since is not None and next_since > since_ts:
                    since_ts = next_since
                elif raw_candles:
                    # Use the timestamp of the last candle + 1 interval
                    last_candle_ts = int(raw_candles[-1][0])
                    since_ts = last_candle_ts + interval_int * 60
                else:
                    break

                # Kraken public API rate limit: 1 request per second is safe.
                await asyncio.sleep(1.0)

        # Final summary
        summary = Table(title="Download Summary", show_header=True, header_style="bold magenta")
        summary.add_column("Field", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Symbol", symbol.upper())
        summary.add_row("Kraken pair", kraken_pair)
        summary.add_row("Interval", interval_str)
        summary.add_row("Date range", f"{start_dt.date()} → {end_dt.date()}")
        summary.add_row("Batches fetched", str(batch_count))
        summary.add_row("Candles saved", str(total_saved))
        if total_skipped:
            summary.add_row("Candles skipped (errors)", str(total_skipped), )
        if dry_run:
            summary.add_row("Mode", "[yellow]DRY RUN (nothing written)[/yellow]")

        console.print()
        console.print(summary)
        console.print()

    finally:
        if db is not None:
            await db.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--symbol",
    required=True,
    help='Trading pair in canonical format, e.g. "BTCUSDT", "ETHUSDT".',
)
@click.option(
    "--interval",
    "interval_str",
    required=True,
    type=click.Choice(list(_INTERVAL_MAP.keys()), case_sensitive=False),
    help="Candle interval (e.g. 15m, 1h, 1d).",
)
@click.option(
    "--start",
    "start_date",
    required=True,
    help="Start date in YYYY-MM-DD format (inclusive).",
)
@click.option(
    "--end",
    "end_date",
    default=None,
    show_default=True,
    help=(
        "End date in YYYY-MM-DD format (inclusive, defaults to today UTC). "
        "Example: --end 2024-06-01"
    ),
)
@click.option(
    "--dsn",
    default=None,
    envvar="TIMESCALEDB_URL",
    show_default=True,
    help=(
        "TimescaleDB connection string. "
        "Defaults to the TIMESCALEDB_URL environment variable or "
        "postgresql://trader:change_me_in_production@localhost:5432/trading."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Fetch and count candles without writing to the database.",
)
def main(
    symbol: str,
    interval_str: str,
    start_date: str,
    end_date: Optional[str],
    dsn: Optional[str],
    dry_run: bool,
) -> None:
    """Download historical OHLCV data from Kraken into TimescaleDB.

    \b
    Examples:
      python scripts/download_historical.py --symbol BTCUSDT --interval 15m --start 2022-01-01
      python scripts/download_historical.py --symbol ETHUSDT --interval 1h  --start 2023-01-01 --end 2024-01-01
      python scripts/download_historical.py --symbol SOLUSDT --interval 1d  --start 2021-01-01 --dry-run
    """
    # Parse dates
    start_dt = _parse_date(start_date)

    if end_date is not None:
        end_dt = _parse_date(end_date)
        # Set end_dt to end-of-day so we include the full final day
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
    else:
        end_dt = datetime.now(tz=timezone.utc)

    if start_dt >= end_dt:
        raise click.BadParameter(
            f"--start ({start_date}) must be before --end ({end_date or 'today'})."
        )

    # Resolve DSN
    resolved_dsn = dsn or os.environ.get(
        "TIMESCALEDB_URL",
        "postgresql://trader:change_me_in_production@localhost:5432/trading",
    )
    # Convert SQLAlchemy asyncpg prefix to plain asyncpg DSN
    for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://"):
        if resolved_dsn.startswith(prefix):
            resolved_dsn = "postgresql://" + resolved_dsn[len(prefix):]
            break

    asyncio.run(
        _download(
            symbol=symbol.upper(),
            interval_str=interval_str.lower(),
            start_dt=start_dt,
            end_dt=end_dt,
            dsn=resolved_dsn,
            dry_run=dry_run,
        )
    )


if __name__ == "__main__":
    main()

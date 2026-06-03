"""
TradeLogExporter — export and summarise closed trade records.

Provides CSV/JSON exports and rolling performance analytics for individual
days or strategies.

Usage
-----
    from reporting.trade_log import TradeLogExporter

    exporter = TradeLogExporter(db=db)
    path = await exporter.export_csv("2024-01-01", "2024-12-31", "/tmp/trades.csv")
    summary = await exporter.daily_summary("2024-06-15")
    perf = await exporter.strategy_performance("crypto_scalp", days=30)
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from config.settings import get_settings
from data.storage.timescale import AsyncTimescaleDB

log = structlog.get_logger(__name__)

__all__ = ["TradeLogExporter"]

# ---------------------------------------------------------------------------
# Column order for CSV / JSON output
# ---------------------------------------------------------------------------
_EXPORT_COLUMNS: list[str] = [
    "id",
    "symbol",
    "side",
    "strategy",
    "entry_price",
    "exit_price",
    "qty",
    "entry_time",
    "exit_time",
    "exit_reason",
    "pnl_abs",
    "pnl_pct",
    "commission_paid",
    "duration_minutes",
    "phase",
]


# ---------------------------------------------------------------------------
# TradeLogExporter
# ---------------------------------------------------------------------------


class TradeLogExporter:
    """Query, aggregate and export trade records from TimescaleDB.

    Parameters
    ----------
    db:
        Connected :class:`~data.storage.timescale.AsyncTimescaleDB` instance.
    """

    def __init__(self, db: AsyncTimescaleDB) -> None:
        self._db = db
        self._settings = get_settings()
        self._log = log.bind(component="TradeLogExporter")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    async def export_csv(
        self,
        start_date: str,
        end_date: str,
        output_path: str,
    ) -> str:
        """Export all trades in a date range to a CSV file.

        Parameters
        ----------
        start_date:
            Inclusive start in ``"YYYY-MM-DD"`` format (interpreted as UTC).
        end_date:
            Inclusive end in ``"YYYY-MM-DD"`` format (end of day UTC).
        output_path:
            Absolute or relative path for the output ``.csv`` file.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        trades = await self._fetch_trades(start_date, end_date)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if not trades:
            self._log.warning(
                "trade_log.export_csv_empty",
                start=start_date,
                end=end_date,
            )
            # Write an empty CSV with headers
            with open(output_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_EXPORT_COLUMNS)
                writer.writeheader()
            return str(Path(output_path).resolve())

        # Determine actual columns available
        available_cols = [c for c in _EXPORT_COLUMNS if c in trades[0]]
        extra_cols = [c for c in trades[0].keys() if c not in _EXPORT_COLUMNS]
        fieldnames = available_cols + extra_cols

        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for trade in trades:
                row = {k: self._serialise_value(v) for k, v in trade.items()}
                writer.writerow(row)

        abs_path = str(Path(output_path).resolve())
        self._log.info(
            "trade_log.csv_exported",
            start=start_date,
            end=end_date,
            n_trades=len(trades),
            path=abs_path,
        )
        return abs_path

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    async def export_json(
        self,
        start_date: str,
        end_date: str,
        output_path: str,
    ) -> str:
        """Export all trades in a date range to a JSON file.

        Parameters
        ----------
        start_date:
            Inclusive start in ``"YYYY-MM-DD"`` format (UTC).
        end_date:
            Inclusive end in ``"YYYY-MM-DD"`` format (end of day UTC).
        output_path:
            Absolute or relative path for the output ``.json`` file.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        trades = await self._fetch_trades(start_date, end_date)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Serialise to JSON-safe dicts
        serialised = [
            {k: self._serialise_value(v) for k, v in t.items()} for t in trades
        ]

        payload = {
            "export_metadata": {
                "start_date": start_date,
                "end_date": end_date,
                "exported_at": datetime.now(tz=timezone.utc).isoformat(),
                "total_trades": len(serialised),
            },
            "trades": serialised,
        }

        with open(output_path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        abs_path = str(Path(output_path).resolve())
        self._log.info(
            "trade_log.json_exported",
            start=start_date,
            end=end_date,
            n_trades=len(trades),
            path=abs_path,
        )
        return abs_path

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    async def daily_summary(self, date: str) -> dict[str, Any]:
        """Return aggregated trading statistics for a single UTC calendar day.

        Parameters
        ----------
        date:
            Day in ``"YYYY-MM-DD"`` format.

        Returns
        -------
        dict
            Keys: ``date``, ``total_trades``, ``wins``, ``losses``,
            ``win_rate``, ``total_pnl_abs``, ``total_pnl_pct``,
            ``total_fees``, ``avg_trade_pct``, ``avg_win_pct``,
            ``avg_loss_pct``, ``largest_win_pct``, ``largest_loss_pct``,
            ``strategies_traded``, ``symbols_traded``.
        """
        trades = await self._fetch_trades(date, date)
        return self._compute_period_summary(trades, label=date)

    # ------------------------------------------------------------------
    # Strategy performance
    # ------------------------------------------------------------------

    async def strategy_performance(
        self,
        strategy_name: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Return rolling performance statistics for a strategy.

        Parameters
        ----------
        strategy_name:
            Strategy identifier (matched against the ``strategy`` column).
        days:
            Look-back window in calendar days (default 30).

        Returns
        -------
        dict
            Keys: ``strategy``, ``period_days``, ``start_date``,
            ``end_date``, ``total_trades``, ``wins``, ``losses``,
            ``win_rate``, ``profit_factor``, ``total_pnl_abs``,
            ``total_pnl_pct``, ``avg_trade_pct``, ``avg_win_pct``,
            ``avg_loss_pct``, ``largest_win_pct``, ``largest_loss_pct``,
            ``total_fees``, ``sharpe_ratio``, ``avg_duration_minutes``.
        """
        end_dt = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        all_trades = await self._fetch_trades(start_date, end_date)

        # Filter to the requested strategy
        trades = [
            t for t in all_trades
            if str(t.get("strategy", "")).lower() == strategy_name.lower()
        ]

        summary = self._compute_period_summary(trades, label=strategy_name)

        # Additional strategy-specific fields
        summary["strategy"] = strategy_name
        summary["period_days"] = days
        summary["start_date"] = start_date
        summary["end_date"] = end_date
        summary.pop("date", None)  # replace generic "date" key

        # Sharpe ratio over the period (trade-level)
        pnl_series = pd.Series(
            [float(t.get("pnl_pct", 0.0)) for t in trades], dtype=float
        )
        sharpe = 0.0
        if len(pnl_series) >= 2 and pnl_series.std(ddof=1) > 0:
            sharpe = float(
                (pnl_series.mean() / pnl_series.std(ddof=1))
                * math.sqrt(len(pnl_series))
            )
        summary["sharpe_ratio"] = round(sharpe, 4)

        # Average duration
        durations = []
        for t in trades:
            dur = t.get("duration_minutes")
            if dur is not None:
                try:
                    durations.append(float(dur))
                except (TypeError, ValueError):
                    pass
            else:
                try:
                    entry = pd.to_datetime(t.get("entry_time"), utc=True)
                    exit_ = pd.to_datetime(t.get("exit_time"), utc=True)
                    if entry is not None and exit_ is not None:
                        durations.append(
                            (exit_ - entry).total_seconds() / 60.0
                        )
                except Exception:
                    pass
        summary["avg_duration_minutes"] = (
            round(float(np.mean(durations)), 2) if durations else 0.0
        )

        self._log.info(
            "trade_log.strategy_performance",
            strategy=strategy_name,
            days=days,
            total_trades=summary.get("total_trades", 0),
        )
        return summary

    # ------------------------------------------------------------------
    # Private: data fetch
    # ------------------------------------------------------------------

    async def _fetch_trades(
        self,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch closed trades from TimescaleDB for the given date range.

        Falls back to an empty list on error without propagating exceptions.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )

        try:
            async with self._db._pool.acquire() as conn:  # type: ignore[union-attr]
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM   trades
                    WHERE  exit_time >= $1
                      AND  exit_time <= $2
                    ORDER  BY exit_time ASC
                    """,
                    start_dt,
                    end_dt,
                )
            trades = [dict(row) for row in rows]
        except Exception as exc:
            self._log.error(
                "trade_log.fetch_error",
                start=start_date,
                end=end_date,
                error=str(exc),
            )
            trades = []

        self._log.debug(
            "trade_log.fetch_complete",
            start=start_date,
            end=end_date,
            n=len(trades),
        )
        return trades

    # ------------------------------------------------------------------
    # Private: period aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_period_summary(
        trades: list[dict],
        label: str = "",
    ) -> dict[str, Any]:
        """Compute aggregate statistics over a list of trade dicts."""
        total = len(trades)
        if total == 0:
            return {
                "date": label,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl_abs": 0.0,
                "total_pnl_pct": 0.0,
                "total_fees": 0.0,
                "avg_trade_pct": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "largest_win_pct": 0.0,
                "largest_loss_pct": 0.0,
                "strategies_traded": [],
                "symbols_traded": [],
                "profit_factor": 0.0,
            }

        pnl_pcts = [float(t.get("pnl_pct", 0.0)) for t in trades]
        pnl_abs = [float(t.get("pnl_abs", 0.0)) for t in trades]
        fees = [float(t.get("commission_paid", t.get("fee", 0.0)) or 0.0) for t in trades]

        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p < 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )

        strategies = sorted(
            set(str(t.get("strategy", "unknown")) for t in trades)
        )
        symbols = sorted(set(str(t.get("symbol", "")) for t in trades))

        return {
            "date": label,
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / total, 4),
            "total_pnl_abs": round(sum(pnl_abs), 4),
            "total_pnl_pct": round(sum(pnl_pcts) * 100, 4),
            "total_fees": round(sum(fees), 4),
            "avg_trade_pct": round(float(np.mean(pnl_pcts)) * 100, 4),
            "avg_win_pct": round(float(np.mean(wins)) * 100, 4) if wins else 0.0,
            "avg_loss_pct": round(float(np.mean(losses)) * 100, 4) if losses else 0.0,
            "largest_win_pct": round(max(wins) * 100, 4) if wins else 0.0,
            "largest_loss_pct": round(min(losses) * 100, 4) if losses else 0.0,
            "strategies_traded": strategies,
            "symbols_traded": symbols,
            "profit_factor": (
                round(profit_factor, 4)
                if profit_factor != float("inf")
                else float("inf")
            ),
        }

    # ------------------------------------------------------------------
    # Private: serialisation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_value(value: Any) -> Any:
        """Convert non-JSON-native types to serialisable primitives."""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, float) and math.isinf(value):
            return str(value)
        return value

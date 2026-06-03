"""
BacktestRunner — vectorised / candle-by-candle backtesting engine.

Usage
-----
    from backtest.runner import BacktestRunner, BacktestConfig, BacktestResult

    config = BacktestConfig(
        strategy_name="crypto_scalp",
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="5m",
        start_date="2022-01-01",
        end_date="2024-12-31",
    )
    runner = BacktestRunner(db=db, strategy=strategy, config=config)
    result = await runner.run()
    print(result.passes_criteria())
"""
from __future__ import annotations

import csv
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from backtest.metrics import BacktestMetrics
from config.settings import get_settings
from data.storage.timescale import AsyncTimescaleDB
from strategies.base import Signal, Strategy

log = structlog.get_logger(__name__)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestRunner",
]

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
_RESULTS_DIR = Path(__file__).parent / "results"


def _ensure_results_dir() -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    Parameters
    ----------
    strategy_name:
        Human-readable label embedded in output filenames.
    symbols:
        List of instrument symbols to backtest (e.g. ``["BTCUSDT"]``).
    timeframe:
        Candle interval string (e.g. ``"5m"``, ``"1h"``).
    start_date:
        Inclusive start in ``"YYYY-MM-DD"`` format.
    end_date:
        Inclusive end in ``"YYYY-MM-DD"`` format.
    initial_capital:
        Starting notional capital in account currency.
    commission_pct:
        Commission rate applied on both entry and exit (0.001 = 0.1%).
    slippage_pct:
        Market-impact slippage applied to fill prices (0.001 = 0.1%).
    max_position_size_pct:
        Maximum fraction of capital allocated to any single position.
    """

    strategy_name: str
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float = 100.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    max_position_size_pct: float = 0.25


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Container for all outputs of a completed backtest run.

    Attributes
    ----------
    config:
        The :class:`BacktestConfig` that produced this result.
    trades:
        List of closed trade dicts.  Keys include: ``symbol``, ``side``,
        ``entry_price``, ``exit_price``, ``qty``, ``entry_time``,
        ``exit_time``, ``exit_reason``, ``pnl_pct``, ``pnl_abs``,
        ``commission_paid``.
    equity_curve:
        Portfolio equity at every simulation candle (DatetimeIndex).
    metrics:
        Dict returned by :meth:`BacktestMetrics.summary`.
    html_report_path:
        Absolute path to the saved CSV/text report.
    """

    config: BacktestConfig
    trades: list[dict]
    equity_curve: pd.Series
    metrics: dict[str, Any]
    html_report_path: str

    def passes_criteria(self) -> bool:
        """Return True when all minimum performance criteria are met."""
        return bool(self.metrics.get("passes_minimum_criteria", False))


# ---------------------------------------------------------------------------
# Open position tracking
# ---------------------------------------------------------------------------


@dataclass
class _OpenPosition:
    """State for a single open simulated position."""

    symbol: str
    side: str          # "BUY" | "SELL"
    entry_price: float
    entry_time: datetime
    qty: float
    stop_price: float
    target_price: float
    commission_paid: float  # entry commission already deducted


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------


class BacktestRunner:
    """Candle-by-candle backtesting engine with stop/target hit detection.

    Parameters
    ----------
    db:
        Connected :class:`~data.storage.timescale.AsyncTimescaleDB` instance.
        Pass ``None`` to force CSV fallback for all data loading.
    strategy:
        A concrete :class:`~strategies.base.Strategy` subclass instance.
    config:
        :class:`BacktestConfig` controlling the run parameters.
    """

    def __init__(
        self,
        db: Optional[AsyncTimescaleDB],
        strategy: Strategy,
        config: BacktestConfig,
    ) -> None:
        self._db = db
        self._strategy = strategy
        self._config = config
        self._settings = get_settings()
        self._log = log.bind(
            strategy=config.strategy_name,
            symbols=config.symbols,
            timeframe=config.timeframe,
            start_date=config.start_date,
            end_date=config.end_date,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> BacktestResult:
        """Execute the full backtest and return a :class:`BacktestResult`.

        Steps
        -----
        1. Load historical OHLCV data from TimescaleDB (or CSV fallback).
        2. Walk candle-by-candle chronologically.
        3. Check open positions for stop/target hits on each candle.
        4. Call ``strategy.on_candle()`` and act on actionable signals.
        5. Build equity curve and compute final metrics.
        6. Save a CSV report to ``backtest/results/``.
        """
        self._log.info("backtest.run_start")

        # ── 1. Load data ─────────────────────────────────────────────────────
        symbol_data: dict[str, pd.DataFrame] = {}
        for symbol in self._config.symbols:
            df = await self._load_symbol_data(symbol)
            if df.empty:
                self._log.warning("backtest.no_data_for_symbol", symbol=symbol)
                continue
            symbol_data[symbol] = df

        if not symbol_data:
            self._log.error("backtest.no_data_loaded")
            return self._empty_result()

        # ── 2. Build unified timeline ─────────────────────────────────────────
        all_timestamps: pd.DatetimeIndex = pd.DatetimeIndex(
            sorted(
                set().union(*[df.index for df in symbol_data.values()])  # type: ignore[arg-type]
            )
        )

        # ── 3. Simulation state ───────────────────────────────────────────────
        capital = self._config.initial_capital
        equity_points: list[tuple[pd.Timestamp, float]] = []
        closed_trades: list[dict] = []
        open_positions: dict[str, _OpenPosition] = {}

        # ── 4. Main loop ──────────────────────────────────────────────────────
        for ts in all_timestamps:
            for symbol in list(open_positions.keys()):
                if symbol not in symbol_data:
                    continue
                df_sym = symbol_data[symbol]
                if ts not in df_sym.index:
                    continue
                candle = df_sym.loc[ts]
                capital, trade = self._check_exits(
                    open_positions, symbol, candle, ts, capital
                )
                if trade is not None:
                    closed_trades.append(trade)

            # Strategy evaluation — per symbol
            for symbol, df_sym in symbol_data.items():
                if ts not in df_sym.index:
                    continue
                if symbol in open_positions:
                    # Only one position per symbol at a time
                    continue

                # Slice history up to and including current candle
                df_up_to_now = df_sym.loc[:ts].copy()
                if len(df_up_to_now) < 2:
                    continue

                try:
                    signal: Optional[Signal] = await self._strategy.on_candle(
                        symbol, self._config.timeframe, df_up_to_now
                    )
                except Exception as exc:
                    self._log.warning(
                        "backtest.strategy_error",
                        symbol=symbol,
                        ts=str(ts),
                        error=str(exc),
                    )
                    continue

                if signal is None or not signal.is_actionable:
                    continue

                # ── Position sizing ──────────────────────────────────────────
                max_alloc = capital * self._config.max_position_size_pct
                if max_alloc <= 0:
                    continue

                candle = df_sym.loc[ts]
                entry_fill = self._apply_slippage(
                    signal.entry_price or float(candle["close"]),
                    signal.side,
                )
                qty = max_alloc / entry_fill
                commission_entry = max_alloc * self._config.commission_pct
                capital -= commission_entry  # deduct entry commission immediately

                open_positions[symbol] = _OpenPosition(
                    symbol=symbol,
                    side=signal.side,
                    entry_price=entry_fill,
                    entry_time=ts.to_pydatetime().replace(tzinfo=timezone.utc),
                    qty=qty,
                    stop_price=signal.stop_price,
                    target_price=signal.target_price,
                    commission_paid=commission_entry,
                )
                self._log.debug(
                    "backtest.position_opened",
                    symbol=symbol,
                    side=signal.side,
                    entry_price=round(entry_fill, 6),
                    qty=round(qty, 8),
                    stop=signal.stop_price,
                    target=signal.target_price,
                )

            # Mark-to-market equity snapshot
            mtm = self._mark_to_market(capital, open_positions, symbol_data, ts)
            equity_points.append((ts, mtm))

        # ── 5. Force-close any remaining open positions at last price ─────────
        for symbol, pos in list(open_positions.items()):
            df_sym = symbol_data.get(symbol)
            if df_sym is not None and not df_sym.empty:
                last_candle = df_sym.iloc[-1]
                last_ts = df_sym.index[-1]
                capital, trade = self._force_close(pos, last_candle, last_ts, capital)
                if trade is not None:
                    closed_trades.append(trade)
        open_positions.clear()

        # ── 6. Build equity curve ─────────────────────────────────────────────
        if equity_points:
            idx, vals = zip(*equity_points)
            equity_curve = pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")
        else:
            equity_curve = pd.Series(dtype=float, name="equity")

        # ── 7. Compute metrics ────────────────────────────────────────────────
        metrics = BacktestMetrics.summary(closed_trades, equity_curve)

        # ── 8. Save report ────────────────────────────────────────────────────
        report_path = self._save_report(closed_trades, metrics, equity_curve)

        self._log.info(
            "backtest.run_complete",
            total_trades=len(closed_trades),
            passes=metrics.get("passes_minimum_criteria"),
            report=report_path,
        )

        return BacktestResult(
            config=self._config,
            trades=closed_trades,
            equity_curve=equity_curve,
            metrics=metrics,
            html_report_path=report_path,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data from TimescaleDB, falling back to an empty frame."""
        start_dt = pd.to_datetime(self._config.start_date, utc=True)
        end_dt = pd.to_datetime(self._config.end_date, utc=True)

        if self._db is not None:
            try:
                # Load in chunks to avoid memory issues for long date ranges
                df = await self._db.get_candles(
                    symbol=symbol,
                    timeframe=self._config.timeframe,
                    limit=1_000_000,
                    end_ts=end_dt,
                )
                if not df.empty:
                    if not df.index.tz:
                        df.index = df.index.tz_localize("UTC")
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    self._log.info(
                        "backtest.data_loaded_db",
                        symbol=symbol,
                        rows=len(df),
                    )
                    return df
            except Exception as exc:
                self._log.warning(
                    "backtest.db_load_failed",
                    symbol=symbol,
                    error=str(exc),
                )

        self._log.warning("backtest.no_data_available", symbol=symbol)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    @staticmethod
    async def load_from_csv(
        symbol: str,
        timeframe: str,
        path: str,
    ) -> pd.DataFrame:
        """Load OHLCV data from a CSV file (DB fallback).

        The CSV must have columns: timestamp,open,high,low,close,volume
        where *timestamp* is parseable by :func:`pandas.to_datetime`.

        Parameters
        ----------
        symbol:
            Instrument symbol (used only for logging).
        timeframe:
            Candle timeframe string (used only for logging).
        path:
            Absolute or relative path to the CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex (UTC) and float OHLCV columns.
            Returns an empty DataFrame when the file cannot be read.
        """
        _log = log.bind(symbol=symbol, timeframe=timeframe, path=path)
        try:
            df = pd.read_csv(
                path,
                parse_dates=["timestamp"],
                index_col="timestamp",
            )
            if not df.index.tz:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            required = {"open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                _log.error("backtest.csv_missing_columns", missing=sorted(missing))
                return pd.DataFrame(columns=list(required))

            for col in required:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_index()
            _log.info("backtest.csv_loaded", rows=len(df))
            return df[list(required)]
        except FileNotFoundError:
            _log.error("backtest.csv_not_found")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        except Exception as exc:
            _log.error("backtest.csv_load_error", error=str(exc))
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply market-impact slippage: buys fill higher, sells lower."""
        slip = self._config.slippage_pct
        if side == "BUY":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        open_positions: dict[str, _OpenPosition],
        symbol: str,
        candle: pd.Series,
        ts: pd.Timestamp,
        capital: float,
    ) -> tuple[float, Optional[dict]]:
        """Check whether a stop or target was hit on this candle.

        Uses intrabar high/low to detect stop or target triggers.  If both
        are hit within the same candle the stop is assumed to have triggered
        first (conservative assumption).

        Returns
        -------
        tuple[float, dict | None]
            Updated capital and closed trade dict (or None if no exit).
        """
        pos = open_positions.get(symbol)
        if pos is None:
            return capital, None

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        exit_price: Optional[float] = None
        exit_reason: str = ""

        if pos.side == "BUY":
            # Stop hit when low goes at or below stop
            if low <= pos.stop_price:
                exit_price = self._apply_slippage(pos.stop_price, "SELL")
                exit_reason = "stop_loss"
            # Target hit when high reaches or exceeds target
            elif high >= pos.target_price:
                exit_price = self._apply_slippage(pos.target_price, "SELL")
                exit_reason = "take_profit"
        else:  # SHORT
            if high >= pos.stop_price:
                exit_price = self._apply_slippage(pos.stop_price, "BUY")
                exit_reason = "stop_loss"
            elif low <= pos.target_price:
                exit_price = self._apply_slippage(pos.target_price, "BUY")
                exit_reason = "take_profit"

        if exit_price is None:
            return capital, None

        return self._close_position(
            open_positions, pos, exit_price, exit_reason, ts, capital
        )

    def _force_close(
        self,
        pos: _OpenPosition,
        candle: pd.Series,
        ts: pd.Timestamp,
        capital: float,
    ) -> tuple[float, Optional[dict]]:
        """Close a position at the last available close price (end of backtest)."""
        close = float(candle["close"])
        fill = self._apply_slippage(close, "SELL" if pos.side == "BUY" else "BUY")
        # We pass a throw-away dict as open_positions so _close_position can pop it
        dummy: dict[str, _OpenPosition] = {pos.symbol: pos}
        return self._close_position(dummy, pos, fill, "end_of_data", ts, capital)

    def _close_position(
        self,
        open_positions: dict[str, _OpenPosition],
        pos: _OpenPosition,
        exit_price: float,
        exit_reason: str,
        ts: pd.Timestamp,
        capital: float,
    ) -> tuple[float, dict]:
        """Simulate position exit: apply exit commission, compute PnL."""
        gross_proceeds = exit_price * pos.qty
        commission_exit = gross_proceeds * self._config.commission_pct
        net_proceeds = gross_proceeds - commission_exit

        if pos.side == "BUY":
            pnl_abs = net_proceeds - (pos.entry_price * pos.qty) - pos.commission_paid
        else:
            # Short: profit when price falls
            pnl_abs = (pos.entry_price * pos.qty) - net_proceeds - pos.commission_paid

        entry_notional = pos.entry_price * pos.qty
        pnl_pct = pnl_abs / entry_notional if entry_notional != 0 else 0.0

        capital += net_proceeds  # cash returned from the position

        exit_dt = ts.to_pydatetime().replace(tzinfo=timezone.utc)
        duration_min = (exit_dt - pos.entry_time).total_seconds() / 60.0

        trade: dict[str, Any] = {
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": round(pos.entry_price, 8),
            "exit_price": round(exit_price, 8),
            "qty": round(pos.qty, 8),
            "entry_time": pos.entry_time.isoformat(),
            "exit_time": exit_dt.isoformat(),
            "exit_reason": exit_reason,
            "pnl_pct": round(pnl_pct, 8),
            "pnl_abs": round(pnl_abs, 8),
            "commission_paid": round(pos.commission_paid + commission_exit, 8),
            "duration_minutes": round(duration_min, 2),
        }

        open_positions.pop(pos.symbol, None)

        self._log.debug(
            "backtest.position_closed",
            symbol=pos.symbol,
            exit_reason=exit_reason,
            pnl_pct=round(pnl_pct * 100, 4),
        )
        return capital, trade

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def _mark_to_market(
        self,
        cash: float,
        open_positions: dict[str, _OpenPosition],
        symbol_data: dict[str, pd.DataFrame],
        ts: pd.Timestamp,
    ) -> float:
        """Compute current portfolio equity: cash + unrealised P&L."""
        unrealised = 0.0
        for symbol, pos in open_positions.items():
            df = symbol_data.get(symbol)
            if df is None:
                continue
            try:
                price = float(df.loc[ts, "close"])
            except (KeyError, TypeError):
                continue
            if pos.side == "BUY":
                unrealised += (price - pos.entry_price) * pos.qty
            else:
                unrealised += (pos.entry_price - price) * pos.qty
        return cash + unrealised

    # ------------------------------------------------------------------
    # Report saving
    # ------------------------------------------------------------------

    def _save_report(
        self,
        trades: list[dict],
        metrics: dict[str, Any],
        equity_curve: pd.Series,
    ) -> str:
        """Save a CSV trade log and a text metrics summary to the results dir."""
        results_dir = _ensure_results_dir()
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = self._config.strategy_name.replace(" ", "_")

        # ── Trade CSV ────────────────────────────────────────────────────────
        trades_path = results_dir / f"{slug}_{run_id}_trades.csv"
        if trades:
            fieldnames = list(trades[0].keys())
            with open(trades_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trades)
        else:
            trades_path.write_text("symbol,side,entry_price,exit_price,qty,entry_time,exit_time,exit_reason,pnl_pct,pnl_abs,commission_paid,duration_minutes\n")

        # ── Metrics text report ───────────────────────────────────────────────
        report_path = results_dir / f"{slug}_{run_id}_report.txt"
        lines: list[str] = [
            "=" * 60,
            f"BACKTEST REPORT — {self._config.strategy_name.upper()}",
            "=" * 60,
            f"Symbols    : {', '.join(self._config.symbols)}",
            f"Timeframe  : {self._config.timeframe}",
            f"Period     : {self._config.start_date} → {self._config.end_date}",
            f"Capital    : {self._config.initial_capital}",
            f"Commission : {self._config.commission_pct*100:.2f}% per side",
            f"Slippage   : {self._config.slippage_pct*100:.2f}% per side",
            "",
            "─" * 60,
            "PERFORMANCE METRICS",
            "─" * 60,
        ]
        metric_rows = [
            ("Total Trades",         str(metrics.get("total_trades", 0))),
            ("Total Return %",       f"{metrics.get('total_return_pct', 0.0):+.4f}%"),
            ("Sharpe Ratio",         f"{metrics.get('sharpe', 0.0):.4f}"),
            ("Sortino Ratio",        f"{metrics.get('sortino', 0.0):.4f}"),
            ("Calmar Ratio",         f"{metrics.get('calmar', 0.0):.4f}"),
            ("Max Drawdown",         f"{metrics.get('max_drawdown', 0.0)*100:.4f}%"),
            ("Win Rate",             f"{metrics.get('win_rate', 0.0)*100:.2f}%"),
            ("Profit Factor",        f"{metrics.get('profit_factor', 0.0):.4f}"),
            ("Avg Trade %",          f"{metrics.get('avg_trade_pct', 0.0):+.4f}%"),
            ("Avg Win %",            f"{metrics.get('avg_win_pct', 0.0):+.4f}%"),
            ("Avg Loss %",           f"{metrics.get('avg_loss_pct', 0.0):+.4f}%"),
            ("Largest Win %",        f"{metrics.get('largest_win_pct', 0.0):+.4f}%"),
            ("Largest Loss %",       f"{metrics.get('largest_loss_pct', 0.0):+.4f}%"),
            ("Avg Duration (min)",   f"{metrics.get('avg_trade_duration_minutes', 0.0):.1f}"),
            ("Passes Criteria",      "YES" if metrics.get("passes_minimum_criteria") else "NO"),
        ]
        for label, value in metric_rows:
            lines.append(f"  {label:<28} {value}")

        lines += ["", "─" * 60, f"Trades CSV : {trades_path}", "=" * 60]
        report_path.write_text("\n".join(lines) + "\n")

        self._log.info(
            "backtest.report_saved",
            report=str(report_path),
            trades_csv=str(trades_path),
        )
        return str(report_path)

    # ------------------------------------------------------------------
    # Empty result helper
    # ------------------------------------------------------------------

    def _empty_result(self) -> BacktestResult:
        empty_equity = pd.Series(dtype=float, name="equity")
        metrics = BacktestMetrics.summary([], empty_equity)
        results_dir = _ensure_results_dir()
        stub = results_dir / f"{self._config.strategy_name}_empty.txt"
        stub.write_text("No data available for this backtest run.\n")
        return BacktestResult(
            config=self._config,
            trades=[],
            equity_curve=empty_equity,
            metrics=metrics,
            html_report_path=str(stub),
        )

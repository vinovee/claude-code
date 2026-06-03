"""
BacktestMetrics — static methods for computing trading performance metrics.

Usage
-----
    from backtest.metrics import BacktestMetrics

    metrics = BacktestMetrics.summary(trades, equity_curve)
    BacktestMetrics.print_summary(metrics)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import structlog
from rich.console import Console
from rich.table import Table
from rich.text import Text

from config.settings import get_settings

log = structlog.get_logger(__name__)

__all__ = ["BacktestMetrics"]

# ---------------------------------------------------------------------------
# Minimum passing criteria (used by passes_minimum_criteria)
# ---------------------------------------------------------------------------
_MIN_SHARPE: float = 1.5
_MAX_DRAWDOWN: float = 0.30   # 30%
_MIN_WIN_RATE: float = 0.45
_MIN_PROFIT_FACTOR: float = 1.5
_MIN_TOTAL_TRADES: int = 200


class BacktestMetrics:
    """Static utility methods for backtest performance analysis.

    All methods are pure functions — no instance state required.
    """

    # ------------------------------------------------------------------
    # Core ratio methods
    # ------------------------------------------------------------------

    @staticmethod
    def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualised Sharpe ratio (risk-free rate assumed zero).

        Parameters
        ----------
        returns:
            Per-period return series (e.g. daily or per-candle returns as
            fractions, not percentages).
        periods_per_year:
            Number of periods in a year for annualisation (default 252 for
            daily returns).

        Returns
        -------
        float
            Annualised Sharpe, or 0.0 when the series is too short or has
            zero standard deviation.
        """
        if returns is None or len(returns) < 2:
            return 0.0

        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0

        mean_r = float(clean.mean())
        std_r = float(clean.std(ddof=1))

        if std_r == 0.0 or math.isnan(std_r):
            return 0.0

        return (mean_r / std_r) * math.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """Maximum peak-to-trough drawdown as a fraction.

        Parameters
        ----------
        equity_curve:
            Equity values over time (e.g. portfolio value at each candle).

        Returns
        -------
        float
            Maximum drawdown fraction, e.g. ``0.25`` = 25%.  Returns 0.0
            for empty or single-element series.
        """
        if equity_curve is None or len(equity_curve) < 2:
            return 0.0

        clean = equity_curve.dropna()
        if len(clean) < 2:
            return 0.0

        running_max = clean.cummax()
        drawdowns = (running_max - clean) / running_max.replace(0, np.nan)
        result = float(drawdowns.max())
        return result if not math.isnan(result) else 0.0

    @staticmethod
    def profit_factor(trades: list[dict]) -> float:
        """Gross profit divided by absolute gross loss across all trades.

        Parameters
        ----------
        trades:
            List of trade dicts, each containing a ``pnl_pct`` key (signed
            fraction, positive = profit).

        Returns
        -------
        float
            Profit factor.  Returns ``float('inf')`` when there are no
            losing trades and ``0.0`` when there are no trades.
        """
        if not trades:
            return 0.0

        gross_profit = sum(t.get("pnl_pct", 0.0) for t in trades if t.get("pnl_pct", 0.0) > 0)
        gross_loss = sum(abs(t.get("pnl_pct", 0.0)) for t in trades if t.get("pnl_pct", 0.0) < 0)

        if gross_loss == 0.0:
            return float("inf") if gross_profit > 0.0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def win_rate(trades: list[dict]) -> float:
        """Fraction of trades with a positive pnl_pct.

        Parameters
        ----------
        trades:
            List of trade dicts, each with a ``pnl_pct`` key.

        Returns
        -------
        float
            Win rate in ``[0.0, 1.0]``.  Returns 0.0 for an empty list.
        """
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("pnl_pct", 0.0) > 0)
        return wins / len(trades)

    @staticmethod
    def calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """Calmar ratio: annualised return divided by maximum drawdown.

        Parameters
        ----------
        annual_return:
            Annualised return as a fraction, e.g. ``0.30`` = 30% p.a.
        max_drawdown:
            Maximum drawdown as a positive fraction, e.g. ``0.15`` = 15%.

        Returns
        -------
        float
            Calmar ratio, or 0.0 when max_drawdown is zero.
        """
        if max_drawdown <= 0.0:
            return 0.0
        return annual_return / max_drawdown

    @staticmethod
    def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualised Sortino ratio (penalises downside deviation only).

        Parameters
        ----------
        returns:
            Per-period return series as fractions.
        periods_per_year:
            Number of periods in a year for annualisation (default 252).

        Returns
        -------
        float
            Annualised Sortino ratio, or 0.0 when insufficient data.
        """
        if returns is None or len(returns) < 2:
            return 0.0

        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0

        mean_r = float(clean.mean())
        downside = clean[clean < 0.0]

        if len(downside) == 0:
            # No losing periods — effectively infinite Sortino; cap at 10
            return 10.0

        downside_std = float(downside.std(ddof=1))
        if downside_std == 0.0 or math.isnan(downside_std):
            return 0.0

        return (mean_r / downside_std) * math.sqrt(periods_per_year)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def summary(trades: list[dict], equity_curve: pd.Series) -> dict[str, Any]:
        """Compute a comprehensive performance summary.

        Parameters
        ----------
        trades:
            List of closed trade dicts.  Expected keys per trade:

            * ``pnl_pct``        — fractional P&L (signed)
            * ``entry_time``     — entry datetime (optional, for duration)
            * ``exit_time``      — exit datetime (optional, for duration)

        equity_curve:
            Portfolio equity value at each time step (not per-trade — use
            the full candle-by-candle series).

        Returns
        -------
        dict
            Metric dict with keys: sharpe, sortino, calmar, max_drawdown,
            profit_factor, win_rate, total_trades, total_return_pct,
            avg_trade_pct, avg_win_pct, avg_loss_pct, largest_win_pct,
            largest_loss_pct, avg_trade_duration_minutes,
            passes_minimum_criteria.
        """
        log.debug("backtest_metrics.summary_start", n_trades=len(trades))

        total_trades = len(trades)

        # ── Return series from equity curve ─────────────────────────────────
        if equity_curve is not None and len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
        else:
            returns = pd.Series(dtype=float)

        # ── Core metrics ────────────────────────────────────────────────────
        sharpe = BacktestMetrics.sharpe_ratio(returns)
        sortino = BacktestMetrics.sortino_ratio(returns)
        mdd = BacktestMetrics.max_drawdown(equity_curve)
        pf = BacktestMetrics.profit_factor(trades)
        wr = BacktestMetrics.win_rate(trades)

        # ── Return-based metrics ─────────────────────────────────────────────
        pnls = [t.get("pnl_pct", 0.0) for t in trades]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]

        total_return_pct: float
        if equity_curve is not None and len(equity_curve) > 0:
            first = float(equity_curve.iloc[0])
            last = float(equity_curve.iloc[-1])
            total_return_pct = ((last - first) / first) if first != 0 else 0.0
        else:
            total_return_pct = sum(pnls)

        avg_trade_pct = float(np.mean(pnls)) if pnls else 0.0
        avg_win_pct = float(np.mean(winning_pnls)) if winning_pnls else 0.0
        avg_loss_pct = float(np.mean(losing_pnls)) if losing_pnls else 0.0
        largest_win_pct = float(max(winning_pnls)) if winning_pnls else 0.0
        largest_loss_pct = float(min(losing_pnls)) if losing_pnls else 0.0

        # ── Calmar (approximate annual return from total return) ─────────────
        # Estimate annualised return from equity curve length or trade count
        n_years: float = 1.0
        if equity_curve is not None and len(equity_curve) > 1:
            # Assume 252 * 24 * 12 = ~72,576 five-minute candles per year; use
            # actual length relative to that as proxy.  For robustness just use
            # the simplest approximation.
            n_years = max(len(equity_curve) / 252, 1 / 252)
        annual_return = (1 + total_return_pct) ** (1 / n_years) - 1 if n_years > 0 else total_return_pct
        calmar = BacktestMetrics.calmar_ratio(annual_return, mdd)

        # ── Trade duration ───────────────────────────────────────────────────
        durations_minutes: list[float] = []
        for t in trades:
            try:
                entry = pd.to_datetime(t.get("entry_time"))
                exit_ = pd.to_datetime(t.get("exit_time"))
                if entry is not None and exit_ is not None and pd.notna(entry) and pd.notna(exit_):
                    delta = (exit_ - entry).total_seconds() / 60.0
                    if delta >= 0:
                        durations_minutes.append(delta)
            except Exception:
                pass

        avg_trade_duration_minutes = float(np.mean(durations_minutes)) if durations_minutes else 0.0

        # ── Minimum criteria ─────────────────────────────────────────────────
        passes = (
            sharpe > _MIN_SHARPE
            and mdd < _MAX_DRAWDOWN
            and wr > _MIN_WIN_RATE
            and pf > _MIN_PROFIT_FACTOR
            and total_trades > _MIN_TOTAL_TRADES
        )

        metrics: dict[str, Any] = {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(mdd, 6),
            "profit_factor": round(pf, 4) if pf != float("inf") else float("inf"),
            "win_rate": round(wr, 4),
            "total_trades": total_trades,
            "total_return_pct": round(total_return_pct * 100, 4),
            "avg_trade_pct": round(avg_trade_pct * 100, 6),
            "avg_win_pct": round(avg_win_pct * 100, 6),
            "avg_loss_pct": round(avg_loss_pct * 100, 6),
            "largest_win_pct": round(largest_win_pct * 100, 6),
            "largest_loss_pct": round(largest_loss_pct * 100, 6),
            "avg_trade_duration_minutes": round(avg_trade_duration_minutes, 2),
            "passes_minimum_criteria": passes,
        }

        log.info(
            "backtest_metrics.summary_complete",
            total_trades=total_trades,
            sharpe=metrics["sharpe"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            passes=passes,
        )
        return metrics

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(metrics: dict[str, Any]) -> None:
        """Pretty-print a metrics dict using a Rich table.

        Parameters
        ----------
        metrics:
            Dict returned by :meth:`summary`.
        """
        console = Console()

        passes = metrics.get("passes_minimum_criteria", False)
        status_text = Text("PASS", style="bold green") if passes else Text("FAIL", style="bold red")

        table = Table(
            title=f"Backtest Performance Summary  [{status_text}]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            min_width=52,
        )
        table.add_column("Metric", style="bold", min_width=30)
        table.add_column("Value", justify="right", min_width=18)

        def _fmt_pct(v: float) -> str:
            return f"{v:+.2f}%"

        def _fmt_float(v: float, decimals: int = 4) -> str:
            if v == float("inf"):
                return "∞"
            return f"{v:.{decimals}f}"

        rows: list[tuple[str, str]] = [
            ("Total Trades", str(metrics.get("total_trades", 0))),
            ("Total Return", _fmt_pct(metrics.get("total_return_pct", 0.0))),
            ("Sharpe Ratio  (target > 1.50)", _fmt_float(metrics.get("sharpe", 0.0))),
            ("Sortino Ratio", _fmt_float(metrics.get("sortino", 0.0))),
            ("Calmar Ratio", _fmt_float(metrics.get("calmar", 0.0))),
            ("Max Drawdown  (limit < 30.0%)", _fmt_pct(metrics.get("max_drawdown", 0.0) * 100)),
            ("Win Rate      (target > 45.0%)", _fmt_pct(metrics.get("win_rate", 0.0) * 100)),
            ("Profit Factor (target > 1.50)", _fmt_float(metrics.get("profit_factor", 0.0))),
            ("Avg Trade", _fmt_pct(metrics.get("avg_trade_pct", 0.0))),
            ("Avg Win", _fmt_pct(metrics.get("avg_win_pct", 0.0))),
            ("Avg Loss", _fmt_pct(metrics.get("avg_loss_pct", 0.0))),
            ("Largest Win", _fmt_pct(metrics.get("largest_win_pct", 0.0))),
            ("Largest Loss", _fmt_pct(metrics.get("largest_loss_pct", 0.0))),
            ("Avg Trade Duration (min)", f"{metrics.get('avg_trade_duration_minutes', 0.0):.1f}"),
        ]

        for label, value in rows:
            table.add_row(label, value)

        console.print()
        console.print(table)

        if passes:
            console.print(
                "[bold green]Strategy PASSES all minimum criteria.[/bold green]"
            )
        else:
            # Identify which criteria failed
            failures: list[str] = []
            if metrics.get("sharpe", 0.0) <= _MIN_SHARPE:
                failures.append(f"Sharpe {metrics['sharpe']:.2f} <= {_MIN_SHARPE}")
            if metrics.get("max_drawdown", 1.0) >= _MAX_DRAWDOWN:
                failures.append(f"Max drawdown {metrics['max_drawdown']:.2%} >= {_MAX_DRAWDOWN:.0%}")
            if metrics.get("win_rate", 0.0) <= _MIN_WIN_RATE:
                failures.append(f"Win rate {metrics['win_rate']:.2%} <= {_MIN_WIN_RATE:.0%}")
            if metrics.get("profit_factor", 0.0) <= _MIN_PROFIT_FACTOR:
                failures.append(f"Profit factor {metrics.get('profit_factor', 0.0):.2f} <= {_MIN_PROFIT_FACTOR}")
            if metrics.get("total_trades", 0) <= _MIN_TOTAL_TRADES:
                failures.append(f"Total trades {metrics['total_trades']} <= {_MIN_TOTAL_TRADES}")
            console.print(
                f"[bold red]Strategy FAILS minimum criteria: {'; '.join(failures)}[/bold red]"
            )
        console.print()

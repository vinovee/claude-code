"""
Abstract base class and shared data-transfer objects for all trading strategies.

Usage
-----
    from strategies.base import Strategy, Signal, StrategyStats

Every concrete strategy must subclass :class:`Strategy` and implement the
three abstract properties and the :meth:`on_candle` coroutine.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

__all__ = [
    "Signal",
    "StrategyStats",
    "Strategy",
]


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """Represents a trading signal emitted by a strategy.

    Attributes
    ----------
    strategy:
        Name of the strategy that produced this signal.
    symbol:
        Instrument symbol (e.g. ``"BTCUSDT"``).
    side:
        Direction: ``"BUY"``, ``"SELL"``, or ``"NONE"``.
    strength:
        Signal conviction, normalised to ``[-1.0, 1.0]``.
        Positive = bullish, negative = bearish, 0 = neutral.
    entry_price:
        Suggested entry price.
    stop_price:
        Suggested stop-loss price.
    target_price:
        Suggested take-profit price.
    rr_ratio:
        Risk-reward ratio: ``abs(target - entry) / abs(entry - stop)``.
    confidence:
        Model/rule confidence in ``[0.0, 1.0]``.
    indicators:
        Snapshot of the indicator values that triggered the signal.
    timestamp:
        UTC time when the signal was generated.
    timeframe:
        Candle timeframe used to generate the signal (e.g. ``"5m"``).
    """

    strategy: str
    symbol: str
    side: str           # BUY | SELL | NONE
    strength: float     # -1.0 to 1.0
    entry_price: float
    stop_price: float
    target_price: float
    rr_ratio: float
    confidence: float   # 0.0 to 1.0
    indicators: dict    # snapshot of indicator values that triggered signal
    timestamp: datetime
    timeframe: str

    @property
    def is_actionable(self) -> bool:
        """Return True when this signal meets minimum execution criteria.

        All three conditions must hold:
        * Side is not ``"NONE"`` — there must be a directional view.
        * Strength exceeds 0.5 — moderate conviction at minimum.
        * Risk-reward ratio >= 2.0 — expected return is at least 2× the risk.
        """
        return (
            self.side != "NONE"
            and self.strength > 0.5
            and self.rr_ratio >= 2.0
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Signal(strategy={self.strategy!r}, symbol={self.symbol!r}, "
            f"side={self.side!r}, strength={self.strength:.3f}, "
            f"rr={self.rr_ratio:.2f}, actionable={self.is_actionable})"
        )


# ---------------------------------------------------------------------------
# StrategyStats
# ---------------------------------------------------------------------------


@dataclass
class StrategyStats:
    """Running performance statistics for a single strategy instance.

    Call :meth:`update_stats` after each closed trade with the realised
    ``pnl_pct`` (positive = profit, negative = loss, expressed as a
    fraction, e.g. ``0.05`` for +5 %).

    Properties
    ----------
    win_rate:
        Fraction of winning trades (0.0–1.0).
    profit_factor:
        Sum of gross profits divided by sum of gross losses.
        Returns ``inf`` when there are no losing trades.
    sharpe_ratio:
        Annualised Sharpe ratio computed from the trade-level P&L series
        (assumes 252 trading days, ~6 trades per day as a baseline —
        adjust ``trades_per_year`` if needed).
    """

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    # Internal running sums (not exposed publicly but stored for accuracy).
    _sum_wins: float = field(default=0.0, repr=False, compare=False)
    _sum_losses: float = field(default=0.0, repr=False, compare=False)
    _trade_returns: list[float] = field(
        default_factory=list, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Mutation                                                            #
    # ------------------------------------------------------------------ #

    def update_stats(self, pnl_pct: float) -> None:
        """Record a completed trade.

        Parameters
        ----------
        pnl_pct:
            Realised P&L as a fraction, e.g. ``0.03`` for +3 %.
            Negative values represent losses.
        """
        self.total_trades += 1
        self.total_pnl += pnl_pct
        self._trade_returns.append(pnl_pct)

        if pnl_pct >= 0.0:
            self.wins += 1
            self._sum_wins += pnl_pct
            self.avg_win_pct = self._sum_wins / self.wins
        else:
            self.losses += 1
            self._sum_losses += abs(pnl_pct)
            self.avg_loss_pct = self._sum_losses / self.losses

        log.debug(
            "strategy_stats_updated",
            total_trades=self.total_trades,
            wins=self.wins,
            losses=self.losses,
            total_pnl=round(self.total_pnl, 6),
        )

    # ------------------------------------------------------------------ #
    # Read-only computed properties                                       #
    # ------------------------------------------------------------------ #

    @property
    def win_rate(self) -> float:
        """Fraction of trades that were profitable (0.0–1.0)."""
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def profit_factor(self) -> float:
        """Gross profits divided by gross losses.

        Returns ``float('inf')`` when gross losses are zero (all wins).
        Returns ``0.0`` when there are no trades at all.
        """
        if self.total_trades == 0:
            return 0.0
        if self._sum_losses == 0.0:
            return float("inf")
        return self._sum_wins / self._sum_losses

    @property
    def sharpe_ratio(self, trades_per_year: int = 1512) -> float:
        """Annualised Sharpe ratio from the trade-level return series.

        Uses a risk-free rate of 0 (appropriate for short-term trading).

        Parameters
        ----------
        trades_per_year:
            Assumed number of trades per year for annualisation.
            Default is 1512 (252 trading days × ~6 trades/day).

        Returns
        -------
        float
            Annualised Sharpe ratio, or ``0.0`` when insufficient data.
        """
        n = len(self._trade_returns)
        if n < 2:
            return 0.0

        returns = self._trade_returns
        mean_r = sum(returns) / n
        variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_r = math.sqrt(variance)

        if std_r == 0.0:
            return 0.0

        # Annualise: multiply by sqrt(trades_per_year)
        return (mean_r / std_r) * math.sqrt(trades_per_year)


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Sub-classes must implement:

    * :attr:`name` — unique identifier for the strategy.
    * :attr:`timeframes` — candle timeframes the strategy subscribes to.
    * :attr:`symbols` — instruments the strategy trades.
    * :meth:`on_candle` — core signal-generation logic.

    A :class:`StrategyStats` instance is automatically created at
    construction time and is accessible as ``self.stats``.
    """

    def __init__(self) -> None:
        self.stats: StrategyStats = StrategyStats()
        self._log = structlog.get_logger(self.__class__.__module__).bind(
            strategy=self.name
        )

    # ------------------------------------------------------------------ #
    # Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, human-readable strategy identifier (e.g. ``"crypto_scalp"``)."""

    @property
    @abstractmethod
    def timeframes(self) -> list[str]:
        """Ordered list of candle timeframes this strategy subscribes to.

        Example: ``["5m", "15m"]``
        """

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """List of instrument symbols this strategy trades.

        Example: ``["BTCUSDT", "ETHUSDT"]``
        """

    @abstractmethod
    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Optional[Signal]:
        """Process a new closed candle and optionally emit a signal.

        Parameters
        ----------
        symbol:
            The instrument that produced the new candle.
        timeframe:
            The candle timeframe (e.g. ``"5m"``).
        df:
            Full OHLCV history for *symbol* on *timeframe*, with the
            newest candle appended.  Columns: ``open``, ``high``,
            ``low``, ``close``, ``volume``.  Index: ``datetime`` (UTC).

        Returns
        -------
        Signal | None
            A populated :class:`Signal` when an entry condition fires,
            otherwise ``None``.
        """

    # ------------------------------------------------------------------ #
    # Helpers available to sub-classes                                    #
    # ------------------------------------------------------------------ #

    def _rr_ratio(
        self,
        entry: float,
        stop: float,
        target: float,
    ) -> float:
        """Compute risk-reward ratio from entry, stop, and target prices.

        Returns 0.0 when risk is zero to avoid division by zero.
        """
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk == 0.0:
            return 0.0
        return round(reward / risk, 4)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"symbols={self.symbols!r}, "
            f"timeframes={self.timeframes!r})"
        )

"""
Dual Momentum Strategy — Phase 2.

Implements Gary Antonacci's dual-momentum framework adapted for crypto:

* **Absolute momentum**: only hold an asset when its own trailing 90-day
  return is positive (i.e. it has beaten "cash").
* **Relative momentum**: among assets that pass the absolute filter, rank by
  the same 90-day return and only buy the top quartile (percentile >= 0.75).

The strategy is inherently *daily*, suited to position trading rather than
intraday speculation.

Usage
-----
    from strategies.phase2.dual_momentum import DualMomentumStrategy

    strategy = DualMomentumStrategy()
    signal = await strategy.on_candle("BTCUSDT", "1d", df)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from config.settings import get_settings
from data.indicators.technical import TechnicalIndicators, MarketRegimeDetector
from strategies.base import Signal, Strategy, StrategyStats

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CANDLES: int = 90          # days of history required
_MOMENTUM_LOOKBACK: int = 90    # trailing return window (trading days)
_RELATIVE_RANK_THRESHOLD: float = 0.75   # top quartile
_STOP_PCT: float = 0.08         # 8% trailing stop for daily timeframe
_TARGET_PCT: float = 0.20       # 20% target


# ---------------------------------------------------------------------------
# DualMomentumStrategy
# ---------------------------------------------------------------------------


class DualMomentumStrategy(Strategy):
    """Daily dual-momentum strategy for a configurable crypto universe.

    Signal logic
    ~~~~~~~~~~~~
    For each incoming daily candle on any symbol in the universe:

    1. Require >= 90 days of history.
    2. **Absolute momentum gate**: 90-day return must be positive, otherwise
       skip (stay in cash for this asset).
    3. **Relative momentum rank**: the asset's 90-day return is compared to
       every other symbol whose return has already been recorded in the current
       day's batch.  Only the top quartile (rank >= 0.75) emits a BUY signal.
    4. Signal levels: entry = close; stop = entry × 0.92; target = entry × 1.20.
    5. ``Signal.strength`` = percentile rank in the universe.

    Parameters
    ----------
    symbols:
        Override the default universe.  All symbols must share the same
        feed so that ``_universe_returns`` is populated on each daily close.
    """

    def __init__(self, symbols: Optional[list[str]] = None) -> None:
        super().__init__()
        settings = get_settings()

        self._symbols: list[str] = symbols or [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "ADAUSDT",
        ]

        # symbol → trailing 90-day return (float).  Updated on every daily candle.
        self._universe_returns: dict[str, float] = {}

        # Protect _universe_returns from concurrent coroutine writes.
        self._lock: asyncio.Lock = asyncio.Lock()

        self._log = log.bind(strategy=self.name)
        self._log.debug(
            "strategy_initialised",
            symbols=self._symbols,
            timeframes=self.timeframes,
            paper_trading=settings.paper_trading,
            lookback=_MOMENTUM_LOOKBACK,
            rank_threshold=_RELATIVE_RANK_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dual_momentum"

    @property
    def timeframes(self) -> list[str]:
        return ["1d"]

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Optional[Signal]:
        """Process a new daily candle and emit a BUY signal when dual-momentum
        conditions are satisfied.

        Parameters
        ----------
        symbol:
            Instrument that closed.
        timeframe:
            Expected ``"1d"``.
        df:
            Full daily OHLCV history, newest row last.

        Returns
        -------
        Signal | None
        """
        if symbol not in self._symbols:
            return None

        # ── Guard: need enough history ─────────────────────────────────────────
        if len(df) < _MIN_CANDLES:
            self._log.debug(
                "insufficient_candles",
                symbol=symbol,
                have=len(df),
                need=_MIN_CANDLES,
            )
            return None

        close_series = df["close"].astype(float)
        current_close: float = float(close_series.iloc[-1])
        close_90_ago: float = float(close_series.iloc[-_MOMENTUM_LOOKBACK])

        # ── Step 1: Absolute momentum — must be positive ───────────────────────
        abs_return: float = (current_close - close_90_ago) / close_90_ago

        async with self._lock:
            self._universe_returns[symbol] = abs_return

        if abs_return < 0.0:
            self._log.debug(
                "absolute_momentum_negative_skip",
                symbol=symbol,
                abs_return=round(abs_return, 6),
            )
            return None

        # ── Step 2: Relative momentum — compute percentile rank in universe ────
        async with self._lock:
            universe_snapshot: dict[str, float] = dict(self._universe_returns)

        all_returns: list[float] = list(universe_snapshot.values())

        if len(all_returns) < 2:
            # Cannot rank with only one data point
            self._log.debug(
                "universe_too_small_for_ranking",
                symbol=symbol,
                universe_size=len(all_returns),
            )
            return None

        returns_arr = np.array(all_returns, dtype=float)
        percentile_rank: float = float(
            np.sum(returns_arr <= abs_return) / len(returns_arr)
        )

        self._log.debug(
            "momentum_computed",
            symbol=symbol,
            abs_return=round(abs_return, 6),
            percentile_rank=round(percentile_rank, 4),
            universe_size=len(all_returns),
        )

        if percentile_rank < _RELATIVE_RANK_THRESHOLD:
            self._log.debug(
                "relative_rank_below_threshold_skip",
                symbol=symbol,
                percentile_rank=round(percentile_rank, 4),
                threshold=_RELATIVE_RANK_THRESHOLD,
            )
            return None

        # ── Signal levels ──────────────────────────────────────────────────────
        entry_price: float = current_close
        stop_price: float = entry_price * (1.0 - _STOP_PCT)
        target_price: float = entry_price * (1.0 + _TARGET_PCT)
        rr: float = self._rr_ratio(entry_price, stop_price, target_price)

        strength: float = round(min(max(percentile_rank, 0.0), 1.0), 4)

        indicator_snapshot: dict = {
            "abs_return_90d": round(abs_return, 6),
            "percentile_rank": round(percentile_rank, 4),
            "close": round(current_close, 8),
            "close_90d_ago": round(close_90_ago, 8),
            "universe_size": len(all_returns),
            "universe_returns": {
                sym: round(ret, 6) for sym, ret in universe_snapshot.items()
            },
        }

        signal = Signal(
            strategy=self.name,
            symbol=symbol,
            side="BUY",
            strength=strength,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            rr_ratio=rr,
            confidence=strength,
            indicators=indicator_snapshot,
            timestamp=datetime.now(tz=timezone.utc),
            timeframe=timeframe,
        )

        self._log.info(
            "signal.generated",
            symbol=symbol,
            timeframe=timeframe,
            side="BUY",
            strength=strength,
            rr=rr,
            entry=entry_price,
            stop=stop_price,
            target=target_price,
            abs_return=round(abs_return, 6),
            percentile_rank=round(percentile_rank, 4),
            actionable=signal.is_actionable,
        )

        return signal

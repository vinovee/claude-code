"""
Mean Reversion Strategy — Phase 2.

Trades intraday reversion to VWAP on short (5-minute) candles.  The core idea
is that liquid assets tend to gravitate back toward their daily VWAP; extreme
deviations combined with oversold/overbought RSI readings offer high-probability
fades.

Entry is only taken during the high-volume windows of major trading sessions:

* **London open**: 08:00–09:30 UTC
* **New York open**: 13:30–15:00 UTC
* **New York close**: 19:00–21:00 UTC

Signal logic
~~~~~~~~~~~~
BUY (expect price to rise back to VWAP):
  * ``deviation_pct < -0.01``  (price is >1 % below VWAP)
  * ``RSI(5) < 30``            (short-term oversold)

SELL (expect price to fall back to VWAP):
  * ``deviation_pct > 0.01``   (price is >1 % above VWAP)
  * ``RSI(5) > 70``            (short-term overbought)

Sizing reference levels:
  * Entry:  current close.
  * Stop:   entry ± 1.0 × ATR(14) (tight, mean-reversion stop).
  * Target: current VWAP (dynamic; moves intraday).

Usage
-----
    from strategies.phase2.mean_reversion import MeanReversionStrategy

    strategy = MeanReversionStrategy()
    signal = await strategy.on_candle("BTCUSDT", "5m", df)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, time, timezone
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

_MIN_CANDLES: int = 20          # minimum bars before trading

# VWAP deviation thresholds (as a fraction of VWAP)
_DEV_ENTRY: float = 0.01        # 1 % deviation to trigger entry
_DEV_STRENGTH_SCALE: float = 0.02   # deviation that saturates strength at 1.0

# RSI(5) thresholds
_RSI_PERIOD: int = 5
_RSI_OVERSOLD: float = 30.0
_RSI_OVERBOUGHT: float = 70.0

# ATR multiplier for stop sizing
_ATR_PERIOD: int = 14
_STOP_ATR_MULT: float = 1.0

# Active-session windows (UTC, inclusive of start, exclusive of end)
# Each entry is (window_start_time, window_end_time)
_SESSION_WINDOWS: list[tuple[time, time]] = [
    (time(8, 0),  time(9, 30)),    # London open
    (time(13, 30), time(15, 0)),   # New York open
    (time(19, 0),  time(21, 0)),   # New York close
]


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------


class MeanReversionStrategy(Strategy):
    """5-minute VWAP mean-reversion strategy.

    The strategy triggers on short-lived deviations from the intraday VWAP,
    confirmed by an extreme RSI(5) reading, and sizes stops using ATR(14).
    Entries are only accepted during the three major-session windows listed
    in :data:`_SESSION_WINDOWS`.

    Signal strength is a composite of:
    * How far the price has deviated from VWAP (relative to the saturation
      threshold of 2 %).
    * How extreme the RSI reading is (how far from the neutral 50 level).

    .. math::

        strength = \\text{dev\\_score} \\times
                   \\left(1 - \\frac{|rsi - 50|}{50} \\times 0.5\\right)

    Parameters
    ----------
    symbols:
        Override the default symbol list.
    """

    def __init__(self, symbols: Optional[list[str]] = None) -> None:
        super().__init__()
        settings = get_settings()

        self._symbols: list[str] = symbols or [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
        ]

        self._log = log.bind(strategy=self.name)
        self._log.debug(
            "strategy_initialised",
            symbols=self._symbols,
            timeframes=self.timeframes,
            paper_trading=settings.paper_trading,
            rsi_period=_RSI_PERIOD,
            atr_period=_ATR_PERIOD,
            dev_entry_threshold=_DEV_ENTRY,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def timeframes(self) -> list[str]:
        return ["5m"]

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    # ------------------------------------------------------------------
    # Session helper
    # ------------------------------------------------------------------

    @staticmethod
    def _is_active_session(ts: datetime) -> bool:
        """Return True when *ts* falls inside one of the major trading windows.

        All comparisons are done in UTC.  If *ts* is tz-aware it is converted
        to UTC first; if it is tz-naive it is assumed to already be UTC.

        Parameters
        ----------
        ts:
            Timestamp to check (the close time of the most recent candle).

        Returns
        -------
        bool
        """
        if ts.tzinfo is not None:
            ts_utc = ts.astimezone(timezone.utc)
        else:
            ts_utc = ts

        t = ts_utc.time().replace(second=0, microsecond=0)
        for window_start, window_end in _SESSION_WINDOWS:
            if window_start <= t < window_end:
                return True
        return False

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Optional[Signal]:
        """Evaluate a new 5-minute candle and return a signal if conditions fire.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTCUSDT"``.
        timeframe:
            Candle resolution — expected ``"5m"`` for this strategy.
        df:
            Full OHLCV history (newest row = latest closed candle).
            Required columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
            Index must be a ``DatetimeIndex`` (UTC) for VWAP daily resets.

        Returns
        -------
        Signal | None
        """
        if symbol not in self._symbols:
            return None

        # ── Guard: minimum history ────────────────────────────────────────────
        if len(df) < _MIN_CANDLES:
            self._log.debug(
                "insufficient_candles",
                symbol=symbol,
                timeframe=timeframe,
                have=len(df),
                need=_MIN_CANDLES,
            )
            return None

        # ── Session filter ────────────────────────────────────────────────────
        # Use the timestamp of the most recent candle.
        if isinstance(df.index, pd.DatetimeIndex):
            candle_ts: datetime = df.index[-1].to_pydatetime()
        else:
            candle_ts = datetime.now(tz=timezone.utc)

        if not self._is_active_session(candle_ts):
            self._log.debug(
                "outside_active_session_skip",
                symbol=symbol,
                timeframe=timeframe,
                candle_ts=candle_ts.isoformat(),
            )
            return None

        # ── Compute indicators ────────────────────────────────────────────────
        vwap_series: pd.Series = TechnicalIndicators.vwap(df)
        rsi_series: pd.Series = TechnicalIndicators.rsi(df, period=_RSI_PERIOD)
        atr_series: pd.Series = TechnicalIndicators.atr(df, period=_ATR_PERIOD)

        # Scalar values for the most recent bar
        close_cur: float = float(df["close"].astype(float).iloc[-1])

        vwap_raw = vwap_series.iloc[-1]
        vwap_val: float = float(vwap_raw) if not pd.isna(vwap_raw) else 0.0

        rsi_raw = rsi_series.dropna()
        rsi_val: float = float(rsi_raw.iloc[-1]) if not rsi_raw.empty else 50.0

        atr_raw = atr_series.dropna()
        atr_val: float = float(atr_raw.iloc[-1]) if not atr_raw.empty else 0.0

        # Guard against zero VWAP (e.g. first bar of the day with no volume)
        if vwap_val <= 0.0:
            self._log.debug(
                "vwap_zero_skip",
                symbol=symbol,
                timeframe=timeframe,
            )
            return None

        # ── VWAP deviation ────────────────────────────────────────────────────
        deviation_pct: float = (close_cur - vwap_val) / vwap_val

        indicator_snapshot: dict = {
            "vwap": round(vwap_val, 8),
            "deviation_pct": round(deviation_pct, 8),
            "rsi_5": round(rsi_val, 4),
            "atr_14": round(atr_val, 8),
            "close": round(close_cur, 8),
            "candle_ts": candle_ts.isoformat(),
        }

        # ── Entry condition checks ────────────────────────────────────────────
        buy_dev = deviation_pct < -_DEV_ENTRY
        buy_rsi = rsi_val < _RSI_OVERSOLD

        sell_dev = deviation_pct > _DEV_ENTRY
        sell_rsi = rsi_val > _RSI_OVERBOUGHT

        is_buy = buy_dev and buy_rsi
        is_sell = sell_dev and sell_rsi

        if not is_buy and not is_sell:
            self._log.debug(
                "no_signal",
                symbol=symbol,
                timeframe=timeframe,
                deviation_pct=round(deviation_pct, 6),
                rsi=round(rsi_val, 2),
            )
            return None

        # Resolve simultaneous edge case by deviation direction
        if is_buy and is_sell:
            side = "BUY" if deviation_pct <= 0.0 else "SELL"
        elif is_buy:
            side = "BUY"
        else:
            side = "SELL"

        # ── Entry / stop / target levels ──────────────────────────────────────
        entry_price: float = close_cur
        stop_distance: float = _STOP_ATR_MULT * atr_val

        # Guard against zero ATR (e.g. first bars with constant price)
        if stop_distance <= 0.0:
            stop_distance = entry_price * 0.005  # 0.5% fallback

        if side == "BUY":
            stop_price: float = entry_price - stop_distance
            # Target is the VWAP — price should revert back up
            target_price: float = vwap_val
            # If the VWAP target is below entry (shouldn't normally happen for
            # a BUY, but guard for edge cases like a freshly-reset VWAP)
            if target_price <= entry_price:
                target_price = entry_price + stop_distance * 2.0
        else:
            stop_price = entry_price + stop_distance
            # Target is the VWAP — price should revert back down
            target_price = vwap_val
            if target_price >= entry_price:
                target_price = entry_price - stop_distance * 2.0

        rr: float = self._rr_ratio(entry_price, stop_price, target_price)

        # ── Signal strength ───────────────────────────────────────────────────
        # Component 1 — how extreme is the VWAP deviation?
        dev_score: float = min(abs(deviation_pct) / _DEV_STRENGTH_SCALE, 1.0)

        # Component 2 — RSI distance from neutral (50); further away = stronger
        # The multiplier goes from 0.5 (RSI at 50) to 1.0 (RSI at 0 or 100)
        rsi_distance_factor: float = 1.0 - (abs(rsi_val - 50.0) / 50.0) * 0.5

        strength: float = round(
            min(max(dev_score * rsi_distance_factor, 0.0), 1.0),
            4,
        )

        signal = Signal(
            strategy=self.name,
            symbol=symbol,
            side=side,
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
            side=side,
            strength=strength,
            rr=rr,
            entry=entry_price,
            stop=stop_price,
            target=target_price,
            vwap=round(vwap_val, 8),
            deviation_pct=round(deviation_pct, 6),
            rsi=round(rsi_val, 2),
            atr=round(atr_val, 8),
            actionable=signal.is_actionable,
        )

        return signal

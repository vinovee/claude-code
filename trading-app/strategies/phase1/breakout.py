"""
Breakout Strategy — Phase 1.

Identifies confirmed breakouts above the 20-period rolling high or below
the 20-period rolling low, requiring above-average volume and a trending
market (ADX > 25) to avoid false breaks.

Usage
-----
    from strategies.phase1.breakout import BreakoutStrategy

    strategy = BreakoutStrategy()
    signal = await strategy.on_candle("BTCUSDT", "1h", df)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import structlog

from config.settings import get_settings
from data.indicators.technical import TechnicalIndicators
from strategies.base import Signal, Strategy

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CANDLES: int = 30

# Volume confirmation: must exceed the 20-bar SMA by this multiple
_VOL_RATIO_BREAKOUT: float = 2.0

# ADX threshold for a confirmed trend
_ADX_THRESHOLD: float = 25.0

# ATR multipliers
_STOP_ATR_MULT: float = 1.0   # Stop is placed at the candle's low/high
_TARGET_RR_MULT: float = 3.0  # Target = entry + 3 × risk

# Lookback window for rolling high/low used in breakout detection
_BREAKOUT_PERIOD: int = 20

# Strength scaling — normalisation caps
_ADX_STRONG: float = 50.0    # ADX value treated as "maximum strength"
_VOL_STRONG: float = 4.0     # Vol ratio treated as "maximum volume strength"


# ---------------------------------------------------------------------------
# BreakoutStrategy
# ---------------------------------------------------------------------------


class BreakoutStrategy(Strategy):
    """Hourly breakout strategy for major crypto pairs.

    Signal logic
    ~~~~~~~~~~~~
    BUY — all of the following must hold:

    * Close breaks above the 20-period rolling high (previous bars only).
    * Volume ratio (current / 20-bar SMA) > 2.0.
    * ADX(14) > 25 (confirmed trend / momentum behind the break).

    SELL — mirror:

    * Close breaks below the 20-period rolling low.
    * Volume ratio > 2.0.
    * ADX(14) > 25.

    Sizing reference levels
    ~~~~~~~~~~~~~~~~~~~~~~~
    * Entry:  current close price.
    * Stop:   low of the breakout candle (long) / high of the breakout candle (short).
    * Target: entry ± 3 × (entry − stop).

    Signal strength
    ~~~~~~~~~~~~~~~
    Composed from ADX value (stronger trend → higher score) and the degree of
    volume excess above the 2× threshold.

    Parameters
    ----------
    symbols:
        Override the default symbol list.
    """

    def __init__(self, symbols: Optional[list[str]] = None) -> None:
        super().__init__()
        settings = get_settings()
        self._symbols: list[str] = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._log = log.bind(strategy=self.name)
        self._log.debug(
            "strategy_initialised",
            symbols=self._symbols,
            timeframes=self.timeframes,
            paper_trading=settings.paper_trading,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def timeframes(self) -> list[str]:
        return ["1h"]

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
        """Evaluate a new closed candle and return a Signal if a breakout fires.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTCUSDT"``.
        timeframe:
            Candle resolution — expected ``"1h"`` for this strategy.
        df:
            Full OHLCV history (newest row = latest closed candle).
            Required columns: ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns
        -------
        Signal | None
        """
        if len(df) < _MIN_CANDLES:
            self._log.debug(
                "insufficient_candles",
                symbol=symbol,
                timeframe=timeframe,
                have=len(df),
                need=_MIN_CANDLES,
            )
            return None

        # ── Compute indicators ────────────────────────────────────────────────
        adx_series = TechnicalIndicators.adx(df, period=14)
        atr_series = TechnicalIndicators.atr(df, period=14)
        ema20_series = TechnicalIndicators.ema(df["close"], period=20)
        vol_ratio_series = TechnicalIndicators.volume_sma_ratio(df, period=20)

        # rolling_high/low exclude the current candle (shift(1).rolling.max/min)
        rolling_high_series = TechnicalIndicators.rolling_high(df, period=_BREAKOUT_PERIOD)
        rolling_low_series = TechnicalIndicators.rolling_low(df, period=_BREAKOUT_PERIOD)

        # Current bar scalars
        close_cur = float(df["close"].iloc[-1])
        high_cur = float(df["high"].iloc[-1])
        low_cur = float(df["low"].iloc[-1])

        adx_val = float(adx_series.dropna().iloc[-1]) if not adx_series.dropna().empty else 0.0
        atr_val = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0
        ema20_val = float(ema20_series.iloc[-1])
        vol_ratio = float(vol_ratio_series.iloc[-1]) if not pd.isna(vol_ratio_series.iloc[-1]) else 0.0

        prev_high = float(rolling_high_series.iloc[-1]) if not pd.isna(rolling_high_series.iloc[-1]) else float("inf")
        prev_low = float(rolling_low_series.iloc[-1]) if not pd.isna(rolling_low_series.iloc[-1]) else float("-inf")

        indicator_snapshot = {
            "adx": round(adx_val, 4),
            "atr": round(atr_val, 8),
            "ema20": round(ema20_val, 8),
            "vol_ratio": round(vol_ratio, 4),
            "rolling_high_20": round(prev_high, 8),
            "rolling_low_20": round(prev_low, 8),
            "close": round(close_cur, 8),
            "high": round(high_cur, 8),
            "low": round(low_cur, 8),
        }

        # ── Filter: ADX must confirm trend ────────────────────────────────────
        if adx_val <= _ADX_THRESHOLD:
            self._log.debug(
                "adx_filter_skip",
                symbol=symbol,
                timeframe=timeframe,
                adx=adx_val,
            )
            return None

        # ── Filter: volume must confirm breakout ──────────────────────────────
        if vol_ratio <= _VOL_RATIO_BREAKOUT:
            self._log.debug(
                "volume_filter_skip",
                symbol=symbol,
                timeframe=timeframe,
                vol_ratio=vol_ratio,
            )
            return None

        # ── Breakout detection ────────────────────────────────────────────────
        bull_breakout = close_cur > prev_high
        bear_breakout = close_cur < prev_low

        if not bull_breakout and not bear_breakout:
            return None

        # Resolve simultaneous signals (edge case) by EMA20 direction
        if bull_breakout and bear_breakout:
            side = "BUY" if close_cur >= ema20_val else "SELL"
        elif bull_breakout:
            side = "BUY"
        else:
            side = "SELL"

        # ── Entry / stop / target ─────────────────────────────────────────────
        entry_price = close_cur

        if side == "BUY":
            # Stop at the low of the breakout candle
            stop_price = low_cur
        else:
            # Stop at the high of the breakout candle
            stop_price = high_cur

        risk = abs(entry_price - stop_price)

        # Protect against zero-risk edge case (e.g. doji candle)
        if risk == 0.0:
            risk = atr_val if atr_val > 0.0 else entry_price * 0.001

        if side == "BUY":
            target_price = entry_price + _TARGET_RR_MULT * risk
        else:
            target_price = entry_price - _TARGET_RR_MULT * risk

        rr = self._rr_ratio(entry_price, stop_price, target_price)

        # ── Signal strength ───────────────────────────────────────────────────
        # ADX component: linearly scales from ADX_THRESHOLD (0) to ADX_STRONG (1)
        adx_score = min((adx_val - _ADX_THRESHOLD) / (_ADX_STRONG - _ADX_THRESHOLD), 1.0)

        # Volume component: scales from VOL_RATIO_BREAKOUT (0) to VOL_STRONG (1)
        vol_score = min(
            (vol_ratio - _VOL_RATIO_BREAKOUT) / (_VOL_STRONG - _VOL_RATIO_BREAKOUT),
            1.0,
        )

        # Breakout margin component: how far above/below the breakout level
        if prev_high > 0 and side == "BUY":
            margin_pct = (close_cur - prev_high) / prev_high
        elif prev_low > 0 and side == "SELL":
            margin_pct = (prev_low - close_cur) / prev_low
        else:
            margin_pct = 0.0
        margin_score = min(margin_pct / 0.01, 1.0)  # normalise to 1% breakout = 1.0

        strength = (
            0.40 * adx_score
            + 0.35 * vol_score
            + 0.25 * margin_score
        )
        strength = round(min(max(strength, 0.0), 1.0), 4)

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
            adx=adx_val,
            vol_ratio=vol_ratio,
            actionable=signal.is_actionable,
        )

        return signal

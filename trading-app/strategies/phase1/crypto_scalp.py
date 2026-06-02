"""
Crypto Scalping Strategy — Phase 1.

Trades short-term momentum on major crypto pairs using RSI, MACD,
Bollinger Bands, ATR, and volume confirmation.  Only active in
TRENDING or VOLATILE market regimes; skips RANGING markets.

Usage
-----
    from strategies.phase1.crypto_scalp import CryptoScalpStrategy

    strategy = CryptoScalpStrategy()
    signal = await strategy.on_candle("BTCUSDT", "5m", df)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import structlog

from config.settings import get_settings
from data.indicators.technical import MarketRegime, MarketRegimeDetector, TechnicalIndicators
from strategies.base import Signal, Strategy

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CANDLES: int = 50

# RSI thresholds
_RSI_BULL_CROSS: float = 55.0      # cross above triggers long consideration
_RSI_BEAR_CROSS: float = 45.0      # cross below triggers short consideration
_RSI_BULL_RANGE_LOW: float = 45.0  # rising-RSI long window lower bound
_RSI_BULL_RANGE_HIGH: float = 65.0  # rising-RSI long window upper bound

# Volume
_VOL_RATIO_THRESHOLD: float = 1.5  # current vol must exceed SMA by this multiple

# ATR multipliers
_STOP_ATR_MULT: float = 1.5
_TARGET_ATR_MULT: float = 3.0  # gives 2:1 R:R by design


# ---------------------------------------------------------------------------
# CryptoScalpStrategy
# ---------------------------------------------------------------------------


class CryptoScalpStrategy(Strategy):
    """Short-term scalping strategy for liquid crypto pairs.

    Signal logic
    ~~~~~~~~~~~~
    BUY — all of the following must hold:

    * RSI(14) crosses above 55, **or** RSI is in [45, 65] with a rising slope.
    * MACD histogram is positive and increasing (current > previous).
    * Volume ratio (current / 20-bar SMA) > 1.5.
    * Close is above the Bollinger midline.

    SELL (short) — mirror of the above:

    * RSI(14) crosses below 45.
    * MACD histogram is negative and decreasing.
    * Volume ratio > 1.5.
    * Close is below the Bollinger midline.

    The market regime is checked first; RANGING markets are skipped entirely.

    Parameters
    ----------
    symbols:
        Override the default symbol list.
    """

    def __init__(self, symbols: Optional[list[str]] = None) -> None:
        super().__init__()
        settings = get_settings()
        self._symbols: list[str] = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
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
        return "crypto_scalp"

    @property
    def timeframes(self) -> list[str]:
        return ["5m", "15m"]

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
        """Evaluate a new closed candle and return a Signal if conditions fire.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTCUSDT"``.
        timeframe:
            Candle resolution, e.g. ``"5m"``.
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

        # ── Market regime gate ────────────────────────────────────────────────
        regime = MarketRegimeDetector.detect(df)
        if regime == MarketRegime.RANGING:
            self._log.debug(
                "regime_skip",
                symbol=symbol,
                timeframe=timeframe,
                regime=regime.value,
            )
            return None

        # ── Compute indicators ────────────────────────────────────────────────
        rsi_series = TechnicalIndicators.rsi(df["close"], period=14)
        macd_result = TechnicalIndicators.macd(df["close"], fast=12, slow=26, signal_period=9)
        bb = TechnicalIndicators.bollinger_bands(df["close"], period=20, std_dev=2.0)
        atr_series = TechnicalIndicators.atr(df, period=14)
        vol_ratio_series = TechnicalIndicators.volume_sma_ratio(df, period=20)

        # Latest and previous bar values
        rsi_cur = float(rsi_series.iloc[-1])
        rsi_prev = float(rsi_series.iloc[-2])

        hist_cur = float(macd_result.histogram.iloc[-1])
        hist_prev = float(macd_result.histogram.iloc[-2])

        close_cur = float(df["close"].iloc[-1])
        bb_mid = float(bb.middle.iloc[-1])

        vol_ratio = float(vol_ratio_series.iloc[-1])
        atr_val = float(atr_series.iloc[-1])

        indicator_snapshot = {
            "rsi": round(rsi_cur, 4),
            "rsi_prev": round(rsi_prev, 4),
            "macd_hist": round(hist_cur, 8),
            "macd_hist_prev": round(hist_prev, 8),
            "bb_mid": round(bb_mid, 8),
            "bb_upper": round(float(bb.upper.iloc[-1]), 8),
            "bb_lower": round(float(bb.lower.iloc[-1]), 8),
            "atr": round(atr_val, 8),
            "vol_ratio": round(vol_ratio, 4),
            "regime": regime.value,
        }

        # ── BUY conditions ────────────────────────────────────────────────────
        rsi_bull_cross = (rsi_prev < _RSI_BULL_CROSS) and (rsi_cur >= _RSI_BULL_CROSS)
        rsi_bull_range = (
            _RSI_BULL_RANGE_LOW <= rsi_cur <= _RSI_BULL_RANGE_HIGH
            and rsi_cur > rsi_prev
        )
        macd_bull = hist_cur > 0.0 and hist_cur > hist_prev
        vol_ok = vol_ratio > _VOL_RATIO_THRESHOLD
        price_above_mid = close_cur > bb_mid

        n_bull_conditions = sum([
            rsi_bull_cross or rsi_bull_range,
            macd_bull,
            vol_ok,
            price_above_mid,
        ])

        # ── SELL conditions ───────────────────────────────────────────────────
        rsi_bear_cross = (rsi_prev > _RSI_BEAR_CROSS) and (rsi_cur <= _RSI_BEAR_CROSS)
        macd_bear = hist_cur < 0.0 and hist_cur < hist_prev
        price_below_mid = close_cur < bb_mid

        n_bear_conditions = sum([
            rsi_bear_cross,
            macd_bear,
            vol_ok,
            price_below_mid,
        ])

        # ── Determine side ────────────────────────────────────────────────────
        all_bull = n_bull_conditions == 4
        all_bear = n_bear_conditions == 4

        if not all_bull and not all_bear:
            return None

        # Choose the side; if somehow both fire simultaneously, prefer the one
        # with cleaner RSI alignment (edge case in volatile regimes).
        if all_bull and all_bear:
            side = "BUY" if rsi_cur >= 50.0 else "SELL"
        elif all_bull:
            side = "BUY"
        else:
            side = "SELL"

        # ── Entry / stop / target ─────────────────────────────────────────────
        entry_price = close_cur

        if side == "BUY":
            stop_price = entry_price - _STOP_ATR_MULT * atr_val
            target_price = entry_price + _TARGET_ATR_MULT * atr_val
        else:
            stop_price = entry_price + _STOP_ATR_MULT * atr_val
            target_price = entry_price - _TARGET_ATR_MULT * atr_val

        rr = self._rr_ratio(entry_price, stop_price, target_price)

        # ── Signal strength ───────────────────────────────────────────────────
        # Components:
        #   1. Condition count (all 4 required, already guaranteed here) → base 0.5
        #   2. RSI momentum component (how far past the threshold) → up to 0.2
        #   3. Volume excess component (how much above 1.5×) → up to 0.15
        #   4. Regime bonus (TRENDING > VOLATILE) → up to 0.15
        if side == "BUY":
            rsi_excess = min((rsi_cur - _RSI_BULL_CROSS) / 20.0, 1.0)  # normalised 0–1
        else:
            rsi_excess = min((_RSI_BEAR_CROSS - rsi_cur) / 20.0, 1.0)

        vol_excess = min((vol_ratio - _VOL_RATIO_THRESHOLD) / 1.5, 1.0)
        regime_bonus = 1.0 if regime == MarketRegime.TRENDING else 0.6

        strength = (
            0.50                        # base for all-conditions-met
            + 0.20 * rsi_excess
            + 0.15 * vol_excess
            + 0.15 * regime_bonus
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
            regime=regime.value,
            actionable=signal.is_actionable,
        )

        return signal

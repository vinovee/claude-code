"""
Technical indicators and market-regime detection for the trading application.

All methods in :class:`TechnicalIndicators` accept a ``pandas.DataFrame``
with OHLCV columns (``open``, ``high``, ``low``, ``close``, ``volume``) and
return a ``pd.Series`` (or a tuple of Series for composite indicators).

Implementation strategy
-----------------------
If TA-Lib's C extension is available it is used for speed; otherwise a
pure-pandas/NumPy fallback is used so the module can be imported in any
environment (CI, Docker build without native libs, etc.).
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional TA-Lib import
# ---------------------------------------------------------------------------

try:
    import talib  # type: ignore[import]

    _TALIB_AVAILABLE = True
    logger.debug("TA-Lib C extension loaded — using native implementations")
except ImportError:
    _TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available — using pandas/NumPy fallbacks")

__all__ = [
    "TechnicalIndicators",
    "MarketRegimeDetector",
]


# ---------------------------------------------------------------------------
# TechnicalIndicators
# ---------------------------------------------------------------------------


class TechnicalIndicators:
    """Stateless helper that computes technical indicators from OHLCV data.

    All methods are *static* so callers never need to instantiate the class;
    it exists purely as a namespace.

    DataFrame contract
    ------------------
    Every method accepts a ``pd.DataFrame`` with at minimum these columns:
    ``open``, ``high``, ``low``, ``close``, ``volume``.
    """

    # ------------------------------------------------------------------
    # RSI
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's smoothing).

        Returns a Series aligned to ``df.index``.
        """
        close = df["close"].astype(float)

        if _TALIB_AVAILABLE:
            result = talib.RSI(close.values, timeperiod=period)
            return pd.Series(result, index=df.index, name="rsi")

        # Pandas fallback — Wilder's smoothing via EWMA with alpha = 1/period
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).rename("rsi")

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, and histogram.

        Returns ``(macd_line, signal_line, histogram)``.
        """
        close = df["close"].astype(float)

        if _TALIB_AVAILABLE:
            macd_arr, signal_arr, hist_arr = talib.MACD(
                close.values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal,
            )
            idx = df.index
            return (
                pd.Series(macd_arr, index=idx, name="macd"),
                pd.Series(signal_arr, index=idx, name="macd_signal"),
                pd.Series(hist_arr, index=idx, name="macd_hist"),
            )

        # Pandas fallback
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        macd_line = (fast_ema - slow_ema).rename("macd")
        signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("macd_signal")
        histogram = (macd_line - signal_line).rename("macd_hist")
        return macd_line, signal_line, histogram

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.

        Returns ``(upper, middle, lower)``.
        """
        close = df["close"].astype(float)

        if _TALIB_AVAILABLE:
            upper_arr, mid_arr, lower_arr = talib.BBANDS(
                close.values,
                timeperiod=period,
                nbdevup=std,
                nbdevdn=std,
                matype=0,  # SMA
            )
            idx = df.index
            return (
                pd.Series(upper_arr, index=idx, name="bb_upper"),
                pd.Series(mid_arr, index=idx, name="bb_mid"),
                pd.Series(lower_arr, index=idx, name="bb_lower"),
            )

        # Pandas fallback
        middle = close.rolling(window=period).mean().rename("bb_mid")
        rolling_std = close.rolling(window=period).std(ddof=0)
        upper = (middle + std * rolling_std).rename("bb_upper")
        lower = (middle - std * rolling_std).rename("bb_lower")
        return upper, middle, lower

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        if _TALIB_AVAILABLE:
            result = talib.ATR(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(result, index=df.index, name="atr")

        # Pandas fallback
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().rename("atr")

    # ------------------------------------------------------------------
    # VWAP — intraday, resets daily
    # ------------------------------------------------------------------

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume-Weighted Average Price.

        Resets at the start of each calendar day.  Requires a
        ``DatetimeIndex`` (tz-aware or tz-naive) or a ``ts`` / ``datetime``
        column.  If the index is not datetime-like, VWAP is computed over the
        full DataFrame without daily resets.
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume

        # Attempt daily grouping when the index is datetime-like
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.normalize()  # truncate to day
            cum_tp_vol = tp_vol.groupby(dates).cumsum()
            cum_vol = volume.groupby(dates).cumsum()
        else:
            cum_tp_vol = tp_vol.cumsum()
            cum_vol = volume.cumsum()

        return (cum_tp_vol / cum_vol.replace(0.0, np.nan)).rename("vwap")

    # ------------------------------------------------------------------
    # ADX
    # ------------------------------------------------------------------

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (ADX).

        Returns values in the range 0–100.
        """
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        if _TALIB_AVAILABLE:
            result = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            return pd.Series(result, index=df.index, name="adx")

        # Pandas fallback — Wilder's smoothed DI / DX / ADX
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )

        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        alpha = 1.0 / period
        atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s * 100.0
        minus_di = minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s * 100.0

        dx = (
            (plus_di - minus_di).abs()
            / (plus_di + minus_di).replace(0.0, np.nan)
            * 100.0
        )
        return dx.ewm(alpha=alpha, adjust=False).mean().rename("adx")

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @staticmethod
    def ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Exponential Moving Average of the close price."""
        close = df["close"].astype(float)

        if _TALIB_AVAILABLE:
            result = talib.EMA(close.values, timeperiod=period)
            return pd.Series(result, index=df.index, name=f"ema_{period}")

        return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")

    # ------------------------------------------------------------------
    # OBV
    # ------------------------------------------------------------------

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        if _TALIB_AVAILABLE:
            result = talib.OBV(close.values, volume.values)
            return pd.Series(result, index=df.index, name="obv")

        # Pandas fallback
        direction = np.sign(close.diff()).fillna(0.0)
        return (direction * volume).cumsum().rename("obv")

    # ------------------------------------------------------------------
    # Stochastic
    # ------------------------------------------------------------------

    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator.

        Returns ``(%K, %D)``.
        """
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        if _TALIB_AVAILABLE:
            k_arr, d_arr = talib.STOCH(
                high.values,
                low.values,
                close.values,
                fastk_period=k_period,
                slowk_period=d_period,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0,
            )
            idx = df.index
            return (
                pd.Series(k_arr, index=idx, name="stoch_k"),
                pd.Series(d_arr, index=idx, name="stoch_d"),
            )

        # Pandas fallback
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        range_ = (highest_high - lowest_low).replace(0.0, np.nan)
        pct_k = ((close - lowest_low) / range_ * 100.0).rename("stoch_k")
        pct_d = pct_k.rolling(window=d_period).mean().rename("stoch_d")
        return pct_k, pct_d

    # ------------------------------------------------------------------
    # Volume SMA ratio
    # ------------------------------------------------------------------

    @staticmethod
    def volume_sma_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Ratio of current volume to its *period*-bar simple moving average.

        Values > 1.0 indicate above-average volume.
        """
        volume = df["volume"].astype(float)
        vol_sma = volume.rolling(window=period).mean()
        return (volume / vol_sma.replace(0.0, np.nan)).rename("volume_sma_ratio")

    # ------------------------------------------------------------------
    # compute_all — add all indicator columns and return the enriched df
    # ------------------------------------------------------------------

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute every indicator and attach them as new columns.

        The original DataFrame is *not* mutated; a copy is returned.
        Columns added:
          rsi, macd, macd_signal, macd_hist,
          bb_upper, bb_mid, bb_lower,
          atr, vwap, adx,
          ema_9, ema_21, ema_50, ema_200,
          obv, stoch_k, stoch_d, volume_sma_ratio
        """
        out = df.copy()

        out["rsi"] = cls.rsi(df)

        macd_line, signal_line, histogram = cls.macd(df)
        out["macd"] = macd_line
        out["macd_signal"] = signal_line
        out["macd_hist"] = histogram

        bb_upper, bb_mid, bb_lower = cls.bollinger_bands(df)
        out["bb_upper"] = bb_upper
        out["bb_mid"] = bb_mid
        out["bb_lower"] = bb_lower

        out["atr"] = cls.atr(df)
        out["vwap"] = cls.vwap(df)
        out["adx"] = cls.adx(df)

        for p in (9, 21, 50, 200):
            out[f"ema_{p}"] = cls.ema(df, period=p)

        out["obv"] = cls.obv(df)

        stoch_k, stoch_d = cls.stochastic(df)
        out["stoch_k"] = stoch_k
        out["stoch_d"] = stoch_d

        out["volume_sma_ratio"] = cls.volume_sma_ratio(df)

        return out


# ---------------------------------------------------------------------------
# MarketRegimeDetector
# ---------------------------------------------------------------------------


class MarketRegimeDetector:
    """Classify the current market regime for a given OHLCV DataFrame.

    Regime logic
    ~~~~~~~~~~~~
    * **TRENDING**: ADX > 25
    * **VOLATILE**: ATR / close > 0.03  (ATR% > 3%)
    * **RANGING**: everything else

    The rules are evaluated in priority order: TRENDING is checked first
    because a trending market is often volatile too.
    """

    @staticmethod
    def detect(df: pd.DataFrame) -> Literal["TRENDING", "RANGING", "VOLATILE"]:
        """Return the regime string for the most recent bar in *df*.

        Parameters
        ----------
        df:
            OHLCV DataFrame.  Must contain at least ``high``, ``low``,
            ``close`` columns and have >= 30 rows for reliable readings.

        Returns
        -------
        ``'TRENDING'``, ``'VOLATILE'``, or ``'RANGING'``.
        """
        if df.empty:
            return "RANGING"

        adx_series = TechnicalIndicators.adx(df)
        atr_series = TechnicalIndicators.atr(df)

        # Use the most recent non-NaN values
        adx_clean = adx_series.dropna()
        atr_clean = atr_series.dropna()

        adx_val: float = float(adx_clean.iloc[-1]) if not adx_clean.empty else 0.0
        atr_val: float = float(atr_clean.iloc[-1]) if not atr_clean.empty else 0.0
        close_val: float = float(df["close"].iloc[-1])

        # TRENDING: strong directional movement
        if adx_val > 25.0:
            return "TRENDING"

        # VOLATILE: ATR exceeds 3% of price
        if close_val > 0.0 and (atr_val / close_val) > 0.03:
            return "VOLATILE"

        return "RANGING"

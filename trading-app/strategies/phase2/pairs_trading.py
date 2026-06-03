"""
Pairs Trading Strategy — Phase 2.

Exploits mean-reversion in co-integrated crypto pairs.  For each configured
pair (A, B) the strategy:

1. Maintains a rolling hedge ratio updated every 20 candles via OLS regression
   on log prices (a computationally cheap Kalman-filter approximation).
2. Tracks a spread history of the last 200 values:
       spread = log(price_A) - hedge_ratio × log(price_B)
3. Computes a z-score of the spread and generates:
   - **BUY spread** (long A / short B) when z < −2.0
   - **SELL spread** (short A / long B) when z > +2.0
   - **EXIT** (both legs) when |z| < 0.5

The Signal encodes which leg is long and which is short in ``indicators``.

Usage
-----
    from strategies.phase2.pairs_trading import PairsTradingStrategy

    strategy = PairsTradingStrategy()
    signal = await strategy.on_candle("BTCUSDT", "1h", df_btc)
"""
from __future__ import annotations

import asyncio
from collections import deque
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

_MIN_CANDLES: int = 100                  # minimum candles before trading
_SPREAD_HISTORY_MAXLEN: int = 200        # rolling window for spread statistics
_HEDGE_UPDATE_INTERVAL: int = 20        # recalculate hedge ratio every N candles
_HEDGE_LOOKBACK: int = 60               # OLS regression window (candles)

# Z-score entry / exit thresholds
_ZSCORE_ENTRY_LONG: float = -2.0        # z < this → long spread
_ZSCORE_ENTRY_SHORT: float = 2.0        # z > this → short spread
_ZSCORE_EXIT: float = 0.5               # |z| < this → exit

# Stop distance as a multiple of spread_std
_STOP_SPREAD_STD_MULT: float = 1.5

# Minimum spread std to avoid division-by-zero noise
_MIN_SPREAD_STD: float = 1e-8


# ---------------------------------------------------------------------------
# PairState — per-pair internal state
# ---------------------------------------------------------------------------


class _PairState:
    """Mutable state maintained for a single (symbol_a, symbol_b) pair."""

    __slots__ = (
        "symbol_a",
        "symbol_b",
        "spread_history",
        "hedge_ratio",
        "spread_mean",
        "spread_std",
        "candles_since_hedge_update",
        "log_prices_a",
        "log_prices_b",
    )

    def __init__(self, symbol_a: str, symbol_b: str) -> None:
        self.symbol_a: str = symbol_a
        self.symbol_b: str = symbol_b
        self.spread_history: deque[float] = deque(maxlen=_SPREAD_HISTORY_MAXLEN)
        self.hedge_ratio: float = 1.0
        self.spread_mean: float = 0.0
        self.spread_std: float = _MIN_SPREAD_STD
        self.candles_since_hedge_update: int = 0
        # Rolling log-price buffers for hedge ratio estimation
        self.log_prices_a: deque[float] = deque(maxlen=_HEDGE_LOOKBACK)
        self.log_prices_b: deque[float] = deque(maxlen=_HEDGE_LOOKBACK)


# ---------------------------------------------------------------------------
# PairsTradingStrategy
# ---------------------------------------------------------------------------


class PairsTradingStrategy(Strategy):
    """Hourly pairs-trading strategy on co-integrated crypto pairs.

    For each pair the strategy tracks a log-price spread, estimates a rolling
    hedge ratio via OLS regression updated every :data:`_HEDGE_UPDATE_INTERVAL`
    candles, and generates mean-reversion signals when the z-score of the
    spread exceeds ±2.0.

    The ``Signal.symbol`` is set to the *primary* (A) symbol of the pair.
    The ``Signal.indicators`` dict always contains:

    .. code-block:: python

        {
            "long_symbol": "BTCUSDT",   # or None for EXIT
            "short_symbol": "ETHUSDT",  # or None for EXIT
            "z_score": -2.3,
            "hedge_ratio": 0.84,
            "spread_mean": 0.021,
            "spread_std": 0.009,
            "spread": 0.003,
        }

    Parameters
    ----------
    pairs:
        List of ``(symbol_a, symbol_b)`` tuples.  Both symbols must be
        present in the same data feed.
    """

    def __init__(
        self,
        pairs: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        super().__init__()
        settings = get_settings()

        self._pairs: list[tuple[str, str]] = pairs or [
            ("BTCUSDT", "ETHUSDT"),
            ("ETHUSDT", "SOLUSDT"),
        ]

        # Build state objects, keyed by (symbol_a, symbol_b)
        self._pair_state: dict[tuple[str, str], _PairState] = {
            pair: _PairState(pair[0], pair[1]) for pair in self._pairs
        }

        # Map each symbol to the pairs it participates in so on_candle
        # can quickly look up which pairs need updating.
        self._symbol_to_pairs: dict[str, list[tuple[str, str]]] = {}
        for sym_a, sym_b in self._pairs:
            self._symbol_to_pairs.setdefault(sym_a, []).append((sym_a, sym_b))
            self._symbol_to_pairs.setdefault(sym_b, []).append((sym_a, sym_b))

        # Async lock guards writes to _pair_state
        self._lock: asyncio.Lock = asyncio.Lock()

        self._log = log.bind(strategy=self.name)
        self._log.debug(
            "strategy_initialised",
            pairs=self._pairs,
            timeframes=self.timeframes,
            paper_trading=settings.paper_trading,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "pairs_trading"

    @property
    def timeframes(self) -> list[str]:
        return ["1h"]

    @property
    def symbols(self) -> list[str]:
        """Flat list of every symbol that appears in at least one pair."""
        seen: set[str] = set()
        result: list[str] = []
        for sym_a, sym_b in self._pairs:
            for sym in (sym_a, sym_b):
                if sym not in seen:
                    seen.add(sym)
                    result.append(sym)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_hedge_ratio(self, state: _PairState) -> None:
        """Recompute the OLS hedge ratio from the buffered log-price series."""
        if (
            len(state.log_prices_a) < 2
            or len(state.log_prices_b) < 2
            or len(state.log_prices_a) != len(state.log_prices_b)
        ):
            return

        x = np.array(state.log_prices_b, dtype=float)
        y = np.array(state.log_prices_a, dtype=float)

        # OLS: y = hedge_ratio * x + intercept  →  hedge_ratio = cov(x,y)/var(x)
        x_demean = x - x.mean()
        x_var = float(np.dot(x_demean, x_demean))
        if x_var < _MIN_SPREAD_STD:
            return  # collinear or flat — keep previous ratio

        covariance = float(np.dot(x_demean, y - y.mean()))
        state.hedge_ratio = covariance / x_var

        self._log.debug(
            "hedge_ratio_updated",
            symbol_a=state.symbol_a,
            symbol_b=state.symbol_b,
            hedge_ratio=round(state.hedge_ratio, 6),
            n=len(x),
        )

    def _update_spread_stats(self, state: _PairState) -> None:
        """Refresh rolling mean and std from the spread history deque."""
        if len(state.spread_history) < 2:
            return
        arr = np.array(state.spread_history, dtype=float)
        state.spread_mean = float(arr.mean())
        state.spread_std = max(float(arr.std(ddof=1)), _MIN_SPREAD_STD)

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Optional[Signal]:
        """Process a new 1-hour candle.

        The strategy inspects all pairs that include *symbol*.  When the pair
        state has sufficient history the current spread z-score is evaluated and
        a signal may be returned for the primary (A) symbol of the pair.

        If *symbol* is the secondary leg (B), the function still updates state
        but returns ``None`` (signals are only emitted on A candles to avoid
        double-counting).

        Parameters
        ----------
        symbol:
            Symbol whose candle just closed.
        timeframe:
            Expected ``"1h"``.
        df:
            Full 1-hour OHLCV history for *symbol*.

        Returns
        -------
        Signal | None
        """
        if symbol not in self._symbol_to_pairs:
            return None

        if len(df) < _MIN_CANDLES:
            self._log.debug(
                "insufficient_candles",
                symbol=symbol,
                have=len(df),
                need=_MIN_CANDLES,
            )
            return None

        current_price: float = float(df["close"].astype(float).iloc[-1])
        log_price: float = float(np.log(current_price))

        # We only emit a signal if this symbol is the A-leg of at least one pair
        emitted_signal: Optional[Signal] = None

        for pair_key in self._symbol_to_pairs[symbol]:
            sym_a, sym_b = pair_key

            async with self._lock:
                state = self._pair_state[pair_key]

                # Feed the new log-price into the appropriate buffer
                if symbol == sym_a:
                    state.log_prices_a.append(log_price)
                else:
                    state.log_prices_b.append(log_price)

                state.candles_since_hedge_update += 1

                # ── Update hedge ratio periodically ────────────────────────────
                if state.candles_since_hedge_update >= _HEDGE_UPDATE_INTERVAL:
                    self._update_hedge_ratio(state)
                    state.candles_since_hedge_update = 0

                # We need matching log-price observations for both legs to
                # compute a spread; skip if one buffer is empty.
                if not state.log_prices_a or not state.log_prices_b:
                    continue

                # Use the most recent log-price from each leg's buffer
                lp_a = state.log_prices_a[-1]
                lp_b = state.log_prices_b[-1]
                current_spread: float = lp_a - state.hedge_ratio * lp_b

                state.spread_history.append(current_spread)
                self._update_spread_stats(state)

                # Capture for use outside the lock
                spread_mean = state.spread_mean
                spread_std = state.spread_std
                hedge_ratio = state.hedge_ratio
                n_spread = len(state.spread_history)

            # ── Need enough spread observations ────────────────────────────────
            if n_spread < _MIN_CANDLES:
                continue

            # ── Z-score ────────────────────────────────────────────────────────
            z_score: float = (current_spread - spread_mean) / spread_std

            self._log.debug(
                "spread_computed",
                pair=f"{sym_a}/{sym_b}",
                spread=round(current_spread, 8),
                z_score=round(z_score, 4),
                hedge_ratio=round(hedge_ratio, 6),
                spread_mean=round(spread_mean, 8),
                spread_std=round(spread_std, 8),
            )

            # ── Signal evaluation ──────────────────────────────────────────────
            side: Optional[str] = None
            long_symbol: Optional[str] = None
            short_symbol: Optional[str] = None

            if z_score < _ZSCORE_ENTRY_LONG:
                # Spread is abnormally low → long A, short B
                side = "BUY"
                long_symbol = sym_a
                short_symbol = sym_b
            elif z_score > _ZSCORE_ENTRY_SHORT:
                # Spread is abnormally high → short A, long B
                side = "SELL"
                long_symbol = sym_b
                short_symbol = sym_a
            elif abs(z_score) < _ZSCORE_EXIT:
                # Mean reversion complete → exit
                side = "NONE"
                long_symbol = None
                short_symbol = None

            # No signal condition met
            if side is None:
                continue

            # Only emit a signal when this candle belongs to the A leg,
            # to avoid duplicate signals per pair.
            if symbol != sym_a:
                continue

            # ── Entry / stop / target levels ───────────────────────────────────
            entry_price: float = current_price

            # Stop is expressed as stop distance = 1.5 × spread_std translated
            # back to price space for the A leg.  We use a simple approximation:
            # Δspread ≈ Δlog(price_A) ≈ Δprice_A / price_A
            stop_distance: float = _STOP_SPREAD_STD_MULT * spread_std * entry_price

            if side == "BUY":
                stop_price: float = entry_price - stop_distance
                # Target is mean reversion back to spread_mean
                target_spread_pct: float = abs(current_spread - spread_mean)
                target_price: float = entry_price + target_spread_pct * entry_price
            elif side == "SELL":
                stop_price = entry_price + stop_distance
                target_spread_pct = abs(current_spread - spread_mean)
                target_price = entry_price - target_spread_pct * entry_price
            else:
                # EXIT signal — levels are symbolic (entry = stop = target)
                stop_price = entry_price
                target_price = entry_price

            rr: float = self._rr_ratio(entry_price, stop_price, target_price)
            strength: float = round(min(abs(z_score) / 3.0, 1.0), 4)

            indicator_snapshot: dict = {
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "z_score": round(z_score, 6),
                "hedge_ratio": round(hedge_ratio, 6),
                "spread_mean": round(spread_mean, 8),
                "spread_std": round(spread_std, 8),
                "spread": round(current_spread, 8),
                "n_spread_observations": n_spread,
            }

            emitted_signal = Signal(
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
                pair=f"{sym_a}/{sym_b}",
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                strength=strength,
                z_score=round(z_score, 4),
                long_symbol=long_symbol,
                short_symbol=short_symbol,
                rr=rr,
                actionable=emitted_signal.is_actionable,
            )

            # Return the first actionable signal found; the caller processes
            # one signal per candle event.
            return emitted_signal

        return emitted_signal

"""
News Catalyst Strategy — Phase 1.

Reacts to high-conviction news events (sentiment_score above ±0.80) on
configured symbols, placing directional trades based on headline sentiment.
Price data from the latest cached candle is used for entry; signals expire
15 minutes after the news publication time.

Usage
-----
    from strategies.phase1.news_catalyst import NewsCatalystStrategy

    strategy = NewsCatalystStrategy(news_feed=my_feed, symbols=["BTCUSDT"])
    signal = await strategy.on_news_event(news_item)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd
import structlog

from config.settings import get_settings
from strategies.base import Signal, Strategy

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum absolute sentiment score required to act
_SENTIMENT_THRESHOLD: float = 0.80

# Maximum age of a news item before it is treated as stale
_STALENESS_SECONDS: int = 120  # 2 minutes

# Signal time-to-live from the publication timestamp
_SIGNAL_TTL_SECONDS: int = 900  # 15 minutes

# Risk parameters expressed as fractions of entry price
_STOP_DISTANCE_PCT: float = 0.02   # 2% stop
_TARGET_DISTANCE_PCT: float = 0.05  # 5% target  → 2.5:1 R:R


# ---------------------------------------------------------------------------
# NewsFeed protocol (structural typing for dependency injection)
# ---------------------------------------------------------------------------


@runtime_checkable
class NewsFeed(Protocol):
    """Minimal interface required from a news-feed provider.

    Any object that implements ``get_latest_candle`` satisfies this protocol,
    enabling easy mocking in tests.
    """

    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Return the most recent OHLCV candle for *symbol* / *timeframe*.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTCUSDT"``.
        timeframe:
            Candle resolution, e.g. ``"5m"``.

        Returns
        -------
        pd.DataFrame | None
            Single-row DataFrame with columns ``open``, ``high``, ``low``,
            ``close``, ``volume``, or ``None`` when no data is available.
        """
        ...


# ---------------------------------------------------------------------------
# NewsCatalystStrategy
# ---------------------------------------------------------------------------


class NewsCatalystStrategy(Strategy):
    """Event-driven strategy that acts on high-sentiment news headlines.

    This strategy is *news-driven*, not candle-driven.  The :meth:`on_candle`
    method is a deliberate no-op; trading signals are produced exclusively by
    :meth:`on_news_event`.

    Signal logic
    ~~~~~~~~~~~~
    A signal is generated when **all** of the following hold:

    * ``abs(news_item['sentiment_score']) > 0.80`` — high-conviction sentiment.
    * ``news_item['published_at']`` is less than 2 minutes ago — not stale.
    * A current price is available from the 5-minute candle cache.

    Direction:

    * ``sentiment_score > 0`` → **BUY**.
    * ``sentiment_score < 0`` → **SELL** (short).

    Sizing reference levels:

    * Entry:  latest close price.
    * Stop:   2% from entry in the adverse direction.
    * Target: 5% from entry in the favourable direction → 2.5:1 R:R.

    The signal expiry timestamp (``published_at + 15 min``) is embedded in the
    ``indicators`` dict under the key ``"expires_at"`` so downstream consumers
    can discard stale signals before execution.

    Parameters
    ----------
    news_feed:
        Dependency-injected object satisfying the :class:`NewsFeed` protocol.
        Used to look up the latest candle price for the affected symbol.
    symbols:
        List of symbols this strategy monitors.  Defaults to the same default
        list used by other Phase-1 strategies.
    """

    def __init__(
        self,
        news_feed: Any,
        symbols: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        settings = get_settings()
        self._news_feed: Any = news_feed
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
        return "news_catalyst"

    @property
    def timeframes(self) -> list[str]:
        # 5-minute candles are used as the price source; this strategy does
        # not emit signals on candle events.
        return ["5m"]

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    # ------------------------------------------------------------------
    # Candle handler — intentional no-op
    # ------------------------------------------------------------------

    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Optional[Signal]:
        """No-op: this strategy is news-driven, not candle-driven.

        Subscribes to ``5m`` candles purely to keep the internal price cache
        warm via the data feed infrastructure; it does not generate signals here.
        """
        return None

    # ------------------------------------------------------------------
    # News event handler
    # ------------------------------------------------------------------

    async def on_news_event(self, news_item: dict) -> Optional[Signal]:
        """Evaluate a news event and return a Signal if entry conditions are met.

        Parameters
        ----------
        news_item:
            Dictionary with the following keys:

            ``symbol`` *(str)*
                Instrument symbol, e.g. ``"BTCUSDT"``.
            ``headline`` *(str)*
                Raw headline text (stored in indicators for audit).
            ``sentiment_score`` *(float)*
                Sentiment in ``[-1.0, 1.0]``.  Values beyond ±0.80 trigger entry.
            ``published_at`` *(datetime)*
                UTC publication timestamp.  Must be timezone-aware.
            ``source`` *(str)*
                News source identifier (stored in indicators for audit).

        Returns
        -------
        Signal | None
            A populated :class:`Signal` when all conditions fire, otherwise
            ``None`` with the reason logged at DEBUG level.
        """
        symbol: str = news_item.get("symbol", "")
        headline: str = news_item.get("headline", "")
        sentiment_score: float = float(news_item.get("sentiment_score", 0.0))
        published_at: datetime = news_item.get("published_at")
        source: str = news_item.get("source", "unknown")

        bound_log = self._log.bind(
            symbol=symbol,
            sentiment=sentiment_score,
            source=source,
        )

        # ── Guard: symbol must be in our watch list ───────────────────────────
        if symbol not in self._symbols:
            bound_log.debug("symbol_not_monitored")
            return None

        # ── Guard: sentiment threshold ────────────────────────────────────────
        if abs(sentiment_score) <= _SENTIMENT_THRESHOLD:
            bound_log.debug(
                "sentiment_below_threshold",
                threshold=_SENTIMENT_THRESHOLD,
            )
            return None

        # ── Guard: staleness check ────────────────────────────────────────────
        if published_at is None:
            bound_log.warning("news_item_missing_published_at")
            return None

        # Normalise to UTC-aware datetime
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        now_utc = datetime.now(tz=timezone.utc)
        age_seconds = (now_utc - published_at).total_seconds()

        if age_seconds > _STALENESS_SECONDS:
            bound_log.debug(
                "news_item_stale",
                age_seconds=round(age_seconds, 1),
                max_age_seconds=_STALENESS_SECONDS,
            )
            return None

        # ── Resolve current price from candle cache ───────────────────────────
        entry_price = self._get_current_price(symbol)
        if entry_price is None:
            bound_log.warning(
                "no_price_available",
                symbol=symbol,
                timeframe="5m",
            )
            return None

        # ── Determine direction ───────────────────────────────────────────────
        side = "BUY" if sentiment_score > 0.0 else "SELL"

        # ── Entry / stop / target ─────────────────────────────────────────────
        if side == "BUY":
            stop_price = entry_price * (1.0 - _STOP_DISTANCE_PCT)
            target_price = entry_price * (1.0 + _TARGET_DISTANCE_PCT)
        else:
            stop_price = entry_price * (1.0 + _STOP_DISTANCE_PCT)
            target_price = entry_price * (1.0 - _TARGET_DISTANCE_PCT)

        rr = self._rr_ratio(entry_price, stop_price, target_price)

        # ── Signal expiry — embedded in indicators for downstream consumers ───
        expires_at: datetime = published_at + timedelta(seconds=_SIGNAL_TTL_SECONDS)

        indicator_snapshot: dict = {
            "headline": headline,
            "sentiment_score": round(sentiment_score, 6),
            "source": source,
            "published_at": published_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "age_seconds": round(age_seconds, 1),
            "stop_distance_pct": _STOP_DISTANCE_PCT,
            "target_distance_pct": _TARGET_DISTANCE_PCT,
        }

        # Confidence directly reflects the absolute sentiment magnitude
        confidence = round(min(abs(sentiment_score), 1.0), 6)

        # Signal strength mirrors confidence for a news-driven strategy
        # (no multi-condition aggregation — a single strong score drives action)
        strength = confidence

        signal = Signal(
            strategy=self.name,
            symbol=symbol,
            side=side,
            strength=strength,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            rr_ratio=rr,
            confidence=confidence,
            indicators=indicator_snapshot,
            timestamp=now_utc,
            timeframe="5m",
        )

        bound_log.info(
            "signal.generated",
            side=side,
            strength=strength,
            confidence=confidence,
            rr=rr,
            entry=entry_price,
            stop=stop_price,
            target=target_price,
            expires_at=expires_at.isoformat(),
            actionable=signal.is_actionable,
        )

        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Retrieve the latest close price for *symbol* from the news feed.

        Queries the injected ``news_feed`` for the most recent ``5m`` candle.
        Returns ``None`` when no data is available so that the caller can
        handle the absence gracefully.

        Parameters
        ----------
        symbol:
            Instrument symbol to look up.

        Returns
        -------
        float | None
        """
        try:
            candle_df = self._news_feed.get_latest_candle(symbol, "5m")
            if candle_df is None or candle_df.empty:
                return None
            return float(candle_df["close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001 — never crash on feed errors
            self._log.warning(
                "price_lookup_error",
                symbol=symbol,
                error=str(exc),
            )
            return None

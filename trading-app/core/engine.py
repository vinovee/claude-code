"""
TradingEngine — the central event loop that wires every subsystem together.

Architecture
------------
- Subscribes to Redis 'candles' pub/sub channel.
- Fans out each closed candle to all registered strategies.
- Runs pre-trade risk checks before placing any order.
- Delegates order placement to an OrderRouter (or paper broker).
- Updates Portfolio on every fill.
- Emits Prometheus metrics and Telegram notifications.
- Runs three background tasks: price-update loop, midnight reset, and
  portfolio snapshot persister.

Usage
-----
    engine = TradingEngine(
        settings=get_settings(),
        db=db,
        cache=cache,
        portfolio=portfolio,
        risk_manager=RiskManager(),
        circuit_breaker=circuit_breaker,
        position_sizer=PositionSizer(),
        order_router=order_router,
        strategies=active_strategies,
        broker=PaperBroker(),
        metrics=TradingMetrics(),
    )
    await engine.start()
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

from config.settings import get_settings
from execution.brokers.base import BrokerBase, Order
from monitoring.logger import (
    ORDER_FILLED,
    ORDER_PLACED,
    SIGNAL_GENERATED,
    SYSTEM_START,
    SYSTEM_STOP,
    get_logger,
)
from strategies.base import Signal, Strategy

if TYPE_CHECKING:
    from data.storage.redis_cache import RedisCache
    from data.storage.timescale import AsyncTimescaleDB
    from monitoring.metrics import TradingMetrics
    from monitoring.telegram_bot import TelegramNotifier
    from risk.manager import RiskManager
    from risk.position_sizer import PositionSizer
    from core.portfolio import Portfolio

log = get_logger(__name__)

__all__ = ["TradingEngine"]


# ---------------------------------------------------------------------------
# Placeholder stubs for CircuitBreaker and OrderRouter so that the engine
# compiles even if those dedicated modules have not yet been created.  A real
# implementation should replace these with properly typed classes.
# ---------------------------------------------------------------------------


class _CircuitBreakerProtocol:
    """Minimal duck-type interface expected by TradingEngine."""

    def allow_live_trading(self) -> bool:
        return True

    def allow_paper_trading(self) -> bool:
        return True


class _OrderRouterProtocol:
    """Minimal duck-type interface expected by TradingEngine."""

    async def route(self, signal: Signal, qty: float, **kwargs: Any) -> Order:
        raise NotImplementedError("OrderRouter.route must be implemented")


# ---------------------------------------------------------------------------
# TradingEngine
# ---------------------------------------------------------------------------


class TradingEngine:
    """The main event loop for the autonomous trading application.

    Constructor uses dependency injection so every subsystem can be
    substituted in tests or for different deployment modes.

    Parameters
    ----------
    settings:
        Application settings instance.
    db:
        Connected TimescaleDB async client.
    cache:
        Connected RedisCache async client.
    portfolio:
        Mutable portfolio state tracker.
    risk_manager:
        Pre-trade risk gate.
    circuit_breaker:
        Global kill-switch that can block live or paper trading.
    position_sizer:
        Half-Kelly position sizing calculator.
    order_router:
        Routes approved signals to the correct broker.
    strategies:
        List of active :class:`~strategies.base.Strategy` instances.
    broker:
        Primary broker adapter (paper or live).
    metrics:
        Prometheus metrics wrapper.
    telegram:
        Optional Telegram notifier (``None`` disables notifications).
    """

    def __init__(
        self,
        settings: Any,
        db: "AsyncTimescaleDB",
        cache: "RedisCache",
        portfolio: "Portfolio",
        risk_manager: "RiskManager",
        circuit_breaker: Any,
        position_sizer: "PositionSizer",
        order_router: Any,
        strategies: list[Strategy],
        broker: BrokerBase,
        metrics: "TradingMetrics",
        telegram: "TelegramNotifier | None" = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._cache = cache
        self._portfolio = portfolio
        self._risk_manager = risk_manager
        self._circuit_breaker = circuit_breaker
        self._position_sizer = position_sizer
        self._order_router = order_router
        self._strategies = strategies
        self._broker = broker
        self._metrics = metrics
        self._telegram = telegram

        self._background_tasks: list[asyncio.Task[Any]] = []
        self._running = False

        self._log = log.bind(component="TradingEngine")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise all dependencies and launch background tasks.

        Steps
        -----
        1. Connect TimescaleDB and Redis.
        2. Subscribe to Redis 'candles' channel.
        3. Start the Binance WebSocket feed (or paper feed heartbeat).
        4. Launch background maintenance tasks.
        5. Start broker order-update stream.
        """
        self._log.info(SYSTEM_START, event_type=SYSTEM_START)
        self._running = True

        # ── 1. Persistence / cache connections ───────────────────────────────
        await self._db.connect()
        await self._cache.connect()

        # ── 2. Subscribe to candle events ────────────────────────────────────
        await self._cache.subscribe("candles", self._on_candle)

        # ── 3. Start broker market-data feed ─────────────────────────────────
        await self._start_feed()

        # ── 4. Background maintenance tasks ──────────────────────────────────
        self._background_tasks = [
            asyncio.create_task(
                self._update_prices_loop(), name="engine:update-prices"
            ),
            asyncio.create_task(
                self._midnight_reset_loop(), name="engine:midnight-reset"
            ),
            asyncio.create_task(
                self._snapshot_loop(), name="engine:snapshot"
            ),
        ]

        # ── 5. Start broker order-update stream ──────────────────────────────
        order_stream_task = asyncio.create_task(
            self._broker.stream_order_updates(self._on_order_filled),
            name="engine:order-stream",
        )
        self._background_tasks.append(order_stream_task)

        self._log.info(
            "engine.started",
            paper_trading=self._settings.paper_trading,
            strategies=[s.name for s in self._strategies],
            phase=self._portfolio.phase,
        )

    async def stop(self) -> None:
        """Gracefully stop the engine and release all resources."""
        self._log.info(SYSTEM_STOP, event_type=SYSTEM_STOP)
        self._running = False

        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()

        results = await asyncio.gather(*self._background_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                self._log.warning("engine.task_error_on_shutdown", error=str(result))

        self._background_tasks.clear()

        # Close broker connection
        if hasattr(self._broker, "close"):
            try:
                await self._broker.close()  # type: ignore[attr-defined]
            except Exception:
                self._log.exception("engine.broker_close_failed")

        # Close Redis and DB
        await self._cache.close()
        await self._db.close()

        self._log.info("engine.stopped")

    # ── Feed startup ─────────────────────────────────────────────────────────

    async def _start_feed(self) -> None:
        """Start the appropriate market-data feed based on trading mode."""
        if self._settings.paper_trading:
            # PaperBroker prices are updated by the price-update loop.
            self._log.info("engine.feed.paper_mode")
            return

        # Live mode: start Binance WebSocket feed
        try:
            from data.feeds.binance_ws import BinanceFeed

            symbols: list[str] = []
            timeframes: list[str] = []
            for strategy in self._strategies:
                symbols.extend(strategy.symbols)
                timeframes.extend(strategy.timeframes)
            symbols = list(dict.fromkeys(symbols))        # deduplicate, preserve order
            timeframes = list(dict.fromkeys(timeframes))

            feed = BinanceFeed(
                db=self._db,
                cache=self._cache,
                api_key=self._settings.binance_api_key,
                api_secret=self._settings.binance_secret_key,
            )
            feed_task = asyncio.create_task(
                feed.start(symbols, timeframes), name="engine:binance-feed"
            )
            self._background_tasks.append(feed_task)
            self._log.info(
                "engine.feed.binance_started",
                symbols=symbols,
                timeframes=timeframes,
            )
        except ImportError:
            self._log.warning("engine.feed.binance_unavailable")

    # ── Candle handler ───────────────────────────────────────────────────────

    async def _on_candle(self, message: dict[str, Any]) -> None:
        """Process a closed candle published to the Redis 'candles' channel.

        Expected message shape
        ----------------------
        {
            "symbol":    "BTCUSDT",
            "timeframe": "5m",
            "ts":        1717171200000,   # epoch ms
            "open":      65100.0,
            "high":      65250.0,
            "low":       65050.0,
            "close":     65200.0,
            "volume":    123.45,
        }
        """
        try:
            symbol: str = message["symbol"]
            timeframe: str = message["timeframe"]
            close_price: float = float(message["close"])
        except (KeyError, ValueError, TypeError) as exc:
            self._log.warning("engine.candle.parse_failed", error=str(exc), message=message)
            return

        # ── Update PaperBroker price map ─────────────────────────────────────
        if self._settings.paper_trading and hasattr(self._broker, "update_prices"):
            self._broker.update_prices({symbol: close_price})  # type: ignore[attr-defined]

        # ── Fan out to strategies ─────────────────────────────────────────────
        for strategy in self._strategies:
            if symbol not in strategy.symbols:
                continue
            if timeframe not in strategy.timeframes:
                continue

            try:
                df = await self._fetch_candles(symbol, timeframe)
                if df.empty:
                    self._log.debug(
                        "engine.candle.no_history",
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy=strategy.name,
                    )
                    continue

                signal: Signal | None = await strategy.on_candle(symbol, timeframe, df)

                if signal is not None:
                    self._log.info(
                        SIGNAL_GENERATED,
                        event_type=SIGNAL_GENERATED,
                        strategy=strategy.name,
                        symbol=symbol,
                        timeframe=timeframe,
                        side=signal.side,
                        strength=round(signal.strength, 4),
                        rr=round(signal.rr_ratio, 2),
                        actionable=signal.is_actionable,
                    )
                    self._metrics.signal_strength.labels(
                        strategy=strategy.name, symbol=symbol
                    ).set(signal.strength)

                    if signal.is_actionable:
                        await self._process_signal(signal)

            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception(
                    "engine.strategy.error",
                    strategy=strategy.name,
                    symbol=symbol,
                    timeframe=timeframe,
                )

    async def _fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch the last *limit* candles from the DB (or cache as fallback)."""
        try:
            df = await self._db.get_candles(symbol, timeframe, limit=limit)
            return df
        except Exception:
            self._log.exception(
                "engine.candles.db_fetch_failed",
                symbol=symbol,
                timeframe=timeframe,
            )
            return pd.DataFrame()

    # ── Signal processing ────────────────────────────────────────────────────

    async def _process_signal(self, signal: Signal) -> None:
        """Run risk checks and place an order for an actionable signal.

        Steps
        -----
        1. Check circuit breaker.
        2. Calculate position size.
        3. Run risk manager checks.
        4. Log all check results.
        5. Place order via order router if approved.
        6. Emit metrics.
        """
        # ── 1. Circuit-breaker gate ───────────────────────────────────────────
        is_paper = self._settings.paper_trading
        if is_paper:
            if not self._circuit_breaker.allow_paper_trading():
                self._log.warning(
                    "engine.signal.circuit_breaker_blocked",
                    strategy=signal.strategy,
                    symbol=signal.symbol,
                    mode="paper",
                )
                return
        else:
            if not self._circuit_breaker.allow_live_trading():
                self._log.warning(
                    "engine.signal.circuit_breaker_blocked",
                    strategy=signal.strategy,
                    symbol=signal.symbol,
                    mode="live",
                )
                if self._telegram:
                    await self._telegram.circuit_breaker_alert(
                        reason="Circuit breaker blocked live trade"
                    )
                return

        # ── 2. Position sizing ────────────────────────────────────────────────
        portfolio_state = self._portfolio.to_dict()
        equity: float = portfolio_state["total_equity"]
        peak_equity: float = portfolio_state["peak_equity"]

        try:
            sizing = self._position_sizer.calculate(
                equity=equity,
                peak_equity=peak_equity,
                win_rate=max(portfolio_state["win_rate"], 0.01),
                avg_win_pct=max(portfolio_state["avg_win_pct"], 0.001),
                avg_loss_pct=max(portfolio_state["avg_loss_pct"], 0.001),
                price=signal.entry_price,
            )
        except ValueError as exc:
            self._log.warning(
                "engine.signal.sizing_failed",
                strategy=signal.strategy,
                symbol=signal.symbol,
                error=str(exc),
            )
            return

        size_gbp: float = sizing["position_size_gbp"]
        qty: float = sizing["qty_units"]

        # ── 3. Risk manager approval ──────────────────────────────────────────
        approved, risk_results = await self._risk_manager.approve_trade(
            symbol=signal.symbol,
            entry=signal.entry_price,
            stop=signal.stop_price,
            target=signal.target_price,
            size_gbp=size_gbp,
            side=signal.side,
            portfolio_state=portfolio_state,
            spread_bps=5.0,  # fallback; real spread from order book in future
        )

        # ── 4. Log all risk check results ─────────────────────────────────────
        for result in risk_results:
            level = "info" if result.passed else "warning"
            getattr(self._log, level)(
                "engine.risk_check",
                check=result.check_name,
                status=result.status,
                action=result.action,
                reason=result.reason,
                strategy=signal.strategy,
                symbol=signal.symbol,
            )
            # Emit Prometheus counter for non-PASS results
            if result.status != "PASS":
                self._metrics.risk_checks_triggered.labels(
                    check_name=result.check_name,
                    action=result.action,
                ).inc()
            # Send Telegram alert for WARN/FAIL
            if result.status in ("WARN", "FAIL") and self._telegram:
                await self._telegram.risk_alert(
                    check_name=result.check_name,
                    action=result.action,
                    details={
                        "reason": result.reason,
                        "strategy": signal.strategy,
                        "symbol": signal.symbol,
                    },
                )

        # ── 5. Place order if approved ────────────────────────────────────────
        label = "approved" if approved else "rejected"
        self._metrics.trades_total.labels(
            strategy=signal.strategy,
            symbol=signal.symbol,
            side=signal.side,
            phase=self._portfolio.phase,
        ).inc()

        if not approved:
            self._log.info(
                "engine.signal.rejected",
                strategy=signal.strategy,
                symbol=signal.symbol,
                side=signal.side,
            )
            return

        try:
            order = await self._order_router.route(signal, qty)
            self._log.info(
                ORDER_PLACED,
                event_type=ORDER_PLACED,
                order_id=order.order_id,
                symbol=signal.symbol,
                side=signal.side,
                qty=qty,
                strategy=signal.strategy,
            )
            # Track in open orders
            self._portfolio.open_orders[order.order_id] = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "strategy": signal.strategy,
                "created_at": order.created_at.isoformat() if order.created_at else None,
            }
        except Exception:
            self._log.exception(
                "engine.signal.order_placement_failed",
                strategy=signal.strategy,
                symbol=signal.symbol,
            )
            self._metrics.broker_api_errors.labels(
                broker="router", error_type="order_placement"
            ).inc()

    # ── Order fill handler ───────────────────────────────────────────────────

    async def _on_order_filled(self, order: Order) -> None:
        """Handle a confirmed order fill from the broker stream.

        Steps
        -----
        1. Update portfolio state.
        2. Save the order to the database.
        3. Update Prometheus metrics.
        4. Send Telegram notification.
        """
        if order.status != "FILLED":
            return

        # ── 1. Update portfolio ───────────────────────────────────────────────
        await self._portfolio.on_order_filled(order)

        # ── 2. Persist order to DB ────────────────────────────────────────────
        try:
            await self._db.save_trade({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "fill_price": order.fill_price,
                "fill_qty": order.fill_qty,
                "fee": order.fee,
                "broker": order.broker,
                "order_type": order.order_type,
                "filled_at": order.filled_at,
                "created_at": order.created_at,
                "status": order.status,
            })
        except Exception:
            self._log.exception("engine.order_db_save_failed", order_id=order.order_id)

        # ── 3. Metrics ────────────────────────────────────────────────────────
        portfolio_state = self._portfolio.to_dict()
        self._metrics.update_portfolio(portfolio_state)
        self._metrics.open_positions.labels(
            strategy=order.meta.get("strategy", "unknown")
        ).set(portfolio_state["open_positions"])

        # ── 4. Telegram fill notification ─────────────────────────────────────
        self._log.info(
            ORDER_FILLED,
            event_type=ORDER_FILLED,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=order.fill_price,
            fill_qty=order.fill_qty,
            fee=order.fee,
        )

        if self._telegram:
            await self._telegram.send(
                f"Order filled: {order.side} {order.fill_qty} {order.symbol} "
                f"@ {order.fill_price:.4f} GBP | Fee: {order.fee:.4f}"
            )

    # ── Background tasks ─────────────────────────────────────────────────────

    async def _update_prices_loop(self) -> None:
        """Refresh portfolio unrealised P&L from latest cache prices every 5s."""
        while self._running:
            try:
                prices: dict[str, float] = {}
                for symbol in list(self._portfolio.positions.keys()):
                    # Try to fetch the latest close from Redis for each symbol
                    for strategy in self._strategies:
                        if symbol in strategy.symbols and strategy.timeframes:
                            candle = await self._cache.get_candle(
                                symbol, strategy.timeframes[0]
                            )
                            if candle and "close" in candle:
                                prices[symbol] = float(candle["close"])
                                break

                if prices:
                    await self._portfolio.on_price_update(prices)
                    self._metrics.update_portfolio(self._portfolio.to_dict())

            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("engine.update_prices_loop.error")

            await asyncio.sleep(5)

    async def _midnight_reset_loop(self) -> None:
        """Wait until the next 00:00 UTC then reset daily portfolio stats."""
        while self._running:
            now = datetime.now(tz=timezone.utc)
            # Seconds until the next midnight UTC
            seconds_until_midnight = (
                (24 * 3600)
                - (now.hour * 3600 + now.minute * 60 + now.second)
            )

            try:
                await asyncio.sleep(seconds_until_midnight)
            except asyncio.CancelledError:
                raise

            if not self._running:
                break

            self._portfolio.reset_daily_stats()

            if self._telegram:
                await self._telegram.daily_summary(self._portfolio)

    async def _snapshot_loop(self) -> None:
        """Save portfolio snapshots to TimescaleDB every 60 seconds."""
        while self._running:
            await asyncio.sleep(60)
            if not self._running:
                break
            try:
                await self._portfolio.save_snapshot(self._db)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._log.exception("engine.snapshot_loop.error")

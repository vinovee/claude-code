"""
Order router — selects order type and broker, then dispatches orders.

The router applies a deterministic decision tree based on trading urgency,
strategy type, and current spread before delegating to either the live
:class:`~execution.brokers.binance_broker.BinanceBroker` or the simulated
:class:`~execution.brokers.paper_broker.PaperBroker`.

If a *stop_price* is provided the router places a protective STOP_MARKET
order immediately after the primary order is confirmed filled.

Prometheus metric
-----------------
``order_execution_latency_ms`` — histogram of wall-clock time from
``route()`` entry to primary order return.

Usage
-----
    from execution.order_router import OrderRouter
    from execution.brokers.paper_broker import PaperBroker
    from execution.brokers.binance_broker import BinanceBroker

    broker = PaperBroker()          # or BinanceBroker(...)
    router = OrderRouter(broker)

    order = await router.route(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.01,
        strategy_type="MOMENTUM",
        urgency="HIGH",
        spread_bps=5.0,
    )
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog
from prometheus_client import Histogram

from config.settings import get_settings
from execution.brokers.base import BrokerBase, Order
from execution.brokers.paper_broker import PaperBroker

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

try:
    ORDER_EXECUTION_LATENCY_MS: Histogram = Histogram(
        "order_execution_latency_ms",
        "Wall-clock time from route() entry to primary order return (ms)",
        ["broker", "strategy_type", "urgency", "order_type"],
        buckets=[5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000],
    )
except ValueError:
    from prometheus_client import REGISTRY  # type: ignore[attr-defined]
    ORDER_EXECUTION_LATENCY_MS = REGISTRY._names_to_collectors[  # type: ignore[index]
        "order_execution_latency_ms"
    ]

# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------

# Spread threshold below which MARKET is always preferred (fast fill)
_MARKET_SPREAD_BPS_THRESHOLD: float = 10.0

# Strategy types that prefer LIMIT orders
_LIMIT_PREFERRED_STRATEGIES: frozenset[str] = frozenset({"MEAN_REVERSION"})


# ---------------------------------------------------------------------------
# OrderRouter
# ---------------------------------------------------------------------------


class OrderRouter:
    """Routes trade signals to the appropriate broker with the correct order type.

    Parameters
    ----------
    broker:
        The broker adapter to use.  If ``None``, the router selects
        :class:`~execution.brokers.paper_broker.PaperBroker` when
        ``settings.paper_trading`` is ``True``, otherwise raises
        ``RuntimeError`` (the caller must supply a live broker explicitly).
    """

    def __init__(self, broker: BrokerBase | None = None) -> None:
        self._settings = get_settings()
        self._broker: BrokerBase

        if broker is not None:
            self._broker = broker
        elif self._settings.paper_trading:
            self._broker = PaperBroker()
        else:
            raise RuntimeError(
                "OrderRouter requires an explicit broker when paper_trading=False"
            )

        self._log = logger.bind(
            router="OrderRouter",
            broker=self._broker.BROKER_NAME,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def route(
        self,
        symbol: str,
        side: str,
        qty: float,
        strategy_type: str,
        urgency: str,
        spread_bps: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Determine the optimal order type and dispatch to the broker.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTCUSDT"``.
        side:
            ``"BUY"`` or ``"SELL"``.
        qty:
            Quantity in base-asset units.
        strategy_type:
            Originating strategy: ``"MOMENTUM"``, ``"MEAN_REVERSION"``, or
            ``"NEWS"``.
        urgency:
            Execution urgency: ``"HIGH"``, ``"NORMAL"``, or ``"LOW"``.
        spread_bps:
            Current bid/ask spread in basis points.
        limit_price:
            Limit price to use when order type resolves to LIMIT.
        stop_price:
            If provided, a protective STOP_MARKET order is placed
            immediately after the primary fill.

        Returns
        -------
        Order
            The primary (entry) order.  The protective stop order, if
            placed, can be found in the returned order's ``meta`` dict
            under the key ``"stop_order"``.
        """
        t_start = time.monotonic()

        order_type = self._select_order_type(
            strategy_type=strategy_type,
            urgency=urgency,
            spread_bps=spread_bps,
            limit_price=limit_price,
        )

        self._log.info(
            "order_route_decision",
            symbol=symbol,
            side=side,
            qty=qty,
            strategy_type=strategy_type,
            urgency=urgency,
            spread_bps=spread_bps,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        primary_order = await self._broker.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=limit_price if order_type == "LIMIT" else None,
            stop_price=None,  # primary order never carries a stop price
        )

        latency_ms = (time.monotonic() - t_start) * 1_000
        ORDER_EXECUTION_LATENCY_MS.labels(
            broker=self._broker.BROKER_NAME,
            strategy_type=strategy_type,
            urgency=urgency,
            order_type=order_type,
        ).observe(latency_ms)

        self._log.info(
            "primary_order_placed",
            order_id=primary_order.order_id,
            symbol=symbol,
            status=primary_order.status,
            fill_price=primary_order.fill_price,
            latency_ms=round(latency_ms, 2),
        )

        # ------------------------------------------------------------------
        # Protective stop — placed only after primary fill is confirmed
        # ------------------------------------------------------------------
        if stop_price is not None:
            stop_order = await self._place_stop_order(
                primary_order=primary_order,
                stop_price=stop_price,
            )
            # Attach to primary order meta so callers have a single return value
            primary_order.meta["stop_order"] = stop_order

        return primary_order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_order_type(
        self,
        strategy_type: str,
        urgency: str,
        spread_bps: float,
        limit_price: float | None,
    ) -> str:
        """Apply the order-type decision tree.

        Rules (evaluated top-to-bottom, first match wins):

        1. HIGH urgency  → MARKET (speed matters most).
        2. Spread < 10 bps → MARKET (cheap to cross the spread).
        3. MEAN_REVERSION strategy → LIMIT (price must come to us).
        4. Default → MARKET.

        A LIMIT order type is only returned when *limit_price* is provided;
        if it is missing despite the rules suggesting LIMIT, the router
        falls back to MARKET and logs a warning.
        """
        # Rule 1: urgency HIGH always takes market
        if urgency == "HIGH":
            return "MARKET"

        # Rule 2: tight spread — cheap to cross
        if spread_bps < _MARKET_SPREAD_BPS_THRESHOLD:
            return "MARKET"

        # Rule 3: mean-reversion strategies prefer to be filled passively
        if strategy_type in _LIMIT_PREFERRED_STRATEGIES:
            if limit_price is not None:
                return "LIMIT"
            self._log.warning(
                "limit_price_missing_for_mean_reversion",
                strategy_type=strategy_type,
                fallback="MARKET",
            )
            return "MARKET"

        # Rule 4: default
        return "MARKET"

    async def _place_stop_order(
        self,
        primary_order: Order,
        stop_price: float,
    ) -> Order:
        """Place a STOP_MARKET order opposite to *primary_order*.

        The stop order is placed regardless of whether the primary order is
        already filled — for paper broker this is immediate; for live
        brokers the primary order may still be PENDING (GTC limit), in
        which case the caller should track the stop separately.
        """
        # Protective stop is always on the opposite side
        stop_side = "SELL" if primary_order.side == "BUY" else "BUY"

        self._log.info(
            "placing_stop_order",
            primary_order_id=primary_order.order_id,
            symbol=primary_order.symbol,
            stop_side=stop_side,
            qty=primary_order.qty,
            stop_price=stop_price,
        )

        try:
            stop_order = await self._broker.place_order(
                symbol=primary_order.symbol,
                side=stop_side,
                qty=primary_order.qty,
                order_type="STOP_MARKET",
                price=None,
                stop_price=stop_price,
            )
            self._log.info(
                "stop_order_placed",
                stop_order_id=stop_order.order_id,
                symbol=primary_order.symbol,
                stop_price=stop_price,
            )
            return stop_order
        except Exception as exc:  # noqa: BLE001
            self._log.error(
                "stop_order_failed",
                primary_order_id=primary_order.order_id,
                symbol=primary_order.symbol,
                stop_price=stop_price,
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def broker(self) -> BrokerBase:
        """The underlying broker adapter."""
        return self._broker

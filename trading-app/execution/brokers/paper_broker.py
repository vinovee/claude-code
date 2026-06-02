"""
Paper-trading broker adapter.

Simulates order execution entirely in-memory — no real exchange connection.
Designed to mirror the ``BinanceBroker`` interface so that dashboards and
strategy logic are unaware of whether they are running live or in simulation.

Features
--------
- Immediate MARKET fill at current price + configurable slippage (0.1 %).
- LIMIT / STOP_MARKET orders queued and triggered by :meth:`update_prices`.
- Fee = 0.1 % of order value (standard Binance taker fee).
- Cash balance starts at ``settings.initial_capital_gbp`` and is updated on
  every fill.
- Emits the same Prometheus counters/histograms as :class:`BinanceBroker` so
  dashboards work without modification.
- Structured logging via structlog.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import structlog
from prometheus_client import Counter, Histogram

from config.settings import get_settings
from .base import AccountInfo, BrokerBase, Order, Position

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (shared registry labels match BinanceBroker exactly so
# a single Grafana dashboard covers both brokers).
# ---------------------------------------------------------------------------

# Guard against duplicate registration when the module is reloaded in tests.
try:
    ORDER_PLACED_TOTAL: Counter = Counter(
        "order_placed_total",
        "Total number of orders placed",
        ["broker", "symbol", "side", "order_type"],
    )
except ValueError:
    from prometheus_client import REGISTRY  # type: ignore[attr-defined]
    ORDER_PLACED_TOTAL = REGISTRY._names_to_collectors["order_placed_total"]  # type: ignore[index]

try:
    ORDER_FILL_LATENCY_MS: Histogram = Histogram(
        "order_fill_latency_ms",
        "Time from order creation to fill in milliseconds",
        ["broker", "symbol"],
        buckets=[10, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000],
    )
except ValueError:
    from prometheus_client import REGISTRY  # type: ignore[attr-defined]
    ORDER_FILL_LATENCY_MS = REGISTRY._names_to_collectors["order_fill_latency_ms"]  # type: ignore[index]

try:
    BROKER_API_ERRORS_TOTAL: Counter = Counter(
        "broker_api_errors_total",
        "Total number of broker API errors",
        ["broker", "operation"],
    )
except ValueError:
    from prometheus_client import REGISTRY  # type: ignore[attr-defined]
    BROKER_API_ERRORS_TOTAL = REGISTRY._names_to_collectors["broker_api_errors_total"]  # type: ignore[index]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SLIPPAGE_RATE: float = 0.001   # 0.1 % market-order slippage
_FEE_RATE: float = 0.001        # 0.1 % taker fee (Binance standard)
_BROKER_NAME: str = "paper"


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------


class PaperBroker(BrokerBase):
    """In-memory simulated broker for paper trading.

    Parameters
    ----------
    initial_capital_gbp:
        Starting cash balance in GBP.  Defaults to
        ``settings.initial_capital_gbp``.
    """

    BROKER_NAME = _BROKER_NAME

    def __init__(self, initial_capital_gbp: float | None = None) -> None:
        settings = get_settings()
        self._cash: float = (
            initial_capital_gbp
            if initial_capital_gbp is not None
            else settings.initial_capital_gbp
        )
        # { order_id: Order }
        self._orders: dict[str, Order] = {}
        # Pending LIMIT / STOP_MARKET orders awaiting price trigger
        self._pending_orders: list[Order] = []
        # { symbol: Position }
        self._positions: dict[str, Position] = {}
        # Historical fills for reporting
        self._fills: list[Order] = []
        # Latest known prices { symbol: float }
        self._prices: dict[str, float] = {}
        # Callbacks registered via stream_order_updates
        self._update_callbacks: list[Callable[[Order], Awaitable[None] | None]] = []
        self._log = logger.bind(broker=self.BROKER_NAME)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update the broker's internal price map and trigger pending orders.

        Call this on every candle close (or tick) so that LIMIT and
        STOP_MARKET orders are evaluated against fresh prices.

        Parameters
        ----------
        prices:
            Mapping of ``symbol -> current_price``.
        """
        self._prices.update(prices)
        # Update unrealised PnL for open positions
        for symbol, price in prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                if pos.side == "LONG":
                    pnl = (price - pos.avg_entry_price) * pos.qty
                else:
                    pnl = (pos.avg_entry_price - price) * pos.qty
                self._positions[symbol] = Position(
                    symbol=pos.symbol,
                    side=pos.side,
                    qty=pos.qty,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=price,
                    unrealised_pnl=pnl,
                )

        # Check pending orders — collect triggers first to avoid mutation
        # during iteration.
        triggered: list[Order] = []
        remaining: list[Order] = []
        for order in self._pending_orders:
            price = self._prices.get(order.symbol)
            if price is None:
                remaining.append(order)
                continue
            if self._should_trigger(order, price):
                triggered.append(order)
            else:
                remaining.append(order)

        self._pending_orders = remaining

        # Fill triggered orders synchronously (no async needed in price callback)
        for order in triggered:
            fill_price = self._prices.get(order.symbol, 0.0)
            self._execute_fill(order, fill_price)

    # ------------------------------------------------------------------
    # BrokerBase implementation
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: float | None = None,
        stop_price: float | None = None,
        **kwargs: Any,
    ) -> Order:
        """Simulate order placement.

        MARKET orders fill immediately at current price ± slippage.
        LIMIT / STOP_MARKET orders are queued until :meth:`update_prices`
        triggers them.
        """
        t_start = time.monotonic()
        order_id = str(uuid.uuid4())
        now = _now()

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            status="PENDING",
            fill_price=None,
            fill_qty=None,
            fee=0.0,
            created_at=now,
            filled_at=None,
            broker=self.BROKER_NAME,
        )

        self._orders[order_id] = order

        self._log.info(
            "placing_order",
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
        )

        ORDER_PLACED_TOTAL.labels(
            broker=self.BROKER_NAME,
            symbol=symbol,
            side=side,
            order_type=order_type,
        ).inc()

        if order_type == "MARKET":
            current_price = self._prices.get(symbol)
            if current_price is None:
                # No price available — reject the order
                order.status = "REJECTED"
                self._log.warning(
                    "order_rejected_no_price",
                    order_id=order_id,
                    symbol=symbol,
                )
                await self._notify_callbacks(order)
                return order

            fill_price = self._apply_slippage(current_price, side)
            self._execute_fill(order, fill_price)

            latency_ms = (time.monotonic() - t_start) * 1_000
            ORDER_FILL_LATENCY_MS.labels(
                broker=self.BROKER_NAME, symbol=symbol
            ).observe(latency_ms)

        else:
            # LIMIT or STOP_MARKET — queue for later triggering
            self._pending_orders.append(order)
            self._log.info(
                "order_queued",
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
            )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending (unfilled) order."""
        order = self._orders.get(order_id)
        if order is None:
            return False
        if order.status != "PENDING":
            return False

        order.status = "CANCELLED"
        # Remove from pending queue
        self._pending_orders = [
            o for o in self._pending_orders if o.order_id != order_id
        ]
        self._log.info("order_cancelled", order_id=order_id, symbol=order.symbol)
        await self._notify_callbacks(order)
        return True

    async def get_order(self, order_id: str) -> Order:
        """Return the current state of an order by ID."""
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id!r} not found in paper broker")
        return order

    async def get_positions(self) -> list[Position]:
        """Return all currently open positions."""
        return list(self._positions.values())

    async def get_account(self) -> AccountInfo:
        """Return account snapshot based on simulated cash balance."""
        # Unrealised PnL is not added to equity here — only realised cash.
        # Callers who need mark-to-market equity can sum positions themselves.
        total_unrealised = sum(
            p.unrealised_pnl for p in self._positions.values()
        )
        equity_gbp = self._cash + total_unrealised
        return AccountInfo(
            equity_gbp=equity_gbp,
            cash_gbp=self._cash,
            margin_used=0.0,
            broker=self.BROKER_NAME,
        )

    async def close_position(self, symbol: str) -> Order:
        """Market-close the entire open position for *symbol*."""
        position = self._positions.get(symbol)
        if position is None:
            raise ValueError(f"No open position for {symbol}")

        close_side = "SELL" if position.side == "LONG" else "BUY"
        self._log.info(
            "closing_position",
            symbol=symbol,
            qty=position.qty,
            side=close_side,
        )
        return await self.place_order(
            symbol=symbol,
            side=close_side,
            qty=position.qty,
            order_type="MARKET",
        )

    async def stream_order_updates(
        self,
        callback: Callable[[Order], Awaitable[None] | None],
    ) -> None:
        """Register a callback and keep the coroutine alive indefinitely.

        Unlike live brokers there is no outbound WebSocket; callbacks are
        called synchronously at fill time.  This coroutine simply parks
        until it is cancelled.
        """
        self._update_callbacks.append(callback)
        self._log.info(
            "paper_order_stream_started",
            callback=getattr(callback, "__name__", repr(callback)),
        )
        try:
            # Run forever — callers cancel the task to stop streaming.
            await asyncio.get_event_loop().create_future()
        except asyncio.CancelledError:
            self._update_callbacks.remove(callback)
            raise

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @property
    def fills(self) -> list[Order]:
        """Return an immutable snapshot of all historical fills."""
        return list(self._fills)

    @property
    def cash(self) -> float:
        """Current simulated cash balance in GBP."""
        return self._cash

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_slippage(price: float, side: str) -> float:
        """Return execution price after slippage.

        BUY orders pay a slightly higher price; SELL orders receive less.
        """
        if side == "BUY":
            return price * (1.0 + _SLIPPAGE_RATE)
        return price * (1.0 - _SLIPPAGE_RATE)

    @staticmethod
    def _should_trigger(order: Order, current_price: float) -> bool:
        """Return True if *order* should trigger at *current_price*."""
        if order.order_type == "LIMIT":
            if order.price is None:
                return False
            # BUY limit fills when price falls to or below limit
            if order.side == "BUY":
                return current_price <= order.price
            # SELL limit fills when price rises to or above limit
            return current_price >= order.price

        if order.order_type == "STOP_MARKET":
            if order.stop_price is None:
                return False
            # BUY stop fills when price rises to or above stop
            if order.side == "BUY":
                return current_price >= order.stop_price
            # SELL stop fills when price falls to or below stop
            return current_price <= order.stop_price

        return False

    def _execute_fill(self, order: Order, fill_price: float) -> None:
        """Mark *order* as filled, update cash/positions, and fire callbacks."""
        now = _now()
        fee = fill_price * order.qty * _FEE_RATE

        order.status = "FILLED"
        order.fill_price = fill_price
        order.fill_qty = order.qty
        order.fee = fee
        order.filled_at = now

        # Update cash balance
        order_value = fill_price * order.qty
        if order.side == "BUY":
            self._cash -= order_value + fee
        else:
            self._cash += order_value - fee

        # Update positions
        self._update_position(order, fill_price)

        # Archive in fill history
        self._fills.append(order)

        self._log.info(
            "order_filled",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=order.qty,
            fee=round(fee, 6),
            cash_after=round(self._cash, 4),
        )

        # Fire callbacks asynchronously if an event loop is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._notify_callbacks(order))
            # If no loop is running (synchronous context from update_prices),
            # callbacks will be missed — acceptable for price-triggered fills
            # in backtesting contexts.  For live simulation use asyncio.run().
        except RuntimeError:
            pass

    def _update_position(self, order: Order, fill_price: float) -> None:
        """Update in-memory position after a fill."""
        symbol = order.symbol
        existing = self._positions.get(symbol)

        if existing is None:
            # Open a new position
            side = "LONG" if order.side == "BUY" else "SHORT"
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                qty=order.qty,
                avg_entry_price=fill_price,
                current_price=fill_price,
                unrealised_pnl=0.0,
            )
            return

        # Same direction — average in
        if (existing.side == "LONG" and order.side == "BUY") or (
            existing.side == "SHORT" and order.side == "SELL"
        ):
            total_qty = existing.qty + order.qty
            avg_entry = (
                existing.avg_entry_price * existing.qty + fill_price * order.qty
            ) / total_qty
            self._positions[symbol] = Position(
                symbol=symbol,
                side=existing.side,
                qty=total_qty,
                avg_entry_price=avg_entry,
                current_price=fill_price,
                unrealised_pnl=0.0,
            )
            return

        # Opposite direction — reduce or flip position
        if order.qty < existing.qty:
            remaining_qty = existing.qty - order.qty
            self._positions[symbol] = Position(
                symbol=symbol,
                side=existing.side,
                qty=remaining_qty,
                avg_entry_price=existing.avg_entry_price,
                current_price=fill_price,
                unrealised_pnl=0.0,
            )
        elif order.qty == existing.qty:
            # Position fully closed
            del self._positions[symbol]
        else:
            # Position flipped to the other side
            leftover_qty = order.qty - existing.qty
            new_side = "LONG" if order.side == "BUY" else "SHORT"
            self._positions[symbol] = Position(
                symbol=symbol,
                side=new_side,
                qty=leftover_qty,
                avg_entry_price=fill_price,
                current_price=fill_price,
                unrealised_pnl=0.0,
            )

    async def _notify_callbacks(self, order: Order) -> None:
        """Call all registered update callbacks with *order*."""
        for cb in list(self._update_callbacks):
            try:
                result = cb(order)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001
                self._log.error(
                    "callback_error",
                    order_id=order.order_id,
                    error=str(exc),
                )

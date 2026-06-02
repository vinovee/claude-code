"""
Abstract base classes and shared data models for broker connectors.

All broker implementations must subclass BrokerBase and implement every
abstract method.  The dataclasses defined here are the canonical transfer
objects used throughout the execution layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class Order:
    """Represents a single order placed with a broker.

    ``status`` lifecycle: PENDING -> FILLED | CANCELLED | REJECTED
    """

    order_id: str
    symbol: str
    side: str                   # BUY | SELL
    qty: float
    order_type: str             # MARKET | LIMIT | STOP_MARKET
    price: float | None         # limit price (LIMIT orders)
    stop_price: float | None    # trigger price (STOP_MARKET orders)
    status: str                 # PENDING | FILLED | CANCELLED | REJECTED
    fill_price: float | None    # average execution price when filled
    fill_qty: float | None      # quantity actually filled
    fee: float                  # cumulative commission/fee paid
    created_at: datetime
    filled_at: datetime | None
    broker: str                 # name tag set by each implementation

    # Optional bag for broker-specific extra data (raw API response, etc.)
    meta: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass
class Position:
    """Snapshot of an open position at a particular point in time."""

    symbol: str
    side: str               # LONG | SHORT
    qty: float
    avg_entry_price: float
    current_price: float
    unrealised_pnl: float


@dataclass
class AccountInfo:
    """Snapshot of the trading account."""

    equity_gbp: float
    cash_gbp: float
    margin_used: float
    broker: str


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BrokerBase(ABC):
    """Interface every broker adapter must satisfy.

    All methods are async to allow non-blocking I/O across REST and
    WebSocket APIs.
    """

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Submit a new order.

        Parameters
        ----------
        symbol:
            Exchange symbol, e.g. ``"BTCUSDT"``.
        side:
            ``"BUY"`` or ``"SELL"``.
        qty:
            Quantity (base asset units for crypto).
        order_type:
            ``"MARKET"``, ``"LIMIT"``, or ``"STOP_MARKET"``.
        price:
            Limit price; required for LIMIT orders, ignored for MARKET.
        stop_price:
            Trigger price; required for STOP_MARKET orders.
        **kwargs:
            Broker-specific extras (e.g. ``timeInForce``, ``reduceOnly``).

        Returns
        -------
        Order
            The freshly created order with at minimum ``status="PENDING"``
            or ``status="FILLED"`` for synchronous fills.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Request cancellation of an open order.

        Returns
        -------
        bool
            ``True`` if the cancellation was accepted; ``False`` if the
            order was already terminal (filled, cancelled, rejected).
        """

    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Fetch the current state of a single order by its ID."""

    # ------------------------------------------------------------------
    # Position / account queries
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Return all currently open positions."""

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Return a snapshot of the account (equity, cash, margin)."""

    @abstractmethod
    async def close_position(self, symbol: str) -> Order:
        """Immediately close the entire open position for *symbol*.

        Implementations should place a market order in the opposite
        direction equal to the current open quantity.

        Returns
        -------
        Order
            The closing order (may still be PENDING immediately after
            submission for live brokers).
        """

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    @abstractmethod
    async def stream_order_updates(
        self,
        callback: Callable[[Order], Awaitable[None] | None],
    ) -> None:
        """Subscribe to real-time order-status updates.

        The broker implementation must call *callback* whenever an order
        status changes (fill, partial fill, cancellation, rejection).

        This coroutine should run indefinitely (until cancelled) — callers
        are expected to wrap it in a task:

        .. code-block:: python

            task = asyncio.create_task(broker.stream_order_updates(cb))

        Parameters
        ----------
        callback:
            An async or sync callable that accepts a single :class:`Order`
            argument.  If the callback is a coroutine function it will be
            ``await``-ed; otherwise it is called synchronously.
        """

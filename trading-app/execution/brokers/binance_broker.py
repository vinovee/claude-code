"""
Binance broker adapter.

Uses the python-binance ``AsyncClient`` for REST calls and
``BinanceSocketManager`` for the user data stream (order updates).

Features
--------
- MARKET / LIMIT / STOP_MARKET order placement
- USDT -> GBP equity conversion (live GBPUSDT ticker or 0.79 fallback)
- Tenacity retry (exponential back-off, 3 attempts) on every REST call
- Prometheus metrics: order_placed_total, order_fill_latency_ms,
  broker_api_errors_total
- Structured logging via structlog
"""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import structlog
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
from prometheus_client import Counter, Histogram
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import AccountInfo, BrokerBase, Order, Position

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

ORDER_PLACED_TOTAL = Counter(
    "order_placed_total",
    "Total number of orders placed",
    ["broker", "symbol", "side", "order_type"],
)

ORDER_FILL_LATENCY_MS = Histogram(
    "order_fill_latency_ms",
    "Time from order creation to fill in milliseconds",
    ["broker", "symbol"],
    buckets=[10, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000],
)

BROKER_API_ERRORS_TOTAL = Counter(
    "broker_api_errors_total",
    "Total number of broker API errors",
    ["broker", "operation"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ORDER_TYPE_MAP: dict[str, str] = {
    "MARKET": "MARKET",
    "LIMIT": "LIMIT",
    "STOP_MARKET": "STOP_MARKET",
}

_STATUS_MAP: dict[str, str] = {
    "NEW": "PENDING",
    "PARTIALLY_FILLED": "PENDING",
    "FILLED": "FILLED",
    "CANCELED": "CANCELLED",
    "CANCELLED": "CANCELLED",
    "REJECTED": "REJECTED",
    "EXPIRED": "CANCELLED",
}

_USDT_GBP_FALLBACK = 0.79  # approximate fallback rate


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ms_to_dt(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc)


def _parse_order(raw: dict[str, Any], broker: str = "binance") -> Order:
    """Convert a raw Binance order dict to an :class:`Order` dataclass."""
    status_raw = raw.get("status", "NEW")
    status = _STATUS_MAP.get(status_raw, "PENDING")

    fill_price_str = raw.get("avgPrice") or raw.get("price") or "0"
    fill_price = float(fill_price_str) if float(fill_price_str) > 0 else None

    fill_qty_str = raw.get("executedQty", "0")
    fill_qty = float(fill_qty_str) if float(fill_qty_str) > 0 else None

    # Fee is reported in fills; sum them up when present
    fee = 0.0
    for fill in raw.get("fills", []):
        try:
            fee += float(fill.get("commission", 0))
        except (TypeError, ValueError):
            pass

    created_ms = raw.get("transactTime") or raw.get("time")
    filled_at: datetime | None = None
    if status == "FILLED":
        filled_at = _ms_to_dt(raw.get("updateTime") or created_ms)

    stop_str = raw.get("stopPrice", "0")
    stop_price = float(stop_str) if stop_str and float(stop_str) > 0 else None

    limit_str = raw.get("price", "0")
    limit_price = float(limit_str) if limit_str and float(limit_str) > 0 else None

    return Order(
        order_id=str(raw.get("orderId", raw.get("clientOrderId", ""))),
        symbol=raw.get("symbol", ""),
        side=raw.get("side", ""),
        qty=float(raw.get("origQty", raw.get("quantity", 0))),
        order_type=raw.get("type", "MARKET"),
        price=limit_price,
        stop_price=stop_price,
        status=status,
        fill_price=fill_price,
        fill_qty=fill_qty,
        fee=fee,
        created_at=_ms_to_dt(created_ms) or _now(),
        filled_at=filled_at,
        broker=broker,
        meta=raw,
    )


# ---------------------------------------------------------------------------
# Retry decorator factory
# ---------------------------------------------------------------------------


def _binance_retry(operation: str) -> Any:
    """Return a tenacity retry decorator for a named broker operation."""

    def _after_retry(retry_state: Any) -> None:
        BROKER_API_ERRORS_TOTAL.labels(
            broker="binance", operation=operation
        ).inc()
        logger.warning(
            "binance_api_retry",
            operation=operation,
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception()),
        )

    return retry(
        retry=retry_if_exception_type((BinanceAPIException, Exception)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        after=_after_retry,
        reraise=True,
    )


# ---------------------------------------------------------------------------
# BinanceBroker
# ---------------------------------------------------------------------------


class BinanceBroker(BrokerBase):
    """Live (or testnet) Binance Futures/Spot broker adapter.

    Parameters
    ----------
    api_key:
        Binance API key.
    api_secret:
        Binance API secret.
    testnet:
        If ``True`` the adapter uses Binance's testnet endpoints.
    """

    BROKER_NAME = "binance"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._client: AsyncClient | None = None
        self._log = logger.bind(broker=self.BROKER_NAME, testnet=testnet)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> AsyncClient:
        """Lazily initialise and return the async Binance client."""
        if self._client is None:
            self._client = await AsyncClient.create(
                api_key=self._api_key,
                api_secret=self._api_secret,
                testnet=self._testnet,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._client is not None:
            await self._client.close_connection()
            self._client = None

    # ------------------------------------------------------------------
    # Internal helper with retry
    # ------------------------------------------------------------------

    async def _call(self, operation: str, coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a Binance API call with tenacity retries."""

        @_binance_retry(operation)
        async def _inner() -> Any:
            return await coro_fn(*args, **kwargs)

        try:
            return await _inner()
        except BinanceAPIException as exc:
            BROKER_API_ERRORS_TOTAL.labels(
                broker=self.BROKER_NAME, operation=operation
            ).inc()
            self._log.error(
                "binance_api_error",
                operation=operation,
                code=exc.code,
                message=exc.message,
            )
            raise

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
        """Place an order on Binance."""
        client = await self._get_client()
        binance_type = _ORDER_TYPE_MAP.get(order_type, "MARKET")

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": binance_type,
            "quantity": qty,
            **kwargs,
        }

        if binance_type == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            params["price"] = str(price)
            params.setdefault("timeInForce", "GTC")

        if binance_type == "STOP_MARKET":
            if stop_price is None:
                raise ValueError("stop_price is required for STOP_MARKET orders")
            params["stopPrice"] = str(stop_price)

        t_start = time.monotonic()

        self._log.info(
            "placing_order",
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
        )

        raw = await self._call(
            "place_order",
            client.create_order,
            **params,
        )

        order = _parse_order(raw, broker=self.BROKER_NAME)

        ORDER_PLACED_TOTAL.labels(
            broker=self.BROKER_NAME,
            symbol=symbol,
            side=side,
            order_type=order_type,
        ).inc()

        if order.status == "FILLED":
            latency_ms = (time.monotonic() - t_start) * 1_000
            ORDER_FILL_LATENCY_MS.labels(
                broker=self.BROKER_NAME, symbol=symbol
            ).observe(latency_ms)
            self._log.info(
                "order_filled",
                order_id=order.order_id,
                symbol=symbol,
                fill_price=order.fill_price,
                fill_qty=order.fill_qty,
                fee=order.fee,
                latency_ms=round(latency_ms, 2),
            )
        else:
            self._log.info(
                "order_placed",
                order_id=order.order_id,
                symbol=symbol,
                status=order.status,
            )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its Binance order ID."""
        client = await self._get_client()
        # We need the symbol; store it in a minimal meta lookup.
        # For simplicity, the caller must pass the symbol embedded as
        # "SYMBOL|ORDER_ID" when using this broker, OR we track it in _order_cache.
        # Here we split on "|" if present, otherwise raise.
        if "|" in order_id:
            symbol, oid = order_id.split("|", 1)
        else:
            raise ValueError(
                "BinanceBroker.cancel_order requires order_id in 'SYMBOL|ORDER_ID' format"
            )

        try:
            await self._call(
                "cancel_order",
                client.cancel_order,
                symbol=symbol,
                orderId=int(oid),
            )
            self._log.info("order_cancelled", order_id=order_id, symbol=symbol)
            return True
        except BinanceAPIException as exc:
            if exc.code in (-2011, -2013):  # unknown order / already filled
                return False
            raise

    async def get_order(self, order_id: str) -> Order:
        """Fetch a single order.  ``order_id`` must be ``"SYMBOL|ORDER_ID"``."""
        client = await self._get_client()
        if "|" in order_id:
            symbol, oid = order_id.split("|", 1)
        else:
            raise ValueError(
                "BinanceBroker.get_order requires order_id in 'SYMBOL|ORDER_ID' format"
            )

        raw = await self._call(
            "get_order",
            client.get_order,
            symbol=symbol,
            orderId=int(oid),
        )
        return _parse_order(raw, broker=self.BROKER_NAME)

    async def get_positions(self) -> list[Position]:
        """Return non-zero positions from the Futures account."""
        client = await self._get_client()
        raw_positions: list[dict] = await self._call(
            "get_positions",
            client.futures_position_information,
        )

        positions: list[Position] = []
        for p in raw_positions:
            qty = float(p.get("positionAmt", 0))
            if qty == 0.0:
                continue
            entry = float(p.get("entryPrice", 0))
            mark = float(p.get("markPrice", 0))
            pnl = float(p.get("unrealizedProfit", 0))
            side = "LONG" if qty > 0 else "SHORT"
            positions.append(
                Position(
                    symbol=p["symbol"],
                    side=side,
                    qty=abs(qty),
                    avg_entry_price=entry,
                    current_price=mark,
                    unrealised_pnl=pnl,
                )
            )
        return positions

    async def get_account(self) -> AccountInfo:
        """Return account equity converted from USDT to GBP."""
        client = await self._get_client()

        # Fetch USDT balance
        account_raw: dict = await self._call(
            "get_account",
            client.get_account,
        )
        usdt_free = 0.0
        usdt_locked = 0.0
        for asset in account_raw.get("balances", []):
            if asset.get("asset") == "USDT":
                usdt_free = float(asset.get("free", 0))
                usdt_locked = float(asset.get("locked", 0))
                break

        # Fetch USDT -> GBP conversion rate
        usdt_to_gbp = await self._fetch_usdt_gbp_rate(client)

        equity_gbp = (usdt_free + usdt_locked) * usdt_to_gbp
        cash_gbp = usdt_free * usdt_to_gbp

        # Margin used: for spot accounts this is locked value
        margin_used = usdt_locked * usdt_to_gbp

        self._log.debug(
            "account_fetched",
            usdt_free=usdt_free,
            usdt_locked=usdt_locked,
            usdt_to_gbp=usdt_to_gbp,
            equity_gbp=round(equity_gbp, 2),
        )

        return AccountInfo(
            equity_gbp=equity_gbp,
            cash_gbp=cash_gbp,
            margin_used=margin_used,
            broker=self.BROKER_NAME,
        )

    async def _fetch_usdt_gbp_rate(self, client: AsyncClient) -> float:
        """Return the USDT -> GBP rate from Binance, falling back to 0.79."""
        # Binance quotes GBP as GBPUSDT (how many USDT per 1 GBP)
        try:
            ticker: dict = await self._call(
                "get_gbpusdt",
                client.get_symbol_ticker,
                symbol="GBPUSDT",
            )
            gbp_in_usdt = float(ticker["price"])  # e.g. 1.27
            if gbp_in_usdt <= 0:
                raise ValueError("zero rate returned")
            return 1.0 / gbp_in_usdt  # USDT -> GBP
        except Exception as exc:
            self._log.warning(
                "gbpusdt_rate_fallback",
                reason=str(exc),
                fallback=_USDT_GBP_FALLBACK,
            )
            return _USDT_GBP_FALLBACK

    async def close_position(self, symbol: str) -> Order:
        """Market-close the entire open position for *symbol*."""
        positions = await self.get_positions()
        position = next((p for p in positions if p.symbol == symbol), None)
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
            reduceOnly="true",
        )

    async def stream_order_updates(
        self,
        callback: Callable[[Order], Awaitable[None] | None],
    ) -> None:
        """Stream user-data events (order fills) via BinanceSocketManager."""
        client = await self._get_client()
        bsm = BinanceSocketManager(client)

        self._log.info("order_stream_starting")

        async with bsm.user_socket() as stream:
            while True:
                msg: dict = await stream.recv()
                event_type = msg.get("e")

                if event_type == "executionReport":
                    order = self._parse_execution_report(msg)
                    self._log.info(
                        "order_update_received",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        status=order.status,
                        fill_price=order.fill_price,
                    )

                    if order.status == "FILLED" and order.created_at:
                        latency_ms = (
                            datetime.now(tz=timezone.utc) - order.created_at
                        ).total_seconds() * 1_000
                        ORDER_FILL_LATENCY_MS.labels(
                            broker=self.BROKER_NAME, symbol=order.symbol
                        ).observe(latency_ms)

                    result = callback(order)
                    if inspect.isawaitable(result):
                        await result

    @staticmethod
    def _parse_execution_report(msg: dict) -> Order:
        """Convert a Binance executionReport WebSocket message to an Order."""
        status_raw = msg.get("X", "NEW")
        status = _STATUS_MAP.get(status_raw, "PENDING")

        fill_price = float(msg.get("L", 0)) or None  # last fill price
        fill_qty = float(msg.get("z", 0)) or None    # cumulative filled qty
        fee = float(msg.get("n", 0))

        created_ms = msg.get("T") or msg.get("O")
        filled_at: datetime | None = None
        if status == "FILLED" and msg.get("T"):
            filled_at = _ms_to_dt(msg["T"])

        limit_price = float(msg.get("p", 0)) or None
        stop_price_val = float(msg.get("P", 0)) or None

        return Order(
            order_id=str(msg.get("i", "")),
            symbol=msg.get("s", ""),
            side=msg.get("S", ""),
            qty=float(msg.get("q", 0)),
            order_type=msg.get("o", "MARKET"),
            price=limit_price,
            stop_price=stop_price_val,
            status=status,
            fill_price=fill_price,
            fill_qty=fill_qty,
            fee=fee,
            created_at=_ms_to_dt(created_ms) or datetime.now(tz=timezone.utc),
            filled_at=filled_at,
            broker="binance",
            meta=msg,
        )

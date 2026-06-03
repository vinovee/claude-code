"""
Kraken broker adapter.

Uses ``krakenex`` for authenticated REST calls and polls ``get_closed_orders``
every 2 s for order-update streaming (Kraken's private WebSocket requires a
one-time token exchange; polling is the simpler and fully-supported alternative
for UK retail spot trading).

UK retail note
--------------
Kraken replaced Binance in the UK after Binance exited the UK market in 2023.
Only *spot* trading is available to UK retail clients.  All pairs are quoted in
USDT (or USD) internally but map to Kraken's XBT notation (e.g. XBTUSDT).

Pairs
-----
    BTCUSDT  → XBTUSDT
    ETHUSDT  → ETHUSDT
    SOLUSDT  → SOLUSDT

Order ID format
---------------
Kraken returns a ``txid`` list from ``add_order``.  We join multiple IDs with
``","`` and use the first entry as the canonical ``order_id``.

Order status mapping
--------------------
    pending  → PENDING
    open     → PENDING
    closed   → FILLED
    canceled → CANCELLED
    expired  → CANCELLED

Prometheus metrics
------------------
    order_placed_total{broker, symbol, side, order_type}
    order_fill_latency_ms{broker, symbol}
    broker_api_errors_total{broker, operation}

All metrics use label ``broker="kraken"``.

Structured logging via structlog.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import krakenex
import structlog
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

# Guard against double-registration when both binance and kraken modules are
# imported in the same process — prometheus_client raises ValueError on
# duplicate metric names.  We use a try/except and fall back to looking up
# the already-registered collector.

try:
    ORDER_PLACED_TOTAL = Counter(
        "order_placed_total",
        "Total number of orders placed",
        ["broker", "symbol", "side", "order_type"],
    )
except ValueError:
    from prometheus_client import REGISTRY as _REGISTRY  # type: ignore[assignment]
    ORDER_PLACED_TOTAL = _REGISTRY._names_to_collectors.get(  # type: ignore[assignment]
        "order_placed_total"
    )

try:
    ORDER_FILL_LATENCY_MS = Histogram(
        "order_fill_latency_ms",
        "Time from order creation to fill in milliseconds",
        ["broker", "symbol"],
        buckets=[10, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000],
    )
except ValueError:
    from prometheus_client import REGISTRY as _REGISTRY  # type: ignore[assignment]
    ORDER_FILL_LATENCY_MS = _REGISTRY._names_to_collectors.get(  # type: ignore[assignment]
        "order_fill_latency_ms"
    )

try:
    BROKER_API_ERRORS_TOTAL = Counter(
        "broker_api_errors_total",
        "Total number of broker API errors",
        ["broker", "operation"],
    )
except ValueError:
    from prometheus_client import REGISTRY as _REGISTRY  # type: ignore[assignment]
    BROKER_API_ERRORS_TOTAL = _REGISTRY._names_to_collectors.get(  # type: ignore[assignment]
        "broker_api_errors_total"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROKER_NAME = "kraken"

# Kraken's native name for BTC is XBT; map our canonical "BTC" prefix.
_SYMBOL_MAP: dict[str, str] = {
    "BTCUSDT": "XBTUSDT",
    "BTCUSD": "XBTUSD",
    "BTCGBP": "XBTGBP",
}

# Our order-type strings → Kraken order-type strings
_ORDER_TYPE_MAP: dict[str, str] = {
    "MARKET": "market",
    "LIMIT": "limit",
    "STOP_MARKET": "stop-loss",
}

# Kraken order status → our canonical status
_STATUS_MAP: dict[str, str] = {
    "pending": "PENDING",
    "open": "PENDING",
    "closed": "FILLED",
    "canceled": "CANCELLED",
    "cancelled": "CANCELLED",
    "expired": "CANCELLED",
}

# GBP/USDT fallback exchange rate (approximate)
_USDT_GBP_FALLBACK = 0.79

# Poll interval for streaming order updates (seconds)
_POLL_INTERVAL_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ts_to_dt(ts: float | int | str | None) -> datetime | None:
    """Convert a Unix timestamp (float/str) to a tz-aware datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _to_kraken_pair(symbol: str) -> str:
    """Map our canonical pair name to Kraken's internal pair name.

    BTCUSDT → XBTUSDT
    All other symbols are returned as-is (Kraken accepts ETHUSDT, SOLUSDT …).
    """
    return _SYMBOL_MAP.get(symbol.upper(), symbol.upper())


def _parse_kraken_order(
    txid: str,
    info: dict[str, Any],
    symbol: str,
    side: str,
    qty: float,
    order_type: str,
    price: float | None,
    stop_price: float | None,
    created_at: datetime | None = None,
) -> Order:
    """Build an :class:`Order` from a Kraken ``query_orders_info`` response entry.

    Parameters
    ----------
    txid:
        The Kraken transaction ID (order ID).
    info:
        The value dict for this txid from the Kraken API response.
    symbol, side, qty, order_type, price, stop_price:
        Values from the original placement call (used as fallback when the
        API response does not carry them, e.g. for newly-created orders).
    created_at:
        Override for the creation timestamp.
    """
    status_raw: str = info.get("status", "pending")
    status = _STATUS_MAP.get(status_raw, "PENDING")

    descr: dict[str, Any] = info.get("descr", {})

    # Fill price — Kraken reports the average execution price in ``price``
    # within the order detail, but only once the order is closed.
    exec_price_str: str = str(info.get("price", "0"))
    try:
        exec_price = float(exec_price_str)
        fill_price: float | None = exec_price if exec_price > 0 else None
    except (TypeError, ValueError):
        fill_price = None

    # Filled quantity
    vol_exec_str: str = str(info.get("vol_exec", "0"))
    try:
        vol_exec = float(vol_exec_str)
        fill_qty: float | None = vol_exec if vol_exec > 0 else None
    except (TypeError, ValueError):
        fill_qty = None

    # Fee
    fee_str: str = str(info.get("fee", "0"))
    try:
        fee = float(fee_str)
    except (TypeError, ValueError):
        fee = 0.0

    # Timestamps
    open_ts = _ts_to_dt(info.get("opentm"))
    close_ts = _ts_to_dt(info.get("closetm"))
    filled_at: datetime | None = close_ts if status == "FILLED" else None

    # Derive symbol / side from the order description if not supplied
    inferred_symbol = descr.get("pair", symbol)
    inferred_side = (descr.get("type", side) or side).upper()

    return Order(
        order_id=txid,
        symbol=inferred_symbol or symbol,
        side=inferred_side,
        qty=qty,
        order_type=order_type,
        price=price,
        stop_price=stop_price,
        status=status,
        fill_price=fill_price,
        fill_qty=fill_qty,
        fee=fee,
        created_at=created_at or open_ts or _now(),
        filled_at=filled_at,
        broker=_BROKER_NAME,
        meta=info,
    )


# ---------------------------------------------------------------------------
# Retry decorator factory
# ---------------------------------------------------------------------------


def _kraken_retry(operation: str) -> Any:
    """Return a tenacity retry decorator for a named Kraken API operation."""

    def _after_retry(retry_state: Any) -> None:
        BROKER_API_ERRORS_TOTAL.labels(
            broker=_BROKER_NAME, operation=operation
        ).inc()
        logger.warning(
            "kraken_api_retry",
            operation=operation,
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception()),
        )

    return retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        after=_after_retry,
        reraise=True,
    )


# ---------------------------------------------------------------------------
# KrakenBroker
# ---------------------------------------------------------------------------


class KrakenBroker(BrokerBase):
    """UK-compliant Kraken spot broker adapter.

    Uses ``krakenex.API`` for all REST calls.  Every public method wraps its
    API call with tenacity retries (up to 3 attempts, exponential back-off).

    Parameters
    ----------
    api_key:
        Kraken API key.
    private_key:
        Kraken private (secret) key.
    """

    BROKER_NAME = _BROKER_NAME

    def __init__(self, api_key: str, private_key: str) -> None:
        self._api_key = api_key
        self._private_key = private_key
        self._api: krakenex.API = krakenex.API(key=api_key, secret=private_key)
        self._log = logger.bind(broker=self.BROKER_NAME)
        # Cache of order_id → (symbol, side, qty, order_type, price, stop_price,
        #                       created_at) used to reconstruct Order objects in
        # subsequent get_order / polling calls without requiring the caller to
        # embed metadata in the order ID (unlike the Binance adapter).
        self._order_meta: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Low-level API helper
    # ------------------------------------------------------------------

    def _call(self, operation: str, method: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a Kraken API call (sync) and raise on error.

        Kraken API errors arrive in the ``"error"`` list of the response
        envelope.  Any non-empty error list is converted to a ``RuntimeError``
        so that tenacity can retry.

        Parameters
        ----------
        operation:
            Human-readable name used in logs and metrics.
        method:
            The krakenex method to call: ``"query_private"`` or
            ``"query_public"``.
        data:
            Request parameters dict (passed as second arg to krakenex).
        """
        resp: dict[str, Any] = getattr(self._api, method)(
            *([data] if data is not None else [])
        )
        errors: list[str] = resp.get("error", [])
        if errors:
            msg = "; ".join(errors)
            raise RuntimeError(f"Kraken API error [{operation}]: {msg}")
        return resp.get("result", {})

    def _query_private(self, operation: str, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Thin wrapper: call a private Kraken endpoint with retries."""

        @_kraken_retry(operation)
        def _inner() -> dict[str, Any]:
            return self._api.query_private(endpoint, data or {})

        resp = _inner()
        errors: list[str] = resp.get("error", [])
        if errors:
            msg = "; ".join(errors)
            BROKER_API_ERRORS_TOTAL.labels(
                broker=self.BROKER_NAME, operation=operation
            ).inc()
            raise RuntimeError(f"Kraken API error [{operation}]: {msg}")
        return resp.get("result", {})

    # ------------------------------------------------------------------
    # GBP conversion helper
    # ------------------------------------------------------------------

    def _fetch_usdt_gbp_rate(self) -> float:
        """Return the approximate USDT → GBP conversion rate.

        Calls the Kraken public ticker for USDTGBP (or GBPUSDT equivalent).
        Falls back to a hard-coded constant on failure.
        """
        try:
            resp = self._api.query_public("Ticker", {"pair": "USDTGBP"})
            errors = resp.get("error", [])
            if errors:
                raise RuntimeError("; ".join(errors))
            result = resp.get("result", {})
            # Kraken returns the pair under its internal name, e.g. "USDTGBP"
            pair_data = next(iter(result.values()), {})
            # "c" is the array [last_trade_price, lot_volume]
            last_price = float(pair_data.get("c", [_USDT_GBP_FALLBACK])[0])
            if last_price <= 0:
                raise ValueError("zero rate")
            return last_price
        except Exception as exc:
            self._log.warning(
                "usdtgbp_rate_fallback",
                reason=str(exc),
                fallback=_USDT_GBP_FALLBACK,
            )
            return _USDT_GBP_FALLBACK

    # ------------------------------------------------------------------
    # BrokerBase: place_order
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
        """Place a spot order on Kraken.

        Parameters
        ----------
        symbol:
            Our canonical pair string, e.g. ``"BTCUSDT"`` or ``"ETHUSDT"``.
        side:
            ``"BUY"`` or ``"SELL"``.
        qty:
            Volume in base-asset units.
        order_type:
            ``"MARKET"``, ``"LIMIT"``, or ``"STOP_MARKET"``.
        price:
            Limit price; required for LIMIT orders.
        stop_price:
            Trigger price; required for STOP_MARKET orders.
        **kwargs:
            Additional Kraken-specific parameters passed verbatim to
            ``add_order`` (e.g. ``timeinforce``, ``userref``).
        """
        kraken_pair = _to_kraken_pair(symbol)
        kraken_type = _ORDER_TYPE_MAP.get(order_type, "market")
        kraken_side = side.lower()  # Kraken expects "buy" / "sell"

        params: dict[str, Any] = {
            "pair": kraken_pair,
            "type": kraken_side,
            "ordertype": kraken_type,
            "volume": str(qty),
            **kwargs,
        }

        if kraken_type == "limit":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            params["price"] = str(price)

        if kraken_type == "stop-loss":
            if stop_price is None:
                raise ValueError("stop_price is required for STOP_MARKET orders")
            params["price"] = str(stop_price)

        self._log.info(
            "placing_order",
            symbol=symbol,
            kraken_pair=kraken_pair,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
        )

        t_start = time.monotonic()
        created_at = _now()

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._query_private("place_order", "AddOrder", params),
        )

        # Kraken returns: {"descr": {...}, "txid": ["XXXXXX-XXXXX-XXXXXX"]}
        txid_list: list[str] = result.get("txid", [])
        if not txid_list:
            raise RuntimeError(
                f"Kraken add_order returned no txid for {symbol}; result={result}"
            )

        order_id = txid_list[0]

        # Cache metadata for later lookups
        self._order_meta[order_id] = {
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty,
            "order_type": order_type,
            "price": price,
            "stop_price": stop_price,
            "created_at": created_at,
        }

        ORDER_PLACED_TOTAL.labels(
            broker=self.BROKER_NAME,
            symbol=symbol,
            side=side.upper(),
            order_type=order_type,
        ).inc()

        self._log.info(
            "order_placed",
            order_id=order_id,
            symbol=symbol,
            kraken_pair=kraken_pair,
        )

        # Construct the initial (PENDING) order object — Kraken does not return
        # fill details synchronously for non-IOC market orders.
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            qty=qty,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            status="PENDING",
            fill_price=None,
            fill_qty=None,
            fee=0.0,
            created_at=created_at,
            filled_at=None,
            broker=self.BROKER_NAME,
            meta=result,
        )

        # For MARKET orders, poll once immediately to capture a synchronous fill
        if order_type == "MARKET":
            await asyncio.sleep(0.3)  # brief pause for the exchange to process
            try:
                order = await self.get_order(order_id)
                if order.status == "FILLED":
                    latency_ms = (time.monotonic() - t_start) * 1_000
                    ORDER_FILL_LATENCY_MS.labels(
                        broker=self.BROKER_NAME, symbol=symbol
                    ).observe(latency_ms)
                    self._log.info(
                        "order_filled",
                        order_id=order_id,
                        symbol=symbol,
                        fill_price=order.fill_price,
                        fill_qty=order.fill_qty,
                        fee=order.fee,
                        latency_ms=round(latency_ms, 2),
                    )
            except Exception as exc:
                self._log.warning(
                    "order_immediate_check_failed",
                    order_id=order_id,
                    reason=str(exc),
                )

        return order

    # ------------------------------------------------------------------
    # BrokerBase: cancel_order
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by its Kraken transaction ID.

        Returns
        -------
        bool
            ``True`` if the cancellation was accepted; ``False`` if Kraken
            reports the order is already in a terminal state.
        """
        self._log.info("cancelling_order", order_id=order_id)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._query_private(
                    "cancel_order", "CancelOrder", {"txid": order_id}
                ),
            )
            # result = {"count": N, "pending": bool}
            self._log.info(
                "order_cancelled",
                order_id=order_id,
                count=result.get("count", 0),
            )
            return True
        except RuntimeError as exc:
            err_str = str(exc).lower()
            # Kraken returns "EOrder:Unknown order" for terminal orders
            if "unknown order" in err_str or "invalid order" in err_str:
                self._log.warning(
                    "cancel_order_already_terminal",
                    order_id=order_id,
                    reason=str(exc),
                )
                return False
            BROKER_API_ERRORS_TOTAL.labels(
                broker=self.BROKER_NAME, operation="cancel_order"
            ).inc()
            raise

    # ------------------------------------------------------------------
    # BrokerBase: get_order
    # ------------------------------------------------------------------

    async def get_order(self, order_id: str) -> Order:
        """Fetch the current state of a single order.

        Parameters
        ----------
        order_id:
            The Kraken transaction ID returned by :meth:`place_order`.
        """
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._query_private(
                "get_order",
                "QueryOrders",
                {"txid": order_id, "trades": False},
            ),
        )

        info: dict[str, Any] = result.get(order_id, {})
        meta = self._order_meta.get(order_id, {})

        return _parse_kraken_order(
            txid=order_id,
            info=info,
            symbol=meta.get("symbol", info.get("descr", {}).get("pair", "")),
            side=meta.get("side", ""),
            qty=meta.get("qty", float(info.get("vol", 0))),
            order_type=meta.get("order_type", "MARKET"),
            price=meta.get("price"),
            stop_price=meta.get("stop_price"),
            created_at=meta.get("created_at"),
        )

    # ------------------------------------------------------------------
    # BrokerBase: get_positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Return inferred spot positions from current GBP/USDT balances.

        For UK retail spot trading, Kraken does not maintain explicit position
        objects.  We infer open positions from non-trivial crypto balances and
        use a rough cost-basis heuristic (current price as entry proxy) when
        no stored metadata is available.

        Notes
        -----
        The balance call returns ``{asset: {"balance": "...", "hold_trade": "..."}}``
        for the new Kraken API, or ``{asset: "amount_string"}`` for the
        older ``Balance`` endpoint.  We normalise both.
        """
        # Fetch balances
        raw_balances = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._query_private("get_positions", "Balance", {}),
        )

        positions: list[Position] = []

        # Minimum threshold below which we consider the balance dust (not a position)
        _DUST_THRESHOLD = 1e-8

        # Assets to skip (fiat / stablecoins that are "cash", not "positions")
        _SKIP_ASSETS = {"ZGBP", "ZEUR", "ZUSD", "USDT", "USD.F", "GBP", "EUR", "USD"}

        for asset, value in raw_balances.items():
            # Normalise: newer API returns dict, older returns string
            if isinstance(value, dict):
                balance_str = str(value.get("balance", "0"))
            else:
                balance_str = str(value)

            try:
                balance = float(balance_str)
            except (TypeError, ValueError):
                continue

            if balance <= _DUST_THRESHOLD:
                continue

            # Normalise asset name: Kraken prefixes BTC with X and fiat with Z
            if asset in _SKIP_ASSETS:
                continue
            if asset.startswith("Z"):
                continue  # fiat

            # Try to get a current price for this asset in USDT
            canonical_asset = asset.lstrip("X")  # XXBT → XBT, XETH → ETH
            kraken_ticker_pair = f"{asset}USDT"
            current_price = 0.0
            try:
                ticker_resp = self._api.query_public(
                    "Ticker", {"pair": kraken_ticker_pair}
                )
                ticker_result = ticker_resp.get("result", {})
                pair_data = next(iter(ticker_result.values()), {})
                current_price = float(pair_data.get("c", [0.0])[0])
            except Exception:
                pass

            # For spot we only have LONG positions
            positions.append(
                Position(
                    symbol=f"{canonical_asset}USDT",
                    side="LONG",
                    qty=balance,
                    avg_entry_price=current_price,   # best approximation available
                    current_price=current_price,
                    unrealised_pnl=0.0,              # no cost-basis stored server-side
                )
            )

        self._log.debug("get_positions", count=len(positions))
        return positions

    # ------------------------------------------------------------------
    # BrokerBase: get_account
    # ------------------------------------------------------------------

    async def get_account(self) -> AccountInfo:
        """Return a snapshot of account equity and cash in GBP.

        Equity is calculated as:
            GBP cash balance
            + sum(crypto_balance × current_price_usdt × usdt_gbp_rate)

        Notes
        -----
        Kraken's ``TradeBalance`` endpoint provides margin-account totals
        (``eb`` = equivalent balance, ``tb`` = trade balance) but only for
        margin accounts.  For spot-only UK retail we compute equity manually
        from ``Balance``.
        """
        raw_balances = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._query_private("get_account", "Balance", {}),
        )

        usdt_gbp = self._fetch_usdt_gbp_rate()

        # Accumulate GBP-denominated values
        gbp_cash = 0.0
        crypto_value_usdt = 0.0

        for asset, value in raw_balances.items():
            if isinstance(value, dict):
                balance_str = str(value.get("balance", "0"))
            else:
                balance_str = str(value)

            try:
                balance = float(balance_str)
            except (TypeError, ValueError):
                balance = 0.0

            if balance <= 0:
                continue

            # GBP balance
            if asset in {"ZGBP", "GBP"}:
                gbp_cash += balance
                continue

            # USD/USDT treated as near-GBP via rate
            if asset in {"ZUSD", "USD", "USDT"}:
                crypto_value_usdt += balance
                continue

            # Skip EUR and other fiat
            if asset in {"ZEUR", "EUR"}:
                continue

            # Crypto: convert to USDT via ticker
            kraken_ticker_pair = f"{asset}USDT"
            try:
                ticker_resp = self._api.query_public(
                    "Ticker", {"pair": kraken_ticker_pair}
                )
                ticker_result = ticker_resp.get("result", {})
                pair_data = next(iter(ticker_result.values()), {})
                price_usdt = float(pair_data.get("c", [0.0])[0])
                crypto_value_usdt += balance * price_usdt
            except Exception:
                pass  # asset price unavailable — exclude from equity

        equity_gbp = gbp_cash + crypto_value_usdt * usdt_gbp
        cash_gbp = gbp_cash

        self._log.debug(
            "account_fetched",
            equity_gbp=round(equity_gbp, 2),
            cash_gbp=round(cash_gbp, 2),
            usdt_gbp=usdt_gbp,
        )

        return AccountInfo(
            equity_gbp=equity_gbp,
            cash_gbp=cash_gbp,
            margin_used=0.0,  # spot-only; no margin on UK retail Kraken
            broker=self.BROKER_NAME,
        )

    # ------------------------------------------------------------------
    # BrokerBase: close_position
    # ------------------------------------------------------------------

    async def close_position(self, symbol: str) -> Order:
        """Market-sell the entire spot balance for *symbol*.

        Parameters
        ----------
        symbol:
            Our canonical pair string, e.g. ``"BTCUSDT"``.  The base asset
            is derived by stripping the quote suffix.
        """
        positions = await self.get_positions()
        position = next((p for p in positions if p.symbol == symbol), None)
        if position is None:
            raise ValueError(f"No open position for {symbol}")

        self._log.info(
            "closing_position",
            symbol=symbol,
            qty=position.qty,
        )
        return await self.place_order(
            symbol=symbol,
            side="SELL",
            qty=position.qty,
            order_type="MARKET",
        )

    # ------------------------------------------------------------------
    # BrokerBase: stream_order_updates
    # ------------------------------------------------------------------

    async def stream_order_updates(
        self,
        callback: Callable[[Order], Awaitable[None] | None],
    ) -> None:
        """Poll ``ClosedOrders`` every 2 s and emit fill events via *callback*.

        Kraken's private WebSocket stream requires a one-time authentication
        token obtained from the ``GetWebSocketsToken`` endpoint.  For
        simplicity and reliability, this implementation polls the REST API
        instead.

        The coroutine runs indefinitely; callers should wrap it in an
        ``asyncio.Task``.

        Parameters
        ----------
        callback:
            Async or sync callable accepting a single :class:`Order`.  Called
            whenever a tracked order transitions to FILLED or CANCELLED.
        """
        self._log.info("order_stream_starting", mode="polling", interval_s=_POLL_INTERVAL_SECONDS)

        # Track which orders we have already emitted a terminal-state event for
        seen_terminal: set[str] = set()

        # We also track orders placed since the stream started so we don't miss
        # fills for short-lived market orders.
        while True:
            try:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)

                # Fetch recently closed orders (last 60 s is more than enough)
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._query_private(
                        "poll_closed_orders",
                        "ClosedOrders",
                        {"start": int(time.time()) - 60},
                    ),
                )

                closed: dict[str, Any] = result.get("closed", {})
                for txid, info in closed.items():
                    if txid in seen_terminal:
                        continue

                    status_raw = info.get("status", "")
                    if status_raw not in ("closed", "canceled", "expired"):
                        continue

                    seen_terminal.add(txid)
                    meta = self._order_meta.get(txid, {})

                    order = _parse_kraken_order(
                        txid=txid,
                        info=info,
                        symbol=meta.get(
                            "symbol", info.get("descr", {}).get("pair", "")
                        ),
                        side=meta.get("side", ""),
                        qty=meta.get("qty", float(info.get("vol", 0))),
                        order_type=meta.get("order_type", "MARKET"),
                        price=meta.get("price"),
                        stop_price=meta.get("stop_price"),
                        created_at=meta.get("created_at"),
                    )

                    self._log.info(
                        "order_update_received",
                        order_id=txid,
                        symbol=order.symbol,
                        status=order.status,
                        fill_price=order.fill_price,
                    )

                    if order.status == "FILLED" and order.created_at:
                        latency_ms = (
                            _now() - order.created_at
                        ).total_seconds() * 1_000
                        ORDER_FILL_LATENCY_MS.labels(
                            broker=self.BROKER_NAME, symbol=order.symbol
                        ).observe(latency_ms)

                    result_cb = callback(order)
                    if inspect.isawaitable(result_cb):
                        await result_cb

            except asyncio.CancelledError:
                self._log.info("order_stream_stopping")
                raise
            except Exception as exc:
                BROKER_API_ERRORS_TOTAL.labels(
                    broker=self.BROKER_NAME, operation="poll_closed_orders"
                ).inc()
                self._log.error(
                    "order_stream_error",
                    reason=str(exc),
                    retry_in_s=_POLL_INTERVAL_SECONDS,
                )
                # Continue looping — a transient network error should not
                # terminate the stream permanently.

"""
Portfolio — tracks cash, positions, P&L, and phase transitions.

This class is the single source of truth for all account state during a
trading session.  It is updated by the engine on every order fill, price
tick, and trade close.

Usage
-----
    from core.portfolio import Portfolio
    from config.settings import get_settings

    portfolio = Portfolio()
    await portfolio.on_order_filled(order)
    await portfolio.on_price_update({"BTCUSDT": 65_000.0})
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from config.settings import get_settings
from execution.brokers.base import Order
from monitoring.logger import PHASE_TRANSITION, get_logger

if TYPE_CHECKING:
    from data.storage.timescale import AsyncTimescaleDB

log = get_logger(__name__)

__all__ = ["Portfolio"]


class Portfolio:
    """Mutable account state for the autonomous trading engine.

    All state is held in-memory.  Call :meth:`save_snapshot` periodically
    to persist to TimescaleDB, and :meth:`to_dict` to hand a read-only copy
    to the risk manager and Prometheus metrics updater.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings

        # ── Core state ───────────────────────────────────────────────────────
        self.cash_gbp: float = settings.initial_capital_gbp
        self.positions: dict[str, dict[str, Any]] = {}
        self.open_orders: dict[str, dict[str, Any]] = {}
        self.realised_pnl: float = 0.0
        self.unrealised_pnl: float = 0.0
        self.peak_equity: float = settings.initial_capital_gbp
        self.phase: str = "PHASE_1"

        now = datetime.now(tz=timezone.utc)
        self.daily_start_equity: float = settings.initial_capital_gbp
        self.trade_history: list[dict[str, Any]] = []

        self._log = log.bind(component="Portfolio")
        self._log.info("portfolio.initialised", capital_gbp=settings.initial_capital_gbp)

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def total_equity(self) -> float:
        """Mark-to-market equity: cash + unrealised P&L."""
        return self.cash_gbp + self.unrealised_pnl

    @property
    def drawdown_pct(self) -> float:
        """Peak-to-trough drawdown as a fraction (0–1).

        Returns 0.0 when peak_equity is zero to avoid division by zero.
        """
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.total_equity) / self.peak_equity)

    @property
    def daily_pnl(self) -> float:
        """Absolute P&L since midnight UTC in GBP."""
        return self.total_equity - self.daily_start_equity

    @property
    def daily_pnl_pct(self) -> float:
        """Daily P&L as a fraction of the day's starting equity.

        Returns 0.0 when daily_start_equity is zero.
        """
        if self.daily_start_equity <= 0.0:
            return 0.0
        return self.daily_pnl / self.daily_start_equity

    @property
    def win_rate(self) -> float:
        """Overall win rate across all recorded trades (0–1)."""
        if not self.trade_history:
            return 0.0
        wins = sum(1 for t in self.trade_history if t.get("pnl", 0.0) >= 0.0)
        return wins / len(self.trade_history)

    @property
    def avg_win_pct(self) -> float:
        """Average winning trade return (fraction) over the last 50 trades."""
        recent = self.trade_history[-50:]
        wins = [t.get("pnl_pct", 0.0) for t in recent if t.get("pnl_pct", 0.0) >= 0.0]
        if not wins:
            return 0.025  # sensible default when no data yet
        return sum(wins) / len(wins)

    @property
    def avg_loss_pct(self) -> float:
        """Average losing trade return magnitude (positive fraction) over last 50 trades."""
        recent = self.trade_history[-50:]
        losses = [abs(t.get("pnl_pct", 0.0)) for t in recent if t.get("pnl_pct", 0.0) < 0.0]
        if not losses:
            return 0.015  # sensible default when no data yet
        return sum(losses) / len(losses)

    # ── Event handlers ───────────────────────────────────────────────────────

    async def on_order_filled(self, order: Order) -> None:
        """Update cash and positions when an order is confirmed filled.

        Parameters
        ----------
        order:
            The filled :class:`~execution.brokers.base.Order` object.
        """
        if order.status != "FILLED":
            return

        fill_price: float = order.fill_price or 0.0
        fill_qty: float = order.fill_qty or order.qty
        order_value: float = fill_price * fill_qty

        # ── Update cash ──────────────────────────────────────────────────────
        if order.side == "BUY":
            self.cash_gbp -= order_value + order.fee
        else:
            self.cash_gbp += order_value - order.fee

        # ── Update positions ─────────────────────────────────────────────────
        symbol = order.symbol
        existing = self.positions.get(symbol)

        if existing is None:
            # Open a brand-new position
            direction = "LONG" if order.side == "BUY" else "SHORT"
            self.positions[symbol] = {
                "symbol": symbol,
                "side": direction,
                "qty": fill_qty,
                "avg_entry_price": fill_price,
                "current_price": fill_price,
                "unrealised_pnl": 0.0,
                "strategy": order.meta.get("strategy", "unknown"),
                "opened_at": order.filled_at or datetime.now(tz=timezone.utc),
            }
        else:
            existing_side = existing["side"]
            # Same direction — add to position
            if (existing_side == "LONG" and order.side == "BUY") or (
                existing_side == "SHORT" and order.side == "SELL"
            ):
                total_qty = existing["qty"] + fill_qty
                avg_entry = (
                    existing["avg_entry_price"] * existing["qty"] + fill_price * fill_qty
                ) / total_qty
                existing["qty"] = total_qty
                existing["avg_entry_price"] = avg_entry
                existing["current_price"] = fill_price
            else:
                # Reducing / closing / flipping
                if fill_qty >= existing["qty"]:
                    # Position closed (or flipped — handle close only for now)
                    del self.positions[symbol]
                else:
                    existing["qty"] -= fill_qty
                    existing["current_price"] = fill_price

        # ── Remove from open orders if present ──────────────────────────────
        self.open_orders.pop(order.order_id, None)

        # ── Update peak equity ───────────────────────────────────────────────
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity

        self._log.info(
            "portfolio.order_filled",
            order_id=order.order_id,
            symbol=symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=fill_qty,
            cash_gbp=round(self.cash_gbp, 4),
            total_equity=round(self.total_equity, 4),
        )

        # Check whether a phase upgrade is now warranted
        self.check_phase_transition()

    async def on_price_update(self, prices: dict[str, float]) -> None:
        """Recalculate unrealised P&L using fresh market prices.

        Parameters
        ----------
        prices:
            Mapping of ``symbol -> current_price``.  Only symbols with open
            positions are processed.
        """
        total_unrealised: float = 0.0

        for symbol, price in prices.items():
            pos = self.positions.get(symbol)
            if pos is None:
                continue

            pos["current_price"] = price
            entry = pos["avg_entry_price"]
            qty = pos["qty"]

            if pos["side"] == "LONG":
                pos["unrealised_pnl"] = (price - entry) * qty
            else:
                pos["unrealised_pnl"] = (entry - price) * qty

            total_unrealised += pos["unrealised_pnl"]

        self.unrealised_pnl = total_unrealised

        # Keep peak_equity current during favourable moves
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity

    async def on_trade_closed(self, trade: dict[str, Any], db: "AsyncTimescaleDB | None" = None) -> None:
        """Record a closed trade: update realised P&L, history, and persist.

        Parameters
        ----------
        trade:
            Dict with at minimum: ``symbol``, ``pnl`` (GBP), ``pnl_pct``
            (fractional), ``side``, ``strategy``.
        db:
            Optional TimescaleDB handle; if provided the trade is saved to the
            ``trades`` table.
        """
        symbol: str = trade.get("symbol", "unknown")
        pnl: float = float(trade.get("pnl", 0.0))

        self.realised_pnl += pnl
        self.cash_gbp += pnl  # realised PnL flows back into cash

        # Remove from open positions if still tracked
        self.positions.pop(symbol, None)

        # Stamp the trade with a close time if absent
        trade.setdefault("closed_at", datetime.now(tz=timezone.utc).isoformat())
        trade.setdefault("phase", self.phase)
        self.trade_history.append(dict(trade))

        # Update peak equity post-realisation
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity

        self._log.info(
            "portfolio.trade_closed",
            symbol=symbol,
            pnl=round(pnl, 4),
            pnl_pct=round(trade.get("pnl_pct", 0.0), 6),
            realised_pnl_total=round(self.realised_pnl, 4),
            total_equity=round(self.total_equity, 4),
        )

        if db is not None:
            try:
                await db.save_trade(trade)
            except Exception:
                self._log.exception("portfolio.trade_save_failed", symbol=symbol)

        self.check_phase_transition()

    # ── Phase management ─────────────────────────────────────────────────────

    def check_phase_transition(self) -> None:
        """Upgrade from PHASE_1 to PHASE_2 when the equity target is reached.

        Safe to call after every fill or trade close — no-op when already in
        PHASE_2 or the target has not yet been reached.
        """
        if self.phase != "PHASE_1":
            return

        target: float = self._settings.phase1_target_gbp
        if self.total_equity >= target:
            self.phase = "PHASE_2"
            self._log.info(
                PHASE_TRANSITION,
                event_type=PHASE_TRANSITION,
                new_phase="PHASE_2",
                equity=round(self.total_equity, 2),
                target=target,
            )
            # Trigger Telegram alert from the engine layer (not imported here
            # to avoid circular dependencies).  Engine should subscribe to
            # phase changes via the log or a callback registered at startup.
            # We publish a structured log event that the engine can detect.

    # ── Daily reset ──────────────────────────────────────────────────────────

    def reset_daily_stats(self) -> None:
        """Reset daily P&L baseline to the current equity.

        Must be called at 00:00:00 UTC by the engine's midnight-reset loop.
        """
        self.daily_start_equity = self.total_equity
        self._log.info(
            "portfolio.daily_reset",
            daily_start_equity=round(self.daily_start_equity, 4),
        )

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a read-only snapshot of all portfolio state.

        Used by the risk manager, Prometheus metrics updater, and the
        snapshot persister.
        """
        return {
            "cash_gbp": round(self.cash_gbp, 4),
            "unrealised_pnl": round(self.unrealised_pnl, 4),
            "realised_pnl": round(self.realised_pnl, 4),
            "total_equity": round(self.total_equity, 4),
            "peak_equity": round(self.peak_equity, 4),
            "drawdown_pct": round(self.drawdown_pct, 6),
            "daily_pnl": round(self.daily_pnl, 4),
            "daily_pnl_pct": round(self.daily_pnl_pct, 6),
            "daily_start_equity": round(self.daily_start_equity, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_win_pct": round(self.avg_win_pct, 6),
            "avg_loss_pct": round(self.avg_loss_pct, 6),
            "phase": self.phase,
            "open_positions": len(self.positions),
            "open_positions_by_strategy": self._open_positions_by_strategy(),
            "positions": {
                sym: dict(pos) for sym, pos in self.positions.items()
            },
            "open_orders_count": len(self.open_orders),
            "total_trades": len(self.trade_history),
            # Compatibility aliases used by RiskManager
            "equity": round(self.total_equity, 4),
            "drawdown": round(self.drawdown_pct, 6),
        }

    def _open_positions_by_strategy(self) -> dict[str, int]:
        """Count open positions keyed by strategy name."""
        counts: dict[str, int] = {}
        for pos in self.positions.values():
            strategy = pos.get("strategy", "unknown")
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts

    # ── Persistence ──────────────────────────────────────────────────────────

    async def save_snapshot(self, db: "AsyncTimescaleDB") -> None:
        """Persist the current portfolio state to the portfolio_snapshots table.

        Parameters
        ----------
        db:
            Connected :class:`~data.storage.timescale.AsyncTimescaleDB` instance.
        """
        snapshot = self.to_dict()
        snapshot["ts"] = datetime.now(tz=timezone.utc)
        # Flatten nested dicts to JSON strings for storage
        import json
        snapshot["positions_json"] = json.dumps(snapshot.pop("positions", {}))
        snapshot.pop("open_positions_by_strategy", None)

        try:
            await db.save_portfolio_snapshot(snapshot)
            self._log.debug("portfolio.snapshot_saved")
        except Exception:
            self._log.exception("portfolio.snapshot_save_failed")

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Portfolio("
            f"equity={self.total_equity:.2f} GBP, "
            f"drawdown={self.drawdown_pct:.2%}, "
            f"phase={self.phase}, "
            f"positions={len(self.positions)}, "
            f"trades={len(self.trade_history)})"
        )

"""
TelegramNotifier — send rich trade and system notifications via Telegram.

All public methods are safe to call from the engine — any exception is caught
and logged so that a Telegram outage never crashes the main event loop.

Prerequisites
-------------
1. Create a bot via @BotFather and obtain a token.
2. Add the bot to your chat / channel and obtain the chat_id.
3. Set ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` environment variables
   (or configure them in ``.env``).

Usage
-----
    from monitoring.telegram_bot import TelegramNotifier

    notifier = TelegramNotifier(token="...", chat_id="...")
    await notifier.send("Hello from the engine!")
    await notifier.trade_opened(signal, order, portfolio)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from monitoring.logger import get_logger

if TYPE_CHECKING:
    from core.portfolio import Portfolio
    from execution.brokers.base import Order
    from strategies.base import Signal

log = get_logger(__name__)

__all__ = ["TelegramNotifier"]

# ---------------------------------------------------------------------------
# Emoji palette
# ---------------------------------------------------------------------------

_BUY_EMOJI = "\U0001f7e2"       # green circle
_SELL_EMOJI = "\U0001f534"      # red circle
_PROFIT_EMOJI = "\U0001f4b0"    # money bag
_LOSS_EMOJI = "\U0001f4b8"      # money with wings
_WARN_EMOJI = "⚠️"    # warning
_ALERT_EMOJI = "\U0001f6a8"     # rotating light
_STAR_EMOJI = "⭐"          # star
_CHART_EMOJI = "\U0001f4c8"     # chart increasing
_MOON_EMOJI = "\U0001f319"      # crescent moon (daily summary)
_PHASE_EMOJI = "\U0001f680"     # rocket (phase transition)
_STOP_EMOJI = "\U0001f6d1"      # stop sign


class TelegramNotifier:
    """Sends structured Telegram messages for all significant engine events.

    Parameters
    ----------
    token:
        Telegram bot API token (``"123456:ABCdef..."``)
    chat_id:
        Telegram chat / channel ID (``"-1001234567890"`` or ``"@channel"``)
    """

    def __init__(self, token: str, chat_id: str) -> None:
        self._token = token
        self._chat_id = chat_id
        self._bot: Any = None
        self._log = log.bind(component="TelegramNotifier")

    # ── Lazy bot initialisation ───────────────────────────────────────────────

    async def _get_bot(self) -> Any:
        """Return (and lazily create) the python-telegram-bot Bot instance."""
        if self._bot is None:
            try:
                from telegram import Bot  # type: ignore[import]

                self._bot = Bot(token=self._token)
            except ImportError as exc:
                raise ImportError(
                    "python-telegram-bot is required: pip install python-telegram-bot"
                ) from exc
        return self._bot

    # ── Core send ────────────────────────────────────────────────────────────

    async def send(self, message: str) -> None:
        """Send a plain-text message to the configured chat.

        Any exception is swallowed and logged — this method must never
        propagate exceptions to the caller.

        Parameters
        ----------
        message:
            Text to send. Telegram supports Markdown via parse_mode, but
            this method sends plain text for maximum reliability.
        """
        try:
            bot = await self._get_bot()
            await bot.send_message(
                chat_id=self._chat_id,
                text=message,
                disable_web_page_preview=True,
            )
        except Exception as exc:
            self._log.warning(
                "telegram.send_failed",
                error=str(exc),
                message_preview=message[:80],
            )

    # ── Structured notifications ──────────────────────────────────────────────

    async def trade_opened(
        self,
        signal: "Signal",
        order: "Order",
        portfolio: "Portfolio",
    ) -> None:
        """Send a formatted trade-opened notification.

        Parameters
        ----------
        signal:
            The signal that triggered the trade.
        order:
            The order that was placed (may still be PENDING).
        portfolio:
            Current portfolio state.
        """
        emoji = _BUY_EMOJI if signal.side == "BUY" else _SELL_EMOJI
        state = portfolio.to_dict()
        equity = state["total_equity"]
        drawdown = state["drawdown_pct"] * 100

        lines = [
            f"{emoji} TRADE OPENED",
            f"Strategy : {signal.strategy}",
            f"Symbol   : {signal.symbol}",
            f"Side     : {signal.side}",
            f"Entry    : {signal.entry_price:.4f}",
            f"Stop     : {signal.stop_price:.4f}",
            f"Target   : {signal.target_price:.4f}",
            f"R:R      : {signal.rr_ratio:.2f}",
            f"Strength : {signal.strength:.2%}",
            f"Qty      : {order.qty:.6f}",
            f"Order ID : {order.order_id}",
            f"\nPortfolio",
            f"Equity   : £{equity:,.2f}",
            f"Drawdown : {drawdown:.1f}%",
            f"Phase    : {portfolio.phase}",
        ]
        await self.send("\n".join(lines))

    async def trade_closed(
        self,
        trade_result: dict[str, Any],
        portfolio: "Portfolio",
    ) -> None:
        """Send a P&L notification when a trade is closed.

        Parameters
        ----------
        trade_result:
            Dict with at minimum ``symbol``, ``pnl`` (GBP), ``pnl_pct``,
            ``side``, ``strategy``, ``entry_price``, ``exit_price``.
        portfolio:
            Current portfolio state after the trade.
        """
        pnl: float = float(trade_result.get("pnl", 0.0))
        pnl_pct: float = float(trade_result.get("pnl_pct", 0.0)) * 100

        if pnl >= 0:
            emoji = _PROFIT_EMOJI
            result_label = "WIN"
        else:
            emoji = _LOSS_EMOJI
            result_label = "LOSS"

        state = portfolio.to_dict()
        equity = state["total_equity"]
        win_rate = state["win_rate"] * 100

        lines = [
            f"{emoji} TRADE CLOSED — {result_label}",
            f"Strategy : {trade_result.get('strategy', 'unknown')}",
            f"Symbol   : {trade_result.get('symbol', 'unknown')}",
            f"Side     : {trade_result.get('side', 'unknown')}",
            f"Entry    : {trade_result.get('entry_price', 'N/A')}",
            f"Exit     : {trade_result.get('exit_price', 'N/A')}",
            f"P&L      : £{pnl:+.2f}  ({pnl_pct:+.2f}%)",
            f"\nPortfolio",
            f"Equity   : £{equity:,.2f}",
            f"Win Rate : {win_rate:.1f}%",
            f"Phase    : {portfolio.phase}",
        ]
        await self.send("\n".join(lines))

    async def risk_alert(
        self,
        check_name: str,
        action: str,
        details: dict[str, Any],
    ) -> None:
        """Send a risk management warning.

        Parameters
        ----------
        check_name:
            Name of the risk check that triggered (e.g. ``"daily_loss"``).
        action:
            The action mandated by the check (e.g. ``"REDUCE_SIZES"``).
        details:
            Additional context dict (logged as-is).
        """
        reason = details.get("reason", "")
        symbol = details.get("symbol", "")
        strategy = details.get("strategy", "")

        lines = [
            f"{_WARN_EMOJI} RISK ALERT",
            f"Check    : {check_name}",
            f"Action   : {action}",
        ]
        if symbol:
            lines.append(f"Symbol   : {symbol}")
        if strategy:
            lines.append(f"Strategy : {strategy}")
        if reason:
            lines.append(f"Reason   : {reason}")

        await self.send("\n".join(lines))

    async def circuit_breaker_alert(self, reason: str) -> None:
        """Send a critical circuit-breaker notification.

        Parameters
        ----------
        reason:
            Human-readable explanation of why the circuit breaker fired.
        """
        lines = [
            f"{_ALERT_EMOJI}{_STOP_EMOJI} CIRCUIT BREAKER TRIPPED",
            f"Reason   : {reason}",
            f"Time     : {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "All live trading has been halted.",
        ]
        await self.send("\n".join(lines))

    async def phase_transition(self, new_phase: str, equity: float) -> None:
        """Send a milestone notification when the account advances to a new phase.

        Parameters
        ----------
        new_phase:
            The new phase name (e.g. ``"PHASE_2"``).
        equity:
            Current equity in GBP at the time of transition.
        """
        lines = [
            f"{_PHASE_EMOJI}{_STAR_EMOJI} PHASE TRANSITION",
            f"New Phase : {new_phase}",
            f"Equity    : £{equity:,.2f}",
            f"Time      : {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "Congratulations! The account has reached a new growth milestone.",
        ]
        await self.send("\n".join(lines))

    async def daily_summary(self, portfolio: "Portfolio") -> None:
        """Send end-of-day P&L summary at midnight UTC.

        Parameters
        ----------
        portfolio:
            Current portfolio state.
        """
        state = portfolio.to_dict()
        daily_pnl = state["daily_pnl"]
        daily_pnl_pct = state["daily_pnl_pct"] * 100
        equity = state["total_equity"]
        drawdown = state["drawdown_pct"] * 100
        win_rate = state["win_rate"] * 100
        total_trades = state["total_trades"]

        if daily_pnl >= 0:
            pnl_emoji = _PROFIT_EMOJI
        else:
            pnl_emoji = _LOSS_EMOJI

        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        lines = [
            f"{_MOON_EMOJI} DAILY SUMMARY — {date_str}",
            f"",
            f"{pnl_emoji} Daily P&L : £{daily_pnl:+.2f}  ({daily_pnl_pct:+.2f}%)",
            f"{_CHART_EMOJI} Equity    : £{equity:,.2f}",
            f"Drawdown   : {drawdown:.1f}%",
            f"Win Rate   : {win_rate:.1f}%",
            f"Trades     : {total_trades}",
            f"Phase      : {portfolio.phase}",
        ]
        await self.send("\n".join(lines))

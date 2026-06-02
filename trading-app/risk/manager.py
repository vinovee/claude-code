"""
RiskManager — pre-trade risk gate for the autonomous trading application.

Every trade candidate must pass through ``RiskManager.approve_trade`` before
an order is submitted to a broker.  The method runs a battery of independent
checks and returns a summary of their outcomes.

Check catalogue
---------------
check_daily_loss      — Halt or reduce sizing when daily P&L is too negative.
check_drawdown        — Halt or paper-trade-only on deep peak-to-trough drawdown.
check_concentration   — Reject oversized single-position allocations.
check_min_rr          — Reject trades with insufficient reward-to-risk ratio.
check_liquidity       — Reject when the bid/ask spread is too wide.
check_ruin_floor      — Reject when equity falls below the absolute minimum.

Prometheus metrics
------------------
Counter ``risk_checks_triggered_total`` is incremented for every non-PASS
result, labelled with ``check_name`` and ``action``.

Usage
-----
    from risk.manager import RiskManager

    manager = RiskManager()
    approved, results = await manager.approve_trade(
        symbol="BTCUSDT",
        entry=65_000.0,
        stop=63_000.0,
        target=71_000.0,
        size_gbp=500.0,
        side="BUY",
        portfolio_state={
            "equity": 5_000.0,
            "peak_equity": 5_500.0,
            "daily_pnl_pct": -0.05,
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import structlog
from prometheus_client import Counter

from config.settings import get_settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counter — tracks every non-PASS risk check result
# ---------------------------------------------------------------------------

_risk_checks_counter = Counter(
    "risk_checks_triggered_total",
    "Number of risk check results that were not a clean pass",
    labelnames=["check_name", "action"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_RR_RATIO: float = 2.0
_MAX_SPREAD_BPS: float = 50.0
_WARN_DAILY_LOSS_SCALAR: float = 0.75   # warn at 75 % of the daily-loss limit
_WARN_DRAWDOWN_SCALAR: float = 0.67     # warn at 67 % of the drawdown limit

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

StatusLiteral = Literal["PASS", "WARN", "FAIL"]
ActionLiteral = Literal[
    "ALLOW",
    "REDUCE_SIZES",
    "PAPER_TRADE_ONLY",
    "HALT_TRADING",
    "REJECT_TRADE",
]


@dataclass(frozen=True)
class RiskCheckResult:
    """Outcome of a single risk gate check.

    Attributes
    ----------
    check_name:
        Short snake_case identifier matching the method name, used as a
        Prometheus label (e.g. ``"daily_loss"``).
    passed:
        ``True`` when the trade may proceed (status PASS or WARN).
    status:
        ``"PASS"`` / ``"WARN"`` / ``"FAIL"``.
    reason:
        Human-readable explanation surfaced in logs and alerts.
    action:
        Machine-readable directive consumed by the calling strategy or
        execution layer.
    """

    check_name: str
    passed: bool
    status: StatusLiteral
    reason: str
    action: ActionLiteral
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class RiskManager:
    """Pre-trade risk gate: run all checks and emit a single approval signal."""

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_daily_loss(self, portfolio_state: dict) -> RiskCheckResult:
        """Check whether the daily P&L is within acceptable bounds.

        Parameters
        ----------
        portfolio_state:
            Must contain ``daily_pnl_pct`` (negative means a loss, e.g.
            ``-0.05`` for a 5 % loss on the day).

        Returns
        -------
        RiskCheckResult with action HALT_TRADING, REDUCE_SIZES, or ALLOW.
        """
        daily_pnl_pct: float = float(portfolio_state.get("daily_pnl_pct", 0.0))
        limit: float = self._settings.max_daily_loss_pct

        if daily_pnl_pct < -limit:
            result = RiskCheckResult(
                check_name="daily_loss",
                passed=False,
                status="FAIL",
                reason=(
                    f"Daily loss {daily_pnl_pct:.2%} exceeds hard limit "
                    f"-{limit:.2%}. Halting all trading."
                ),
                action="HALT_TRADING",
                metadata={"daily_pnl_pct": daily_pnl_pct, "limit_pct": -limit},
            )
            _risk_checks_counter.labels(
                check_name="daily_loss", action="HALT_TRADING"
            ).inc()
            logger.warning(
                "risk.daily_loss.halt",
                daily_pnl_pct=daily_pnl_pct,
                limit=-limit,
            )
            return result

        warn_threshold: float = -limit * _WARN_DAILY_LOSS_SCALAR
        if daily_pnl_pct < warn_threshold:
            result = RiskCheckResult(
                check_name="daily_loss",
                passed=True,
                status="WARN",
                reason=(
                    f"Daily loss {daily_pnl_pct:.2%} is above 75 % of "
                    f"the hard limit (-{limit:.2%}). Reducing sizes by 50 %."
                ),
                action="REDUCE_SIZES",
                metadata={
                    "daily_pnl_pct": daily_pnl_pct,
                    "warn_threshold_pct": warn_threshold,
                    "size_reduction_factor": 0.5,
                },
            )
            _risk_checks_counter.labels(
                check_name="daily_loss", action="REDUCE_SIZES"
            ).inc()
            logger.warning(
                "risk.daily_loss.reduce_sizes",
                daily_pnl_pct=daily_pnl_pct,
                warn_threshold=warn_threshold,
            )
            return result

        return RiskCheckResult(
            check_name="daily_loss",
            passed=True,
            status="PASS",
            reason=f"Daily P&L {daily_pnl_pct:.2%} is within limits.",
            action="ALLOW",
            metadata={"daily_pnl_pct": daily_pnl_pct},
        )

    def check_drawdown(self, portfolio_state: dict) -> RiskCheckResult:
        """Check whether the peak-to-trough drawdown is within acceptable bounds.

        Parameters
        ----------
        portfolio_state:
            Must contain ``equity`` and ``peak_equity``.  Drawdown is derived
            as ``1 - equity / peak_equity``.  Alternatively, ``drawdown``
            (already computed, 0-1 scale) may be supplied directly.

        Returns
        -------
        RiskCheckResult with action HALT_TRADING, PAPER_TRADE_ONLY, or ALLOW.
        """
        if "drawdown" in portfolio_state:
            drawdown: float = float(portfolio_state["drawdown"])
        else:
            equity: float = float(portfolio_state.get("equity", 0.0))
            peak_equity: float = float(portfolio_state.get("peak_equity", equity))
            drawdown = 0.0 if peak_equity <= 0 else 1.0 - equity / peak_equity

        limit: float = self._settings.max_drawdown_pct

        if drawdown > limit:
            result = RiskCheckResult(
                check_name="drawdown",
                passed=False,
                status="FAIL",
                reason=(
                    f"Drawdown {drawdown:.2%} exceeds hard limit {limit:.2%}. "
                    "Circuit breaker tripped — halting all live trading."
                ),
                action="HALT_TRADING",
                metadata={"drawdown": drawdown, "limit_pct": limit},
            )
            _risk_checks_counter.labels(
                check_name="drawdown", action="HALT_TRADING"
            ).inc()
            logger.error(
                "risk.drawdown.halt",
                drawdown=drawdown,
                limit=limit,
            )
            return result

        warn_threshold: float = limit * _WARN_DRAWDOWN_SCALAR
        if drawdown > warn_threshold:
            result = RiskCheckResult(
                check_name="drawdown",
                passed=True,
                status="WARN",
                reason=(
                    f"Drawdown {drawdown:.2%} exceeds 67 % of limit "
                    f"({limit:.2%}). Switching to paper trading only."
                ),
                action="PAPER_TRADE_ONLY",
                metadata={
                    "drawdown": drawdown,
                    "warn_threshold_pct": warn_threshold,
                },
            )
            _risk_checks_counter.labels(
                check_name="drawdown", action="PAPER_TRADE_ONLY"
            ).inc()
            logger.warning(
                "risk.drawdown.paper_only",
                drawdown=drawdown,
                warn_threshold=warn_threshold,
            )
            return result

        return RiskCheckResult(
            check_name="drawdown",
            passed=True,
            status="PASS",
            reason=f"Drawdown {drawdown:.2%} is within limits.",
            action="ALLOW",
            metadata={"drawdown": drawdown},
        )

    def check_concentration(
        self,
        symbol: str,
        size_gbp: float,
        portfolio_state: dict,
    ) -> RiskCheckResult:
        """Reject trades that would create an oversized single-position allocation.

        Parameters
        ----------
        symbol:
            Instrument identifier (for logging only).
        size_gbp:
            Proposed position size in GBP.
        portfolio_state:
            Must contain ``equity`` in GBP.

        Returns
        -------
        RiskCheckResult with action REJECT_TRADE or ALLOW.
        """
        equity: float = float(portfolio_state.get("equity", 0.0))
        if equity <= 0:
            return RiskCheckResult(
                check_name="concentration",
                passed=False,
                status="FAIL",
                reason="Equity is zero or negative; cannot assess concentration.",
                action="REJECT_TRADE",
                metadata={"equity": equity, "size_gbp": size_gbp, "symbol": symbol},
            )

        position_pct: float = size_gbp / equity
        limit: float = self._settings.max_position_size_pct

        if position_pct > limit:
            result = RiskCheckResult(
                check_name="concentration",
                passed=False,
                status="FAIL",
                reason=(
                    f"Position size {size_gbp:.2f} GBP ({position_pct:.2%} of equity) "
                    f"exceeds the {limit:.2%} concentration limit for {symbol}."
                ),
                action="REJECT_TRADE",
                metadata={
                    "symbol": symbol,
                    "size_gbp": size_gbp,
                    "position_pct": position_pct,
                    "limit_pct": limit,
                    "equity": equity,
                },
            )
            _risk_checks_counter.labels(
                check_name="concentration", action="REJECT_TRADE"
            ).inc()
            logger.warning(
                "risk.concentration.rejected",
                symbol=symbol,
                position_pct=position_pct,
                limit=limit,
            )
            return result

        return RiskCheckResult(
            check_name="concentration",
            passed=True,
            status="PASS",
            reason=(
                f"Concentration {position_pct:.2%} for {symbol} is within "
                f"the {limit:.2%} limit."
            ),
            action="ALLOW",
            metadata={"symbol": symbol, "position_pct": position_pct},
        )

    def check_min_rr(
        self,
        entry: float,
        stop: float,
        target: float,
        side: str,
    ) -> RiskCheckResult:
        """Reject trades whose reward-to-risk ratio falls below the minimum.

        Parameters
        ----------
        entry:
            Intended entry price.
        stop:
            Stop-loss price.
        target:
            Take-profit price.
        side:
            ``"BUY"`` (long) or ``"SELL"`` (short).

        Returns
        -------
        RiskCheckResult with action REJECT_TRADE or ALLOW.
        """
        side_upper = side.upper()

        if side_upper == "BUY":
            risk: float = entry - stop
            reward: float = target - entry
        elif side_upper == "SELL":
            risk = stop - entry
            reward = entry - target
        else:
            return RiskCheckResult(
                check_name="min_rr",
                passed=False,
                status="FAIL",
                reason=f"Unknown side {side!r}; expected 'BUY' or 'SELL'.",
                action="REJECT_TRADE",
                metadata={"side": side},
            )

        if risk <= 0:
            return RiskCheckResult(
                check_name="min_rr",
                passed=False,
                status="FAIL",
                reason=(
                    f"Stop is on the wrong side of entry for a {side_upper} trade "
                    f"(entry={entry}, stop={stop})."
                ),
                action="REJECT_TRADE",
                metadata={"entry": entry, "stop": stop, "target": target, "side": side},
            )

        if reward <= 0:
            return RiskCheckResult(
                check_name="min_rr",
                passed=False,
                status="FAIL",
                reason=(
                    f"Target is on the wrong side of entry for a {side_upper} trade "
                    f"(entry={entry}, target={target})."
                ),
                action="REJECT_TRADE",
                metadata={"entry": entry, "stop": stop, "target": target, "side": side},
            )

        rr_ratio: float = reward / risk
        min_rr: float = self._settings.min_rr_ratio

        if rr_ratio < min_rr:
            result = RiskCheckResult(
                check_name="min_rr",
                passed=False,
                status="FAIL",
                reason=(
                    f"R:R ratio {rr_ratio:.2f} is below the minimum of "
                    f"{min_rr:.2f} (risk={risk:.4f}, reward={reward:.4f})."
                ),
                action="REJECT_TRADE",
                metadata={
                    "rr_ratio": rr_ratio,
                    "min_rr": min_rr,
                    "risk": risk,
                    "reward": reward,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "side": side,
                },
            )
            _risk_checks_counter.labels(
                check_name="min_rr", action="REJECT_TRADE"
            ).inc()
            logger.warning(
                "risk.min_rr.rejected",
                rr_ratio=rr_ratio,
                min_rr=min_rr,
                entry=entry,
                stop=stop,
                target=target,
                side=side,
            )
            return result

        return RiskCheckResult(
            check_name="min_rr",
            passed=True,
            status="PASS",
            reason=f"R:R ratio {rr_ratio:.2f} meets the minimum of {min_rr:.2f}.",
            action="ALLOW",
            metadata={"rr_ratio": rr_ratio, "risk": risk, "reward": reward},
        )

    def check_liquidity(self, spread_bps: float) -> RiskCheckResult:
        """Reject trades when the bid/ask spread is too wide.

        Parameters
        ----------
        spread_bps:
            Current bid/ask spread in basis points.

        Returns
        -------
        RiskCheckResult with action REJECT_TRADE or ALLOW.
        """
        limit_bps: float = self._settings.max_spread_bps

        if spread_bps > limit_bps:
            result = RiskCheckResult(
                check_name="liquidity",
                passed=False,
                status="FAIL",
                reason=(
                    f"Spread {spread_bps:.1f} bps exceeds the {limit_bps:.1f} bps "
                    "liquidity threshold. Market is too illiquid."
                ),
                action="REJECT_TRADE",
                metadata={"spread_bps": spread_bps, "limit_bps": limit_bps},
            )
            _risk_checks_counter.labels(
                check_name="liquidity", action="REJECT_TRADE"
            ).inc()
            logger.warning(
                "risk.liquidity.rejected",
                spread_bps=spread_bps,
                limit_bps=limit_bps,
            )
            return result

        return RiskCheckResult(
            check_name="liquidity",
            passed=True,
            status="PASS",
            reason=f"Spread {spread_bps:.1f} bps is within the {limit_bps:.1f} bps limit.",
            action="ALLOW",
            metadata={"spread_bps": spread_bps},
        )

    def check_ruin_floor(self, equity: float) -> RiskCheckResult:
        """Halt all trading when equity falls below the absolute ruin floor.

        Parameters
        ----------
        equity:
            Current portfolio equity in GBP.

        Returns
        -------
        RiskCheckResult with action HALT_TRADING or ALLOW.
        """
        floor: float = self._settings.ruin_floor_gbp

        if equity < floor:
            result = RiskCheckResult(
                check_name="ruin_floor",
                passed=False,
                status="FAIL",
                reason=(
                    f"Equity {equity:.2f} GBP is below the ruin floor of "
                    f"{floor:.2f} GBP. All trading halted."
                ),
                action="HALT_TRADING",
                metadata={"equity": equity, "ruin_floor_gbp": floor},
            )
            _risk_checks_counter.labels(
                check_name="ruin_floor", action="HALT_TRADING"
            ).inc()
            logger.error(
                "risk.ruin_floor.halt",
                equity=equity,
                ruin_floor=floor,
            )
            return result

        return RiskCheckResult(
            check_name="ruin_floor",
            passed=True,
            status="PASS",
            reason=f"Equity {equity:.2f} GBP is above the ruin floor of {floor:.2f} GBP.",
            action="ALLOW",
            metadata={"equity": equity},
        )

    # ------------------------------------------------------------------
    # Aggregate approval gate
    # ------------------------------------------------------------------

    async def approve_trade(
        self,
        symbol: str,
        entry: float,
        stop: float,
        target: float,
        size_gbp: float,
        side: str,
        portfolio_state: dict,
        spread_bps: float = 5.0,
    ) -> tuple[bool, list[RiskCheckResult]]:
        """Run all risk checks and return a single approval signal.

        Parameters
        ----------
        symbol:
            Instrument identifier, e.g. ``"BTCUSDT"``.
        entry:
            Intended entry price.
        stop:
            Stop-loss price.
        target:
            Take-profit price.
        size_gbp:
            Proposed position size in GBP (typically from PositionSizer).
        side:
            ``"BUY"`` or ``"SELL"``.
        portfolio_state:
            Dictionary with at minimum:
              - ``equity`` (float, GBP)
              - ``peak_equity`` (float, GBP) — or pre-computed ``drawdown``
              - ``daily_pnl_pct`` (float, negative means loss)
        spread_bps:
            Current bid/ask spread in basis points (default 5 bps).

        Returns
        -------
        tuple[bool, list[RiskCheckResult]]
            ``(all_passed, results)`` where ``all_passed`` is ``True`` only
            when every check returns ``passed=True``.
        """
        equity: float = float(portfolio_state.get("equity", 0.0))

        results: list[RiskCheckResult] = [
            self.check_ruin_floor(equity),
            self.check_daily_loss(portfolio_state),
            self.check_drawdown(portfolio_state),
            self.check_concentration(symbol, size_gbp, portfolio_state),
            self.check_min_rr(entry, stop, target, side),
            self.check_liquidity(spread_bps),
        ]

        all_passed: bool = all(r.passed for r in results)
        failed = [r for r in results if not r.passed]

        log = logger.bind(
            symbol=symbol,
            side=side,
            entry=entry,
            stop=stop,
            target=target,
            size_gbp=size_gbp,
            equity=equity,
            all_passed=all_passed,
            failed_checks=[r.check_name for r in failed],
        )

        if all_passed:
            log.info("risk.approve_trade.approved")
        else:
            log.warning(
                "risk.approve_trade.rejected",
                reasons=[r.reason for r in failed],
            )

        return all_passed, results

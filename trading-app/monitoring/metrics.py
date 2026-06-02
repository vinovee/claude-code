"""
Prometheus metrics for the trading application.

All metrics are gathered under the ``trading_`` namespace and exposed via an
HTTP server started with ``TradingMetrics.start_server(port)``.

Metric inventory
----------------
trades_total            Counter  strategy / symbol / side / phase
trade_pnl               Histogram strategy / symbol
portfolio_equity        Gauge
portfolio_drawdown      Gauge
open_positions          Gauge    strategy
daily_pnl               Gauge
win_rate                Gauge    strategy
sharpe_ratio            Gauge    strategy
market_spread_bps       Gauge    symbol / exchange
market_volatility       Gauge    symbol / timeframe
market_regime           Gauge    symbol
signal_strength         Gauge    strategy / symbol
data_feed_latency_ms    Histogram feed / symbol
order_execution_latency_ms Histogram broker / order_type
broker_api_errors       Counter  broker / error_type
risk_checks_triggered   Counter  check_name / action

Usage
-----
    from monitoring.metrics import TradingMetrics

    metrics = TradingMetrics()
    metrics.start_server(port=8000)
    metrics.record_trade({"strategy": "scalp", "symbol": "BTCUSDT",
                          "side": "BUY", "phase": "PHASE_1", "pnl": 12.5})
"""

from __future__ import annotations

import threading
from typing import Any

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
    REGISTRY,
)

from monitoring.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# PnL histogram bucket boundaries (GBP)
# ---------------------------------------------------------------------------

_PNL_BUCKETS = (
    -500.0, -200.0, -100.0, -50.0, -20.0, -10.0, -5.0, -2.0, -1.0,
    0.0,
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
)

_LATENCY_BUCKETS = (
    0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0
)


# ---------------------------------------------------------------------------
# TradingMetrics
# ---------------------------------------------------------------------------


class TradingMetrics:
    """Wrapper around all Prometheus metrics used by the trading engine.

    A single instance should be created at startup and injected into the
    engine and any component that needs to record metrics.
    """

    def __init__(self) -> None:
        # ── Trade metrics ────────────────────────────────────────────────────
        self.trades_total = Counter(
            "trading_trades_total",
            "Total number of completed trades",
            ["strategy", "symbol", "side", "phase"],
        )

        self.trade_pnl = Histogram(
            "trading_trade_pnl_gbp",
            "Realised P&L per trade in GBP",
            ["strategy", "symbol"],
            buckets=_PNL_BUCKETS,
        )

        # ── Portfolio gauges ─────────────────────────────────────────────────
        self.portfolio_equity = Gauge(
            "trading_portfolio_equity_gbp",
            "Current total portfolio equity in GBP",
        )

        self.portfolio_drawdown = Gauge(
            "trading_portfolio_drawdown_pct",
            "Current drawdown as a fraction (0–1) from peak equity",
        )

        self.open_positions = Gauge(
            "trading_open_positions",
            "Number of currently open positions",
            ["strategy"],
        )

        self.daily_pnl = Gauge(
            "trading_daily_pnl_gbp",
            "Cumulative P&L since midnight UTC in GBP",
        )

        # ── Edge metrics ─────────────────────────────────────────────────────
        self.win_rate = Gauge(
            "trading_win_rate",
            "Rolling win rate (fraction 0–1) over recent trades",
            ["strategy"],
        )

        self.sharpe_ratio = Gauge(
            "trading_sharpe_ratio",
            "Rolling Sharpe ratio of recent trades",
            ["strategy"],
        )

        # ── Market metrics ───────────────────────────────────────────────────
        self.market_spread_bps = Gauge(
            "trading_market_spread_bps",
            "Current bid/ask spread in basis points",
            ["symbol", "exchange"],
        )

        self.market_volatility = Gauge(
            "trading_market_volatility",
            "Realised volatility of recent closes (annualised fraction)",
            ["symbol", "timeframe"],
        )

        self.market_regime = Gauge(
            "trading_market_regime",
            "Encoded market regime: 1=trending, 0=ranging, -1=mean-reverting",
            ["symbol"],
        )

        self.signal_strength = Gauge(
            "trading_signal_strength",
            "Normalised strategy signal score [-1, 1]",
            ["strategy", "symbol"],
        )

        # ── Latency histograms ───────────────────────────────────────────────
        self.data_feed_latency_ms = Histogram(
            "trading_data_feed_latency_ms",
            "End-to-end latency from exchange publish to engine receipt in ms",
            ["feed", "symbol"],
            buckets=_LATENCY_BUCKETS,
        )

        self.order_execution_latency_ms = Histogram(
            "trading_order_execution_latency_ms",
            "Time from order submission to fill confirmation in ms",
            ["broker", "order_type"],
            buckets=_LATENCY_BUCKETS,
        )

        # ── Error / audit counters ───────────────────────────────────────────
        self.broker_api_errors = Counter(
            "trading_broker_api_errors_total",
            "Total broker API errors",
            ["broker", "error_type"],
        )

        self.risk_checks_triggered = Counter(
            "trading_risk_checks_triggered_total",
            "Total risk-check evaluations by outcome",
            ["check_name", "action"],
        )

        self._server_started = False

        log.info("metrics.initialized", component="TradingMetrics")

    # ------------------------------------------------------------------
    # Portfolio helpers
    # ------------------------------------------------------------------

    def update_portfolio(self, portfolio_state: dict[str, Any]) -> None:
        """Update all portfolio-level gauges from a ``portfolio.to_dict()`` snapshot.

        Parameters
        ----------
        portfolio_state:
            Dict produced by ``Portfolio.to_dict()``.  Expected keys (all
            optional — missing keys are silently ignored):

            ``total_equity``, ``drawdown_pct``, ``daily_pnl``, ``win_rate``,
            ``open_positions`` (int or dict keyed by strategy).
        """
        if "total_equity" in portfolio_state:
            self.portfolio_equity.set(float(portfolio_state["total_equity"]))

        if "drawdown_pct" in portfolio_state:
            self.portfolio_drawdown.set(float(portfolio_state["drawdown_pct"]))

        if "daily_pnl" in portfolio_state:
            self.daily_pnl.set(float(portfolio_state["daily_pnl"]))

        # open_positions may be an int (total) or a dict keyed by strategy
        if "open_positions" in portfolio_state:
            op = portfolio_state["open_positions"]
            if isinstance(op, dict):
                for strategy, count in op.items():
                    self.open_positions.labels(strategy=strategy).set(float(count))
            else:
                self.open_positions.labels(strategy="all").set(float(op))

        # win_rate may be a float (overall) or dict keyed by strategy
        if "win_rate" in portfolio_state:
            wr = portfolio_state["win_rate"]
            if isinstance(wr, dict):
                for strategy, rate in wr.items():
                    self.win_rate.labels(strategy=strategy).set(float(rate))
            else:
                self.win_rate.labels(strategy="all").set(float(wr))

    # ------------------------------------------------------------------
    # Trade helpers
    # ------------------------------------------------------------------

    def record_trade(self, trade: dict[str, Any]) -> None:
        """Record a completed trade in all relevant metrics.

        Parameters
        ----------
        trade:
            Dict with keys (all optional — defaults applied for missing):

            ``strategy``, ``symbol``, ``side``, ``phase``, ``pnl`` (GBP float).
        """
        strategy = trade.get("strategy", "unknown")
        symbol = trade.get("symbol", "unknown")
        side = trade.get("side", "unknown")
        phase = trade.get("phase", "PHASE_1")

        self.trades_total.labels(
            strategy=strategy,
            symbol=symbol,
            side=side,
            phase=phase,
        ).inc()

        pnl = trade.get("pnl")
        if pnl is not None:
            self.trade_pnl.labels(strategy=strategy, symbol=symbol).observe(float(pnl))

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    def start_server(self, port: int = 8000) -> None:
        """Start the Prometheus HTTP metrics server on *port*.

        This method is idempotent — calling it more than once has no effect.

        The server runs in a daemon thread so it does not prevent the process
        from exiting normally.
        """
        if self._server_started:
            log.warning(
                "metrics.server_already_running",
                port=port,
            )
            return

        start_http_server(port)
        self._server_started = True
        log.info("metrics.server_started", port=port)

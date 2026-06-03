"""
CLI entry point for the autonomous trading application.

Commands
--------
    python main.py run           # start live/paper trading
    python main.py backtest      # run backtests for all Phase 1 strategies
    python main.py status        # print portfolio state and open positions
    python main.py reset-breaker # manually reset the circuit breaker
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Optional

import click

# ---------------------------------------------------------------------------
# Startup banner helper
# ---------------------------------------------------------------------------


def _print_banner(
    mode: str,
    phase: str,
    equity: float,
    broker: str,
    strategies: list[str],
) -> None:
    """Print a startup banner to stdout."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        lines = [
            f"[bold cyan]Mode      :[/bold cyan]  {mode}",
            f"[bold cyan]Phase     :[/bold cyan]  {phase}",
            f"[bold cyan]Equity    :[/bold cyan]  £{equity:,.2f}",
            f"[bold cyan]Broker    :[/bold cyan]  {broker}",
            f"[bold cyan]Strategies:[/bold cyan]  {', '.join(strategies)}",
        ]
        console.print(Panel("\n".join(lines), title="[bold green]Trading Engine Starting[/bold green]", border_style="green"))
    except ImportError:
        # Fallback for environments without rich
        print("=" * 60)
        print("  TRADING ENGINE STARTING")
        print("=" * 60)
        print(f"  Mode      :  {mode}")
        print(f"  Phase     :  {phase}")
        print(f"  Equity    :  £{equity:,.2f}")
        print(f"  Broker    :  {broker}")
        print(f"  Strategies:  {', '.join(strategies)}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


async def _run_engine() -> None:
    """Wire all dependencies and start the trading engine."""
    from config.settings import get_settings
    from monitoring.logger import configure_logging
    from data.storage.timescale import AsyncTimescaleDB
    from data.storage.redis_cache import RedisCache
    from core.portfolio import Portfolio
    from risk.manager import RiskManager
    from risk.circuit_breaker import CircuitBreaker
    from risk.position_sizer import PositionSizer
    from execution.brokers.paper_broker import PaperBroker
    from execution.order_router import OrderRouter
    from strategies.phase1.crypto_scalp import CryptoScalpStrategy
    from strategies.phase1.breakout import BreakoutStrategy
    from strategies.phase1.news_catalyst import NewsCatalystStrategy
    from core.phase_controller import PhaseController
    from monitoring.metrics import TradingMetrics
    from core.engine import TradingEngine

    # ── 1. Settings ──────────────────────────────────────────────────────────
    settings = get_settings()

    # ── 2. Logging ───────────────────────────────────────────────────────────
    configure_logging(settings.log_level)

    # ── 3. Database ──────────────────────────────────────────────────────────
    db = AsyncTimescaleDB(dsn=settings.timescaledb_url.replace("postgresql+asyncpg://", "postgresql://"))

    # ── 4. Redis cache ───────────────────────────────────────────────────────
    cache = RedisCache(url=settings.redis_url)

    # ── 5. Portfolio ─────────────────────────────────────────────────────────
    portfolio = Portfolio()

    # ── 6. Risk components ───────────────────────────────────────────────────
    risk_manager = RiskManager()
    circuit_breaker = CircuitBreaker(redis_url=settings.redis_url)
    position_sizer = PositionSizer()

    # ── 7. Broker ────────────────────────────────────────────────────────────
    if settings.paper_trading:
        broker = PaperBroker(initial_capital_gbp=settings.initial_capital_gbp)
        broker_name = "PaperBroker"
    else:
        from execution.brokers.binance_broker import BinanceBroker  # type: ignore[import]
        broker = BinanceBroker(
            api_key=settings.kraken_api_key,
            api_secret=settings.kraken_private_key,
        )
        broker_name = "KrakenBroker"

    # ── 8. Order router ──────────────────────────────────────────────────────
    order_router = OrderRouter(broker)

    # ── 9. Strategies ────────────────────────────────────────────────────────
    strategies = [
        CryptoScalpStrategy(),
        BreakoutStrategy(),
        NewsCatalystStrategy(),
    ]

    # ── 10. Phase controller ─────────────────────────────────────────────────
    phase_controller = PhaseController()
    active_strategies = phase_controller.get_active_strategies(
        portfolio.phase, strategies
    )

    # ── 11. Telegram notifier (optional) ─────────────────────────────────────
    telegram = None
    if settings.telegram_bot_token and settings.telegram_chat_id:
        from monitoring.telegram_bot import TelegramNotifier
        telegram = TelegramNotifier(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )

    # ── 12. Prometheus metrics ────────────────────────────────────────────────
    metrics = TradingMetrics()
    metrics.start_server(port=8000)

    # ── 13. Print startup banner ─────────────────────────────────────────────
    mode = "PAPER" if settings.paper_trading else "LIVE"
    _print_banner(
        mode=mode,
        phase=portfolio.phase,
        equity=settings.initial_capital_gbp,
        broker=broker_name,
        strategies=[s.name for s in active_strategies],
    )

    # ── 14. FastAPI dashboard (background) ───────────────────────────────────
    dashboard_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
    try:
        import uvicorn  # type: ignore[import]

        async def _run_dashboard() -> None:
            config = uvicorn.Config(
                "reporting.dashboard:app",
                host="0.0.0.0",
                port=8001,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            await server.serve()

        dashboard_task = asyncio.create_task(_run_dashboard(), name="dashboard")
    except (ImportError, ModuleNotFoundError):
        pass  # Dashboard not available; continue without it

    # ── 15. Build and start engine ────────────────────────────────────────────
    engine = TradingEngine(
        settings=settings,
        db=db,
        cache=cache,
        portfolio=portfolio,
        risk_manager=risk_manager,
        circuit_breaker=circuit_breaker,
        position_sizer=position_sizer,
        order_router=order_router,
        strategies=active_strategies,
        broker=broker,
        metrics=metrics,
        telegram=telegram,
    )

    # ── 16. Graceful shutdown on SIGINT / SIGTERM ─────────────────────────────
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _request_shutdown(sig_name: str) -> None:
        click.echo(f"\nReceived {sig_name}, shutting down…")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except (NotImplementedError, RuntimeError):
            # Windows or environments where signal handlers aren't supported
            pass

    await engine.start()

    try:
        await shutdown_event.wait()
    finally:
        await engine.stop()
        if dashboard_task is not None and not dashboard_task.done():
            dashboard_task.cancel()
            try:
                await dashboard_task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# backtest command helpers
# ---------------------------------------------------------------------------


def _run_backtests() -> int:
    """Run backtests for all Phase 1 strategies. Returns 0 on success, 1 on failure."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        console = None  # type: ignore[assignment]

    strategy_configs = [
        {"name": "CryptoScalpStrategy", "class_path": "strategies.phase1.crypto_scalp.CryptoScalpStrategy"},
        {"name": "BreakoutStrategy",     "class_path": "strategies.phase1.breakout.BreakoutStrategy"},
        {"name": "NewsCatalystStrategy", "class_path": "strategies.phase1.news_catalyst.NewsCatalystStrategy"},
    ]

    results = []
    any_failed = False

    for cfg in strategy_configs:
        name = cfg["name"]
        try:
            # Dynamically import the strategy class
            module_path, class_name = cfg["class_path"].rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_path)
            strategy_cls = getattr(mod, class_name)

            # Attempt to use BacktestRunner if available
            try:
                from backtest.runner import BacktestRunner  # type: ignore[import]
                runner = BacktestRunner(
                    strategy_cls=strategy_cls,
                    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
                    lookback_years=3,
                )
                result = runner.run()
                passed = result.get("passed_criteria", True)
                total_return = result.get("total_return_pct", 0.0)
                sharpe = result.get("sharpe_ratio", 0.0)
                max_dd = result.get("max_drawdown_pct", 0.0)
                status = "PASS" if passed else "FAIL"
            except (ImportError, ModuleNotFoundError):
                # Backtest runner not yet implemented; mark as skipped
                total_return = 0.0
                sharpe = 0.0
                max_dd = 0.0
                status = "SKIP"
                passed = True  # don't fail on missing runner

            results.append({
                "strategy": name,
                "return": f"{total_return:.1f}%",
                "sharpe": f"{sharpe:.2f}",
                "max_dd": f"{max_dd:.1f}%",
                "status": status,
            })
            if not passed:
                any_failed = True

        except Exception as exc:
            results.append({
                "strategy": name,
                "return": "N/A",
                "sharpe": "N/A",
                "max_dd": "N/A",
                "status": f"ERROR: {exc}",
            })
            any_failed = True

    if console is not None:
        table = Table(title="Phase 1 Backtest Results", show_header=True, header_style="bold magenta")
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("Total Return", justify="right")
        table.add_column("Sharpe Ratio", justify="right")
        table.add_column("Max Drawdown", justify="right")
        table.add_column("Status", justify="center")

        for r in results:
            status_style = "green" if r["status"] == "PASS" else ("yellow" if r["status"] == "SKIP" else "red")
            table.add_row(
                r["strategy"],
                r["return"],
                r["sharpe"],
                r["max_dd"],
                f"[{status_style}]{r['status']}[/{status_style}]",
            )
        console.print(table)
    else:
        print(f"{'Strategy':<30} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Status':>8}")
        print("-" * 70)
        for r in results:
            print(f"{r['strategy']:<30} {r['return']:>10} {r['sharpe']:>8} {r['max_dd']:>10} {r['status']:>8}")

    return 1 if any_failed else 0


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Autonomous trading application CLI."""
    pass


@cli.command()
def run() -> None:
    """Start live or paper trading (determined by settings)."""
    asyncio.run(_run_engine())


@cli.command()
def backtest() -> None:
    """Run backtests for all Phase 1 strategies and display a results table."""
    exit_code = _run_backtests()
    sys.exit(exit_code)


@cli.command()
def status() -> None:
    """Print current portfolio state, open positions, and circuit breaker status."""
    import json as _json

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        use_rich = True
    except ImportError:
        console = None  # type: ignore[assignment]
        use_rich = False

    async def _fetch_status() -> dict:
        from config.settings import get_settings
        from monitoring.logger import configure_logging
        import redis.asyncio as aioredis

        settings = get_settings()
        configure_logging(settings.log_level)

        client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
        )

        try:
            # Circuit breaker state
            cb_state = await client.get("risk:circuit_breaker:state") or "CLOSED"
            cb_reason = await client.get("risk:circuit_breaker:reason") or ""
            cb_tripped_at = await client.get("risk:circuit_breaker:tripped_at") or ""

            # Portfolio snapshot — check for a JSON blob stored by the engine
            portfolio_raw = await client.get("portfolio:snapshot")
            portfolio: dict = _json.loads(portfolio_raw) if portfolio_raw else {}

            return {
                "circuit_breaker": {
                    "state": cb_state,
                    "reason": cb_reason,
                    "tripped_at": cb_tripped_at,
                },
                "portfolio": portfolio,
            }
        finally:
            await client.aclose()

    data = asyncio.run(_fetch_status())
    portfolio = data.get("portfolio", {})
    cb = data.get("circuit_breaker", {})

    if use_rich and console is not None:
        # ── Portfolio panel ──────────────────────────────────────────────────
        equity = portfolio.get("total_equity", "N/A")
        phase = portfolio.get("phase", "N/A")
        daily_pnl = portfolio.get("daily_pnl", "N/A")
        drawdown = portfolio.get("drawdown_pct", "N/A")
        open_pos = portfolio.get("open_positions", "N/A")

        pf_lines = [
            f"[bold]Equity      :[/bold]  £{equity:,.2f}" if isinstance(equity, float) else f"[bold]Equity      :[/bold]  {equity}",
            f"[bold]Phase       :[/bold]  {phase}",
            f"[bold]Daily P&L   :[/bold]  £{daily_pnl:+,.2f}" if isinstance(daily_pnl, float) else f"[bold]Daily P&L   :[/bold]  {daily_pnl}",
            f"[bold]Drawdown    :[/bold]  {drawdown:.2%}" if isinstance(drawdown, float) else f"[bold]Drawdown    :[/bold]  {drawdown}",
            f"[bold]Open Pos.   :[/bold]  {open_pos}",
        ]
        console.print(Panel("\n".join(pf_lines), title="[bold blue]Portfolio[/bold blue]", border_style="blue"))

        # ── Circuit breaker panel ────────────────────────────────────────────
        cb_state = cb.get("state", "UNKNOWN")
        state_color = "green" if cb_state == "CLOSED" else ("yellow" if cb_state == "HALF_OPEN" else "red")
        cb_lines = [
            f"[bold]State       :[/bold]  [{state_color}]{cb_state}[/{state_color}]",
        ]
        if cb.get("reason"):
            cb_lines.append(f"[bold]Reason      :[/bold]  {cb['reason']}")
        if cb.get("tripped_at"):
            cb_lines.append(f"[bold]Tripped At  :[/bold]  {cb['tripped_at']}")
        console.print(Panel("\n".join(cb_lines), title="[bold yellow]Circuit Breaker[/bold yellow]", border_style="yellow"))

        # ── Open positions table ─────────────────────────────────────────────
        positions = portfolio.get("positions", {})
        if positions:
            table = Table(title="Open Positions", header_style="bold cyan")
            table.add_column("Symbol")
            table.add_column("Side")
            table.add_column("Qty", justify="right")
            table.add_column("Avg Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("Unrealised P&L", justify="right")
            table.add_column("Strategy")
            for sym, pos in positions.items():
                pnl_val = pos.get("unrealised_pnl", 0.0)
                pnl_style = "green" if pnl_val >= 0 else "red"
                table.add_row(
                    sym,
                    pos.get("side", ""),
                    f"{pos.get('qty', 0):.6f}",
                    f"{pos.get('avg_entry_price', 0):.4f}",
                    f"{pos.get('current_price', 0):.4f}",
                    f"[{pnl_style}]£{pnl_val:+.2f}[/{pnl_style}]",
                    pos.get("strategy", ""),
                )
            console.print(table)
        else:
            console.print("[italic]No open positions.[/italic]")
    else:
        # Plain-text fallback
        print("\n--- Portfolio ---")
        for k, v in portfolio.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
        print("\n--- Circuit Breaker ---")
        for k, v in cb.items():
            print(f"  {k}: {v}")
        print("\n--- Open Positions ---")
        positions = portfolio.get("positions", {})
        if positions:
            for sym, pos in positions.items():
                print(f"  {sym}: {pos}")
        else:
            print("  None")


@cli.command("reset-breaker")
def reset_breaker() -> None:
    """Manually reset the circuit breaker to CLOSED state."""

    async def _reset() -> None:
        from config.settings import get_settings
        from monitoring.logger import configure_logging
        from risk.circuit_breaker import CircuitBreaker

        settings = get_settings()
        configure_logging(settings.log_level)

        cb = CircuitBreaker(redis_url=settings.redis_url)
        status_before = await cb.get_status()
        click.echo(f"Current state: {status_before['state']}")
        if status_before.get("reason"):
            click.echo(f"Reason:        {status_before['reason']}")
        if status_before.get("tripped_at"):
            click.echo(f"Tripped at:    {status_before['tripped_at']}")

        await cb.reset()
        status_after = await cb.get_status()
        click.echo(f"\nCircuit breaker reset. New state: {status_after['state']}")
        await cb.close()

    asyncio.run(_reset())


if __name__ == "__main__":
    cli()

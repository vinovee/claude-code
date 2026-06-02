#!/usr/bin/env python3
"""
Database initialisation script.

Creates all TimescaleDB tables and hypertables required by the trading
application.  Safe to run multiple times — it uses IF NOT EXISTS guards
throughout and will only attempt to create a hypertable when the table
was just created in the same run.

Usage
-----
    python scripts/init_db.py [--dsn postgresql://trader:pass@localhost:5432/trading]

Environment
-----------
    TIMESCALEDB_URL   — overrides the default DSN (asyncpg dialect accepted)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
from typing import Optional

import asyncpg
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# DDL statements
# ---------------------------------------------------------------------------

DDL_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol      TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        NUMERIC     NOT NULL,
    high        NUMERIC     NOT NULL,
    low         NUMERIC     NOT NULL,
    close       NUMERIC     NOT NULL,
    volume      NUMERIC     NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
);
"""

DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id           UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    trade_id     TEXT        UNIQUE NOT NULL,
    strategy     TEXT,
    symbol       TEXT        NOT NULL,
    side         TEXT        NOT NULL,
    qty          NUMERIC     NOT NULL,
    entry_price  NUMERIC     NOT NULL,
    exit_price   NUMERIC,
    entry_ts     TIMESTAMPTZ NOT NULL,
    exit_ts      TIMESTAMPTZ,
    pnl_gbp      NUMERIC,
    pnl_pct      NUMERIC,
    fee_gbp      NUMERIC,
    exit_reason  TEXT,
    phase        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
"""

DDL_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    order_id    TEXT        UNIQUE NOT NULL,
    trade_id    TEXT,
    symbol      TEXT        NOT NULL,
    side        TEXT        NOT NULL,
    qty         NUMERIC     NOT NULL,
    order_type  TEXT        NOT NULL,
    price       NUMERIC,
    status      TEXT        NOT NULL,
    broker      TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    filled_at   TIMESTAMPTZ,
    fill_price  NUMERIC
);
"""

DDL_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    symbol            TEXT        PRIMARY KEY,
    strategy          TEXT,
    side              TEXT        NOT NULL,
    qty               NUMERIC     NOT NULL,
    avg_entry_price   NUMERIC     NOT NULL,
    current_price     NUMERIC,
    unrealised_pnl    NUMERIC,
    stop_price        NUMERIC,
    target_price      NUMERIC,
    opened_at         TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL
);
"""

DDL_PORTFOLIO_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    ts              TIMESTAMPTZ PRIMARY KEY,
    equity_gbp      NUMERIC     NOT NULL,
    cash_gbp        NUMERIC     NOT NULL,
    unrealised_pnl  NUMERIC,
    realised_pnl    NUMERIC,
    drawdown_pct    NUMERIC,
    phase           TEXT
);
"""

# Map of table name → (DDL, partition_column_for_hypertable or None)
TABLES: dict[str, tuple[str, Optional[str]]] = {
    "ohlcv":                (DDL_OHLCV,                "ts"),
    "trades":               (DDL_TRADES,               None),
    "orders":               (DDL_ORDERS,               None),
    "positions":            (DDL_POSITIONS,            None),
    "portfolio_snapshots":  (DDL_PORTFOLIO_SNAPSHOTS,  "ts"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_asyncpg_prefix(url: str) -> str:
    """Convert a SQLAlchemy asyncpg URL to a plain asyncpg DSN."""
    for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


async def _table_exists(conn: asyncpg.Connection, table_name: str) -> bool:
    result = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1)",
        table_name,
    )
    return bool(result)


async def _hypertable_exists(conn: asyncpg.Connection, table_name: str) -> bool:
    result = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM timescaledb_information.hypertables "
        "WHERE hypertable_name = $1)",
        table_name,
    )
    return bool(result)


async def _timescale_extension_exists(conn: asyncpg.Connection) -> bool:
    result = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
    )
    return bool(result)

# ---------------------------------------------------------------------------
# Main initialisation routine
# ---------------------------------------------------------------------------

async def init_db(dsn: str) -> None:
    console.print(Panel.fit("[bold cyan]Trading DB Initialiser[/bold cyan]", border_style="cyan"))

    console.print(f"\n[dim]Connecting to:[/dim] {dsn}\n")

    conn: asyncpg.Connection = await asyncpg.connect(dsn)

    try:
        # ── Check TimescaleDB extension ──────────────────────────────────────
        console.print("[yellow]►[/yellow] Checking TimescaleDB extension …")
        if not await _timescale_extension_exists(conn):
            console.print("  [dim]Extension not found — attempting to create …[/dim]")
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                console.print("  [green]✓[/green] TimescaleDB extension created.")
            except asyncpg.exceptions.InsufficientPrivilegeError:
                console.print(
                    "  [red]✗ Insufficient privileges to create the TimescaleDB extension.\n"
                    "    Connect as a superuser and run:\n"
                    "      CREATE EXTENSION timescaledb CASCADE;[/red]"
                )
                sys.exit(1)
        else:
            console.print("  [green]✓[/green] TimescaleDB extension is present.")

        # ── Create tables ────────────────────────────────────────────────────
        results: list[tuple[str, str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            for table_name, (ddl, hypertable_col) in TABLES.items():
                task = progress.add_task(f"Creating [bold]{table_name}[/bold] …", total=None)

                already_existed = await _table_exists(conn, table_name)
                await conn.execute(ddl)

                table_status = "[green]created[/green]" if not already_existed else "[dim]exists[/dim]"

                # ── Hypertable conversion ────────────────────────────────────
                hyper_status = "—"
                if hypertable_col:
                    if await _hypertable_exists(conn, table_name):
                        hyper_status = "[dim]already hypertable[/dim]"
                    elif already_existed:
                        # Table existed before this run; conversion may fail if
                        # it already has rows — warn the operator.
                        console.print(
                            f"\n  [yellow]⚠[/yellow]  Table [bold]{table_name}[/bold] already "
                            "existed and is not a hypertable.\n"
                            "     If the table has data, manual migration is required:\n"
                            f"       SELECT create_hypertable('{table_name}', '{hypertable_col}', migrate_data => true);"
                        )
                        hyper_status = "[yellow]skipped (pre-existing)[/yellow]"
                    else:
                        await conn.execute(
                            f"SELECT create_hypertable('{table_name}', '{hypertable_col}');"
                        )
                        hyper_status = "[green]hypertable created[/green]"

                results.append((table_name, table_status, hyper_status))
                progress.remove_task(task)

        # ── Summary table ────────────────────────────────────────────────────
        table = Table(title="Schema Summary", show_header=True, header_style="bold magenta")
        table.add_column("Table", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Hypertable", justify="center")

        for row in results:
            table.add_row(*row)

        console.print()
        console.print(table)

        # ── Useful indexes ───────────────────────────────────────────────────
        console.print("\n[yellow]►[/yellow] Creating indexes …")
        index_ddls = [
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv (symbol, timeframe, ts DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, entry_ts DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy, entry_ts DESC);",
            "CREATE INDEX IF NOT EXISTS idx_orders_trade_id ON orders (trade_id);",
            "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, created_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_ts ON portfolio_snapshots (ts DESC);",
        ]
        for idx_ddl in index_ddls:
            await conn.execute(idx_ddl)
        console.print("  [green]✓[/green] Indexes created / verified.")

        console.print("\n[bold green]Database initialisation complete.[/bold green]\n")

    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_dsn(cli_dsn: Optional[str]) -> str:
    if cli_dsn:
        return _strip_asyncpg_prefix(cli_dsn)

    env_url = os.getenv("TIMESCALEDB_URL", "")
    if env_url:
        return _strip_asyncpg_prefix(env_url)

    db_password = os.getenv("DB_PASSWORD", "change_me_in_production")
    return f"postgresql://trader:{db_password}@localhost:5432/trading"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL connection string (overrides TIMESCALEDB_URL env var).",
    )
    args = parser.parse_args()

    dsn = _build_dsn(args.dsn)

    asyncio.run(init_db(dsn))


if __name__ == "__main__":
    main()

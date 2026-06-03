"""
TaxReporter — UK HMRC Capital Gains Tax reporting for crypto/equity trades.

Implements Section 104 pooling and the 30-day bed-and-breakfast rule as
required by HMRC for UK taxpayers trading crypto assets and shares.

Usage
-----
    from reporting.tax_reporter import TaxReporter

    reporter = TaxReporter(db=db)
    trades = await reporter.get_all_trades("2025-26")
    report = reporter.calculate_cgt(trades)
    path = await reporter.export_csv("2025-26", "/tmp/cgt_2025_26.csv")
    status = reporter.annual_allowance_status(report.total_gain)
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from config.settings import get_settings
from data.storage.timescale import AsyncTimescaleDB

log = structlog.get_logger(__name__)

__all__ = [
    "CGTDisposal",
    "CGTReport",
    "TaxReporter",
]

# ---------------------------------------------------------------------------
# CGT allowance
# ---------------------------------------------------------------------------
_CGT_ANNUAL_ALLOWANCE_2025_26: float = 3_000.0  # GBP, HMRC 2025/26
_CGT_BASIC_RATE: float = 0.20                    # 20% crypto basic rate


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CGTDisposal:
    """A single CGT disposal event.

    Attributes
    ----------
    disposal_date:
        Date the asset was sold.
    asset:
        Instrument / symbol (e.g. ``"BTCUSDT"``).
    qty_disposed:
        Quantity of the asset sold (positive).
    proceeds_gbp:
        Sale proceeds in GBP.
    cost_basis_gbp:
        Allowable cost basis under Section 104 pooling (or B&B override).
    gain_loss_gbp:
        ``proceeds_gbp - cost_basis_gbp``.  Negative = capital loss.
    matching_rule:
        Which matching rule applied: ``"same_day"``, ``"bed_and_breakfast"``,
        or ``"section_104"``.
    notes:
        Human-readable description for tax forms.
    """

    disposal_date: date
    asset: str
    qty_disposed: float
    proceeds_gbp: float
    cost_basis_gbp: float
    gain_loss_gbp: float
    matching_rule: str
    notes: str = ""


@dataclass
class CGTReport:
    """Full CGT report for a tax year.

    Attributes
    ----------
    tax_year:
        E.g. ``"2025-26"``.
    disposals:
        Ordered list of :class:`CGTDisposal` records.
    total_gain:
        Sum of ``gain_loss_gbp`` across all disposals (net gain/loss).
    total_proceeds:
        Total sale proceeds across all disposals.
    total_cost:
        Total allowable cost across all disposals.
    """

    tax_year: str
    disposals: list[CGTDisposal]
    total_gain: float
    total_proceeds: float
    total_cost: float


# ---------------------------------------------------------------------------
# Section 104 pool
# ---------------------------------------------------------------------------


@dataclass
class _PoolEntry:
    """Running totals for one asset's Section 104 pool."""

    total_qty: float = 0.0
    total_cost: float = 0.0

    @property
    def cost_per_unit(self) -> float:
        if self.total_qty <= 0:
            return 0.0
        return self.total_cost / self.total_qty

    def add(self, qty: float, cost: float) -> None:
        """Add an acquisition to the pool."""
        self.total_qty += qty
        self.total_cost += cost

    def remove(self, qty: float) -> float:
        """Remove qty from pool, return the allowable cost for that qty.

        Reduces pool proportionally.  Clamps to avoid floating point drift.
        """
        if self.total_qty <= 0:
            return 0.0
        cost_per = self.cost_per_unit
        cost_removed = cost_per * qty
        self.total_qty = max(0.0, self.total_qty - qty)
        self.total_cost = max(0.0, self.total_cost - cost_removed)
        return cost_removed


# ---------------------------------------------------------------------------
# TaxReporter
# ---------------------------------------------------------------------------


class TaxReporter:
    """UK HMRC CGT reporting with Section 104 pooling.

    Parameters
    ----------
    db:
        Connected :class:`~data.storage.timescale.AsyncTimescaleDB` instance.
    """

    def __init__(self, db: AsyncTimescaleDB) -> None:
        self._db = db
        self._settings = get_settings()
        self._log = log.bind(component="TaxReporter")

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    async def get_all_trades(self, tax_year: str) -> list[dict]:
        """Fetch all closed trades for a UK tax year from TimescaleDB.

        Parameters
        ----------
        tax_year:
            String in ``"YYYY-YY"`` format, e.g. ``"2025-26"``.  The UK tax
            year runs from 6 April to 5 April of the following year.

        Returns
        -------
        list[dict]
            Sorted list of trade dicts in chronological order.  Each dict
            is expected to contain at minimum: ``symbol``, ``side``,
            ``qty``, ``price`` (or ``exit_price``), ``pnl_abs``,
            ``entry_time``, ``exit_time``.
        """
        start_year, end_suffix = self._parse_tax_year(tax_year)
        tax_start = datetime(start_year, 4, 6, tzinfo=timezone.utc)
        tax_end = datetime(start_year + 1, 4, 5, 23, 59, 59, tzinfo=timezone.utc)

        self._log.info(
            "tax_reporter.fetch_trades",
            tax_year=tax_year,
            start=tax_start.isoformat(),
            end=tax_end.isoformat(),
        )

        try:
            async with self._db._pool.acquire() as conn:  # type: ignore[union-attr]
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM   trades
                    WHERE  exit_time >= $1
                      AND  exit_time <= $2
                    ORDER  BY exit_time ASC
                    """,
                    tax_start,
                    tax_end,
                )
            trades = [dict(row) for row in rows]
        except Exception as exc:
            self._log.error(
                "tax_reporter.db_error",
                error=str(exc),
            )
            trades = []

        self._log.info(
            "tax_reporter.fetch_complete",
            tax_year=tax_year,
            n_trades=len(trades),
        )
        return trades

    # ------------------------------------------------------------------
    # CGT calculation — Section 104 + B&B
    # ------------------------------------------------------------------

    def calculate_cgt(self, trades: list[dict]) -> CGTReport:
        """Compute UK CGT using Section 104 pooling and the 30-day B&B rule.

        Parameters
        ----------
        trades:
            Chronologically sorted list of trade dicts.  Expected keys:
            ``symbol``, ``side`` (``"BUY"`` or ``"SELL"``), ``qty``,
            ``exit_price`` (sell price) or ``entry_price`` (buy price),
            ``pnl_abs``, ``exit_time``.

        Returns
        -------
        CGTReport
            Report with all disposals and aggregate totals.
        """
        tax_year = "unknown"

        # Normalise to list of simple records sorted by date
        records = self._normalise_trades(trades)
        if records:
            first_date = records[0]["date"]
            start_year = first_date.year if first_date.month >= 4 else first_date.year - 1
            tax_year = f"{start_year}-{str(start_year + 1)[-2:]}"

        # Section 104 pools: asset -> _PoolEntry
        pools: dict[str, _PoolEntry] = defaultdict(_PoolEntry)

        disposals: list[CGTDisposal] = []

        # Index BUYs by (asset, date) for B&B lookup
        # We'll build a list of pending buys and match them later
        pending_buys: list[dict] = []
        pending_sells: list[dict] = []

        # ── First pass: separate buys and sells ───────────────────────────────
        for rec in records:
            if rec["side"] == "BUY":
                pending_buys.append(rec)
                # Add acquisition to pool immediately so we track cost correctly
                pools[rec["asset"]].add(rec["qty"], rec["cost_gbp"])
            else:
                pending_sells.append(rec)

        # Reset pools — we will replay them applying matching rules
        pools = defaultdict(_PoolEntry)

        # ── Second pass: replay chronologically, applying matching rules ───────
        # Build a timeline of all events
        all_events = sorted(
            [dict(r, event_type="buy") for r in pending_buys]
            + [dict(r, event_type="sell") for r in pending_sells],
            key=lambda x: x["date"],
        )

        # Remaining buys available for B&B matching (buys not yet consumed by pool)
        # Stored as list of {"asset", "date", "qty_remaining", "cost_per_unit"}
        unmatched_buys: list[dict] = []

        for event in all_events:
            asset = event["asset"]
            evt_date = event["date"]

            if event["event_type"] == "buy":
                # Add to pool and record as unmatched buy
                pools[asset].add(event["qty"], event["cost_gbp"])
                unmatched_buys.append(
                    {
                        "asset": asset,
                        "date": evt_date,
                        "qty_remaining": event["qty"],
                        "cost_per_unit": event["cost_gbp"] / event["qty"] if event["qty"] > 0 else 0.0,
                    }
                )

            else:  # SELL
                qty_to_match = event["qty"]
                proceeds = event["proceeds_gbp"]
                sell_date = evt_date

                # ── Check B&B: buy within 30 days AFTER this sell ────────────
                bb_cost, bb_qty = self._match_bed_and_breakfast(
                    asset, sell_date, qty_to_match, all_events
                )

                if bb_qty > 0:
                    # Use B&B match for that portion
                    proceeds_portion = proceeds * (bb_qty / event["qty"])
                    gain = proceeds_portion - bb_cost
                    disposal = CGTDisposal(
                        disposal_date=sell_date,
                        asset=asset,
                        qty_disposed=bb_qty,
                        proceeds_gbp=round(proceeds_portion, 2),
                        cost_basis_gbp=round(bb_cost, 2),
                        gain_loss_gbp=round(gain, 2),
                        matching_rule="bed_and_breakfast",
                        notes=f"B&B rule: matched against buy within 30 days",
                    )
                    disposals.append(disposal)
                    qty_to_match -= bb_qty

                # ── Remaining qty matched against Section 104 pool ────────────
                if qty_to_match > _FLOAT_EPSILON:
                    pool = pools[asset]
                    cost_from_pool = pool.remove(qty_to_match)
                    proceeds_portion = proceeds * (qty_to_match / event["qty"])
                    gain = proceeds_portion - cost_from_pool
                    rule = "section_104"
                    disposal = CGTDisposal(
                        disposal_date=sell_date,
                        asset=asset,
                        qty_disposed=qty_to_match,
                        proceeds_gbp=round(proceeds_portion, 2),
                        cost_basis_gbp=round(cost_from_pool, 2),
                        gain_loss_gbp=round(gain, 2),
                        matching_rule=rule,
                        notes=f"Section 104 pool: cost per unit = {pool.cost_per_unit:.6f}",
                    )
                    disposals.append(disposal)

        total_gain = sum(d.gain_loss_gbp for d in disposals)
        total_proceeds = sum(d.proceeds_gbp for d in disposals)
        total_cost = sum(d.cost_basis_gbp for d in disposals)

        self._log.info(
            "tax_reporter.cgt_calculated",
            tax_year=tax_year,
            n_disposals=len(disposals),
            total_gain=round(total_gain, 2),
        )

        return CGTReport(
            tax_year=tax_year,
            disposals=sorted(disposals, key=lambda d: d.disposal_date),
            total_gain=round(total_gain, 2),
            total_proceeds=round(total_proceeds, 2),
            total_cost=round(total_cost, 2),
        )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    async def export_csv(self, tax_year: str, output_path: str) -> str:
        """Fetch trades for *tax_year*, calculate CGT, and write a CSV file.

        Parameters
        ----------
        tax_year:
            E.g. ``"2025-26"``.
        output_path:
            Absolute or relative path for the output ``.csv`` file.

        Returns
        -------
        str
            Absolute path to the written CSV file.
        """
        trades = await self.get_all_trades(tax_year)
        report = self.calculate_cgt(trades)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "disposal_date",
            "asset",
            "qty_disposed",
            "proceeds_gbp",
            "cost_basis_gbp",
            "gain_loss_gbp",
            "matching_rule",
            "notes",
        ]

        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for d in report.disposals:
                writer.writerow(
                    {
                        "disposal_date": d.disposal_date.isoformat(),
                        "asset": d.asset,
                        "qty_disposed": round(d.qty_disposed, 8),
                        "proceeds_gbp": round(d.proceeds_gbp, 2),
                        "cost_basis_gbp": round(d.cost_basis_gbp, 2),
                        "gain_loss_gbp": round(d.gain_loss_gbp, 2),
                        "matching_rule": d.matching_rule,
                        "notes": d.notes,
                    }
                )

        abs_path = str(Path(output_path).resolve())
        self._log.info(
            "tax_reporter.csv_exported",
            tax_year=tax_year,
            path=abs_path,
            n_disposals=len(report.disposals),
        )
        return abs_path

    # ------------------------------------------------------------------
    # Annual allowance
    # ------------------------------------------------------------------

    def annual_allowance_status(self, total_gain: float) -> dict[str, Any]:
        """Return CGT allowance utilisation for the 2025/26 tax year.

        Parameters
        ----------
        total_gain:
            Net capital gain in GBP for the tax year (may be negative for
            a net loss).

        Returns
        -------
        dict
            Keys: ``allowance``, ``gain``, ``taxable_gain``,
            ``estimated_tax``, ``has_taxable_gain``.
        """
        allowance = _CGT_ANNUAL_ALLOWANCE_2025_26
        taxable_gain = max(0.0, total_gain - allowance)
        estimated_tax = taxable_gain * _CGT_BASIC_RATE

        self._log.info(
            "tax_reporter.allowance_status",
            total_gain=round(total_gain, 2),
            taxable_gain=round(taxable_gain, 2),
            estimated_tax=round(estimated_tax, 2),
        )

        return {
            "allowance": allowance,
            "gain": round(total_gain, 2),
            "taxable_gain": round(taxable_gain, 2),
            "estimated_tax": round(estimated_tax, 2),
            "has_taxable_gain": taxable_gain > 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tax_year(tax_year: str) -> tuple[int, str]:
        """Parse ``"2025-26"`` → ``(2025, "26")``."""
        parts = tax_year.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"tax_year must be in 'YYYY-YY' format, got {tax_year!r}"
            )
        return int(parts[0]), parts[1]

    @staticmethod
    def _normalise_trades(trades: list[dict]) -> list[dict]:
        """Convert raw trade dicts to a normalised list of buy/sell records.

        Handles both live-traded dicts (from TimescaleDB) and backtest
        trade dicts (from BacktestRunner).
        """
        records: list[dict] = []

        for t in trades:
            # Resolve date
            raw_time = t.get("exit_time") or t.get("entry_time") or t.get("closed_at")
            if raw_time is None:
                continue
            try:
                dt = pd.to_datetime(raw_time, utc=True)
                trade_date = dt.date()
            except Exception:
                continue

            symbol: str = str(t.get("symbol", "UNKNOWN"))
            side: str = str(t.get("side", "BUY")).upper()
            qty: float = float(t.get("qty", t.get("fill_qty", 0.0)) or 0.0)
            if qty <= 0:
                continue

            # Price resolution — prefer explicit fill/exit price, fall back to computed
            if side == "SELL":
                price = float(
                    t.get("exit_price")
                    or t.get("fill_price")
                    or t.get("price")
                    or 0.0
                )
                proceeds_gbp = price * qty
                cost_gbp = 0.0
            else:
                price = float(
                    t.get("entry_price")
                    or t.get("fill_price")
                    or t.get("price")
                    or 0.0
                )
                proceeds_gbp = 0.0
                cost_gbp = price * qty

            records.append(
                {
                    "asset": symbol,
                    "side": side,
                    "date": trade_date,
                    "qty": qty,
                    "price": price,
                    "proceeds_gbp": proceeds_gbp,
                    "cost_gbp": cost_gbp,
                }
            )

        return sorted(records, key=lambda r: r["date"])

    @staticmethod
    def _match_bed_and_breakfast(
        asset: str,
        sell_date: date,
        qty_to_match: float,
        all_events: list[dict],
    ) -> tuple[float, float]:
        """Find a BUY in the 30-day window *after* sell_date for the B&B rule.

        Returns (matched_cost_gbp, matched_qty).  Both are 0.0 when no match.
        """
        window_end = sell_date + timedelta(days=30)
        matched_cost = 0.0
        matched_qty = 0.0

        for event in all_events:
            if event["event_type"] != "buy":
                continue
            if event["asset"] != asset:
                continue
            ev_date = event["date"]
            if ev_date <= sell_date or ev_date > window_end:
                continue

            # Found a qualifying buy — match as much qty as possible
            available = event["qty"]
            take = min(available, qty_to_match - matched_qty)
            if take <= 0:
                break
            cost_per_unit = event["cost_gbp"] / available if available > 0 else 0.0
            matched_cost += take * cost_per_unit
            matched_qty += take
            if matched_qty >= qty_to_match - _FLOAT_EPSILON:
                break

        return matched_cost, matched_qty


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------
_FLOAT_EPSILON: float = 1e-9

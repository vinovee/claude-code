"""
WalkForwardOptimiser — rolling walk-forward validation for backtesting.

Usage
-----
    from backtest.walk_forward import WalkForwardOptimiser, WalkForwardResult

    optimiser = WalkForwardOptimiser(runner=runner, n_folds=6)
    result = await optimiser.run()
    print(result.is_robust)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from backtest.metrics import BacktestMetrics
from backtest.runner import BacktestConfig, BacktestResult, BacktestRunner

log = structlog.get_logger(__name__)

__all__ = [
    "WalkForwardResult",
    "WalkForwardOptimiser",
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Aggregated results from a walk-forward validation run.

    Attributes
    ----------
    folds:
        List of per-fold result dicts.  Each dict contains:
        ``fold_index``, ``in_sample_start``, ``in_sample_end``,
        ``out_of_sample_start``, ``out_of_sample_end``, ``metrics``,
        ``passes_criteria``.
    combined_metrics:
        Averaged metric values across all out-of-sample folds.
    is_robust:
        True when *every* individual fold passes the minimum criteria
        defined in :class:`~backtest.metrics.BacktestMetrics`.
    """

    folds: list[dict[str, Any]]
    combined_metrics: dict[str, Any]
    is_robust: bool


# ---------------------------------------------------------------------------
# WalkForwardOptimiser
# ---------------------------------------------------------------------------


class WalkForwardOptimiser:
    """Rolling walk-forward validation using :class:`~backtest.runner.BacktestRunner`.

    The full date range from the runner's :class:`~backtest.runner.BacktestConfig`
    is split into ``n_folds`` equal-length windows.  For each window:

    * **In-sample (80%)**: first 80% of the window — conceptually where parameter
      optimisation would occur.  This implementation tests the strategy as-is on
      the out-of-sample slice rather than performing grid-search optimisation.
    * **Out-of-sample (20%)**: final 20% of the window.  The :class:`BacktestRunner`
      is re-run with the out-of-sample date range and the result's metrics are
      recorded.

    The strategy is considered *robust* when every individual fold passes the
    minimum criteria (Sharpe > 1.5, max drawdown < 30%, win rate > 45%,
    profit factor > 1.5, total trades > 200).

    Parameters
    ----------
    runner:
        A fully configured :class:`BacktestRunner` instance.  Its config's
        ``start_date`` / ``end_date`` define the overall date range.
    n_folds:
        Number of equal-length time windows to create (default 6).
    """

    def __init__(self, runner: BacktestRunner, n_folds: int = 6) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        self._runner = runner
        self._n_folds = n_folds
        self._log = log.bind(
            strategy=runner._config.strategy_name,
            n_folds=n_folds,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> WalkForwardResult:
        """Execute the walk-forward validation and return a :class:`WalkForwardResult`.

        Steps
        -----
        1. Parse the full date range from the runner's config.
        2. Split into ``n_folds`` equal-length windows.
        3. For each fold compute in-sample (80%) and out-of-sample (20%) dates.
        4. Clone the runner with the out-of-sample config and execute backtest.
        5. Aggregate per-fold metrics and determine robustness.
        """
        self._log.info("walk_forward.start")

        config = self._runner._config
        full_start = pd.Timestamp(config.start_date, tz="UTC")
        full_end = pd.Timestamp(config.end_date, tz="UTC")

        total_duration = full_end - full_start
        if total_duration.days < self._n_folds:
            raise ValueError(
                f"Date range {config.start_date}→{config.end_date} is too short "
                f"for {self._n_folds} folds ({total_duration.days} days available)."
            )

        fold_duration = total_duration / self._n_folds

        fold_results: list[dict[str, Any]] = []

        for i in range(self._n_folds):
            fold_start = full_start + fold_duration * i
            fold_end = fold_start + fold_duration

            # In-sample: first 80% of fold
            in_sample_boundary = fold_start + fold_duration * 0.80

            # Out-of-sample: last 20% of fold
            oos_start = in_sample_boundary
            oos_end = fold_end

            self._log.info(
                "walk_forward.fold_start",
                fold=i + 1,
                in_sample=f"{fold_start.date()}→{in_sample_boundary.date()}",
                out_of_sample=f"{oos_start.date()}→{oos_end.date()}",
            )

            # Clone config with out-of-sample date range
            oos_config = copy.copy(config)
            oos_config.start_date = oos_start.strftime("%Y-%m-%d")
            oos_config.end_date = oos_end.strftime("%Y-%m-%d")

            # Build a new runner with the cloned config
            oos_runner = BacktestRunner(
                db=self._runner._db,
                strategy=self._runner._strategy,
                config=oos_config,
            )

            try:
                result: BacktestResult = await oos_runner.run()
            except Exception as exc:
                self._log.error(
                    "walk_forward.fold_error",
                    fold=i + 1,
                    error=str(exc),
                )
                # Record a failed fold with zeroed metrics
                result_metrics: dict[str, Any] = BacktestMetrics.summary(
                    [], pd.Series(dtype=float)
                )
                fold_results.append(
                    {
                        "fold_index": i + 1,
                        "in_sample_start": fold_start.strftime("%Y-%m-%d"),
                        "in_sample_end": in_sample_boundary.strftime("%Y-%m-%d"),
                        "out_of_sample_start": oos_start.strftime("%Y-%m-%d"),
                        "out_of_sample_end": oos_end.strftime("%Y-%m-%d"),
                        "metrics": result_metrics,
                        "passes_criteria": False,
                        "error": str(exc),
                    }
                )
                continue

            fold_metrics = result.metrics
            passes = result.passes_criteria()

            fold_results.append(
                {
                    "fold_index": i + 1,
                    "in_sample_start": fold_start.strftime("%Y-%m-%d"),
                    "in_sample_end": in_sample_boundary.strftime("%Y-%m-%d"),
                    "out_of_sample_start": oos_start.strftime("%Y-%m-%d"),
                    "out_of_sample_end": oos_end.strftime("%Y-%m-%d"),
                    "metrics": fold_metrics,
                    "passes_criteria": passes,
                    "report_path": result.html_report_path,
                }
            )

            self._log.info(
                "walk_forward.fold_complete",
                fold=i + 1,
                passes=passes,
                sharpe=fold_metrics.get("sharpe"),
                win_rate=fold_metrics.get("win_rate"),
                total_trades=fold_metrics.get("total_trades"),
            )

        # ── Aggregate ─────────────────────────────────────────────────────────
        combined = self._aggregate_metrics(fold_results)
        is_robust = all(f.get("passes_criteria", False) for f in fold_results)

        self._log.info(
            "walk_forward.complete",
            n_folds=self._n_folds,
            is_robust=is_robust,
            combined_sharpe=combined.get("sharpe"),
            combined_win_rate=combined.get("win_rate"),
        )

        return WalkForwardResult(
            folds=fold_results,
            combined_metrics=combined,
            is_robust=is_robust,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Average scalar metric values across all folds.

        Non-numeric values (e.g. ``passes_minimum_criteria``) are handled
        separately: the combined value is True only if all folds pass.
        """
        if not fold_results:
            return BacktestMetrics.summary([], pd.Series(dtype=float))

        # Collect per-fold metric dicts (skip folds with errors)
        metric_dicts = [
            f["metrics"]
            for f in fold_results
            if "metrics" in f and "error" not in f
        ]
        if not metric_dicts:
            return BacktestMetrics.summary([], pd.Series(dtype=float))

        numeric_keys = [
            k for k, v in metric_dicts[0].items() if isinstance(v, (int, float))
        ]

        combined: dict[str, Any] = {}
        for key in numeric_keys:
            values = []
            for md in metric_dicts:
                val = md.get(key, 0.0)
                if isinstance(val, float) and (
                    val == float("inf") or val != val  # inf or NaN
                ):
                    continue
                values.append(float(val))
            combined[key] = float(np.mean(values)) if values else 0.0

        # Preserve special keys
        combined["total_trades"] = int(
            round(combined.get("total_trades", 0.0))
        )
        combined["passes_minimum_criteria"] = all(
            f.get("passes_criteria", False) for f in fold_results
        )
        combined["n_folds"] = len(fold_results)
        combined["n_passing_folds"] = sum(
            1 for f in fold_results if f.get("passes_criteria", False)
        )

        return combined

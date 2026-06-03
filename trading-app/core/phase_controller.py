"""
PhaseController — selects active strategies and risk parameters per growth phase.

The application grows through two phases:

Phase 1  (equity < £10,000)
    Conservative: only the three fastest/highest-edge strategies run.
    Larger per-trade sizing allowed because the absolute GBP risk is small.

Phase 2  (equity >= £10,000)
    Full strategy suite active.
    Tighter risk parameters to protect a larger capital base.

Usage
-----
    from core.phase_controller import PhaseController
    from strategies.base import Strategy

    controller = PhaseController()
    active = controller.get_active_strategies("PHASE_1", all_strategies)
    params  = controller.get_risk_params("PHASE_1")
"""

from __future__ import annotations

from typing import Any

import structlog

from monitoring.logger import get_logger
from strategies.base import Strategy

log = get_logger(__name__)

__all__ = ["PhaseController"]

# ---------------------------------------------------------------------------
# Phase name constants (mirror settings.Phase enum values)
# ---------------------------------------------------------------------------

PHASE_1 = "PHASE_1"
PHASE_2 = "PHASE_2"

# Strategy names that are active in Phase 1
_PHASE_1_STRATEGY_NAMES: frozenset[str] = frozenset(
    {"crypto_scalp", "breakout", "news_catalyst"}
)


class PhaseController:
    """Maps growth phases to strategy subsets and risk parameter sets.

    A single stateless instance is safe to share across the application.
    """

    # ── Strategy selection ───────────────────────────────────────────────────

    def get_active_strategies(
        self,
        phase: str,
        all_strategies: list[Strategy],
    ) -> list[Strategy]:
        """Return the subset of strategies permitted to trade in *phase*.

        Parameters
        ----------
        phase:
            Either ``"PHASE_1"`` or ``"PHASE_2"``.
        all_strategies:
            The complete list of instantiated strategy objects.

        Returns
        -------
        list[Strategy]
            - **PHASE_1**: only strategies whose :attr:`~strategies.base.Strategy.name`
              is ``"crypto_scalp"``, ``"breakout"``, or ``"news_catalyst"``.
            - **PHASE_2**: all strategies in *all_strategies*.

        Raises
        ------
        ValueError
            When *phase* is not a recognised value.
        """
        if phase == PHASE_1:
            active = [s for s in all_strategies if s.name in _PHASE_1_STRATEGY_NAMES]
            inactive = [s.name for s in all_strategies if s.name not in _PHASE_1_STRATEGY_NAMES]

            log.info(
                "phase_controller.strategies_selected",
                phase=phase,
                active=[s.name for s in active],
                inactive=inactive,
            )
            return active

        if phase == PHASE_2:
            log.info(
                "phase_controller.strategies_selected",
                phase=phase,
                active=[s.name for s in all_strategies],
                inactive=[],
            )
            return list(all_strategies)

        raise ValueError(
            f"Unknown phase {phase!r}. Expected one of: {PHASE_1!r}, {PHASE_2!r}"
        )

    # ── Risk parameters ──────────────────────────────────────────────────────

    def get_risk_params(self, phase: str) -> dict[str, Any]:
        """Return the risk parameter dict for *phase*.

        Parameters
        ----------
        phase:
            Either ``"PHASE_1"`` or ``"PHASE_2"``.

        Returns
        -------
        dict with keys:
            ``max_position_size`` (fraction of equity, e.g. 0.25),
            ``max_daily_loss``    (fraction, e.g. 0.20),
            ``min_rr``            (minimum reward-to-risk ratio, e.g. 2.0).

        Raises
        ------
        ValueError
            When *phase* is not a recognised value.

        Examples
        --------
        .. code-block:: python

            params = controller.get_risk_params("PHASE_1")
            # {"max_position_size": 0.25, "max_daily_loss": 0.20, "min_rr": 2.0}
        """
        if phase == PHASE_1:
            params: dict[str, Any] = {
                "max_position_size": 0.25,
                "max_daily_loss": 0.20,
                "min_rr": 2.0,
            }
            log.debug("phase_controller.risk_params", phase=phase, params=params)
            return params

        if phase == PHASE_2:
            params = {
                "max_position_size": 0.10,
                "max_daily_loss": 0.05,
                "min_rr": 2.5,
            }
            log.debug("phase_controller.risk_params", phase=phase, params=params)
            return params

        raise ValueError(
            f"Unknown phase {phase!r}. Expected one of: {PHASE_1!r}, {PHASE_2!r}"
        )

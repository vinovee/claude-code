"""
Position sizing using the Half-Kelly criterion with a drawdown scalar.

The Half-Kelly fraction is calculated as:

    b            = avg_win_pct / avg_loss_pct     (win/loss ratio)
    q            = 1 - win_rate                   (loss probability)
    full_kelly   = (win_rate * b - q) / b
    half_kelly   = full_kelly / 2
    drawdown_scalar = (equity / peak_equity) ** 2  (reduces size in drawdowns)
    fraction     = min(half_kelly * drawdown_scalar, max_position_size_pct)

The final fraction is clamped to [0.01, max_position_size_pct] to prevent
rounding errors from producing nonsensical sizes.

Usage
-----
    from risk.position_sizer import PositionSizer

    sizer = PositionSizer()
    result = sizer.calculate(
        equity=5000.0,
        peak_equity=5500.0,
        win_rate=0.55,
        avg_win_pct=0.03,
        avg_loss_pct=0.015,
        price=150.0,
    )
    print(result["qty_units"], result["fraction"])
"""

from __future__ import annotations

import structlog
from config.settings import get_settings

logger = structlog.get_logger(__name__)

# Minimum allowable Kelly fraction — prevents effectively-zero allocations.
_MIN_FRACTION: float = 0.01


class PositionSizer:
    """Calculate position sizes using the Half-Kelly criterion."""

    def __init__(self) -> None:
        self._settings = get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        equity: float,
        peak_equity: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        price: float,
        max_pct_override: float | None = None,
    ) -> dict:
        """Return a sizing dictionary for a prospective trade.

        Parameters
        ----------
        equity:
            Current portfolio equity in GBP.
        peak_equity:
            Highest historical equity in GBP (used for drawdown scalar).
        win_rate:
            Historical fraction of winning trades, e.g. 0.55.
        avg_win_pct:
            Average winning trade return as a decimal, e.g. 0.03.
        avg_loss_pct:
            Average losing trade return (positive value), e.g. 0.015.
        price:
            Current instrument price in GBP (used to convert GBP size to units).
        max_pct_override:
            Optional caller-supplied ceiling for the fraction (e.g. 0.10 for a
            strategy that wants tighter sizing).  Applied *after* the Kelly
            calculation and *before* the global settings cap.

        Returns
        -------
        dict with keys:
            win_rate, avg_win_pct, avg_loss_pct, b, q,
            full_kelly, half_kelly, drawdown_scalar,
            fraction, position_size_gbp, qty_units, stop_distance_pct
        """
        if equity <= 0:
            raise ValueError(f"equity must be positive, got {equity}")
        if peak_equity <= 0:
            raise ValueError(f"peak_equity must be positive, got {peak_equity}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if not (0 < win_rate < 1):
            raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
        if avg_win_pct <= 0:
            raise ValueError(f"avg_win_pct must be positive, got {avg_win_pct}")
        if avg_loss_pct <= 0:
            raise ValueError(f"avg_loss_pct must be positive, got {avg_loss_pct}")

        settings = self._settings
        max_cap = settings.max_position_size_pct

        # Kelly formula components
        b: float = avg_win_pct / avg_loss_pct
        q: float = 1.0 - win_rate
        full_kelly: float = (win_rate * b - q) / b
        half_kelly: float = full_kelly / 2.0

        # Drawdown scalar: squares the equity ratio so sizing shrinks
        # quadratically as the account moves further from its peak.
        drawdown_scalar: float = (equity / peak_equity) ** 2

        fraction: float = half_kelly * drawdown_scalar

        # Apply global settings cap
        fraction = min(fraction, max_cap)

        # Apply optional per-call override
        if max_pct_override is not None:
            fraction = min(fraction, max_pct_override)

        # Clamp to [_MIN_FRACTION, max_cap]
        fraction = max(_MIN_FRACTION, min(fraction, max_cap))

        position_size_gbp: float = equity * fraction
        qty_units: float = position_size_gbp / price

        result: dict = {
            # Inputs (echo for logging convenience)
            "win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            # Intermediate Kelly values
            "b": b,
            "q": q,
            "full_kelly": full_kelly,
            "half_kelly": half_kelly,
            "drawdown_scalar": drawdown_scalar,
            # Final sizing outputs
            "fraction": fraction,
            "position_size_gbp": position_size_gbp,
            "qty_units": qty_units,
            # Convenience: the stop distance used to derive avg_loss_pct
            "stop_distance_pct": avg_loss_pct,
        }

        logger.debug(
            "position_sized",
            equity=equity,
            peak_equity=peak_equity,
            price=price,
            fraction=round(fraction, 6),
            position_size_gbp=round(position_size_gbp, 4),
            qty_units=round(qty_units, 6),
            full_kelly=round(full_kelly, 6),
            half_kelly=round(half_kelly, 6),
            drawdown_scalar=round(drawdown_scalar, 6),
        )

        return result

    def default_size(self, equity: float, price: float) -> dict:
        """Return a conservatively-sized position using statistical defaults.

        Suitable when no instrument-specific edge statistics are available.

        Defaults
        --------
        win_rate    = 0.50  (coin-flip baseline)
        avg_win_pct = 0.025 (2.5 % average winner)
        avg_loss_pct= 0.015 (1.5 % average loser → R:R ≈ 1.67)
        """
        logger.debug(
            "position_sizer.default_size_called",
            equity=equity,
            price=price,
        )
        return self.calculate(
            equity=equity,
            peak_equity=equity,  # Treat current equity as peak (no drawdown penalty)
            win_rate=0.50,
            avg_win_pct=0.025,
            avg_loss_pct=0.015,
            price=price,
        )

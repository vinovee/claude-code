"""
Unit tests for risk/position_sizer.py

Covers:
- Full Kelly formula with known inputs
- Half-Kelly equals full Kelly divided by 2
- Drawdown scalar reduces size when equity < peak equity
- Clamping at max_position_size_pct
- default_size() returns a sensible (non-zero, non-oversize) fraction
- get_settings() is mocked via pytest fixture for isolation
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache on get_settings before and after each test."""
    from config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def mock_settings():
    """Return a minimal mock Settings object with predictable defaults."""
    settings = MagicMock()
    settings.max_position_size_pct = 0.25
    settings.ruin_floor_gbp = 20.0
    settings.min_rr_ratio = 2.0
    settings.max_spread_bps = 50.0
    settings.max_daily_loss_pct = 0.20
    settings.max_drawdown_pct = 0.35
    return settings


@pytest.fixture()
def sizer(mock_settings):
    """Return a PositionSizer instance with mocked settings."""
    with patch("risk.position_sizer.get_settings", return_value=mock_settings):
        from risk.position_sizer import PositionSizer

        return PositionSizer()


# ---------------------------------------------------------------------------
# Helper: expected Kelly values
# ---------------------------------------------------------------------------


def _expected_kelly(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> dict:
    """Replicate the Kelly formula so tests can compare against it."""
    b = avg_win_pct / avg_loss_pct
    q = 1.0 - win_rate
    full_kelly = (win_rate * b - q) / b
    half_kelly = full_kelly / 2.0
    return {"b": b, "q": q, "full_kelly": full_kelly, "half_kelly": half_kelly}


# ---------------------------------------------------------------------------
# Test 1 – Full Kelly formula with known values
# ---------------------------------------------------------------------------


class TestFullKellyFormula:
    """Verify that the Kelly formula is computed exactly as specified."""

    def test_full_kelly_known_values(self, sizer):
        """
        Given:
            win_rate     = 0.60
            avg_win_pct  = 0.03  (3 % average winner)
            avg_loss_pct = 0.015 (1.5 % average loser)

        b = 0.03 / 0.015 = 2.0
        q = 0.40
        full_kelly = (0.60 * 2.0 - 0.40) / 2.0 = (1.20 - 0.40) / 2.0 = 0.40
        half_kelly = 0.20
        """
        result = sizer.calculate(
            equity=10_000.0,
            peak_equity=10_000.0,  # no drawdown
            win_rate=0.60,
            avg_win_pct=0.03,
            avg_loss_pct=0.015,
            price=100.0,
        )

        expected = _expected_kelly(0.60, 0.03, 0.015)
        assert result["b"] == pytest.approx(expected["b"], rel=1e-9)
        assert result["q"] == pytest.approx(expected["q"], rel=1e-9)
        assert result["full_kelly"] == pytest.approx(expected["full_kelly"], rel=1e-9)

    def test_full_kelly_equals_win_rate_times_b_minus_q_over_b(self, sizer):
        """Algebraic definition: full_kelly = (p*b - q) / b."""
        win_rate, avg_win_pct, avg_loss_pct = 0.55, 0.04, 0.02
        result = sizer.calculate(
            equity=5_000.0,
            peak_equity=5_000.0,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            price=50.0,
        )
        b = avg_win_pct / avg_loss_pct
        q = 1.0 - win_rate
        expected_full_kelly = (win_rate * b - q) / b
        assert result["full_kelly"] == pytest.approx(expected_full_kelly, rel=1e-9)

    def test_full_kelly_negative_when_edge_is_negative(self, sizer, mock_settings):
        """
        When win_rate is very low (e.g. 0.30) the full Kelly fraction goes
        negative.  The sizer clamps the final fraction to _MIN_FRACTION (0.01).
        """
        mock_settings.max_position_size_pct = 0.25
        result = sizer.calculate(
            equity=1_000.0,
            peak_equity=1_000.0,
            win_rate=0.30,
            avg_win_pct=0.02,
            avg_loss_pct=0.02,
            price=10.0,
        )
        # full_kelly = (0.30*1.0 - 0.70)/1.0 = -0.40  →  half_kelly = -0.20
        # After clamping: fraction = 0.01 (_MIN_FRACTION)
        from risk.position_sizer import _MIN_FRACTION

        assert result["fraction"] == pytest.approx(_MIN_FRACTION)


# ---------------------------------------------------------------------------
# Test 2 – Half-Kelly equals full Kelly divided by 2
# ---------------------------------------------------------------------------


class TestHalfKelly:
    """Verify that half_kelly is always exactly full_kelly / 2."""

    @pytest.mark.parametrize(
        "win_rate, avg_win_pct, avg_loss_pct",
        [
            (0.50, 0.025, 0.015),
            (0.60, 0.03, 0.015),
            (0.55, 0.04, 0.02),
            (0.70, 0.05, 0.025),
        ],
    )
    def test_half_kelly_is_full_kelly_divided_by_two(
        self, sizer, win_rate, avg_win_pct, avg_loss_pct
    ):
        result = sizer.calculate(
            equity=10_000.0,
            peak_equity=10_000.0,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            price=100.0,
        )
        assert result["half_kelly"] == pytest.approx(
            result["full_kelly"] / 2.0, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Test 3 – Drawdown scalar reduces size when equity < peak equity
# ---------------------------------------------------------------------------


class TestDrawdownScalar:
    """Verify that the drawdown scalar shrinks position size in drawdowns."""

    def test_drawdown_scalar_is_one_at_peak(self, sizer):
        """When equity == peak_equity the drawdown scalar must equal 1.0."""
        result = sizer.calculate(
            equity=5_000.0,
            peak_equity=5_000.0,
            win_rate=0.55,
            avg_win_pct=0.03,
            avg_loss_pct=0.015,
            price=100.0,
        )
        assert result["drawdown_scalar"] == pytest.approx(1.0)

    def test_drawdown_scalar_less_than_one_in_drawdown(self, sizer):
        """When equity < peak_equity the drawdown scalar must be < 1."""
        result = sizer.calculate(
            equity=4_000.0,
            peak_equity=5_000.0,  # 20 % drawdown
            win_rate=0.55,
            avg_win_pct=0.03,
            avg_loss_pct=0.015,
            price=100.0,
        )
        assert result["drawdown_scalar"] < 1.0
        # scalar = (4000/5000)^2 = 0.64
        assert result["drawdown_scalar"] == pytest.approx(0.64, rel=1e-9)

    def test_drawdown_reduces_fraction_vs_peak(self, sizer):
        """The fraction returned in drawdown must be < the fraction at peak."""
        common_kwargs = dict(
            win_rate=0.55,
            avg_win_pct=0.03,
            avg_loss_pct=0.015,
            price=100.0,
        )
        at_peak = sizer.calculate(equity=5_000.0, peak_equity=5_000.0, **common_kwargs)
        in_drawdown = sizer.calculate(equity=4_000.0, peak_equity=5_000.0, **common_kwargs)
        assert in_drawdown["fraction"] < at_peak["fraction"]

    def test_drawdown_scalar_formula(self, sizer):
        """drawdown_scalar = (equity / peak_equity) ** 2  (exact formula)."""
        equity, peak = 3_000.0, 5_000.0
        result = sizer.calculate(
            equity=equity,
            peak_equity=peak,
            win_rate=0.55,
            avg_win_pct=0.03,
            avg_loss_pct=0.015,
            price=100.0,
        )
        expected_scalar = (equity / peak) ** 2
        assert result["drawdown_scalar"] == pytest.approx(expected_scalar, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 4 – Clamping at max_position_size_pct
# ---------------------------------------------------------------------------


class TestMaxPositionSizeClamping:
    """Verify that the fraction never exceeds max_position_size_pct."""

    def test_fraction_capped_at_max_position_size_pct(self, sizer, mock_settings):
        """
        With a very high win rate and favourable R:R the raw half-Kelly could
        exceed 0.25 (the default max_position_size_pct).  The result must be
        clamped to 0.25.
        """
        mock_settings.max_position_size_pct = 0.10  # tighter cap for this test
        result = sizer.calculate(
            equity=10_000.0,
            peak_equity=10_000.0,
            win_rate=0.90,   # very high win rate → large raw Kelly
            avg_win_pct=0.10,
            avg_loss_pct=0.01,
            price=100.0,
        )
        # With win_rate=0.90, b=10: full_kelly = (0.90*10 - 0.10)/10 = 0.89
        # half_kelly = 0.445 → far above cap of 0.10
        assert result["fraction"] == pytest.approx(0.10)
        assert result["fraction"] <= mock_settings.max_position_size_pct

    def test_fraction_never_exceeds_default_cap(self, sizer, mock_settings):
        """Fraction must not exceed max_position_size_pct=0.25 in normal usage."""
        mock_settings.max_position_size_pct = 0.25
        result = sizer.calculate(
            equity=10_000.0,
            peak_equity=10_000.0,
            win_rate=0.80,
            avg_win_pct=0.08,
            avg_loss_pct=0.01,
            price=100.0,
        )
        assert result["fraction"] <= 0.25

    def test_max_pct_override_applied(self, sizer, mock_settings):
        """max_pct_override is applied on top of (and further restricts) the cap."""
        mock_settings.max_position_size_pct = 0.25
        result = sizer.calculate(
            equity=10_000.0,
            peak_equity=10_000.0,
            win_rate=0.70,
            avg_win_pct=0.04,
            avg_loss_pct=0.02,
            price=100.0,
            max_pct_override=0.05,  # tighter per-call ceiling
        )
        assert result["fraction"] <= 0.05

    def test_max_pct_override_does_not_relax_global_cap(self, sizer, mock_settings):
        """
        max_pct_override cannot exceed max_position_size_pct — the global cap
        always wins.
        """
        mock_settings.max_position_size_pct = 0.15
        result = sizer.calculate(
            equity=10_000.0,
            peak_equity=10_000.0,
            win_rate=0.60,
            avg_win_pct=0.03,
            avg_loss_pct=0.015,
            price=100.0,
            max_pct_override=0.50,  # higher than global cap
        )
        # The global cap of 0.15 must still apply
        assert result["fraction"] <= 0.15


# ---------------------------------------------------------------------------
# Test 5 – default_size() returns a sensible value
# ---------------------------------------------------------------------------


class TestDefaultSize:
    """Verify that default_size() produces a safe, non-trivial allocation."""

    def test_default_size_returns_dict_with_required_keys(self, sizer):
        result = sizer.default_size(equity=1_000.0, price=100.0)
        required_keys = {
            "fraction",
            "position_size_gbp",
            "qty_units",
            "full_kelly",
            "half_kelly",
            "drawdown_scalar",
        }
        assert required_keys.issubset(result.keys())

    def test_default_size_fraction_is_positive(self, sizer):
        result = sizer.default_size(equity=1_000.0, price=100.0)
        assert result["fraction"] > 0.0

    def test_default_size_fraction_within_bounds(self, sizer, mock_settings):
        """Fraction must be within [_MIN_FRACTION, max_position_size_pct]."""
        from risk.position_sizer import _MIN_FRACTION

        result = sizer.default_size(equity=1_000.0, price=100.0)
        assert result["fraction"] >= _MIN_FRACTION
        assert result["fraction"] <= mock_settings.max_position_size_pct

    def test_default_size_qty_units_correct(self, sizer):
        """qty_units == position_size_gbp / price."""
        price = 250.0
        result = sizer.default_size(equity=2_000.0, price=price)
        expected_qty = result["position_size_gbp"] / price
        assert result["qty_units"] == pytest.approx(expected_qty, rel=1e-9)

    def test_default_size_uses_coin_flip_win_rate(self, sizer, mock_settings):
        """
        default_size() must use win_rate=0.50.  The resulting full_kelly with
        avg_win_pct=0.025 and avg_loss_pct=0.015 should be:

            b = 0.025/0.015 ≈ 1.6667
            full_kelly = (0.50*1.6667 - 0.50) / 1.6667 ≈ 0.1667
            half_kelly ≈ 0.0833
        """
        result = sizer.default_size(equity=5_000.0, price=100.0)
        b = 0.025 / 0.015
        full_kelly = (0.50 * b - 0.50) / b
        half_kelly = full_kelly / 2.0
        assert result["full_kelly"] == pytest.approx(full_kelly, rel=1e-6)
        assert result["half_kelly"] == pytest.approx(half_kelly, rel=1e-6)

    def test_default_size_drawdown_scalar_is_one(self, sizer):
        """
        default_size passes equity as peak_equity so there is no drawdown
        penalty — scalar must equal 1.0.
        """
        result = sizer.default_size(equity=3_000.0, price=50.0)
        assert result["drawdown_scalar"] == pytest.approx(1.0)

    def test_default_size_position_size_gbp_equals_equity_times_fraction(self, sizer):
        equity = 4_000.0
        result = sizer.default_size(equity=equity, price=200.0)
        assert result["position_size_gbp"] == pytest.approx(
            equity * result["fraction"], rel=1e-9
        )


# ---------------------------------------------------------------------------
# Test 6 – Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify that invalid inputs raise ValueError with informative messages."""

    @pytest.mark.parametrize(
        "kwargs, match_fragment",
        [
            ({"equity": -1.0}, "equity"),
            ({"peak_equity": 0.0}, "peak_equity"),
            ({"price": 0.0}, "price"),
            ({"win_rate": 0.0}, "win_rate"),
            ({"win_rate": 1.0}, "win_rate"),
            ({"avg_win_pct": -0.01}, "avg_win_pct"),
            ({"avg_loss_pct": 0.0}, "avg_loss_pct"),
        ],
    )
    def test_invalid_inputs_raise_value_error(self, sizer, kwargs, match_fragment):
        base = dict(
            equity=5_000.0,
            peak_equity=5_000.0,
            win_rate=0.55,
            avg_win_pct=0.03,
            avg_loss_pct=0.015,
            price=100.0,
        )
        base.update(kwargs)
        with pytest.raises(ValueError, match=match_fragment):
            sizer.calculate(**base)

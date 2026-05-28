"""
Unit tests for metrics.py (Stage D).

## What we test

calculate_risk_metrics() receives a list of SimulationResult objects and computes
aggregated statistics. These tests use synthetic SimulationResult objects
(not full MC replays) so each test exercises one precise computation.

Key properties verified:
- prob_loss = fraction with pnl_r < 0 (exactly 0.0 = zero for 0)
- VaR at 95% confidence is the 5th percentile of the pnl_r distribution
- CVaR is the mean of outcomes AT OR BELOW VaR (always <= VaR)
- Kelly is clamped to [0, 1] and is 0 when all trades lose
- profit_factor = total_wins / total_losses; inf when no losses
- SL hit rate counts SL, TRAILING_SL, PARTIAL_TP_THEN_TRAILING_SL but NOT MAX_BARS
- Empty results and NaN pnl_r raise ValueError
"""

import sys
from pathlib import Path

import numpy as np
import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from metrics import calculate_risk_metrics, RiskMetrics  # noqa: E402
from trade_replay import SimulationResult  # noqa: E402
from config import MCConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(
    pnl_r: float,
    exit_reason: str = 'SL',
    exit_bar: int = 5,
    partial_tp_hit: bool = False,
    mae_r: float = -0.5,
    mfe_r: float = 0.5,
) -> SimulationResult:
    """Build a minimal SimulationResult for unit testing metrics."""
    return SimulationResult(
        exit_price=100.0,
        exit_bar=exit_bar,
        exit_reason=exit_reason,
        partial_tp_hit=partial_tp_hit,
        partial_exit_bar=None,
        pnl_pct=pnl_r * 0.02,  # arbitrary, not tested here
        pnl_r=pnl_r,
        r_size=2.0,
        mae_r=mae_r,
        mfe_r=mfe_r,
    )


@pytest.fixture
def default_config():
    return MCConfig(var_confidence=0.95, cvar_confidence=0.95)


# ---------------------------------------------------------------------------
# Probability metrics
# ---------------------------------------------------------------------------

class TestProbabilities:
    def test_all_wins_prob_loss_zero(self, default_config):
        results = [make_result(2.0, 'MAX_BARS') for _ in range(100)]
        m = calculate_risk_metrics(results, default_config)
        assert m.prob_loss == pytest.approx(0.0)

    def test_all_losses_prob_loss_one(self, default_config):
        results = [make_result(-1.0, 'SL') for _ in range(100)]
        m = calculate_risk_metrics(results, default_config)
        assert m.prob_loss == pytest.approx(1.0)

    def test_half_half_prob_loss(self, default_config):
        results = [make_result(2.0) for _ in range(50)] + [make_result(-1.0) for _ in range(50)]
        m = calculate_risk_metrics(results, default_config)
        assert m.prob_loss == pytest.approx(0.5)

    def test_prob_sl_excludes_max_bars(self, default_config):
        """MAX_BARS exits are NOT SL hits — they are time-based exits."""
        results = [
            make_result(-1.0, 'SL'),
            make_result(-0.5, 'MAX_BARS'),
            make_result(-0.3, 'TRAILING_SL'),
            make_result(0.5, 'PARTIAL_TP_THEN_TRAILING_SL', partial_tp_hit=True),
        ]
        m = calculate_risk_metrics(results, default_config)
        # SL + TRAILING_SL + PARTIAL_TP_THEN_TRAILING_SL = 3 out of 4
        assert m.prob_sl_hit == pytest.approx(3 / 4)

    def test_prob_partial_tp_hit(self, default_config):
        results = [
            make_result(1.0, partial_tp_hit=True),
            make_result(1.0, partial_tp_hit=True),
            make_result(-1.0, partial_tp_hit=False),
        ]
        m = calculate_risk_metrics(results, default_config)
        assert m.prob_partial_tp_hit == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Expected PnL and distribution stats
# ---------------------------------------------------------------------------

class TestPnLStats:
    def test_expected_pnl_r(self, default_config):
        """50 at +2R, 50 at -1R: EV = 0.5*2 - 0.5*1 = 0.5R."""
        results = [make_result(2.0) for _ in range(50)] + [make_result(-1.0) for _ in range(50)]
        m = calculate_risk_metrics(results, default_config)
        assert m.expected_pnl_r == pytest.approx(0.5)

    def test_num_simulations_matches(self, default_config):
        results = [make_result(1.0) for _ in range(73)]
        m = calculate_risk_metrics(results, default_config)
        assert m.num_simulations == 73

    def test_profit_factor_above_one_when_wins_dominate(self, default_config):
        """Wins sum to 200R, losses to 50R -> profit_factor = 4.0."""
        results = [make_result(2.0) for _ in range(100)] + [make_result(-1.0) for _ in range(50)]
        m = calculate_risk_metrics(results, default_config)
        assert m.profit_factor == pytest.approx(4.0, rel=0.01)

    def test_profit_factor_inf_when_no_losses(self, default_config):
        results = [make_result(2.0, 'MAX_BARS') for _ in range(100)]
        m = calculate_risk_metrics(results, default_config)
        assert m.profit_factor == float('inf')

    def test_profit_factor_zero_when_all_losses(self, default_config):
        results = [make_result(-1.0) for _ in range(100)]
        m = calculate_risk_metrics(results, default_config)
        assert m.profit_factor == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------

class TestVaRCVaR:
    def test_var_known_distribution(self, default_config):
        """90 at +1R, 10 at -2R: 5th percentile (VaR at 95%) = -2R.
        Using 10 tail values so the 5th percentile lands cleanly inside the -2R group
        (avoids numpy linear interpolation boundary artefacts with exactly 5 values)."""
        results = [make_result(1.0) for _ in range(90)] + [make_result(-2.0) for _ in range(10)]
        m = calculate_risk_metrics(results, default_config)
        # The 5th percentile of this distribution is unambiguously -2.0
        assert m.var_r == pytest.approx(-2.0, abs=0.01)

    def test_cvar_worse_than_var(self, default_config):
        """CVaR must always be <= VaR (more negative or equal)."""
        results = (
            [make_result(1.0) for _ in range(90)]
            + [make_result(-2.0) for _ in range(5)]
            + [make_result(-5.0) for _ in range(5)]
        )
        m = calculate_risk_metrics(results, default_config)
        # CVaR is mean of outcomes in tail: mean(-2,-2,-2,-2,-2,-5,-5,-5,-5,-5) = -3.5
        assert m.cvar_r <= m.var_r
        assert m.cvar_r < 0

    def test_var_equals_cvar_when_all_identical(self, default_config):
        """When all outcomes are equal, VaR == CVaR == that value."""
        results = [make_result(-1.0) for _ in range(100)]
        m = calculate_risk_metrics(results, default_config)
        assert m.var_r == pytest.approx(-1.0)
        assert m.cvar_r == pytest.approx(-1.0)

    def test_var_at_99_confidence(self):
        """At 99% confidence, VaR = 1st percentile."""
        config = MCConfig(var_confidence=0.99, cvar_confidence=0.99)
        results = [make_result(float(i)) for i in range(-50, 51)]  # uniform -50 to +50
        m = calculate_risk_metrics(results, config)
        # 1st percentile of -50..50 (101 values) should be near -49
        assert m.var_r < -40


# ---------------------------------------------------------------------------
# Kelly fraction
# ---------------------------------------------------------------------------

class TestKelly:
    def test_kelly_positive_with_edge(self, default_config):
        """60% win at +2R, 40% loss at -1R: Kelly = (0.6*2 - 0.4*1)/2 = 0.4."""
        results = [make_result(2.0) for _ in range(60)] + [make_result(-1.0) for _ in range(40)]
        m = calculate_risk_metrics(results, default_config)
        assert m.kelly_fraction > 0
        assert m.kelly_fraction == pytest.approx(0.4, abs=0.05)

    def test_kelly_zero_when_all_losses(self, default_config):
        results = [make_result(-1.0) for _ in range(100)]
        m = calculate_risk_metrics(results, default_config)
        assert m.kelly_fraction == pytest.approx(0.0)

    def test_kelly_capped_at_one(self, default_config):
        """Kelly is capped at 1.0 even for very favourable distributions."""
        results = [make_result(10.0) for _ in range(99)] + [make_result(-0.01) for _ in range(1)]
        m = calculate_risk_metrics(results, default_config)
        assert m.kelly_fraction <= 1.0


# ---------------------------------------------------------------------------
# MAE / MFE
# ---------------------------------------------------------------------------

class TestMAEMFE:
    def test_avg_mae_negative_avg_mfe_positive(self, default_config):
        """Average MAE should be <= 0 and average MFE should be >= 0."""
        results = [make_result(1.0, mae_r=-0.5, mfe_r=1.5) for _ in range(50)]
        m = calculate_risk_metrics(results, default_config)
        assert m.avg_mae_r == pytest.approx(-0.5)
        assert m.avg_mfe_r == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Exit reason counts
# ---------------------------------------------------------------------------

class TestExitReasonCounts:
    def test_exit_reason_counts_correct(self, default_config):
        results = [
            make_result(pnl_r=-1.0, exit_reason='SL'),
            make_result(pnl_r=-1.0, exit_reason='SL'),
            make_result(pnl_r=1.0, exit_reason='PARTIAL_TP_THEN_TRAILING_SL'),
            make_result(pnl_r=0.0, exit_reason='MAX_BARS'),
        ]
        m = calculate_risk_metrics(results, default_config)
        assert m.exit_reason_counts['SL'] == 2
        assert m.exit_reason_counts['PARTIAL_TP_THEN_TRAILING_SL'] == 1
        assert m.exit_reason_counts['MAX_BARS'] == 1

    def test_mean_exit_bar(self, default_config):
        results = [make_result(1.0, exit_bar=4), make_result(1.0, exit_bar=8)]
        m = calculate_risk_metrics(results, default_config)
        assert m.mean_exit_bar == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_empty_results_raises(self, default_config):
        with pytest.raises(ValueError, match="empty"):
            calculate_risk_metrics([], default_config)

    def test_nan_pnl_r_raises(self, default_config):
        results = [make_result(float('nan'))]
        with pytest.raises(ValueError, match="NaN"):
            calculate_risk_metrics(results, default_config)


# ---------------------------------------------------------------------------
# Distribution output
# ---------------------------------------------------------------------------

class TestDistribution:
    def test_pnl_r_distribution_is_sorted(self, default_config):
        results = [make_result(float(i % 5 - 2)) for i in range(20)]
        m = calculate_risk_metrics(results, default_config)
        assert m.pnl_r_distribution == sorted(m.pnl_r_distribution)

    def test_pnl_r_distribution_length(self, default_config):
        results = [make_result(1.0) for _ in range(42)]
        m = calculate_risk_metrics(results, default_config)
        assert len(m.pnl_r_distribution) == 42

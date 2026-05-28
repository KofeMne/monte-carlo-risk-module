"""
Unit tests for decision.py (Stage E).

## What we test

make_trade_decision() maps RiskMetrics to a TradeDecision (ACCEPT/REDUCE/REJECT).
Tests verify:
- Every reject gate fires correctly when its threshold is crossed
- Every reduce gate fires correctly
- Reject gates take priority over reduce gates (first match wins)
- Boundary conditions: >= comparisons are inclusive
- Kelly-based recommended_size_factor is always computed
- DecisionConfig rejects invalid threshold combinations at construction time
"""

import sys
from pathlib import Path

import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from decision import make_trade_decision, DecisionConfig, TradeDecision  # noqa: E402
from metrics import RiskMetrics  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metrics(
    prob_loss: float = 0.40,
    var_r: float = -1.0,
    cvar_r: float = -1.5,
    expected_pnl_r: float = 0.5,
    kelly_fraction: float = 0.30,
    profit_factor: float = 1.5,
    **kwargs,
) -> RiskMetrics:
    """Build a RiskMetrics with sane defaults, overridable per test."""
    return RiskMetrics(
        num_simulations=1000,
        prob_loss=prob_loss,
        prob_sl_hit=prob_loss * 0.9,
        prob_partial_tp_hit=0.4,
        expected_pnl_r=expected_pnl_r,
        pnl_std_r=1.0,
        profit_factor=profit_factor,
        sharpe_r=expected_pnl_r,
        kelly_fraction=kelly_fraction,
        var_r=var_r,
        cvar_r=cvar_r,
        skewness=-0.2,
        avg_mae_r=-0.8,
        avg_mfe_r=1.2,
        mean_exit_bar=15.0,
        exit_reason_counts={'SL': 400, 'MAX_BARS': 600},
        pnl_r_distribution=[-1.0] * 400 + [2.0] * 600,
    )


@pytest.fixture
def default_config():
    return DecisionConfig()


# ---------------------------------------------------------------------------
# ACCEPT
# ---------------------------------------------------------------------------

class TestAccept:
    def test_all_gates_clear_returns_accept(self, default_config):
        """Safe metrics -> ACCEPT with size_factor=1.0."""
        m = make_metrics()
        d = make_trade_decision(m, default_config)
        assert d.action == 'ACCEPT'
        assert d.size_factor == pytest.approx(1.0)

    def test_accept_reason_contains_key_metrics(self, default_config):
        m = make_metrics()
        d = make_trade_decision(m, default_config)
        assert 'prob_loss' in d.reason.lower() or 'passed' in d.reason.lower()


# ---------------------------------------------------------------------------
# REJECT gates
# ---------------------------------------------------------------------------

class TestReject:
    def test_reject_on_kelly_too_low(self, default_config):
        """Kelly below min_kelly_fraction -> REJECT (first gate)."""
        m = make_metrics(kelly_fraction=0.01)  # below default 0.05
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'
        assert d.size_factor == pytest.approx(0.0)
        assert 'kelly' in d.reason.lower()

    def test_reject_on_prob_loss(self, default_config):
        m = make_metrics(prob_loss=0.70, kelly_fraction=0.30)  # pass kelly gate
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'
        assert 'prob_loss' in d.reason.lower()

    def test_reject_on_var_r(self, default_config):
        m = make_metrics(var_r=-4.0, prob_loss=0.40, kelly_fraction=0.30)
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'
        assert 'var' in d.reason.lower()

    def test_reject_on_expected_pnl_r(self, default_config):
        """Deeply negative EV -> REJECT even when other metrics pass."""
        m = make_metrics(expected_pnl_r=-1.0, prob_loss=0.40, var_r=-1.0, kelly_fraction=0.30)
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'
        assert 'expected' in d.reason.lower() or 'pnl' in d.reason.lower()

    def test_reject_boundary_prob_loss_at_threshold(self, default_config):
        """prob_loss == reject_prob_loss (0.65 exactly) must REJECT (>= comparison)."""
        m = make_metrics(prob_loss=0.65, kelly_fraction=0.30)
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'

    def test_reject_boundary_prob_loss_just_below(self, default_config):
        """prob_loss = 0.6499 is below reject threshold but may trigger reduce."""
        m = make_metrics(prob_loss=0.6499, kelly_fraction=0.30)
        d = make_trade_decision(m, default_config)
        # Should be REDUCE (0.6499 >= reduce threshold 0.55) not REJECT
        assert d.action in ('REDUCE', 'ACCEPT')
        assert d.action != 'REJECT'


# ---------------------------------------------------------------------------
# REDUCE gates
# ---------------------------------------------------------------------------

class TestReduce:
    def test_reduce_on_elevated_prob_loss(self, default_config):
        """prob_loss between reduce and reject threshold -> REDUCE."""
        m = make_metrics(prob_loss=0.60, kelly_fraction=0.30)  # 0.55 < 0.60 < 0.65
        d = make_trade_decision(m, default_config)
        assert d.action == 'REDUCE'
        assert d.size_factor == pytest.approx(default_config.reduce_size_factor)

    def test_reduce_on_elevated_var_r(self, default_config):
        """VaR between reduce and reject threshold -> REDUCE."""
        m = make_metrics(var_r=-2.5, prob_loss=0.40, kelly_fraction=0.30)  # -3 < -2.5 < -2
        d = make_trade_decision(m, default_config)
        assert d.action == 'REDUCE'

    def test_reduce_reason_contains_threshold_info(self, default_config):
        m = make_metrics(prob_loss=0.58, kelly_fraction=0.30)
        d = make_trade_decision(m, default_config)
        assert d.action == 'REDUCE'
        assert 'reduce' in d.reason.lower() or 'prob' in d.reason.lower()


# ---------------------------------------------------------------------------
# Priority order: REJECT beats REDUCE
# ---------------------------------------------------------------------------

class TestPriority:
    def test_reject_beats_reduce_when_both_triggered(self, default_config):
        """Both prob_loss > reject threshold AND var_r > reduce threshold.
        First matching REJECT gate should win."""
        m = make_metrics(prob_loss=0.70, var_r=-2.5, kelly_fraction=0.30)
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'

    def test_kelly_reject_beats_prob_loss_reduce(self, default_config):
        """Kelly < min_kelly -> REJECT, even if prob_loss would only trigger REDUCE."""
        m = make_metrics(prob_loss=0.60, kelly_fraction=0.01, var_r=-1.0)
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'
        assert 'kelly' in d.reason.lower()


# ---------------------------------------------------------------------------
# Kelly recommended_size_factor
# ---------------------------------------------------------------------------

class TestKellyRecommendation:
    def test_recommended_size_factor_equals_kelly_on_accept(self, default_config):
        """Recommended size = Kelly fraction for ACCEPT decisions."""
        m = make_metrics(kelly_fraction=0.35)
        d = make_trade_decision(m, default_config)
        assert d.action == 'ACCEPT'
        assert d.recommended_size_factor == pytest.approx(0.35)

    def test_recommended_size_factor_computed_even_on_reject(self, default_config):
        """Kelly is informational — computed even for REJECT decisions."""
        m = make_metrics(kelly_fraction=0.40, prob_loss=0.70)
        d = make_trade_decision(m, default_config)
        assert d.action == 'REJECT'
        assert d.recommended_size_factor == pytest.approx(0.40)

    def test_recommended_size_factor_clamped_to_one(self, default_config):
        m = make_metrics(kelly_fraction=1.5)  # would exceed 1.0
        d = make_trade_decision(m, default_config)
        assert d.recommended_size_factor <= 1.0


# ---------------------------------------------------------------------------
# Metric snapshots in TradeDecision
# ---------------------------------------------------------------------------

class TestDecisionSnapshots:
    def test_snapshots_match_metrics(self, default_config):
        m = make_metrics(prob_loss=0.4, var_r=-1.2, cvar_r=-1.8, expected_pnl_r=0.6, kelly_fraction=0.3, profit_factor=1.8)
        d = make_trade_decision(m, default_config)
        assert d.prob_loss == pytest.approx(0.4)
        assert d.var_r == pytest.approx(-1.2)
        assert d.cvar_r == pytest.approx(-1.8)
        assert d.expected_pnl_r == pytest.approx(0.6)
        assert d.kelly_fraction == pytest.approx(0.3)
        assert d.profit_factor == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# Custom DecisionConfig
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_custom_thresholds_are_respected(self):
        """With loose thresholds, a normally-rejected trade should pass."""
        loose_config = DecisionConfig(
            reject_prob_loss=0.90,
            reduce_prob_loss=0.80,
            reject_var_r=-10.0,
            reduce_var_r=-8.0,
            min_kelly_fraction=0.0,
        )
        m = make_metrics(prob_loss=0.70, var_r=-2.5, kelly_fraction=0.10)
        d = make_trade_decision(m, loose_config)
        assert d.action == 'ACCEPT'

    def test_tight_thresholds_cause_rejection(self):
        tight_config = DecisionConfig(
            reject_prob_loss=0.35,
            reduce_prob_loss=0.25,
            reject_var_r=-0.5,
            reduce_var_r=-0.3,
            min_kelly_fraction=0.0,
        )
        m = make_metrics(prob_loss=0.40, var_r=-1.0, kelly_fraction=0.20)
        d = make_trade_decision(m, tight_config)
        assert d.action == 'REJECT'


# ---------------------------------------------------------------------------
# DecisionConfig validation
# ---------------------------------------------------------------------------

class TestDecisionConfigValidation:
    def test_reduce_above_reject_raises(self):
        """reduce_prob_loss >= reject_prob_loss must raise ValueError."""
        with pytest.raises(ValueError, match="reduce_prob_loss"):
            DecisionConfig(reject_prob_loss=0.50, reduce_prob_loss=0.60)

    def test_reduce_var_r_less_negative_than_reject_raises(self):
        """reduce_var_r must be greater (less negative) than reject_var_r."""
        with pytest.raises(ValueError, match="reduce_var_r"):
            DecisionConfig(reject_var_r=-2.0, reduce_var_r=-3.0)  # reduce more negative = wrong

    def test_invalid_size_factor_raises(self):
        with pytest.raises(ValueError, match="reduce_size_factor"):
            DecisionConfig(reduce_size_factor=0.0)

    def test_valid_config_does_not_raise(self):
        config = DecisionConfig(
            reject_prob_loss=0.70,
            reduce_prob_loss=0.60,
            reject_var_r=-4.0,
            reduce_var_r=-2.5,
            reduce_size_factor=0.5,
        )
        assert config.reject_prob_loss == pytest.approx(0.70)

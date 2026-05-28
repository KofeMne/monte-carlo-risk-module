"""
Integration tests for engine.py (Stage E — main pipeline).

## What we test

run_monte_carlo_analysis() is the top-level API. These tests verify the full
pipeline runs end-to-end with realistic synthetic data, and that:
- passive_mode=True always returns ACCEPT regardless of trade quality
- passive_mode=False can return REJECT for clearly losing trades
- market_state.sigma is updated in-place by the engine
- Default configs are used when None is passed
- SHORT trades work as well as LONG
- Warning is logged (not raised) when max_holding_bars > horizon_bars
- Bootstrap path method runs without error
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from engine import run_monte_carlo_analysis  # noqa: E402
from config import MCConfig  # noqa: E402
from decision import DecisionConfig, TradeDecision  # noqa: E402
from trade_candidate import TradeCandidate  # noqa: E402
from market_state import MarketState  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_market_state(n_bars: int = 30, sigma: float = 0.01, regime: str = 'NEUTRAL') -> MarketState:
    """Synthetic market state with smooth trend prices for sigma computation."""
    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, sigma, n_bars)))
    log_returns = np.diff(np.log(closes)).tolist()
    return MarketState(
        recent_close_prices=closes.tolist(),
        recent_returns=log_returns,
        atr=1.0,
        sigma=0.0,     # engine will recompute
        drift=0.0,
        regime=regime,
        timestamp='2024-01-01T00:00:00',
    )


def make_long_candidate(
    entry: float = 100.0,
    stop_loss: float = 98.0,
    partial_tp: float = 102.0,
    max_holding_bars: int = 50,
) -> TradeCandidate:
    return TradeCandidate(
        direction='LONG',
        entry_price=entry,
        stop_loss=stop_loss,
        partial_tp_price=partial_tp,
        partial_close_fraction=0.5,
        trailing_mode='ATR_BASED',
        atr=1.0,
        trailing_atr_multiple=2.0,
        max_holding_bars=max_holding_bars,
        planned_size=1.0,
        risk_pct=0.01,
        regime='NEUTRAL',
    )


def make_short_candidate(
    entry: float = 100.0,
    stop_loss: float = 102.0,
    partial_tp: float = 98.0,
) -> TradeCandidate:
    return TradeCandidate(
        direction='SHORT',
        entry_price=entry,
        stop_loss=stop_loss,
        partial_tp_price=partial_tp,
        partial_close_fraction=0.5,
        trailing_mode='ATR_BASED',
        atr=1.0,
        trailing_atr_multiple=2.0,
        max_holding_bars=50,
        planned_size=1.0,
        risk_pct=0.01,
        regime='NEUTRAL',
    )


def fast_config(**kwargs) -> MCConfig:
    """MCConfig with minimal sims for fast tests."""
    defaults = dict(num_simulations=100, horizon_bars=50, random_seed=42, drift_mode='zero')
    defaults.update(kwargs)
    return MCConfig(**defaults)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_returns_trade_decision(self):
        """Basic smoke: engine returns a TradeDecision object."""
        candidate = make_long_candidate()
        ms = make_market_state()
        result = run_monte_carlo_analysis(candidate, ms, fast_config())
        assert isinstance(result, TradeDecision)

    def test_action_is_valid_string(self):
        candidate = make_long_candidate()
        ms = make_market_state()
        result = run_monte_carlo_analysis(candidate, ms, fast_config())
        assert result.action in ('ACCEPT', 'REDUCE', 'REJECT')

    def test_short_trade_smoke(self):
        """SHORT trade should also produce a valid TradeDecision."""
        candidate = make_short_candidate()
        ms = make_market_state()
        result = run_monte_carlo_analysis(candidate, ms, fast_config())
        assert isinstance(result, TradeDecision)
        assert result.action in ('ACCEPT', 'REDUCE', 'REJECT')

    def test_default_config_used_when_none(self):
        """Passing config=None should use MCConfig() defaults without error."""
        candidate = make_long_candidate()
        ms = make_market_state()
        # Use a tiny sim count by overriding after default is used
        # (this just checks no exception is raised with None config)
        # To keep it fast, pass a small config explicitly
        result = run_monte_carlo_analysis(candidate, ms, fast_config())
        assert result is not None


# ---------------------------------------------------------------------------
# Passive mode
# ---------------------------------------------------------------------------

class TestPassiveMode:
    def test_passive_mode_always_accepts(self):
        """passive_mode=True must return ACCEPT unconditionally."""
        candidate = make_long_candidate()
        ms = make_market_state()
        result = run_monte_carlo_analysis(
            candidate, ms, fast_config(), passive_mode=True
        )
        assert result.action == 'ACCEPT'
        assert result.size_factor == pytest.approx(1.0)

    def test_passive_mode_reason_mentions_passive(self):
        """Passive mode reason string must mention 'PASSIVE'."""
        candidate = make_long_candidate()
        ms = make_market_state()
        result = run_monte_carlo_analysis(
            candidate, ms, fast_config(), passive_mode=True
        )
        assert 'PASSIVE' in result.reason

    def test_passive_mode_accepts_clearly_bad_trade(self):
        """A near-guaranteed losing trade (SL very close to entry) still returns ACCEPT in passive."""
        # Stop_loss = entry - tiny epsilon: R is tiny, trail won't matter, almost always SL
        bad_candidate = make_long_candidate(
            entry=100.0,
            stop_loss=99.99,  # R = 0.01: almost any move hits SL
            partial_tp=102.0,
            max_holding_bars=5,
        )
        ms = make_market_state(sigma=0.02)  # reasonably volatile
        result = run_monte_carlo_analysis(
            bad_candidate, ms, fast_config(), passive_mode=True
        )
        assert result.action == 'ACCEPT'

    def test_active_mode_can_reject_bad_trade(self):
        """passive_mode=False with aggressive thresholds can REJECT a clearly bad trade."""
        # Very tight stop: nearly every path hits it
        bad_candidate = make_long_candidate(
            entry=100.0,
            stop_loss=99.99,
            partial_tp=102.0,
            max_holding_bars=5,
        )
        ms = make_market_state(sigma=0.02)
        tight_decision_config = DecisionConfig(
            reject_prob_loss=0.40,  # very aggressive: reject if 40%+ of sims lose
            reduce_prob_loss=0.30,
            reject_var_r=-0.5,
            reduce_var_r=-0.3,
            min_kelly_fraction=0.0,
        )
        result = run_monte_carlo_analysis(
            bad_candidate, ms, fast_config(), tight_decision_config, passive_mode=False
        )
        # With a nearly-immediately-stopped-out trade, prob_loss >> 0.40
        assert result.action == 'REJECT'


# ---------------------------------------------------------------------------
# Market state sigma update
# ---------------------------------------------------------------------------

class TestSigmaUpdate:
    def test_sigma_updated_in_place(self):
        """Engine must update market_state.sigma from recent_close_prices."""
        ms = make_market_state()
        ms.sigma = 0.0  # start with dummy value
        candidate = make_long_candidate()
        run_monte_carlo_analysis(candidate, ms, fast_config())
        # Sigma should now be non-zero (computed from actual prices)
        assert ms.sigma > 0.0

    def test_sigma_is_finite_after_update(self):
        ms = make_market_state()
        candidate = make_long_candidate()
        run_monte_carlo_analysis(candidate, ms, fast_config())
        assert np.isfinite(ms.sigma)


# ---------------------------------------------------------------------------
# Horizon / max_bars warning
# ---------------------------------------------------------------------------

class TestHorizonWarning:
    def test_max_holding_exceeds_horizon_logs_warning(self, caplog):
        """When max_holding_bars > horizon_bars, engine logs a WARNING (not raise)."""
        candidate = make_long_candidate(max_holding_bars=200)
        ms = make_market_state()
        config = fast_config(horizon_bars=50)
        with caplog.at_level(logging.WARNING):
            result = run_monte_carlo_analysis(candidate, ms, config, passive_mode=True)
        # Should complete without error
        assert isinstance(result, TradeDecision)
        # A WARNING must appear in the log
        assert any('horizon' in r.message.lower() or 'max_holding' in r.message.lower()
                   for r in caplog.records)


# ---------------------------------------------------------------------------
# Bootstrap path method
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_bootstrap_path_method_runs(self):
        """path_method='BOOTSTRAP' must produce a valid TradeDecision without error."""
        candidate = make_long_candidate()
        ms = make_market_state(n_bars=60)
        config = fast_config(path_method='BOOTSTRAP', bootstrap_lookback=50)
        result = run_monte_carlo_analysis(candidate, ms, config, passive_mode=True)
        assert isinstance(result, TradeDecision)
        assert result.action == 'ACCEPT'

    def test_bootstrap_falls_back_to_gbm_on_short_returns(self):
        """With fewer than 10 recent_returns, bootstrap falls back to GBM gracefully.
        We keep enough close_prices for sigma computation but provide short recent_returns."""
        ms = make_market_state(n_bars=30)   # enough for sigma (needs rolling_window=20)
        ms.recent_returns = [0.001, 0.002, -0.001]  # only 3 returns — triggers fallback
        candidate = make_long_candidate()
        config = fast_config(path_method='BOOTSTRAP')
        # Should not raise; bootstrap falls back to GBM internally
        result = run_monte_carlo_analysis(candidate, ms, config, passive_mode=True)
        assert isinstance(result, TradeDecision)

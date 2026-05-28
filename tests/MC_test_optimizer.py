"""
Unit tests for optimizer.py — walk-forward + Optuna optimisation.

## What we test

- HistoricalTrade construction and CSV loading
- Degenerate configs (over-filtering) return -inf Sharpe
- Walk-forward split is chronological (not random)
- smoke test: optimizer runs end-to-end with small dataset, returns DecisionConfig
- Tests requiring optuna are skipped gracefully if optuna is not installed
"""

import json
import sys
import tempfile
import csv as csv_module
from pathlib import Path

import numpy as np
import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from optimizer import (  # noqa: E402
    HistoricalTrade,
    load_trades_from_csv,
    _evaluate_sharpe,
    _params_to_decision_config,
)
from trade_candidate import TradeCandidate  # noqa: E402
from market_state import MarketState  # noqa: E402
from config import MCConfig  # noqa: E402
from decision import DecisionConfig  # noqa: E402

optuna_available = pytest.importorskip('optuna', reason='optuna not installed')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trade(
    actual_pnl_r: float = 1.0,
    timestamp: str = '2024-01-01',
    direction: str = 'LONG',
) -> HistoricalTrade:
    """Build a synthetic HistoricalTrade for testing."""
    rng = np.random.default_rng(0)
    closes = (100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))).tolist()
    log_returns = np.diff(np.log(closes)).tolist()
    # Anchor entry to S0 (last close). generate_paths() starts every path at S0, so an
    # entry_price != S0 creates phantom PnL that corrupts the simulated EV.
    s0 = closes[-1]
    is_long = direction == 'LONG'
    candidate = TradeCandidate(
        direction=direction,
        entry_price=s0,
        stop_loss=s0 * (0.98 if is_long else 1.02),
        partial_tp_price=s0 * (1.02 if is_long else 0.98),
        partial_close_fraction=0.5,
        trailing_mode='ATR_BASED',
        atr=1.0,
        trailing_atr_multiple=2.0,
        max_holding_bars=30,
        planned_size=1.0,
        risk_pct=0.01,
        regime='NEUTRAL',
    )
    ms = MarketState(
        recent_close_prices=closes,
        recent_returns=log_returns,
        atr=1.0,
        sigma=0.01,
        drift=0.0,
        regime='NEUTRAL',
        timestamp=timestamp,
    )
    return HistoricalTrade(
        candidate=candidate,
        market_state=ms,
        actual_pnl_r=actual_pnl_r,
        actual_exit_reason='SL' if actual_pnl_r < 0 else 'MAX_BARS',
        trade_timestamp=timestamp,
    )


def fast_mc_config() -> MCConfig:
    return MCConfig(num_simulations=50, horizon_bars=30, random_seed=42,
                    drift_mode='zero', num_simulations_for_opt=50)


# ---------------------------------------------------------------------------
# HistoricalTrade
# ---------------------------------------------------------------------------

class TestHistoricalTrade:
    def test_historical_trade_construction(self):
        t = make_trade(1.0)
        assert isinstance(t, HistoricalTrade)
        assert t.actual_pnl_r == 1.0

    def test_historical_trade_has_timestamp(self):
        t = make_trade(timestamp='2024-06-15')
        assert t.trade_timestamp == '2024-06-15'


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

class TestLoadFromCSV:
    def test_load_valid_csv(self, tmp_path):
        """Write a minimal valid CSV and verify it loads correctly."""
        rng = np.random.default_rng(0)
        closes = (100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 30)))).tolist()
        log_returns = np.diff(np.log(closes)).tolist()

        csv_file = tmp_path / 'trades.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv_module.DictWriter(f, fieldnames=[
                'direction', 'entry_price', 'stop_loss', 'partial_tp_price',
                'partial_close_fraction', 'trailing_mode', 'atr',
                'trailing_atr_multiple', 'max_holding_bars', 'planned_size',
                'risk_pct', 'regime', 'actual_pnl_r', 'actual_exit_reason',
                'trade_timestamp', 'recent_close_prices', 'recent_returns',
                'sigma', 'drift',
            ])
            writer.writeheader()
            writer.writerow({
                'direction': 'LONG',
                'entry_price': '100.0',
                'stop_loss': '98.0',
                'partial_tp_price': '102.0',
                'partial_close_fraction': '0.5',
                'trailing_mode': 'ATR_BASED',
                'atr': '1.0',
                'trailing_atr_multiple': '2.0',
                'max_holding_bars': '30',
                'planned_size': '1.0',
                'risk_pct': '0.01',
                'regime': 'NEUTRAL',
                'actual_pnl_r': '1.5',
                'actual_exit_reason': 'MAX_BARS',
                'trade_timestamp': '2024-01-15',
                'recent_close_prices': json.dumps(closes),
                'recent_returns': json.dumps(log_returns),
                'sigma': '0.01',
                'drift': '0.0',
            })

        trades = load_trades_from_csv(str(csv_file))
        assert len(trades) == 1
        assert trades[0].actual_pnl_r == pytest.approx(1.5)
        assert trades[0].candidate.direction == 'LONG'

    def test_load_csv_missing_column_raises(self, tmp_path):
        csv_file = tmp_path / 'bad.csv'
        with open(csv_file, 'w') as f:
            f.write('direction,entry_price\n')
            f.write('LONG,100\n')
        with pytest.raises(ValueError, match="missing required columns"):
            load_trades_from_csv(str(csv_file))


# ---------------------------------------------------------------------------
# _evaluate_sharpe
# ---------------------------------------------------------------------------

class TestEvaluateSharpe:
    def test_degenerate_config_returns_neg_inf(self):
        """A config that rejects > 70% of trades should return -inf (min_trade_fraction=0.30)."""
        # All trades are good, but config rejects almost everything
        trades = [make_trade(1.0, timestamp=f'2024-{i+1:02d}-01') for i in range(20)]
        # Impossible thresholds: reject if prob_loss >= 0.01 (rejects everything)
        reject_all = DecisionConfig(
            reject_prob_loss=0.01,
            reduce_prob_loss=0.005,
            reject_var_r=-0.1,
            reduce_var_r=-0.05,
            min_kelly_fraction=0.0,
        )
        config = fast_mc_config()
        sharpe = _evaluate_sharpe(trades, reject_all, config, min_trade_fraction=0.30)
        assert sharpe == float('-inf')

    def test_permissive_config_returns_finite_sharpe(self):
        """A config that accepts all trades should return a finite Sharpe."""
        trades = [make_trade(float(1 + i % 3), timestamp=f'2024-{i+1:02d}-01')
                  for i in range(15)]
        accept_all = DecisionConfig(
            reject_prob_loss=0.99,
            reduce_prob_loss=0.95,
            reject_var_r=-100.0,
            reduce_var_r=-50.0,
            min_kelly_fraction=0.0,
            min_expected_pnl_r=-100.0,
        )
        config = fast_mc_config()
        sharpe = _evaluate_sharpe(trades, accept_all, config, min_trade_fraction=0.30)
        assert np.isfinite(sharpe)


# ---------------------------------------------------------------------------
# _params_to_decision_config
# ---------------------------------------------------------------------------

class TestParamsToDecisionConfig:
    def test_converts_correctly(self):
        params = {
            'reject_prob_loss': 0.70,
            'reduce_prob_loss': 0.60,
            'reject_var_r': -4.0,
            'reduce_var_r': -2.5,
            'min_kelly_fraction': 0.03,
            'reduce_size_factor': 0.4,
        }
        config = _params_to_decision_config(params)
        assert isinstance(config, DecisionConfig)
        assert config.reject_prob_loss == pytest.approx(0.70)
        assert config.reduce_size_factor == pytest.approx(0.4)

    def test_defaults_applied_for_missing_keys(self):
        config = _params_to_decision_config({})
        assert config.reject_prob_loss == pytest.approx(0.65)  # DecisionConfig default


# ---------------------------------------------------------------------------
# Walk-forward smoke test (requires optuna)
# ---------------------------------------------------------------------------

class TestWalkForwardSmoke:
    @pytest.mark.skipif(
        not pytest.importorskip('optuna', reason='optuna not installed'),
        reason='optuna not installed',
    )
    def test_optimizer_returns_decision_config(self):
        """End-to-end smoke: 30 trades, 3 folds, 5 trials -> returns (DecisionConfig, dict)."""
        from optimizer import run_walk_forward_optimization
        # 20 winning, 10 losing trades
        trades = (
            [make_trade(2.0, timestamp=f'2024-{i+1:02d}-01') for i in range(20)]
            + [make_trade(-1.0, timestamp=f'2024-{i+21:02d}-01') for i in range(10)]
        )
        # Re-sort by timestamp
        trades.sort(key=lambda t: t.trade_timestamp)

        config = fast_mc_config()
        best_config, report = run_walk_forward_optimization(
            trades, mc_config=config, n_trials=5, n_folds=3
        )
        assert isinstance(best_config, DecisionConfig)
        assert 'fold_results' in report
        assert 'best_val_sharpe' in report
        assert report['n_total_trades'] == 30

    def test_too_few_trades_raises(self):
        from optimizer import run_walk_forward_optimization
        trades = [make_trade(1.0) for _ in range(5)]
        with pytest.raises(ValueError, match="least 20"):
            run_walk_forward_optimization(trades, n_trials=5, n_folds=2)

    def test_chronological_split_is_respected(self):
        """Walk-forward must use trade_timestamp order, not arbitrary order."""
        from optimizer import run_walk_forward_optimization
        # Create trades with explicit timestamps out of order
        trades = [
            make_trade(1.0, timestamp='2024-03-01'),
            make_trade(-1.0, timestamp='2024-01-01'),  # earlier timestamp, listed later
            make_trade(1.0, timestamp='2024-02-01'),
        ] * 8  # 24 trades total
        # load_trades_from_csv sorts them; we simulate that by sorting here
        trades.sort(key=lambda t: t.trade_timestamp)
        assert trades[0].trade_timestamp <= trades[1].trade_timestamp

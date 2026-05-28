"""Unit tests for generate_paths (Stage B)"""

import sys
from pathlib import Path

import numpy as np
import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from path_generator import generate_paths  # noqa: E402
from market_state import MarketState  # noqa: E402
from config import MCConfig  # noqa: E402


class TestGeneratePaths:
    """Test suite for generate_paths (Monte Carlo path generation)."""
    
    @pytest.fixture
    def default_config(self):
        """Default config for path generation."""
        return MCConfig(
            num_simulations=100,
            horizon_bars=50,
            use_log_returns=True,
            drift_mode='zero',
            sigma_mode='historical',
            rolling_window=20,
            random_seed=42,
            var_confidence=0.95,
            cvar_confidence=0.95,
        )
    
    @pytest.fixture
    def market_state_standard(self):
        """Standard market state for testing."""
        return MarketState(
            recent_close_prices=[100.0, 101.5, 100.8, 102.3, 101.9],
            recent_returns=[0.015, -0.0069, 0.0149, -0.0039],
            atr=2.5,
            sigma=0.02,
            drift=0.0001,
            regime="TRENDING_UP",
            timestamp="2026-04-19T10:00:00",
        )
    
    # ============== OUTPUT SHAPE & STRUCTURE ==============
    
    def test_output_shape_correct(self, market_state_standard, default_config):
        """Verify output shape: (num_simulations, horizon_bars + 1)."""
        paths = generate_paths(market_state_standard, default_config)
        expected_shape = (default_config.num_simulations, default_config.horizon_bars + 1)
        assert paths.shape == expected_shape
    
    def test_output_is_numpy_array(self, market_state_standard, default_config):
        """Output should be np.ndarray."""
        paths = generate_paths(market_state_standard, default_config)
        assert isinstance(paths, np.ndarray)
    
    def test_output_is_float_type(self, market_state_standard, default_config):
        """Output should be float data."""
        paths = generate_paths(market_state_standard, default_config)
        assert paths.dtype in [np.float32, np.float64]
    
    # ============== STARTING PRICE ==============
    
    def test_first_column_is_starting_price(self, market_state_standard, default_config):
        """All paths start at the most recent close price."""
        paths = generate_paths(market_state_standard, default_config)
        S0 = market_state_standard.recent_close_prices[-1]
        starting_prices = paths[:, 0]
        
        np.testing.assert_array_almost_equal(
            starting_prices,
            np.full(default_config.num_simulations, S0),
        )
    
    def test_consistent_starting_price_across_runs(self, market_state_standard, default_config):
        """Starting price should be identical for each simulation."""
        paths = generate_paths(market_state_standard, default_config)
        unique_starts = np.unique(paths[:, 0])
        assert len(unique_starts) == 1
    
    # ============== VALUE VALIDATION ==============
    
    def test_no_nan_values(self, market_state_standard, default_config):
        """Paths should not contain NaN."""
        paths = generate_paths(market_state_standard, default_config)
        assert not np.any(np.isnan(paths))
    
    def test_no_inf_values(self, market_state_standard, default_config):
        """Paths should not contain Inf."""
        paths = generate_paths(market_state_standard, default_config)
        assert not np.any(np.isinf(paths))
    
    def test_all_prices_positive(self, market_state_standard, default_config):
        """All simulated prices should be > 0."""
        paths = generate_paths(market_state_standard, default_config)
        assert np.all(paths > 0)
    
    def test_finite_values_only(self, market_state_standard, default_config):
        """All values should be finite (not NaN, not Inf)."""
        paths = generate_paths(market_state_standard, default_config)
        assert np.isfinite(paths).all()
    
    # ============== REPRODUCIBILITY ==============
    
    def test_same_seed_produces_same_paths(self, market_state_standard):
        """Fixed seed -> deterministic output (exact equality)."""
        config = MCConfig(num_simulations=100, horizon_bars=50, random_seed=42)
        paths1 = generate_paths(market_state_standard, config)
        paths2 = generate_paths(market_state_standard, config)
        
        np.testing.assert_array_equal(paths1, paths2)
    
    def test_different_seeds_produce_different_paths(self, market_state_standard):
        """Different seeds -> different paths."""
        config_seed42 = MCConfig(num_simulations=100, horizon_bars=50, random_seed=42)
        config_seed123 = MCConfig(num_simulations=100, horizon_bars=50, random_seed=123)
        
        paths1 = generate_paths(market_state_standard, config_seed42)
        paths2 = generate_paths(market_state_standard, config_seed123)
        
        assert not np.array_equal(paths1, paths2)
    
    def test_none_seed_produces_different_paths(self, market_state_standard):
        """None seed -> randomness (different each time)."""
        config = MCConfig(num_simulations=100, horizon_bars=50, random_seed=None)
        
        paths1 = generate_paths(market_state_standard, config)
        paths2 = generate_paths(market_state_standard, config)
        
        assert not np.array_equal(paths1, paths2)
    
    # ============== EDGE CASES ==============
    
    def test_zero_sigma_produces_flat_paths(self, market_state_standard, default_config):
        """sigma=0 with drift=0 -> all prices equal to starting price."""
        state_zero_vol = MarketState(
            recent_close_prices=market_state_standard.recent_close_prices,
            recent_returns=[0.0, 0.0, 0.0, 0.0],
            atr=0.0,
            sigma=0.0,
            drift=0.0,
            regime="FLAT",
            timestamp="2026-04-19T10:00:00",
        )
        
        paths = generate_paths(state_zero_vol, default_config)
        S0 = state_zero_vol.recent_close_prices[-1]
        np.testing.assert_array_almost_equal(paths, np.full_like(paths, S0))
    
    def test_drift_mode_zero_ignores_market_drift(self, market_state_standard):
        """With drift_mode='zero', market_state.drift should be ignored.
        
        Two paths with same S0 and sigma but different drift values should be
        identical when using the same seed and drift_mode='zero'.
        """
        state_with_positive_drift = MarketState(
            recent_close_prices=market_state_standard.recent_close_prices,
            recent_returns=[0.01, 0.01, 0.01, 0.01],
            atr=2.5,
            sigma=0.02,
            drift=0.01,  # Positive drift
            regime="TRENDING_UP",
            timestamp="2026-04-19T10:00:00",
        )
        
        state_with_negative_drift = MarketState(
            recent_close_prices=market_state_standard.recent_close_prices,
            recent_returns=[0.01, 0.01, 0.01, 0.01],
            atr=2.5,
            sigma=0.02,
            drift=-0.01,  # Negative drift
            regime="TRENDING_UP",
            timestamp="2026-04-19T10:00:00",
        )
        
        config_zero_drift = MCConfig(
            num_simulations=100,
            horizon_bars=50,
            drift_mode='zero',
            random_seed=42,  # Fixed seed
        )
        
        # Generate paths with different drift but same seed and sigma
        paths_pos = generate_paths(state_with_positive_drift, config_zero_drift)
        paths_neg = generate_paths(state_with_negative_drift, config_zero_drift)
        
        # With drift_mode='zero', both should ignore drift and produce identical paths
        np.testing.assert_array_equal(paths_pos, paths_neg)
    
    def test_multiple_simulations_are_different(self, market_state_standard, default_config):
        """Different paths in same generation should differ."""
        paths = generate_paths(market_state_standard, default_config)
        unique_paths = len(np.unique(paths, axis=0))
        assert unique_paths > 1
    
    # ============== CONFIG PARAMETER VARIATION ==============
    
    def test_more_simulations_produces_more_rows(self, market_state_standard):
        """num_simulations controls row count."""
        config100 = MCConfig(num_simulations=100, horizon_bars=50)
        config500 = MCConfig(num_simulations=500, horizon_bars=50)
        
        paths100 = generate_paths(market_state_standard, config100)
        paths500 = generate_paths(market_state_standard, config500)
        
        assert paths100.shape[0] == 100
        assert paths500.shape[0] == 500
        assert paths100.shape[1] == paths500.shape[1]
    
    def test_longer_horizon_produces_more_columns(self, market_state_standard):
        """horizon_bars controls column count."""
        config_short = MCConfig(num_simulations=100, horizon_bars=20)
        config_long = MCConfig(num_simulations=100, horizon_bars=100)
        
        paths_short = generate_paths(market_state_standard, config_short)
        paths_long = generate_paths(market_state_standard, config_long)
        
        assert paths_short.shape[1] == 21
        assert paths_long.shape[1] == 101
        assert paths_short.shape[0] == paths_long.shape[0]
    
    # ============== INPUT VALIDATION ==============
    
    def test_empty_prices_raises_error(self, default_config):
        """Empty price list should raise ValueError."""
        bad_state = MarketState(
            recent_close_prices=[],
            recent_returns=[],
            atr=0.0,
            sigma=0.02,
            drift=0.0,
            regime="FLAT",
            timestamp="2026-04-19T10:00:00",
        )
        
        with pytest.raises(ValueError, match="empty"):
            generate_paths(bad_state, default_config)
    
    def test_negative_sigma_raises_error(self, default_config):
        """Negative sigma should raise ValueError."""
        bad_state = MarketState(
            recent_close_prices=[100.0, 101.0, 102.0],
            recent_returns=[0.01, 0.01],
            atr=0.0,
            sigma=-0.01,
            drift=0.0,
            regime="FLAT",
            timestamp="2026-04-19T10:00:00",
        )
        
        with pytest.raises(ValueError, match="non-negative"):
            generate_paths(bad_state, default_config)
    
    def test_zero_num_simulations_raises_error(self, market_state_standard):
        """num_simulations must be > 0."""
        bad_config = MCConfig(num_simulations=0, horizon_bars=50)
        
        with pytest.raises(ValueError, match="positive"):
            generate_paths(market_state_standard, bad_config)
    
    def test_negative_num_simulations_raises_error(self, market_state_standard):
        """num_simulations must be positive."""
        bad_config = MCConfig(num_simulations=-10, horizon_bars=50)
        
        with pytest.raises(ValueError, match="positive"):
            generate_paths(market_state_standard, bad_config)
    
    def test_zero_horizon_raises_error(self, market_state_standard):
        """horizon_bars must be > 0."""
        bad_config = MCConfig(num_simulations=100, horizon_bars=0)
        
        with pytest.raises(ValueError, match="positive"):
            generate_paths(market_state_standard, bad_config)
    
    def test_negative_horizon_raises_error(self, market_state_standard):
        """horizon_bars must be positive."""
        bad_config = MCConfig(num_simulations=100, horizon_bars=-5)
        
        with pytest.raises(ValueError, match="positive"):
            generate_paths(market_state_standard, bad_config)
    
    def test_zero_price_raises_error(self, default_config):
        """Zero price should raise ValueError."""
        bad_state = MarketState(
            recent_close_prices=[100.0, 0.0, 102.0],
            recent_returns=[],
            atr=0.0,
            sigma=0.02,
            drift=0.0,
            regime="FLAT",
            timestamp="2026-04-19T10:00:00",
        )
        
        with pytest.raises(ValueError, match="positive"):
            generate_paths(bad_state, default_config)
    
    def test_negative_price_raises_error(self, default_config):
        """Negative price should raise ValueError."""
        bad_state = MarketState(
            recent_close_prices=[100.0, -50.0, 102.0],
            recent_returns=[],
            atr=0.0,
            sigma=0.02,
            drift=0.0,
            regime="FLAT",
            timestamp="2026-04-19T10:00:00",
        )
        
        with pytest.raises(ValueError, match="positive"):
            generate_paths(bad_state, default_config)
    
    def test_nan_sigma_raises_error(self, default_config):
        """NaN sigma should raise ValueError."""
        bad_state = MarketState(
            recent_close_prices=[100.0, 101.0, 102.0],
            recent_returns=[0.01, 0.01],
            atr=0.0,
            sigma=np.nan,
            drift=0.0,
            regime="FLAT",
            timestamp="2026-04-19T10:00:00",
        )
        
        with pytest.raises(ValueError, match="finite"):
            generate_paths(bad_state, default_config)
    
    def test_inf_drift_raises_error(self):
        """Inf drift should raise ValueError."""
        bad_state = MarketState(
            recent_close_prices=[100.0, 101.0, 102.0],
            recent_returns=[0.01, 0.01],
            atr=0.0,
            sigma=0.02,
            drift=np.inf,
            regime="FLAT",
            timestamp="2026-04-19T10:00:00",
        )
        
        # Must use drift_mode='historical' to test drift validation
        config = MCConfig(num_simulations=100, horizon_bars=50, drift_mode='historical')
        
        with pytest.raises(ValueError, match="finite"):
            generate_paths(bad_state, config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

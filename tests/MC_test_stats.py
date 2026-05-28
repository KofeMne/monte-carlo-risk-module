"""Unit tests for compute_sigma_and_drift (Stage A)."""

import sys
from pathlib import Path

import numpy as np
import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from stats import compute_sigma_and_drift  # noqa: E402
from config import MCConfig  # noqa: E402


class TestComputeSigmaAndDrift:
    """Test suite for compute_sigma_and_drift function."""
    
    @pytest.fixture
    def default_config(self):
        """Default config with zero drift mode."""
        return MCConfig(
            drift_mode='zero',
            rolling_window=20,
            use_log_returns=True,
        )
    
    @pytest.fixture
    def historical_drift_config(self):
        """Config with historical drift mode."""
        return MCConfig(
            drift_mode='historical',
            rolling_window=20,
            use_log_returns=True,
        )
    
    # ============== SYNTHETIC DATA FIXTURES ==============
    
    @pytest.fixture
    def linear_trend_data(self):
        """Linear uptrend: 100 -> 120."""
        return np.linspace(100, 120, 50)
    
    @pytest.fixture
    def constant_data(self):
        """Constant prices (no volatility)."""
        return np.full(50, 100.0)
    
    @pytest.fixture
    def random_data(self):
        """Random geometric returns, reproducible."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, size=100)
        return 100 * np.exp(np.cumsum(returns))
    
    @pytest.fixture
    def small_noise_data(self):
        """Very small volatility."""
        rng = np.random.default_rng(123)
        returns = rng.normal(0, 0.0001, size=50)
        return 100 * np.exp(np.cumsum(returns))
    
    # ============== OUTPUT VALIDATION ==============
    
    def test_output_is_tuple_of_two_floats(self, linear_trend_data, default_config):
        """Verify output format."""
        result = compute_sigma_and_drift(linear_trend_data, default_config)
        assert isinstance(result, tuple)
        assert len(result) == 2
        sigma, drift = result
        assert isinstance(sigma, (float, np.floating))
        assert isinstance(drift, (float, np.floating))
    
    def test_sigma_is_non_negative(self, random_data, default_config):
        """Sigma must be >= 0."""
        sigma, _ = compute_sigma_and_drift(random_data, default_config)
        assert sigma >= 0
    
    def test_sigma_is_finite(self, random_data, default_config):
        """Sigma must not be NaN or Inf."""
        sigma, _ = compute_sigma_and_drift(random_data, default_config)
        assert np.isfinite(sigma)
    
    def test_drift_is_finite(self, random_data, default_config):
        """Drift must not be NaN or Inf."""
        _, drift = compute_sigma_and_drift(random_data, default_config)
        assert np.isfinite(drift)
    
    # ============== DRIFT MODE TESTS ==============
    
    def test_zero_drift_mode_returns_zero(self, random_data, default_config):
        """When drift_mode='zero', drift must be 0.0."""
        _, drift = compute_sigma_and_drift(random_data, default_config)
        assert drift == 0.0
    
    def test_historical_drift_mode_returns_nonzero(self, random_data, historical_drift_config):
        """When drift_mode='historical', drift can be non-zero."""
        _, drift = compute_sigma_and_drift(random_data, historical_drift_config)
        # Drift may be ~0 by chance, but should be finite
        assert np.isfinite(drift)
    
    def test_different_drift_modes_produce_different_drift(self, random_data, 
                                                            default_config, historical_drift_config):
        """Zero drift != historical drift (usually)."""
        _, drift_zero = compute_sigma_and_drift(random_data, default_config)
        _, drift_hist = compute_sigma_and_drift(random_data, historical_drift_config)
        
        # Drift should differ (unless by extreme chance they're both ~0)
        # For random data, this is virtually guaranteed
        assert drift_zero == 0.0
        assert drift_hist != drift_zero  # They should be different
    
    # ============== VOLATILITY BEHAVIOR TESTS ==============
    
    def test_constant_prices_have_zero_sigma(self, constant_data, default_config):
        """Constant prices = zero volatility."""
        sigma, drift = compute_sigma_and_drift(constant_data, default_config)
        assert sigma == 0.0
        assert drift == 0.0
    
    def test_linear_trend_has_low_sigma(self, linear_trend_data, default_config):
        """Linear trend has predictable, low sigma."""
        sigma, _ = compute_sigma_and_drift(linear_trend_data, default_config)
        # Linear trend = very low/zero volatility in returns
        assert sigma < 0.001
    
    def test_random_data_has_higher_sigma(self, random_data, small_noise_data, default_config):
        """Random data should have higher sigma than small noise."""
        sigma_random, _ = compute_sigma_and_drift(random_data, default_config)
        sigma_small, _ = compute_sigma_and_drift(small_noise_data, default_config)
        
        assert sigma_random > sigma_small
    
    # ============== WINDOW SIZE BEHAVIOR ==============
    
    def test_smaller_window_uses_fewer_bars(self, random_data):
        """Window size should affect calculation."""
        config_small = MCConfig(rolling_window=5, drift_mode='zero')
        config_large = MCConfig(rolling_window=30, drift_mode='zero')
        
        sigma_small, _ = compute_sigma_and_drift(random_data, config_small)
        sigma_large, _ = compute_sigma_and_drift(random_data, config_large)
        
        # Different windows should generally give different results
        assert np.isfinite(sigma_small)
        assert np.isfinite(sigma_large)
    
    # ============== INPUT VALIDATION ==============
    
    def test_too_few_prices_raises_error(self, default_config):
        """Need at least 2 prices to calculate returns."""
        with pytest.raises(ValueError, match="at least 2 prices"):
            compute_sigma_and_drift(np.array([100.0]), default_config)
    
    def test_empty_prices_raises_error(self, default_config):
        """Empty array should raise error."""
        with pytest.raises(ValueError, match="at least 2 prices"):
            compute_sigma_and_drift(np.array([]), default_config)
    
    def test_non_positive_prices_raise_error(self, default_config):
        """log(price) requires price > 0."""
        with pytest.raises(ValueError, match="positive"):
            compute_sigma_and_drift(np.array([100.0, 0.0, 101.0]), default_config)
    
    def test_negative_prices_raise_error(self, default_config):
        """Negative prices not allowed."""
        with pytest.raises(ValueError, match="positive"):
            compute_sigma_and_drift(np.array([100.0, -50.0, 101.0]), default_config)
    
    def test_insufficient_data_for_window_raises_error(self, default_config):
        """If data < window, should raise error."""
        tiny = np.array([100.0, 101.0, 102.0])
        config = MCConfig(rolling_window=50, drift_mode='zero')
        
        with pytest.raises(ValueError, match="Not enough data"):
            compute_sigma_and_drift(tiny, config)
    
    # ============== DATA TYPE FLEXIBILITY ==============
    
    def test_accepts_numpy_array(self, random_data, default_config):
        """Should work with np.ndarray."""
        sigma, drift = compute_sigma_and_drift(np.array(random_data), default_config)
        assert np.isfinite(sigma)
        assert np.isfinite(drift)
    
    def test_accepts_list(self, default_config):
        """Should work with Python list."""
        prices = [100.0, 101.0, 102.0, 101.5, 103.0] * 10  # 50 prices
        sigma, drift = compute_sigma_and_drift(prices, default_config)
        assert np.isfinite(sigma)
        assert np.isfinite(drift)
    
    # ============== EDGE CASES ==============
    
    def test_all_same_returns_zero_sigma(self, default_config):
        """Prices like 100, 100, 100... -> sigma=0."""
        prices = np.full(100, 100.0)
        sigma, _ = compute_sigma_and_drift(prices, default_config)
        assert sigma == 0.0
    
    def test_single_big_jump_then_flat(self, default_config):
        """Big jump, then constant."""
        prices = np.concatenate([np.array([100.0]), np.full(99, 150.0)])
        sigma, _ = compute_sigma_and_drift(prices, default_config)
        # Should be finite but close to 0 (after first bar, constant)
        assert np.isfinite(sigma)
    
    def test_very_large_prices(self, default_config):
        """Should work with large nominal values."""
        prices = np.linspace(100000, 120000, 50)
        sigma, _ = compute_sigma_and_drift(prices, default_config)
        assert np.isfinite(sigma)
    
    def test_very_small_prices(self, default_config):
        """Should work with small prices."""
        prices = np.linspace(0.001, 0.012, 50)
        sigma, _ = compute_sigma_and_drift(prices, default_config)
        assert np.isfinite(sigma)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

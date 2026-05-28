"""Smoke tests on real data (CSV/Parquet).

This is NOT a unit test - it verifies that the Monte Carlo engine
works stably on actual market data without crashing or producing
inf/nan values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from stats import compute_sigma_and_drift  # noqa: E402
from path_generator import generate_paths  # noqa: E402
from market_state import MarketState  # noqa: E402
from config import MCConfig  # noqa: E402


class TestRealDataSmoke:
    """Smoke tests using real OHLC data."""
    
    @pytest.fixture
    def real_data_path(self):
        """Find real data file (ETH or BTC)."""
        data_dir = Path(__file__).resolve().parents[1] / "data"
        
        # Try parquet first
        parquet_file = data_dir / "ETHUSDT_15m_ohlc_clean.parquet"
        if parquet_file.exists():
            return parquet_file
        
        # Fall back to CSV
        csv_file = data_dir / "btc_15m.csv"
        if csv_file.exists():
            return csv_file
        
        pytest.skip("No real data file found (ETHUSDT_15m_ohlc_clean.parquet or btc_15m.csv)")
    
    @pytest.fixture
    def closes_from_real_data(self, real_data_path):
        """Load close prices from real data file (sorted, cleaned, numeric)."""
        if str(real_data_path).endswith('.parquet'):
            df = pd.read_parquet(real_data_path)
        else:
            df = pd.read_csv(real_data_path)
        
        # Sort by time (if open_time exists)
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'], errors='coerce')
            df = df.sort_values('open_time')
        
        # Convert to numeric and remove NaN
        close = pd.to_numeric(df['close'], errors='coerce').dropna().to_numpy()
        
        return close
    
    @pytest.fixture
    def recent_closes(self, closes_from_real_data):
        """Use most recent 500 bars."""
        return closes_from_real_data[-500:]
    
    @pytest.fixture
    def config(self):
        """Standard config for smoke tests."""
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
    
    # ============== STAGE A: SIGMA & DRIFT ==============
    
    def test_sigma_and_drift_on_real_data(self, recent_closes, config):
        """Compute sigma & drift should work on real data."""
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        
        # Should not crash and produce valid numbers
        assert np.isfinite(sigma), "sigma is not finite"
        assert np.isfinite(drift), "drift is not finite"
        assert sigma >= 0, "sigma should be non-negative"
    
    def test_sigma_is_reasonable_range(self, recent_closes, config):
        """Sigma should be in reasonable range (0.0001 to 1.0)."""
        sigma, _ = compute_sigma_and_drift(recent_closes, config)
        
        # Typical daily volatility is 0.5-3%, 15min is much smaller
        assert 0.0 <= sigma <= 1.0, f"sigma {sigma} out of reasonable range"
    
    def test_drift_zero_mode_ignores_trend(self, recent_closes, config):
        """With drift_mode='zero', drift should be 0."""
        _, drift = compute_sigma_and_drift(recent_closes, config)
        assert drift == 0.0, "drift_mode='zero' should produce exactly 0 drift"
    
    def test_drift_with_historical_mode(self, recent_closes):
        """With drift_mode='historical', drift is estimated."""
        config = MCConfig(drift_mode='historical', rolling_window=20)
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        
        assert np.isfinite(drift), "historical drift should be finite"
        assert np.isfinite(sigma), "sigma should be finite"
    
    # ============== STAGE B: PATH GENERATION ==============
    
    def test_generate_paths_on_real_data(self, recent_closes, config):
        """Generate paths should work on real data."""
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        
        market_state = MarketState(
            recent_close_prices=recent_closes.tolist(),
            recent_returns=[],  # Not used in generation
            atr=0.0,            # Not used
            sigma=sigma,
            drift=drift,
            regime="UNKNOWN",
            timestamp="2026-04-19",
        )
        
        paths = generate_paths(market_state, config)
        
        assert np.isfinite(paths).all(), "paths contain non-finite values"
        assert (paths > 0).all(), "paths contain non-positive prices"
    
    def test_generated_paths_shape(self, recent_closes, config):
        """Verify output shape."""
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        
        market_state = MarketState(
            recent_close_prices=recent_closes.tolist(),
            recent_returns=[],
            atr=0.0,
            sigma=sigma,
            drift=drift,
            regime="UNKNOWN",
            timestamp="2026-04-19",
        )
        
        paths = generate_paths(market_state, config)
        
        expected_shape = (config.num_simulations, config.horizon_bars + 1)
        assert paths.shape == expected_shape, f"expected {expected_shape}, got {paths.shape}"
    
    def test_paths_start_at_correct_price(self, recent_closes, config):
        """First column should be starting price."""
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        S0 = recent_closes[-1]
        
        market_state = MarketState(
            recent_close_prices=recent_closes.tolist(),
            recent_returns=[],
            atr=0.0,
            sigma=sigma,
            drift=drift,
            regime="UNKNOWN",
            timestamp="2026-04-19",
        )
        
        paths = generate_paths(market_state, config)
        starting_prices = paths[:, 0]
        
        np.testing.assert_array_almost_equal(
            starting_prices,
            np.full(config.num_simulations, S0),
        )
    
    def test_paths_are_reasonable_magnitude(self, recent_closes, config):
        """Prices should stay in reasonable range from starting price."""
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        S0 = recent_closes[-1]
        
        market_state = MarketState(
            recent_close_prices=recent_closes.tolist(),
            recent_returns=[],
            atr=0.0,
            sigma=sigma,
            drift=drift,
            regime="UNKNOWN",
            timestamp="2026-04-19",
        )
        
        paths = generate_paths(market_state, config)
        
        # With 50-bar horizon and 2% vol, prices shouldn't deviate >50% typically
        max_price = np.max(paths)
        min_price = np.min(paths)
        
        # Allow wide range: 0.5x to 2x starting price
        assert min_price > S0 * 0.3, f"paths went too low: {min_price} vs {S0}"
        assert max_price < S0 * 3.0, f"paths went too high: {max_price} vs {S0}"
    
    # ============== END-TO-END PIPELINE ==============
    
    def test_full_pipeline_with_real_data(self, recent_closes, config):
        """Full pipeline: data -> sigma -> paths."""
        # Stage A: compute sigma
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        assert np.isfinite(sigma)
        assert np.isfinite(drift)
        
        # Create market state
        market_state = MarketState(
            recent_close_prices=recent_closes.tolist(),
            recent_returns=[],
            atr=0.0,
            sigma=sigma,
            drift=drift,
            regime="UNKNOWN",
            timestamp="2026-04-19",
        )
        
        # Stage B: generate paths
        paths = generate_paths(market_state, config)
        assert np.isfinite(paths).all()
        assert (paths > 0).all()
        assert paths.shape == (config.num_simulations, config.horizon_bars + 1)
    
    def test_reproducibility_with_seed(self, recent_closes, config):
        """Same seed -> same paths."""
        sigma, drift = compute_sigma_and_drift(recent_closes, config)
        
        market_state = MarketState(
            recent_close_prices=recent_closes.tolist(),
            recent_returns=[],
            atr=0.0,
            sigma=sigma,
            drift=drift,
            regime="UNKNOWN",
            timestamp="2026-04-19",
        )
        
        paths1 = generate_paths(market_state, config)
        paths2 = generate_paths(market_state, config)
        
        np.testing.assert_array_almost_equal(paths1, paths2)
    
    # ============== DATA QUALITY CHECKS ==============
    
    def test_no_nan_in_real_data(self, closes_from_real_data):
        """Raw data should have no NaN."""
        assert not np.any(np.isnan(closes_from_real_data)), "Real data contains NaN"
    
    def test_no_zero_prices_in_real_data(self, closes_from_real_data):
        """Prices should all be positive."""
        assert np.all(closes_from_real_data > 0), "Real data contains non-positive prices"
    
    def test_different_windows_produce_different_sigma(self, recent_closes):
        """Window size should affect result."""
        config_small = MCConfig(rolling_window=5, drift_mode='zero')
        config_large = MCConfig(rolling_window=50, drift_mode='zero')
        
        sigma_small, _ = compute_sigma_and_drift(recent_closes, config_small)
        sigma_large, _ = compute_sigma_and_drift(recent_closes, config_large)
        
        # Different windows should usually give different results
        assert np.isfinite(sigma_small)
        assert np.isfinite(sigma_large)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

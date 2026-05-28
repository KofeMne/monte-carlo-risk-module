"""
From recent close prices, calculates:

- returns
- rolling volatility estimate (sigma)
- optional drift
"""

import numpy as np
import pandas as pd
from config import MCConfig


def compute_sigma_and_drift(closes: np.ndarray | pd.Series, config: MCConfig) -> tuple[float, float]:
    """Compute sigma and drift for Monte Carlo simulation.
    
    Uses log returns with rolling standard deviation. No look-ahead bias.
    
    Args:
        closes: Array or Series of close prices
        config: MCConfig object with rolling_window and drift_mode settings
        
    Returns:
        Tuple of (sigma, drift)
        
    Raises:
        ValueError: If closes has fewer than 2 prices, contains non-positive values,
                   or has insufficient data for the rolling window
    """
    # Validate input data
    closes_array = np.asarray(closes, dtype=float)
    if closes_array.ndim != 1:
        raise ValueError("closes must be a 1D array")
    
    if len(closes_array) < 2:
        raise ValueError("Need at least 2 prices to calculate returns")
    
    if np.any(closes_array <= 0):
        raise ValueError("Prices must be positive for log returns")
    
    window = config.rolling_window
    if len(closes_array) - 1 < window:
        raise ValueError(
            f"Not enough data for rolling window: have {len(closes_array) - 1} "
            f"returns but need {window}"
        )
    
    # Calculate log returns once (no repetition)
    log_returns = np.diff(np.log(closes_array))
    
    # Use only the most recent 'window' returns to avoid look-ahead bias
    recent_returns = log_returns[-window:]
    
    # Calculate sigma (rolling volatility)
    sigma = np.std(recent_returns, ddof=1)  # ddof=1 for sample std dev
    
    # Calculate drift based on config
    if config.drift_mode == 'zero':
        drift = 0.0
    elif config.drift_mode == 'historical':
        drift = np.mean(recent_returns)
    else:
        raise ValueError(f"Unknown drift_mode: {config.drift_mode}")
    
    return float(sigma), float(drift)
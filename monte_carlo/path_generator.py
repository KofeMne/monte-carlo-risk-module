"""
Generate Monte Carlo price paths for future market scenarios.

## Two simulation approaches

### GBM (Geometric Brownian Motion) — default
Uses the classic lognormal model: each log-return is drawn from N(drift, sigma²).
Fast, analytically tractable, and appropriate for short-horizon risk assessment.

Limitation: real market returns are NOT normally distributed. They have fat tails
(extreme moves happen more often than GBM predicts) and negative skewness
(crashes are sharper than rallies). GBM underestimates tail risk.

Formula: log_return_t = (drift − 0.5·σ²) + σ·Z_t,  Z_t ~ N(0,1)
         S_t = S_0 · exp(Σ log_returns)

### BOOTSTRAP — historical resampling
Draws log-returns with replacement directly from the observed historical return
series instead of generating them from a parametric distribution.

Why this is better for risk management: the bootstrap automatically captures
fat tails, negative skewness, and volatility clustering that exist in the actual
data. You are not assuming a distribution — you are using the one the market
actually produced. This makes tail-event estimates (VaR, CVaR) more realistic.

When to prefer BOOTSTRAP: whenever you have at least 60–100 bars of recent history
and want the most realistic tail scenarios.
When to prefer GBM: fast sensitivity checks, or when historical data is sparse.

### Regime-conditioned sigma
Before path generation, sigma is scaled by the regime multiplier from
config.regime_sigma_multipliers (e.g. {'CRASH': 1.5}). This makes simulated paths
more extreme in high-volatility regimes, which translates to more conservative
risk decisions — exactly when you need them most.
"""

import numpy as np
from config import MCConfig
from market_state import MarketState


def generate_paths(market_state: MarketState, config: MCConfig) -> np.ndarray:
    """Generate Monte Carlo price paths.

    Creates num_simulations price paths, each spanning horizon_bars future bars.
    Supports GBM (parametric) and BOOTSTRAP (historical resampling) methods.

    Args:
        market_state: Current market snapshot. Provides starting price, sigma,
                      drift, recent_returns (for bootstrap), and regime label.
        config:       MCConfig controlling simulation count, horizon, method, seed.

    Returns:
        2D array of shape (num_simulations, horizon_bars + 1) where:
          - Column 0 is the starting price S0 (repeated for all paths)
          - Columns 1: are simulated future prices
          - Each row is one complete independent price path

    Raises:
        ValueError: If market_state has insufficient data or config is invalid.
    """
    # --- Validate config ---
    if config.num_simulations <= 0:
        raise ValueError("num_simulations must be positive")
    if config.horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    # --- Validate market data ---
    if not market_state.recent_close_prices:
        raise ValueError("market_state.recent_close_prices is empty")

    closes_array = np.asarray(market_state.recent_close_prices)
    if np.any(closes_array <= 0):
        raise ValueError("all prices must be positive")

    S0 = float(market_state.recent_close_prices[-1])

    # --- Validate and apply regime sigma multiplier ---
    # Scaling sigma upward in stressed regimes makes path scenarios more extreme,
    # giving risk metrics a more conservative (realistic) view during high-vol periods.
    sigma = float(market_state.sigma)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")

    regime_mult = config.regime_sigma_multipliers.get(market_state.regime, 1.0)
    if regime_mult != 1.0:
        sigma = sigma * regime_mult

    drift = 0.0 if config.drift_mode == "zero" else float(market_state.drift)
    if not np.isfinite(drift):
        raise ValueError("drift must be finite")

    num_paths = config.num_simulations
    num_steps = config.horizon_bars

    # --- Initialise price matrix ---
    prices = np.empty((num_paths, num_steps + 1), dtype=float)
    prices[:, 0] = S0

    # Local RNG — does not affect global numpy random state.
    rng = np.random.default_rng(config.random_seed)

    # --- Path generation ---
    if config.path_method == 'BOOTSTRAP':
        _fill_bootstrap(prices, market_state, config, rng, S0, num_paths, num_steps, regime_mult)
    else:
        # Default: GBM
        _fill_gbm(prices, sigma, drift, rng, S0, num_paths, num_steps)

    return prices


def _fill_gbm(prices: np.ndarray, sigma: float, drift: float,
              rng: np.random.Generator, S0: float,
              num_paths: int, num_steps: int) -> None:
    """Fill price matrix using Geometric Brownian Motion.

    GBM formula (Itô-corrected drift):
        log_return_t = (drift − 0.5·σ²) + σ·Z_t,  Z_t ~ N(0,1)
        S_t = S_0 · exp(cumsum(log_returns))

    The −0.5·σ² Itô correction ensures that E[S_t] = S_0·exp(drift·t),
    i.e. the expected price grows at the drift rate (not inflated by variance).
    """
    adjusted_drift = drift - 0.5 * sigma ** 2
    random_shocks = rng.standard_normal((num_paths, num_steps))
    log_returns = adjusted_drift + sigma * random_shocks
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    prices[:, 1:] = S0 * np.exp(cumulative_log_returns)


def _fill_bootstrap(prices: np.ndarray, market_state: MarketState,
                    config: MCConfig, rng: np.random.Generator,
                    S0: float, num_paths: int, num_steps: int,
                    regime_mult: float = 1.0) -> None:
    """Fill price matrix using historical bootstrap resampling.

    Draws log-returns with replacement from the recent return history.
    This preserves the empirical distribution of returns — fat tails, skewness,
    and any autocorrelation present in the data — without assuming normality.

    Requires market_state.recent_returns to be populated.
    Falls back to GBM if fewer than 10 historical returns are available.

    The bootstrap_lookback cap (default 252 bars ≈ 1 trading year) limits
    the pool to recent market conditions. Using very old data risks including
    regimes that are no longer relevant.

    Fix #2: when drift_mode='zero' the pool is demeaned so the conservative
    zero-drift contract holds (raw resampling would inject the recent trend).
    Fix #1: regime_sigma_multipliers are applied here too — deviations around the
    pool mean are scaled by regime_mult so stressed regimes widen the tails
    (previously this only affected GBM).
    """
    returns = market_state.recent_returns
    if not returns or len(returns) < 10:
        # Not enough history for bootstrap — fall back to GBM with current sigma/drift
        sigma = float(market_state.sigma)
        drift = 0.0 if config.drift_mode == "zero" else float(market_state.drift)
        _fill_gbm(prices, sigma, drift, rng, S0, num_paths, num_steps)
        return

    # Limit pool to the most recent bootstrap_lookback bars
    pool = np.asarray(returns[-config.bootstrap_lookback:], dtype=float)

    # Remove any non-finite values that could corrupt paths
    pool = pool[np.isfinite(pool)]
    if len(pool) < 10:
        sigma = float(market_state.sigma)
        drift = 0.0 if config.drift_mode == "zero" else float(market_state.drift)
        _fill_gbm(prices, sigma, drift, rng, S0, num_paths, num_steps)
        return

    # Fix #2 — honor drift_mode='zero': raw returns carry their own sample mean,
    # so resampling them directly injects the recent trend. Demean to keep paths driftless.
    if config.drift_mode == "zero":
        pool = pool - pool.mean()

    # Fix #1 — apply the regime sigma multiplier in bootstrap too. Scale deviations
    # around the (preserved) pool mean so stressed regimes widen the tails without
    # shifting the central tendency.
    if regime_mult != 1.0:
        m = pool.mean()
        pool = m + (pool - m) * regime_mult

    # Sample with replacement: shape (num_paths, num_steps)
    sampled = rng.choice(pool, size=(num_paths, num_steps), replace=True)

    # Convert resampled log-returns to prices via cumulative sum + exponentiation
    cumulative = np.cumsum(sampled, axis=1)
    prices[:, 1:] = S0 * np.exp(cumulative)

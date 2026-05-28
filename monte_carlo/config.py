"""
Monte Carlo simulation configuration.

## Design decisions

- `path_method`: Two simulation modes are supported.
  - 'GBM' (Geometric Brownian Motion): assumes log-normally distributed returns.
    Fast, analytically tractable, good for broad scenario generation.
  - 'BOOTSTRAP': resamples directly from the observed historical return series.
    Naturally captures fat tails, negative skewness, and volatility clustering
    that GBM misses. Default — preferred for BTC-like assets and momentum/
    breakout strategies where realistic tails AND a preserved trend matter.

- `drift_mode`: Default is 'historical' so the bootstrap pool keeps its sample
  mean (the trend a breakout strategy depends on). Switch to 'zero' for
  mean-reversion strategies or pure tail stress-testing — but note that under
  BOOTSTRAP a zero-drift config demeans the pool and deletes the trend.

- `rolling_window`: Uses only the most recent N bars to estimate sigma/drift.
  This avoids look-ahead bias and keeps volatility estimates current.

- `regime_sigma_multipliers`: A dict mapping regime labels to sigma scaling
  factors. Defaults are calibrated from BTC 15m: HIGH_VOL=2.0 (p90/p50 ≈ 2.2x)
  and CRASH=3.5 (p99/p50 ≈ 4.7x). Applied in both GBM and BOOTSTRAP modes.

- `bootstrap_lookback`: 500 bars gives stable tails on 15m crypto without
  reaching back into regimes that no longer apply.

- `num_simulations_for_opt`: A reduced simulation count for use inside the
  Optuna optimizer where many MC runs are evaluated per trial. Lower than
  num_simulations to keep wall time acceptable; accuracy is sufficient for
  threshold optimization since it averages over many trials.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCConfig:
    """Configuration for Monte Carlo simulations.

    Attributes:
        num_simulations:    Number of Monte Carlo paths to simulate per trade evaluation.
        horizon_bars:       Maximum number of bars to simulate per path.
        use_log_returns:    Whether to use log returns internally (always True; kept for API compat).
        drift_mode:         'zero' (conservative default) | 'historical' | 'custom'.
        sigma_mode:         'historical' (rolling std of log returns).
        rolling_window:     Number of recent bars used to compute sigma and drift.
        random_seed:        Seed for reproducibility. None = random each run.
        var_confidence:     Confidence level for VaR (e.g. 0.95 → worst 5th percentile).
        cvar_confidence:    Confidence level for CVaR (same convention as var_confidence).
        path_method:        'GBM' for parametric simulation, 'BOOTSTRAP' for historical resampling.
        bootstrap_lookback: Number of historical bars to draw from in BOOTSTRAP mode.
        regime_sigma_multipliers: Mapping of regime name → sigma scale factor.
                            Applied before path generation. E.g. {'CRASH': 1.5, 'HIGH_VOL': 1.25}.
        num_simulations_for_opt: Reduced sim count used inside the Optuna optimizer to keep
                            optimization wall time manageable (default 500).
    """
    num_simulations: int = 10_000
    horizon_bars: int = 500
    use_log_returns: bool = True
    drift_mode: str = 'historical'
    sigma_mode: str = 'historical'
    rolling_window: int = 20
    random_seed: Optional[int] = None
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95

    # Path simulation method
    path_method: str = 'BOOTSTRAP'
    bootstrap_lookback: int = 500

    # Regime-conditioned volatility — calibrated from BTC 15m tail ratios.
    regime_sigma_multipliers: dict = field(
        default_factory=lambda: {'HIGH_VOL': 2.0, 'CRASH': 3.5}
    )

    # Optimizer-specific reduced simulation count
    num_simulations_for_opt: int = 1_000


# Default configuration instance — BTC 15m calibrated (see MC_TUNING_RESULTS §2).
default_config = MCConfig()

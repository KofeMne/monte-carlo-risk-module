"""
Compute risk metrics from a collection of Monte Carlo trade replay results.

## What these metrics tell you

After running replay_trade() across N simulated paths, you have a distribution of
possible trade outcomes. This module distils that distribution into actionable numbers:

### Probability metrics
- prob_loss: How often does this trade lose money? (primary filter signal)
- prob_sl_hit: How often does the hard stop fire? A high SL rate suggests the stop
  is too tight or the trade has poor initial positioning.
- prob_partial_tp_hit: How often does the partial TP fire? Low rates suggest the
  TP target is unrealistic for current volatility.

### PnL distribution metrics
- expected_pnl_r: The mean outcome in R multiples. Positive = theoretical edge.
  Zero or negative = the trade has no edge in the simulation.
- pnl_std_r: Standard deviation of outcomes. Wide spread = high uncertainty.
- profit_factor: Total wins / total losses (in R). Industry standard measure.
  > 1.0 means the system makes more than it loses in aggregate.
- sharpe_r: expected_pnl_r / pnl_std_r. Risk-adjusted quality. Higher is better.
  Analogous to the Sharpe ratio but on individual trade outcomes rather than a
  time series. Useful for comparing two trades with similar EV but different spread.

### Kelly criterion
- kelly_fraction: The theoretically optimal fraction of capital to risk on this trade.
  Derived from Kelly's formula: f* = (W * avg_win - L * avg_loss) / avg_win
  where W = win rate, L = loss rate, normalised by avg_win size.
  Interpretation: if kelly_fraction = 0.3, Kelly says risk 30% of your risk budget.
  With 1% base risk, that means 0.3% per trade.
  Values near 0 = almost no edge. Values > 0.5 = strong edge (verify before trusting).
  We cap at [0, 1] — negative Kelly means the trade has no expected edge.

### Tail risk metrics
- var_r: Value at Risk in R multiples at config.var_confidence (e.g. 95%).
  VaR = the (1 - confidence) worst-percentile outcome.
  Example: var_r = -2.5 at 95% means 5% of simulations lose more than 2.5R.
- cvar_r: Conditional VaR (Expected Shortfall). The mean of all outcomes
  at or below VaR. Answers: "when things go badly, how badly on average?"
  CVaR is always worse (more negative) than VaR.
- skewness: Statistical skewness of the pnl_r distribution.
  Negative skewness = the distribution has a heavier left tail (occasional large losses).
  This is common in trend-following strategies. Important for tail risk assessment.

### Excursion metrics
- avg_mae_r: Average Maximum Adverse Excursion across all simulations.
  Tells you how wide the drawdown typically gets before exit.
  If avg_mae_r is much worse than -1R, the sim suggests price regularly goes through
  the stop level during the trade — a sign the stop is placed at a noisy level.
- avg_mfe_r: Average Maximum Favorable Excursion.
  The average best-case profit reached. If avg_mfe_r >> final pnl, the trade is
  giving back a lot of profit — consider tighter trailing or earlier full TP.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import MCConfig
from trade_replay import SimulationResult

# Exit reasons that represent a hard stop firing (vs. MAX_BARS which is time-based).
_SL_REASONS = frozenset({'SL', 'TRAILING_SL', 'PARTIAL_TP_THEN_TRAILING_SL'})


@dataclass
class RiskMetrics:
    """Aggregated risk statistics from N Monte Carlo trade replays.

    All R-multiple fields use 1R = |entry_price - stop_loss| / entry_price.
    Negative values represent losses; positive values represent profits.
    """

    num_simulations: int
    """Number of simulated paths that produced these metrics."""

    prob_loss: float
    """Fraction of simulations where final pnl_r < 0. Primary filter signal.
    Most V2 trades with real edge have prob_loss in the 0.35-0.55 range
    because they have 3R reward targets with ~40-50% win rates."""

    prob_sl_hit: float
    """Fraction of simulations where a hard or trailing stop fired.
    Includes: SL, TRAILING_SL, PARTIAL_TP_THEN_TRAILING_SL.
    Excludes MAX_BARS (time-based exit, not a loss event per se)."""

    prob_partial_tp_hit: float
    """Fraction of simulations where the partial TP price was reached."""

    expected_pnl_r: float
    """Mean pnl_r across all simulations. Positive = theoretical edge.
    For a 3R:1R trade with 40% win rate: EV = 0.4*3 - 0.6*1 = +0.6R."""

    pnl_std_r: float
    """Standard deviation of pnl_r. Measures outcome uncertainty."""

    profit_factor: float
    """Sum of all positive pnl_r / abs(sum of all negative pnl_r).
    > 1.0 means the strategy makes more than it loses in aggregate.
    float('inf') if there are no losses (all wins)."""

    sharpe_r: float
    """expected_pnl_r / pnl_std_r. Risk-adjusted return quality.
    0.0 if pnl_std_r is zero (all outcomes identical)."""

    kelly_fraction: float
    """Optimal position size fraction from Kelly criterion. Capped to [0, 1].
    Near 0 = no simulated edge. > 0.3 = strong edge (validate carefully)."""

    var_r: float
    """Value at Risk in R multiples at (1 - var_confidence) quantile.
    Negative = loss. E.g. -2.5 means 5% of simulations lose > 2.5R."""

    cvar_r: float
    """Conditional VaR: mean of all outcomes <= var_r.
    Always <= var_r. Captures the severity of tail losses."""

    skewness: float
    """Skewness of pnl_r distribution. Negative = heavier left tail (crash risk)."""

    avg_mae_r: float
    """Average Maximum Adverse Excursion across all simulations (in R).
    Always <= 0. How deep does the trade typically go against us?"""

    avg_mfe_r: float
    """Average Maximum Favorable Excursion across all simulations (in R).
    Always >= 0. How much profit does the trade typically reach before exit?"""

    mean_exit_bar: float
    """Average bar number at which trades exit. Useful for checking if
    max_holding_bars is binding (i.e. many trades hitting the time cap)."""

    exit_reason_counts: dict
    """Count of each exit reason string across all simulations.
    Keys: 'SL', 'TRAILING_SL', 'PARTIAL_TP_THEN_TRAILING_SL', 'MAX_BARS'."""

    pnl_r_distribution: list
    """Sorted list of all pnl_r values. For downstream plotting or percentile queries."""


def calculate_risk_metrics(
    results: list,
    config: MCConfig,
) -> RiskMetrics:
    """Compute aggregated risk statistics from a list of SimulationResult objects.

    Args:
        results: List of SimulationResult, one per simulated path. Must not be empty.
        config:  MCConfig providing var_confidence and cvar_confidence levels.

    Returns:
        RiskMetrics dataclass with all computed statistics.

    Raises:
        ValueError: If results is empty or if any pnl_r value is NaN
                    (NaN indicates a bug in the replay logic).
    """
    if not results:
        raise ValueError(
            "results list is empty — cannot compute metrics from zero simulations."
        )

    # --- Extract arrays for vectorised computation ---
    pnl_r_arr = np.array([r.pnl_r for r in results], dtype=float)
    exit_bars = np.array([r.exit_bar for r in results], dtype=float)
    mae_arr = np.array([r.mae_r for r in results], dtype=float)
    mfe_arr = np.array([r.mfe_r for r in results], dtype=float)
    exit_reasons = [r.exit_reason for r in results]
    partial_hits = np.array([r.partial_tp_hit for r in results], dtype=bool)

    # NaN in pnl_r indicates a bug upstream (e.g. division by zero in replay).
    if np.any(np.isnan(pnl_r_arr)):
        raise ValueError(
            "pnl_r contains NaN values — check replay logic for division by zero "
            "or invalid candidate parameters."
        )

    n = len(results)

    # --- Probability metrics ---
    prob_loss = float(np.mean(pnl_r_arr < 0))
    prob_sl_hit = float(np.mean(np.isin(exit_reasons, list(_SL_REASONS))))
    prob_partial_tp_hit = float(np.mean(partial_hits))

    # --- PnL distribution stats ---
    expected_pnl_r = float(np.mean(pnl_r_arr))
    pnl_std_r = float(np.std(pnl_r_arr, ddof=1)) if n > 1 else 0.0

    # Sharpe: risk-adjusted return. Undefined if std is 0 (all outcomes identical).
    sharpe_r = expected_pnl_r / pnl_std_r if pnl_std_r > 0 else 0.0

    # Profit factor: total wins / total losses (magnitude).
    win_mask = pnl_r_arr > 0
    loss_mask = pnl_r_arr < 0
    total_wins = float(np.sum(pnl_r_arr[win_mask])) if np.any(win_mask) else 0.0
    total_losses = float(np.sum(np.abs(pnl_r_arr[loss_mask]))) if np.any(loss_mask) else 0.0
    if total_losses > 0:
        profit_factor = total_wins / total_losses
    elif total_wins > 0:
        profit_factor = float('inf')  # all wins, no losses
    else:
        profit_factor = 0.0  # all zeros (all MAX_BARS exits at entry)

    # Skewness of pnl_r distribution.
    # scipy is not a required dependency — compute manually using the standardised 3rd moment.
    if pnl_std_r > 0 and n >= 3:
        skewness = float(
            np.mean(((pnl_r_arr - expected_pnl_r) / pnl_std_r) ** 3)
        )
    else:
        skewness = 0.0

    # --- Kelly criterion ---
    # Kelly formula: f* = (W * avg_win - L * avg_loss) / avg_win
    # where W = win rate, L = loss rate (fraction of trades).
    # We normalise by avg_win so the result is a fraction of the risk budget.
    # Negative Kelly means no theoretical edge — clamp to 0.
    win_rate = float(np.mean(win_mask))
    loss_rate = float(np.mean(loss_mask))
    avg_win_r = float(np.mean(pnl_r_arr[win_mask])) if win_rate > 0 else 0.0
    avg_loss_r = float(np.mean(np.abs(pnl_r_arr[loss_mask]))) if loss_rate > 0 else 0.0

    if avg_win_r > 0:
        kelly_raw = (win_rate * avg_win_r - loss_rate * avg_loss_r) / avg_win_r
        kelly_fraction = float(np.clip(kelly_raw, 0.0, 1.0))
    else:
        kelly_fraction = 0.0

    # --- Tail risk metrics (VaR and CVaR) ---
    # VaR at confidence level c means: (1-c)% of outcomes are WORSE than VaR.
    # alpha = 1 - confidence (e.g. 0.05 for 95% confidence).
    # np.quantile at alpha gives the bottom (1-c)% threshold of the distribution.
    alpha_var = 1.0 - config.var_confidence
    var_r = float(np.quantile(pnl_r_arr, alpha_var))

    # CVaR (Expected Shortfall): mean of all outcomes AT OR BELOW VaR.
    # This captures the severity of losses in the tail, not just the threshold.
    alpha_cvar = 1.0 - config.cvar_confidence
    cvar_threshold = np.quantile(pnl_r_arr, alpha_cvar)
    tail_outcomes = pnl_r_arr[pnl_r_arr <= cvar_threshold]
    cvar_r = float(np.mean(tail_outcomes)) if len(tail_outcomes) > 0 else float(var_r)

    # --- Excursion metrics ---
    avg_mae_r = float(np.mean(mae_arr))
    avg_mfe_r = float(np.mean(mfe_arr))

    # --- Duration and exit breakdown ---
    mean_exit_bar = float(np.mean(exit_bars))
    exit_reason_counts = dict(Counter(exit_reasons))

    # Sorted pnl_r for downstream plotting (percentile queries, histograms).
    pnl_r_distribution = sorted(pnl_r_arr.tolist())

    return RiskMetrics(
        num_simulations=n,
        prob_loss=prob_loss,
        prob_sl_hit=prob_sl_hit,
        prob_partial_tp_hit=prob_partial_tp_hit,
        expected_pnl_r=expected_pnl_r,
        pnl_std_r=pnl_std_r,
        profit_factor=profit_factor,
        sharpe_r=sharpe_r,
        kelly_fraction=kelly_fraction,
        var_r=var_r,
        cvar_r=cvar_r,
        skewness=skewness,
        avg_mae_r=avg_mae_r,
        avg_mfe_r=avg_mfe_r,
        mean_exit_bar=mean_exit_bar,
        exit_reason_counts=exit_reason_counts,
        pnl_r_distribution=pnl_r_distribution,
    )

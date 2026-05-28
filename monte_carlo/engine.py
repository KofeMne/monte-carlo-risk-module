"""
Main Monte Carlo engine — orchestrates the full risk evaluation pipeline.

## Pipeline overview

    MarketState + TradeCandidate
          |
          v
    1. compute_sigma_and_drift()   [stats.py]
          Update market_state.sigma and market_state.drift from recent prices.
          Uses rolling window to stay current without look-ahead bias.
          |
          v
    2. generate_paths()            [path_generator.py]
          Simulate N future price paths (GBM or Bootstrap).
          Regime sigma multiplier applied here if configured.
          |
          v
    3. replay_trade() x N          [trade_replay.py]
          Apply V2 exit rules (SL, partial TP, trailing, time cap)
          to each simulated path. Produces N SimulationResult objects.
          |
          v
    4. calculate_risk_metrics()    [metrics.py]
          Aggregate: prob_loss, VaR_r, CVaR_r, Kelly, Sharpe, MAE/MFE, etc.
          |
          v
    5. make_trade_decision()       [decision.py]
          Apply threshold gates -> ACCEPT / REDUCE / REJECT.
          |
          v
    TradeDecision (with metrics snapshot and Kelly recommendation)

## Passive mode (default: True)

The engine defaults to passive_mode=True. In passive mode, the full pipeline
runs and all metrics are computed and logged — but the returned decision is
always ACCEPT with size_factor=1.0.

Why start in passive mode? Before the MC filter can control execution, you need
evidence that its risk estimates are meaningful. Run passive_mode=True for at
least 30-50 live trades. Then compare:
  - simulated prob_loss vs actual win/loss rate
  - simulated VaR_r vs the worst actual outcome
  - simulated expected_pnl_r vs actual average

If simulated and actual metrics are reasonably correlated, switch to
passive_mode=False. If they diverge significantly, investigate the replay
assumptions (close-only paths, fixed ATR, etc.) before going live.

## How to integrate with V2

After V2 constructs a TradePlan (entry, SL, TP, size), convert it to a
TradeCandidate and populate a MarketState from current market data:

    candidate = TradeCandidate(
        direction='LONG',
        entry_price=plan.entry_price,
        stop_loss=plan.sl_price,
        partial_tp_price=plan.partial_tp_price,
        partial_close_fraction=0.5,
        trailing_mode='ATR_BASED',
        atr=current_atr,
        trailing_atr_multiple=2.0,
        max_holding_bars=50,
        planned_size=plan.quantity,
        risk_pct=0.01,
        regime='TRENDING',
    )
    market_state = MarketState(
        recent_close_prices=recent_closes[-30:],
        recent_returns=recent_log_returns[-30:],
        atr=current_atr,
        sigma=0.0,    # engine will recompute from recent_close_prices
        drift=0.0,
        regime='TRENDING',
        timestamp=str(current_time),
    )
    decision = run_monte_carlo_analysis(candidate, market_state)
    if decision.action == 'REJECT':
        skip trade
    elif decision.action == 'REDUCE':
        size = plan.quantity * decision.size_factor

## Performance note

The replay loop is a plain Python for-loop over N paths. At 10,000 simulations
and 500 bars each, this runs in approximately 1-3 seconds. This is acceptable
for pre-trade filtering on a signal that fires a few times per day.

Vectorizing the trailing-stop state machine would require more complex numpy
indexing and is reserved as a future optimisation if latency becomes an issue.
"""

import logging
from typing import Optional

from config import MCConfig
from decision import DecisionConfig, TradeDecision, make_trade_decision
from market_state import MarketState
from metrics import calculate_risk_metrics
from path_generator import generate_paths
from stats import compute_sigma_and_drift
from trade_candidate import TradeCandidate
from trade_replay import replay_trade

logger = logging.getLogger(__name__)


def run_monte_carlo_analysis(
    candidate: TradeCandidate,
    market_state: MarketState,
    config: Optional[MCConfig] = None,
    decision_config: Optional[DecisionConfig] = None,
    passive_mode: bool = True,
) -> TradeDecision:
    """Run the full Monte Carlo risk evaluation pipeline for a single trade.

    Args:
        candidate:       The V2 trade to evaluate. All parameters (entry, SL,
                         partial TP, trailing) must be fully specified.
        market_state:    Current market snapshot. recent_close_prices and
                         recent_returns must contain at least rolling_window + 1 bars.
                         market_state.sigma and .drift are updated in-place.
        config:          MCConfig. Defaults to MCConfig() (conservative defaults).
        decision_config: DecisionConfig. Defaults to DecisionConfig() (conservative).
        passive_mode:    If True (default), always returns ACCEPT regardless of metrics.
                         Set to False only after validating that simulated metrics
                         correlate with actual trade outcomes.

    Returns:
        TradeDecision with action (ACCEPT/REDUCE/REJECT), size_factor,
        Kelly-based recommended_size_factor, reason string, and metric snapshots.
    """
    # --- Default config objects ---
    if config is None:
        config = MCConfig()
    if decision_config is None:
        decision_config = DecisionConfig()

    # --- Sanity check: warn if trade can outlive the simulation horizon ---
    # When max_holding_bars > horizon_bars, some paths will be truncated at horizon.
    # Exit statistics (mean_exit_bar, MAX_BARS rate) will be underestimated.
    if candidate.max_holding_bars > config.horizon_bars:
        logger.warning(
            "candidate.max_holding_bars=%d exceeds config.horizon_bars=%d. "
            "Paths will be truncated at horizon — exit duration stats may be underestimated.",
            candidate.max_holding_bars,
            config.horizon_bars,
        )

    # --- Step 1: Recompute sigma and drift from recent price history ---
    # We always recompute rather than trusting the sigma already on market_state.
    # Stale sigma (e.g. computed hours earlier) can over/under-estimate current vol.
    # The rolling window (default 20 bars) keeps the estimate responsive to recent moves.
    sigma, drift = compute_sigma_and_drift(market_state.recent_close_prices, config)
    market_state.sigma = sigma
    market_state.drift = drift

    # --- Step 2: Generate price paths ---
    # Path method (GBM vs Bootstrap) is determined by config.path_method.
    # Regime sigma multiplier from config.regime_sigma_multipliers is applied
    # inside generate_paths() based on market_state.regime.
    paths = generate_paths(market_state, config)

    # --- Step 3: Replay the trade on every simulated path ---
    # Each call to replay_trade() is independent — paths share no state.
    # A plain loop is used for clarity and correctness over premature vectorisation.
    n = config.num_simulations
    results = [replay_trade(paths[i], candidate) for i in range(n)]

    # --- Step 4: Aggregate results into risk metrics ---
    metrics = calculate_risk_metrics(results, config)

    # --- Step 5: Apply decision thresholds ---
    raw_decision = make_trade_decision(metrics, decision_config)

    # --- Step 6: Passive mode override ---
    # In passive mode, we return ACCEPT but log the real decision. This creates
    # the audit trail needed to validate simulation quality before going live.
    if passive_mode:
        decision = TradeDecision(
            action='ACCEPT',
            size_factor=1.0,
            recommended_size_factor=raw_decision.recommended_size_factor,
            reason=(
                f"PASSIVE MODE — trade accepted unconditionally. "
                f"Real decision would have been: {raw_decision.action}. "
                f"Reason: {raw_decision.reason}"
            ),
            prob_loss=metrics.prob_loss,
            var_r=metrics.var_r,
            cvar_r=metrics.cvar_r,
            expected_pnl_r=metrics.expected_pnl_r,
            kelly_fraction=metrics.kelly_fraction,
            profit_factor=metrics.profit_factor,
        )
    else:
        decision = raw_decision

    # --- Step 7: Log a single structured INFO line ---
    # One line per trade evaluation — easy to grep, parse, and compare with actual outcomes.
    logger.info(
        "MC | dir=%s | sims=%d | prob_loss=%.3f | sl_rate=%.3f | "
        "VaR_r=%.3f | CVaR_r=%.3f | EV_r=%.3f | Kelly=%.3f | PF=%.2f | "
        "skew=%.2f | MAE_r=%.3f | MFE_r=%.3f | action=%s%s",
        candidate.direction,
        n,
        metrics.prob_loss,
        metrics.prob_sl_hit,
        metrics.var_r,
        metrics.cvar_r,
        metrics.expected_pnl_r,
        metrics.kelly_fraction,
        metrics.profit_factor,
        metrics.skewness,
        metrics.avg_mae_r,
        metrics.avg_mfe_r,
        decision.action,
        " [PASSIVE]" if passive_mode else "",
    )

    if passive_mode and raw_decision.action != 'ACCEPT':
        logger.info(
            "MC PASSIVE — would have %s: %s",
            raw_decision.action,
            raw_decision.reason,
        )

    return decision

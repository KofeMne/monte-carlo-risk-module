"""
Turn Monte Carlo risk metrics into a trade decision: ACCEPT, REDUCE, or REJECT.

## Decision philosophy

The decision layer is conservative by design. When in doubt, it rejects.
This is because:
- False negative (missing a good trade): you lose one potential win.
- False positive (taking a bad trade): you take a real loss.
For a strategy with limited daily signals, missing one good trade is less
damaging than taking a bad one. The thresholds are set accordingly.

## Rule-based gates vs. Kelly sizing

Two separate mechanisms work together:

1. RULE-BASED GATES (DecisionConfig thresholds): Binary filters.
   If any reject threshold is breached, the trade is rejected regardless of
   other metrics. If any reduce threshold is breached but no reject, position
   size is cut. This is the primary execution control.

2. KELLY SIZING (recommended_size_factor): A continuous suggestion.
   Even for ACCEPT decisions, Kelly tells you the theoretically optimal fraction.
   A trade can pass all gates (ACCEPT) but have a low Kelly fraction, suggesting
   the risk budget is too aggressive. This is informational — the caller decides
   whether to use it.

## Default thresholds and their rationale

reject_prob_loss = 0.65: Most V2 trades with positive EV have prob_loss 35-55%.
  A trade where 65%+ of simulations lose money has deeply negative expectancy
  in the simulation. Reject it.

reject_var_r = -3.0: In the 5th percentile scenario, losing more than 3R is
  unacceptable. That means there's a 5% chance the trade loses 3x its risk.

min_kelly_fraction = 0.05: A Kelly fraction below 5% means the simulation
  sees almost no positive expectancy. Kelly below 5% is a sign to skip.

reduce_prob_loss = 0.55: Between 55% and 65% loss rate — the trade has
  questionable edge. Take it, but at half size.

reduce_var_r = -2.0: Between -2R and -3R tail loss — elevated but not extreme.
  Take it, but smaller.

## Conservative start

On first deployment, run with passive_mode=True in the engine. The decision
is computed but overridden to ACCEPT. Review the logged decisions for 30-50 trades
before switching passive_mode=False and letting the gates affect execution.
"""

import numpy as np
from dataclasses import dataclass, field
from metrics import RiskMetrics


@dataclass
class DecisionConfig:
    """Configurable thresholds for the trade decision layer.

    All thresholds are conservative defaults suitable for initial go-live.
    Tune only after collecting out-of-sample validation data (Stage F).

    Attributes:
        reject_prob_loss:   Reject if prob_loss >= this. (0.65 = reject when 65%+ of
                            simulations lose money)
        reduce_prob_loss:   Reduce size if prob_loss >= this (and < reject threshold).
        reject_var_r:       Reject if VaR_r <= this. (e.g. -3.0 = reject if the 5th
                            percentile outcome is worse than -3R)
        reduce_var_r:       Reduce size if VaR_r <= this (and > reject threshold).
        min_expected_pnl_r: Reject if mean simulated outcome is below this. Catches
                            deeply negative-EV trades that pass other gates.
        min_kelly_fraction: Reject if Kelly fraction < this. Near-zero Kelly means the
                            simulation sees essentially no positive expectancy.
        reduce_size_factor: Position multiplier applied on REDUCE. 0.5 = half size.
    """
    reject_prob_loss: float = 0.65
    reduce_prob_loss: float = 0.55
    reject_var_r: float = -3.0
    reduce_var_r: float = -2.0
    min_expected_pnl_r: float = -0.5
    min_kelly_fraction: float = 0.05
    reduce_size_factor: float = 0.5

    def __post_init__(self) -> None:
        """Validate that reduce thresholds are strictly inside reject thresholds."""
        if self.reduce_prob_loss >= self.reject_prob_loss:
            raise ValueError(
                f"reduce_prob_loss ({self.reduce_prob_loss}) must be strictly less than "
                f"reject_prob_loss ({self.reject_prob_loss}). "
                "The reduce zone must be inside the reject zone."
            )
        # reject_var_r is MORE negative than reduce_var_r (e.g. -3.0 < -2.0)
        if self.reduce_var_r <= self.reject_var_r:
            raise ValueError(
                f"reduce_var_r ({self.reduce_var_r}) must be greater than (less negative than) "
                f"reject_var_r ({self.reject_var_r}). "
                "E.g. reduce=-2.0, reject=-3.0 is valid: the reduce zone is between -2R and -3R."
            )
        if not (0.0 < self.reduce_size_factor < 1.0):
            raise ValueError(
                f"reduce_size_factor ({self.reduce_size_factor}) must be in (0, 1). "
                "0.5 means half size."
            )


@dataclass
class TradeDecision:
    """The outcome of the decision layer for a single trade.

    Attributes:
        action:                   'ACCEPT', 'REDUCE', or 'REJECT'.
        size_factor:              Rule-based position multiplier.
                                  1.0 = full risk, reduce_size_factor = reduced, 0.0 = skip.
        recommended_size_factor:  Kelly-based position fraction. Informational only —
                                  the caller decides whether to apply it. Always computed
                                  regardless of the rule-based action.
        reason:                   Human-readable explanation of the decision.
                                  First matching gate's message (reject/reduce gates are
                                  evaluated in priority order; first hit wins).
        prob_loss:                Snapshot from RiskMetrics (for logging without re-accessing).
        var_r:                    Snapshot from RiskMetrics.
        cvar_r:                   Snapshot from RiskMetrics.
        expected_pnl_r:           Snapshot from RiskMetrics.
        kelly_fraction:           Snapshot from RiskMetrics.
        profit_factor:            Snapshot from RiskMetrics.
    """
    action: str
    size_factor: float
    recommended_size_factor: float
    reason: str
    prob_loss: float
    var_r: float
    cvar_r: float
    expected_pnl_r: float
    kelly_fraction: float
    profit_factor: float


def make_trade_decision(metrics: RiskMetrics, config: DecisionConfig) -> TradeDecision:
    """Evaluate Monte Carlo risk metrics and return an ACCEPT / REDUCE / REJECT decision.

    Gates are evaluated in priority order. The first matching gate determines the
    action — multiple gates can be breached but only the first is reported in reason.
    This keeps logs readable and unambiguous.

    Args:
        metrics: RiskMetrics from calculate_risk_metrics().
        config:  DecisionConfig with thresholds. Defaults are conservative.

    Returns:
        TradeDecision with action, size_factor, Kelly recommendation, and reason string.
    """
    # Kelly-based recommended size is always computed, regardless of decision.
    # This gives the caller an optimal-sizing suggestion even for ACCEPT decisions.
    recommended_size_factor = float(np.clip(metrics.kelly_fraction, 0.0, 1.0))

    # Shared snapshot fields for all decision paths.
    snapshot = dict(
        prob_loss=metrics.prob_loss,
        var_r=metrics.var_r,
        cvar_r=metrics.cvar_r,
        expected_pnl_r=metrics.expected_pnl_r,
        kelly_fraction=metrics.kelly_fraction,
        profit_factor=metrics.profit_factor,
        recommended_size_factor=recommended_size_factor,
    )

    # --- REJECT gates (checked in this priority order) ---

    # Gate R1: Kelly too low — simulation sees almost no positive expectancy.
    if metrics.kelly_fraction < config.min_kelly_fraction:
        return TradeDecision(
            action='REJECT',
            size_factor=0.0,
            reason=(
                f"Kelly fraction={metrics.kelly_fraction:.4f} < "
                f"min_kelly threshold={config.min_kelly_fraction:.4f}. "
                "Simulation sees negligible positive expectancy — skip trade."
            ),
            **snapshot,
        )

    # Gate R2: Too many losing simulations.
    if metrics.prob_loss >= config.reject_prob_loss:
        return TradeDecision(
            action='REJECT',
            size_factor=0.0,
            reason=(
                f"prob_loss={metrics.prob_loss:.3f} >= "
                f"reject threshold={config.reject_prob_loss:.3f}. "
                "Majority of simulations lose money."
            ),
            **snapshot,
        )

    # Gate R3: Tail loss too severe (VaR deeper than reject threshold).
    if metrics.var_r <= config.reject_var_r:
        return TradeDecision(
            action='REJECT',
            size_factor=0.0,
            reason=(
                f"VaR_r={metrics.var_r:.3f}R <= "
                f"reject threshold={config.reject_var_r:.3f}R. "
                f"Tail loss too severe (the VaR-quantile outcome is below the reject threshold)."
            ),
            **snapshot,
        )

    # Gate R4: Deeply negative expected value.
    if metrics.expected_pnl_r < config.min_expected_pnl_r:
        return TradeDecision(
            action='REJECT',
            size_factor=0.0,
            reason=(
                f"expected_pnl_r={metrics.expected_pnl_r:.3f}R < "
                f"min_expected_pnl threshold={config.min_expected_pnl_r:.3f}R. "
                "Mean simulated outcome is too negative."
            ),
            **snapshot,
        )

    # --- REDUCE gates ---

    # Gate D1: Elevated loss probability.
    if metrics.prob_loss >= config.reduce_prob_loss:
        return TradeDecision(
            action='REDUCE',
            size_factor=config.reduce_size_factor,
            reason=(
                f"prob_loss={metrics.prob_loss:.3f} >= "
                f"reduce threshold={config.reduce_prob_loss:.3f}. "
                f"Elevated loss rate — reducing to {config.reduce_size_factor:.0%} size."
            ),
            **snapshot,
        )

    # Gate D2: Elevated tail loss.
    if metrics.var_r <= config.reduce_var_r:
        return TradeDecision(
            action='REDUCE',
            size_factor=config.reduce_size_factor,
            reason=(
                f"VaR_r={metrics.var_r:.3f}R <= "
                f"reduce threshold={config.reduce_var_r:.3f}R. "
                f"Elevated tail risk — reducing to {config.reduce_size_factor:.0%} size."
            ),
            **snapshot,
        )

    # --- ACCEPT ---
    return TradeDecision(
        action='ACCEPT',
        size_factor=1.0,
        reason=(
            f"All risk checks passed. "
            f"prob_loss={metrics.prob_loss:.3f}, "
            f"VaR_r={metrics.var_r:.3f}R, "
            f"EV={metrics.expected_pnl_r:.3f}R, "
            f"Kelly={metrics.kelly_fraction:.3f}."
        ),
        **snapshot,
    )

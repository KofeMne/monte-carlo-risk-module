"""
Standalone diagnostic runner for the Monte Carlo risk module.

## Purpose

Run this file directly to examine the full MC pipeline in isolation —
no V2 strategy, no QuantConnect, no live data required.

It constructs synthetic TradeCandidate and MarketState objects, runs the
engine under several labelled scenarios, and prints all risk metrics to
the terminal so you can see exactly what the module computes and why.

## How to run

    cd monte_carlo
    python main.py

## Important: entry price anchoring

generate_paths() starts every simulated path at market_state.recent_close_prices[-1]
(the current market price S0). Every candidate must therefore use that same price
as entry_price. Using a fixed number like 100.0 when S0 has drifted to 103.2 creates
phantom PnL that corrupts all metrics. All helpers below anchor entry to S0.

## Scenarios covered

  1. Symmetric LONG — low prob_loss, passes all gates → ACCEPT
  2. Wide-TP LONG — high win per trade, but 69% of sims hit SL → REJECT by prob_loss gate
     (positive EV! Demonstrates system conservatism — optimizer can tune thresholds)
  3. SHORT trade — mirrors LONG logic, direction sanity check
  4. Passive vs. active mode — shows how passive mode creates audit trail
  5. GBM vs. Bootstrap — compare tail width on same trade
  6. Crash regime — sigma multiplier widens tails, downgrade in VaR visible
"""

import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import MCConfig
from decision import DecisionConfig, TradeDecision
from engine import run_monte_carlo_analysis
from market_state import MarketState
from trade_candidate import TradeCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)
_log = logging.getLogger("mc_main")


# ---------------------------------------------------------------------------
# Synthetic data builders — entry always anchored to S0 (last close)
# ---------------------------------------------------------------------------

def _make_ms(n_bars: int = 60, sigma: float = 0.015,
             regime: str = "NEUTRAL", seed: int = 0) -> MarketState:
    """Build a synthetic MarketState. The last close price is S0 for all paths."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0002, sigma, n_bars)   # tiny upward drift
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    return MarketState(
        recent_close_prices=closes.tolist(),
        recent_returns=log_returns.tolist(),
        atr=closes[-1] * sigma * 1.5,
        sigma=0.0,           # engine recomputes this
        drift=0.0,
        regime=regime,
        timestamp="2025-01-01T09:30:00",
    )


def _long(ms: MarketState,
          sl_pct: float = 0.03,
          tp_pct: float = 0.03,
          trail_mult: float = 2.0,
          max_bars: int = 50) -> TradeCandidate:
    """LONG candidate anchored to last close of market state.

    sl_pct / tp_pct are fractions of entry price, e.g. 0.03 = 3%.
    ATR is set to half the SL distance (consistent with typical ATR-to-SL ratios).
    """
    entry = ms.recent_close_prices[-1]
    return TradeCandidate(
        direction="LONG",
        entry_price=entry,
        stop_loss=entry * (1.0 - sl_pct),
        partial_tp_price=entry * (1.0 + tp_pct),
        partial_close_fraction=0.5,
        trailing_mode="ATR_BASED",
        atr=entry * sl_pct * 0.5,
        trailing_atr_multiple=trail_mult,
        max_holding_bars=max_bars,
        planned_size=1.0,
        risk_pct=0.01,
        regime="NEUTRAL",
    )


def _short(ms: MarketState,
           sl_pct: float = 0.03,
           tp_pct: float = 0.03,
           trail_mult: float = 2.0,
           max_bars: int = 50) -> TradeCandidate:
    """SHORT candidate anchored to last close of market state."""
    entry = ms.recent_close_prices[-1]
    return TradeCandidate(
        direction="SHORT",
        entry_price=entry,
        stop_loss=entry * (1.0 + sl_pct),
        partial_tp_price=entry * (1.0 - tp_pct),
        partial_close_fraction=0.5,
        trailing_mode="ATR_BASED",
        atr=entry * sl_pct * 0.5,
        trailing_atr_multiple=trail_mult,
        max_holding_bars=max_bars,
        planned_size=1.0,
        risk_pct=0.01,
        regime="NEUTRAL",
    )


def _config(**kw) -> MCConfig:
    defaults = dict(num_simulations=2_000, horizon_bars=60,
                    random_seed=42, drift_mode="zero")
    defaults.update(kw)
    return MCConfig(**defaults)


def _default_dc() -> DecisionConfig:
    """Conservative defaults: reject if prob_loss >= 0.65 or VaR <= -3R."""
    return DecisionConfig()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_W = 72

def _section(title: str) -> None:
    print(f"\n{'=' * _W}")
    print(f"  {title}")
    print('=' * _W)


def _sub(label: str) -> None:
    print(f"\n{'-' * _W}")
    print(f"  {label}")
    print('-' * _W)


def _print_decision(d: TradeDecision, label: str = "") -> None:
    mark = {"ACCEPT": "✓", "REDUCE": "~", "REJECT": "✗"}.get(d.action, "?")
    pfx = f"[{label}] " if label else ""
    print(f"\n  {pfx}ACTION : {mark} {d.action}")
    print(f"  {pfx}REASON : {d.reason}")
    print()
    print(f"    prob_loss      : {d.prob_loss:.3f}   "
          "(fraction of sims that ended with pnl_r < 0)")
    print(f"    var_r          : {d.var_r:+.3f} R  "
          "(worst 5th-percentile outcome)")
    print(f"    cvar_r         : {d.cvar_r:+.3f} R  "
          "(mean of worst 5% tail)")
    print(f"    expected_pnl_r : {d.expected_pnl_r:+.3f} R  "
          "(mean across all sims)")
    print(f"    kelly_fraction : {d.kelly_fraction:.3f}   "
          "(optimal position fraction from Kelly criterion)")
    print(f"    profit_factor  : {d.profit_factor:.3f}   "
          "(sum_wins / sum_losses, >1 = edge)")
    print()
    print(f"    size_factor    : {d.size_factor:.2f}     "
          "(rule-based multiplier applied to position)")
    print(f"    recommended    : {d.recommended_size_factor:.2f}     "
          "(Kelly suggestion, informational)")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_1_good_long() -> None:
    """
    Scenario 1 — Symmetric LONG (SL=3%, TP=3% from entry).

    With zero-drift GBM and equal SL/TP distances in log space, theory predicts
    P(SL first) ≈ 0.49. The partial TP + trailing mechanism creates slight positive
    asymmetry (locks in gains on the 50% that fires the TP), so EV > 0 and
    prob_loss should come in just below the 0.55 REDUCE threshold.

    Expected outcome: ACCEPT with size_factor=1.0.

    What to verify:
      - prob_loss around 0.40-0.50
      - var_r better than -3.0R (default reject)
      - expected_pnl_r slightly positive
      - kelly_fraction > 0.05 (positive expectancy)
    """
    _section("Scenario 1 — Symmetric LONG  (SL=3%, TP=3%)")
    ms = _make_ms(sigma=0.015)
    entry = ms.recent_close_prices[-1]
    print(f"  entry={entry:.2f} | SL={entry*0.97:.2f} (-3%) | "
          f"partial_TP={entry*1.03:.2f} (+3%) | sigma≈1.5% | 2000 sims")

    c = _long(ms, sl_pct=0.03, tp_pct=0.03)
    d = run_monte_carlo_analysis(c, ms, _config(), _default_dc(), passive_mode=False)
    _print_decision(d, "Scenario 1")


def scenario_2_wide_tp_reject() -> None:
    """
    Scenario 2 — Wide-TP LONG (SL=3%, TP=7%): positive EV, REJECTs on prob_loss.

    Classical trading tension: the 2.3:1 reward-to-risk ratio sounds attractive,
    but in a zero-drift market the math works against it.

    Barrier-hitting probability (zero-drift GBM formula):
        P(SL first) = log(TP/entry) / log(TP/SL)
                    = log(1.07) / log(1.07/0.97) ≈ 0.069 / 0.100 ≈ 0.69

    So 69% of simulated paths hit the SL. But the 31% that hit TP earn 2.33R on
    the partial position + continued trailing gains. The expected PnL is POSITIVE:
        EV ≈ 0.31 * 2.33R - 0.69 * 1R ≈ +0.02 to +0.10R (after partial TP effect)

    The default reject threshold is prob_loss >= 0.65. This trade trips it.

    KEY INSIGHT: The system REJECTs a mathematically profitable trade because it
    is conservative about frequency of loss. The optimizer (optimizer.py) can tune
    this threshold upward if historical data shows the high-prob_loss / high-EV
    profile actually delivers Sharpe improvement in practice.

    Expected outcome: REJECT (prob_loss gate fires, not VaR or Kelly).
    """
    _section("Scenario 2 — Wide-TP LONG  (SL=3%, TP=7%): positive EV, REJECTED")
    ms = _make_ms(sigma=0.015)
    entry = ms.recent_close_prices[-1]
    print(f"  entry={entry:.2f} | SL={entry*0.97:.2f} (-3%) | "
          f"partial_TP={entry*1.07:.2f} (+7%) | sigma≈1.5% | 2000 sims")
    print("  Theory: P(SL first) ≈ 69%. Default reject_prob_loss = 0.65.")
    print("  Note: EV is POSITIVE — this is a conservatism trade-off, not a bad trade.")

    c = _long(ms, sl_pct=0.03, tp_pct=0.07)
    d = run_monte_carlo_analysis(c, ms, _config(), _default_dc(), passive_mode=False)
    _print_decision(d, "Scenario 2")

    # Show that loosening prob_loss threshold to 0.75 would ACCEPT this trade
    print()
    print("  --- Retry with looser prob_loss threshold (0.75) ---")
    loose = DecisionConfig(
        reject_prob_loss=0.75,
        reduce_prob_loss=0.65,
        reject_var_r=-3.0,
        reduce_var_r=-2.0,
        min_kelly_fraction=0.05,
        reduce_size_factor=0.5,
    )
    ms2 = _make_ms(sigma=0.015)
    c2 = _long(ms2, sl_pct=0.03, tp_pct=0.07)
    d2 = run_monte_carlo_analysis(c2, ms2, _config(), loose, passive_mode=False)
    _print_decision(d2, "Loose thresholds")
    print("  (Use the optimizer to find the right threshold for your trade history.)")


def scenario_3_short() -> None:
    """
    Scenario 3 — SHORT trade (SL=3% above, TP=3% below).

    Mirrors scenario 1 but direction is reversed. With symmetric 3% barriers
    and zero drift, P(SL first) ≈ 0.49 for SHORT as well. The engine should
    produce coherent output: profit when price falls, loss when it rises.

    What to verify: no sign errors, sensible metrics, action matches scenario 1.
    """
    _section("Scenario 3 — SHORT trade  (SL=3%, TP=3%)")
    ms = _make_ms(sigma=0.015, seed=1)
    entry = ms.recent_close_prices[-1]
    print(f"  entry={entry:.2f} | SL={entry*1.03:.2f} (+3%) | "
          f"partial_TP={entry*0.97:.2f} (-3%) | sigma≈1.5% | 2000 sims")

    c = _short(ms, sl_pct=0.03, tp_pct=0.03)
    d = run_monte_carlo_analysis(c, ms, _config(), _default_dc(), passive_mode=False)
    _print_decision(d, "Scenario 3 SHORT")


def scenario_4_passive_vs_active() -> None:
    """
    Scenario 4 — Passive mode vs. active mode on the wide-TP trade.

    passive_mode=True (default, safe for initial go-live):
      - Always returns ACCEPT with size_factor=1.0
      - Still runs the full pipeline and logs real metrics
      - Reason string contains what the real decision would have been
      - Purpose: build the audit trail before enabling live rejection

    passive_mode=False (enable after validation):
      - Returns the real ACCEPT / REDUCE / REJECT decision
      - Use only after verifying simulated prob_loss correlates with actual loss rate
    """
    _section("Scenario 4 — Passive vs. Active mode  (wide-TP LONG)")

    _sub("4a — PASSIVE mode  (always returns ACCEPT)")
    ms = _make_ms(sigma=0.015, seed=2)
    c = _long(ms, sl_pct=0.03, tp_pct=0.07)
    d = run_monte_carlo_analysis(c, ms, _config(), _default_dc(), passive_mode=True)
    _print_decision(d, "PASSIVE")
    print()
    print("  ^ Note: action=ACCEPT but reason shows the real decision.")
    print("  Review these logs for 30-50 trades before enabling active mode.")

    _sub("4b — ACTIVE mode  (real decision returned)")
    ms2 = _make_ms(sigma=0.015, seed=2)
    c2 = _long(ms2, sl_pct=0.03, tp_pct=0.07)
    d2 = run_monte_carlo_analysis(c2, ms2, _config(), _default_dc(), passive_mode=False)
    _print_decision(d2, "ACTIVE")


def scenario_5_gbm_vs_bootstrap() -> None:
    """
    Scenario 5 — GBM vs. Bootstrap path generation.

    GBM draws returns from a normal distribution — thin tails, no volatility clustering.
    Bootstrap resamples from the actual historical return series — fat tails,
    negative skewness, and any autocorrelation that exists in real data.

    On synthetic data (which IS normally distributed), differences are small.
    On real market data, Bootstrap typically produces:
      - More negative VaR_r and CVaR_r  (fatter left tail)
      - Slightly different prob_loss     (tail events more frequent)
      - More realistic MAE distribution

    What to verify: both complete without error; compare VaR difference.
    """
    _section("Scenario 5 — GBM vs. Bootstrap  (same symmetric LONG)")

    _sub("5a — GBM  (parametric, normal returns)")
    ms_g = _make_ms(sigma=0.015, n_bars=80, seed=3)
    c_g = _long(ms_g, sl_pct=0.03, tp_pct=0.03)
    d_g = run_monte_carlo_analysis(
        c_g, ms_g, _config(path_method="GBM"), _default_dc(), passive_mode=False
    )
    _print_decision(d_g, "GBM")

    _sub("5b — Bootstrap  (resample from 80 bars of history)")
    ms_b = _make_ms(sigma=0.015, n_bars=80, seed=3)
    c_b = _long(ms_b, sl_pct=0.03, tp_pct=0.03)
    d_b = run_monte_carlo_analysis(
        c_b, ms_b, _config(path_method="BOOTSTRAP", bootstrap_lookback=70),
        _default_dc(), passive_mode=False,
    )
    _print_decision(d_b, "Bootstrap")

    print()
    print("  Delta (Bootstrap - GBM):")
    print(f"    prob_loss      : {d_b.prob_loss - d_g.prob_loss:+.3f}")
    print(f"    var_r          : {d_b.var_r - d_g.var_r:+.3f} R")
    print(f"    cvar_r         : {d_b.cvar_r - d_g.cvar_r:+.3f} R")
    print(f"    expected_pnl_r : {d_b.expected_pnl_r - d_g.expected_pnl_r:+.3f} R")
    print()
    print("  Interpretation:")
    print("  Bootstrap is more conservative here because it resamples from only 80 specific")
    print("  historical bars. Those 80 bars have sample-specific quirks (clustering of adverse")
    print("  returns) that make paths more adversarial than the idealized GBM assumption.")
    print("  With real market data (200-500 bars), Bootstrap would capture fat tails and")
    print("  skewness that GBM misses — typically producing lower VaR_r and CVaR_r.")


def scenario_6_crash_regime() -> None:
    """
    Scenario 6 — Crash regime sigma multiplier.

    config.regime_sigma_multipliers = {'CRASH': 1.5} means: when the market
    state reports regime='CRASH', sigma is scaled by 1.5x before path generation.

    Why this matters: a flat historical sigma computed from calm periods will
    underestimate how much prices can move during a crash. Multiplying sigma
    by 1.5 produces more extreme paths — the risk module becomes more conservative
    exactly when the market is most dangerous.

    Effect to observe:
      - prob_loss increases (more paths hit the SL with wider swings)
      - VaR_r worsens (fatter tails from wider sigma)
      - Action may downgrade from ACCEPT → REDUCE or REDUCE → REJECT
    """
    _section("Scenario 6 — Crash regime sigma multiplier  (1.5× sigma)")

    _sub("6a — NEUTRAL regime  (sigma × 1.0)")
    ms_n = _make_ms(sigma=0.015, regime="NEUTRAL", seed=4)
    c_n = _long(ms_n, sl_pct=0.03, tp_pct=0.03)
    cfg_n = _config(regime_sigma_multipliers={})
    d_n = run_monte_carlo_analysis(c_n, ms_n, cfg_n, _default_dc(), passive_mode=False)
    _print_decision(d_n, "NEUTRAL")

    _sub("6b — CRASH regime  (sigma × 1.5)")
    ms_c = _make_ms(sigma=0.015, regime="CRASH", seed=4)
    c_c = _long(ms_c, sl_pct=0.03, tp_pct=0.03)
    cfg_c = _config(regime_sigma_multipliers={"CRASH": 1.5})
    d_c = run_monte_carlo_analysis(c_c, ms_c, cfg_c, _default_dc(), passive_mode=False)
    _print_decision(d_c, "CRASH 1.5×")

    print()
    print("  Effect of 1.5× sigma in CRASH regime:")
    print(f"    prob_loss : {d_n.prob_loss:.3f}  →  {d_c.prob_loss:.3f}"
          f"  (Δ {d_c.prob_loss - d_n.prob_loss:+.3f})")
    print(f"    var_r     : {d_n.var_r:+.3f} R  →  {d_c.var_r:+.3f} R"
          f"  (Δ {d_c.var_r - d_n.var_r:+.3f} R)")
    print(f"    action    : {d_n.action}  →  {d_c.action}")
    print()
    print("  Why the effect is small for a symmetric trade:")
    print("  With SL=3% below and TP=3% above, wider sigma makes BOTH barriers more")
    print("  accessible proportionally. SL and TP both become easier to reach, so the")
    print("  hit-rate ratio stays roughly the same → prob_loss barely changes.")
    print("  VaR is capped at -1.000R (the SL exit is always exactly -1R), so it cannot")
    print("  worsen. The crash multiplier has the most visible impact when:")
    print("   - The trade is near one barrier (SL almost within reach at normal sigma)")
    print("   - The VaR gate is already close to its reject threshold (-3.0R)")
    print("   - Or you use a trend-following setup where drift amplifies sigma changes.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure UTF-8 output so the glyphs below (✓ ✗ σ → ≈) don't raise
    # UnicodeEncodeError on consoles using a non-UTF-8 codepage (e.g. Windows cp1251).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"\n{'#' * _W}")
    print("  Monte Carlo Risk Module — Standalone Diagnostic")
    print("  All entries are anchored to S0 (last close of each market state).")
    print(f"{'#' * _W}")
    print("""
  Reading the output
  ------------------
  Each scenario prints:
    - ACTION : ✓ ACCEPT  ~REDUCE  ✗ REJECT
    - Key risk metrics with plain-English labels
    - size_factor  : position multiplier the engine recommends (rule-based)
    - recommended  : Kelly-based suggestion (informational, may differ)

  INFO lines from the engine logger show one structured line per trade,
  useful for production log parsing. They appear ABOVE each scenario block
  because Python's buffered stdout flushes after the logger.

  About var_r = -1.000 R in all scenarios
  ----------------------------------------
  The SL exit always produces exactly -1R by construction (that is what 1R means).
  With 3% SL and sigma=1.5%, the SL is about 2 sigma away, so it fires in ~50% of
  simulations. The 5th-percentile (VaR at 95%) lands in the -1R SL bucket.
  The VaR gate (-3.0R reject threshold) is designed for trades where partial exits
  or trailing stops create a distribution of losses across multiple R-multiples.
  In real V2 trades, partial TP + aggressive trailing can produce losses between
  -1R and -0.3R on the "partial then trail" paths, making VaR more informative.
  """)

    scenario_1_good_long()
    scenario_2_wide_tp_reject()
    scenario_3_short()
    scenario_4_passive_vs_active()
    scenario_5_gbm_vs_bootstrap()
    scenario_6_crash_regime()

    print(f"\n{'#' * _W}")
    print("  All scenarios complete.")
    print()
    print("  Stage F — what comes next:")
    print("  1. Collect 30-50 real V2 trades with actual_pnl_r (passive_mode=True).")
    print("  2. Compare simulated prob_loss vs actual loss rate in the logs.")
    print("  3. If correlation is good: switch passive_mode=False.")
    print("  4. Run optimizer.py:run_walk_forward_optimization() to tune thresholds.")
    print(f"{'#' * _W}\n")


if __name__ == "__main__":
    main()

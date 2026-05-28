"""
Apply V2 trade exit rules to a single simulated price path.

## Purpose

This module replicates the V2 strategy's exit logic (stop-loss, partial take-profit,
ATR trailing stop, time cap) on a Monte Carlo-generated price path. Running this
for all N simulated paths produces the distribution of possible trade outcomes
that the metrics and decision layers consume.

## Key design decisions

### Close-only approximation
GBM and bootstrap paths contain only one price per bar (the simulated "close").
Real V2 exit logic uses OHLC bars: SL is triggered by bar.low (LONG) and partial TP
by bar.high (LONG). With close-only paths we check both conditions against the
single close price.

Consequence: this UNDERESTIMATES how often intrabar moves trigger the SL or
partial TP. The actual SL hit rate in live trading will be slightly higher than
the simulated probability. This is a known, documented conservative approximation.
It is acceptable for risk-screening purposes — if a trade looks bad with
close-only paths, it is genuinely risky.

### Exit price discipline
When the SL fires, the exit price is set to `candidate.stop_loss`, not to the
path price that crossed it. This models a stop-limit or stop-market order filling
at the stated price. Same logic applies to partial TP (exits at partial_tp_price)
and trailing SL (exits at trailing_stop value). Only MAX_BARS exits at the actual
path price, because there is no stated target — the trade simply expires.

### Trailing stop initialisation
After partial TP fires, the trailing stop starts at `candidate.stop_loss` (the
original hard stop). This means the trailing stop has room to move up toward
entry before tightening. It is more conservative than starting at entry (breakeven)
because it allows for some initial adverse movement before the trail tightens.
This mirrors the V2 live strategy behaviour where trailing takes over from the
existing stop position.

### WORST_CASE same-bar rule
On each bar, the SL/trailing SL check runs before the partial TP check. With
close-only paths, a bar can only have one price, so both cannot simultaneously
trigger on the same bar — but the ordering ensures correctness if edge cases arise.

### MAE / MFE tracking
Maximum Adverse Excursion (MAE): the worst unrealised PnL reached during the trade
(always <= 0 for a trade that hits SL at -1R). Tells you how deep the drawdown got.
Maximum Favorable Excursion (MFE): the best unrealised PnL reached. Tells you how
much profit was available before exit. Large MFE with small final PnL = TP too early.
"""

from dataclasses import dataclass
from typing import Optional
from trade_candidate import TradeCandidate


@dataclass
class SimulationResult:
    """Outcome of replaying a single trade on a single simulated price path.

    Fields:
        exit_price:       The price at which the trade was closed.
                          For SL/trailing exits: the stated stop price (models order fill).
                          For MAX_BARS: the actual path price at the final bar.
        exit_bar:         1-indexed bar number when the trade exited.
        exit_reason:      One of:
                            'SL'                           - hard stop hit before partial TP
                            'TRAILING_SL'                  - trailing stop hit (no partial TP)
                            'PARTIAL_TP_THEN_TRAILING_SL'  - partial TP fired, then trailing stop
                            'MAX_BARS'                     - forced exit at time cap
        partial_tp_hit:   True if the partial take-profit price was reached.
        partial_exit_bar: Bar index when partial TP fired, or None.
        pnl_pct:          Blended profit/loss as a fraction of entry price.
                          Positive = profit, negative = loss. Blended when partial TP fired:
                          (partial_fraction x partial_pct) + (remaining_fraction x exit_pct).
        pnl_r:            pnl_pct expressed in R multiples (1R = |entry - stop_loss| / entry).
                          A full SL hit -> pnl_r ~= -1.0. A full TP hit at 3R -> pnl_r ~= +3.0.
        r_size:           |entry_price - stop_loss| in price units (1R in dollar terms).
        mae_r:            Maximum Adverse Excursion in R. Worst (most negative) unrealised
                          pnl_r during the trade. Always <= 0 on a losing trade.
        mfe_r:            Maximum Favorable Excursion in R. Best (most positive) unrealised
                          pnl_r during the trade. Always >= 0 on a winning trade.
    """
    exit_price: float
    exit_bar: int
    exit_reason: str
    partial_tp_hit: bool
    partial_exit_bar: Optional[int]
    pnl_pct: float
    pnl_r: float
    r_size: float
    mae_r: float
    mfe_r: float


# Exit reason constants — avoids magic strings scattered through the codebase.
_SL = 'SL'
_TRAILING_SL = 'TRAILING_SL'
_PARTIAL_THEN_TRAILING_SL = 'PARTIAL_TP_THEN_TRAILING_SL'
_MAX_BARS = 'MAX_BARS'


def replay_trade(path, candidate: TradeCandidate) -> SimulationResult:
    """Apply V2 trade exit rules to a single simulated price path.

    Args:
        path:      1D array-like of shape (horizon_bars + 1,).
                   path[0] is the starting price S0.
                   path[1:] are simulated future bar prices (close-only).
        candidate: Fully specified V2 trade. Defines entry, hard SL, partial TP
                   rules, trailing stop behaviour, and time cap.

    Returns:
        SimulationResult with exit details, blended PnL in % and R multiples,
        and MAE/MFE excursion statistics.

    Raises:
        ValueError: If partial_tp_price is set but partial_close_fraction is None.
    """
    # --- Input validation ---
    if candidate.partial_tp_price is not None and candidate.partial_close_fraction is None:
        raise ValueError(
            "partial_tp_price is set but partial_close_fraction is None. "
            "Both must be provided together — partial_tp_price defines WHEN to exit, "
            "partial_close_fraction defines HOW MUCH to close."
        )

    entry = candidate.entry_price
    sl = candidate.stop_loss
    r_size = abs(entry - sl)

    # r_unit is the 1R move as a fraction of entry price (used for R-multiple conversion).
    # Guard against zero: a trade with stop_loss == entry_price is degenerate.
    r_unit = r_size / entry if entry != 0 and r_size > 0 else 0.0

    is_long = (candidate.direction == 'LONG')
    use_trailing = (candidate.trailing_mode == 'ATR_BASED')
    partial_tp = candidate.partial_tp_price
    partial_frac = candidate.partial_close_fraction

    # Clamp max_bar to path length so we never index out of bounds.
    max_bar = min(candidate.max_holding_bars, len(path) - 1)

    # Edge case: path has only S0, no future bars to check.
    if max_bar <= 0:
        return _make_result(
            exit_price=float(path[0]),
            exit_bar=0,
            exit_reason=_MAX_BARS,
            partial_tp_hit=False,
            partial_exit_bar=None,
            partial_exit_price=None,
            partial_frac=partial_frac,
            entry=entry,
            r_unit=r_unit,
            r_size=r_size,
            is_long=is_long,
            worst_pnl_r=0.0,
            best_pnl_r=0.0,
        )

    # --- State machine ---
    partial_done = False
    partial_exit_bar: Optional[int] = None
    partial_exit_price: Optional[float] = None

    # Trailing stop is None until partial TP fires.
    # Once activated, it starts at candidate.stop_loss (the original hard stop) so
    # it has room to tighten as price moves in our favour.
    trailing_stop: Optional[float] = None

    # MAE/MFE tracking: worst and best unrealised pnl_r seen during the trade.
    worst_pnl_r = 0.0
    best_pnl_r = 0.0

    for bar_idx in range(1, max_bar + 1):
        price = float(path[bar_idx])

        # --- Update MAE/MFE before checking exits ---
        # Unrealised pnl at this bar's price, normalised to R multiples.
        unrealised_pct = (price - entry) / entry if is_long else (entry - price) / entry
        unrealised_r = unrealised_pct / r_unit if r_unit > 0 else 0.0
        worst_pnl_r = min(worst_pnl_r, unrealised_r)
        best_pnl_r = max(best_pnl_r, unrealised_r)

        # --- Step A: SL / Trailing SL check (WORST_CASE: always before partial TP) ---
        # Checking SL first implements the WORST_CASE same-bar rule from V2:
        # if both SL and TP could trigger on the same bar, we assume SL hits first.
        if partial_done and use_trailing and trailing_stop is not None:
            # After partial TP + trailing activated: check trailing stop.
            trailing_hit = (price <= trailing_stop) if is_long else (price >= trailing_stop)
            if trailing_hit:
                return _make_result(
                    exit_price=trailing_stop,
                    exit_bar=bar_idx,
                    exit_reason=_PARTIAL_THEN_TRAILING_SL,
                    partial_tp_hit=True,
                    partial_exit_bar=partial_exit_bar,
                    partial_exit_price=partial_exit_price,
                    partial_frac=partial_frac,
                    entry=entry,
                    r_unit=r_unit,
                    r_size=r_size,
                    is_long=is_long,
                    worst_pnl_r=worst_pnl_r,
                    best_pnl_r=best_pnl_r,
                )
        else:
            # Hard SL check (no trailing yet, or trailing_mode is OFF).
            sl_hit = (price <= sl) if is_long else (price >= sl)
            if sl_hit:
                return _make_result(
                    exit_price=sl,
                    exit_bar=bar_idx,
                    exit_reason=_SL,
                    partial_tp_hit=partial_done,
                    partial_exit_bar=partial_exit_bar,
                    partial_exit_price=partial_exit_price,
                    partial_frac=partial_frac,
                    entry=entry,
                    r_unit=r_unit,
                    r_size=r_size,
                    is_long=is_long,
                    worst_pnl_r=worst_pnl_r,
                    best_pnl_r=best_pnl_r,
                )

        # --- Step B: Partial TP check ---
        # Only checked if partial TP has not yet fired and a target is defined.
        if not partial_done and partial_tp is not None:
            ptp_hit = (price >= partial_tp) if is_long else (price <= partial_tp)
            if ptp_hit:
                partial_done = True
                partial_exit_bar = bar_idx
                partial_exit_price = partial_tp  # exit at stated TP price, not path price

                if use_trailing:
                    # Initialise trailing stop at the original SL level.
                    # The trailing stop can only tighten (move toward entry/profit)
                    # from here. Starting at the hard SL means we allow initial
                    # adverse movement before the trail tightens — consistent with
                    # how the live V2 strategy manages the stop after partial exit.
                    trailing_stop = sl

        # --- Step C: Update trailing stop (runs AFTER exit checks) ---
        # The update applies to the NEXT bar's check, not the current bar.
        # This matches compute_trailing_stop() in entry_exit.py which updates
        # on bar close and the new stop is effective from the next bar onward.
        if partial_done and use_trailing and trailing_stop is not None:
            atr = candidate.atr
            mult = candidate.trailing_atr_multiple
            if is_long:
                # Trail moves UP: new trail = max(old_trail, price - k*ATR)
                trailing_stop = max(trailing_stop, price - atr * mult)
            else:
                # Trail moves DOWN: new trail = min(old_trail, price + k*ATR)
                trailing_stop = min(trailing_stop, price + atr * mult)

    # --- MAX_BARS forced exit ---
    # No SL or TP triggered within max_holding_bars. Exit at the actual path price
    # (no stated target — best we can do is take the market price at time cap).
    # Under zero drift, the average MAX_BARS pnl_r is near 0, which is realistic:
    # a trade that neither wins nor loses simply expires.
    exit_price = float(path[max_bar])
    return _make_result(
        exit_price=exit_price,
        exit_bar=max_bar,
        exit_reason=_MAX_BARS,
        partial_tp_hit=partial_done,
        partial_exit_bar=partial_exit_bar,
        partial_exit_price=partial_exit_price,
        partial_frac=partial_frac,
        entry=entry,
        r_unit=r_unit,
        r_size=r_size,
        is_long=is_long,
        worst_pnl_r=worst_pnl_r,
        best_pnl_r=best_pnl_r,
    )


def _straight_pnl_pct(exit_price: float, entry: float, is_long: bool) -> float:
    """Compute simple PnL as a fraction of entry price.

    LONG:  profit when exit > entry  -> (exit - entry) / entry
    SHORT: profit when exit < entry  -> (entry - exit) / entry
    """
    if is_long:
        return (exit_price - entry) / entry
    return (entry - exit_price) / entry


def _make_result(
    *,
    exit_price: float,
    exit_bar: int,
    exit_reason: str,
    partial_tp_hit: bool,
    partial_exit_bar: Optional[int],
    partial_exit_price: Optional[float],
    partial_frac: Optional[float],
    entry: float,
    r_unit: float,
    r_size: float,
    is_long: bool,
    worst_pnl_r: float,
    best_pnl_r: float,
) -> SimulationResult:
    """Compute blended PnL and package all fields into a SimulationResult."""
    if partial_tp_hit and partial_exit_price is not None and partial_frac is not None:
        # Blended PnL: two-leg exit.
        # Leg 1 (partial): partial_frac x PnL from entry to partial_tp_price
        # Leg 2 (remainder): (1 - partial_frac) x PnL from entry to final exit_price
        remaining_frac = 1.0 - partial_frac
        leg1_pct = _straight_pnl_pct(partial_exit_price, entry, is_long) * partial_frac
        leg2_pct = _straight_pnl_pct(exit_price, entry, is_long) * remaining_frac
        pnl_pct = leg1_pct + leg2_pct
    else:
        pnl_pct = _straight_pnl_pct(exit_price, entry, is_long)

    pnl_r = pnl_pct / r_unit if r_unit > 0 else 0.0

    return SimulationResult(
        exit_price=exit_price,
        exit_bar=exit_bar,
        exit_reason=exit_reason,
        partial_tp_hit=partial_tp_hit,
        partial_exit_bar=partial_exit_bar,
        pnl_pct=pnl_pct,
        pnl_r=pnl_r,
        r_size=r_size,
        mae_r=worst_pnl_r,
        mfe_r=best_pnl_r,
    )

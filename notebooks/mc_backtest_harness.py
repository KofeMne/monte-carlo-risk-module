"""
Local backtest harness for the V2 BOS strategy — produces a trade log from raw candles.

WHY THIS EXISTS
---------------
`code/main.py` is a QuantConnect algorithm: it only runs inside the LEAN engine and
trades ETHUSD. The Monte Carlo optimizer, however, needs a list of *completed trades*
with real R-multiple outcomes (`actual_pnl_r`) to tune the DecisionConfig thresholds.

This harness reuses the SAME strategy modules under `code/` (BOS detection, fixed stop,
RR take-profit, partial-at-1R, ATR trailing, RSI gate, swing detection, ATR) and replays
them bar-by-bar over an OHLC candle file (e.g. data/processed/btc_15m.csv) to emit that
trade log — locally, no QuantConnect.

It mirrors the config in code/main.py.Initialize():
    SL = fixed 1%   |  TP = RR 3.0  |  partial 0.5 at 1R  |  ATR(14) x2 trailing
    BOS buffer = 2.0 x ATR  |  RSI(14) THRESHOLD 55/45  |  same-bar = WORST_CASE

KNOWN APPROXIMATIONS (documented, same spirit as the QC algo)
-------------------------------------------------------------
* Entry is taken at open[t+1] (the plan's entry price), with the BOS confirmed on the
  close of bar t — the same "decide on close, record entry at next open" model the QC
  algo uses. Exit management starts the bar AFTER entry (matches QC's OPEN->next-bar flow).
* No fees / slippage / spread (R-multiples are relative, so ranking is largely preserved).
* Holding is capped at MAX_HOLD bars so the simulated and actual outcomes stay comparable
  (the live strategy has no explicit time cap).
* One position at a time; cooldown=0; no per-day trade cap (matches the QC defaults).

OUTPUT
------
A DataFrame / CSV in the optimizer's schema (see optimizer.load_trades_from_csv). For each
trade we record BOTH:
  - the actual outcome of the strategy (actual_pnl_r, actual_exit_reason), and
  - a self-consistent MC candidate whose entry_price == S0 == close[signal_t] (so the MC
    paths, which start at S0, are not corrupted by an entry!=S0 mismatch). SL/partial-TP are
    placed at the same proportional distances the live trade used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from entry_exit_rules.entry_exit import (  # noqa: E402
    Bar, SwingLevels, PositionDirection, TakeProfitMode, SameBarSlTpRule,
    update_last_swing_levels, detect_bos_signal, plan_trade_from_signal,
    check_exit_rules, check_partial_1r_reached, compute_trailing_stop,
)
from stop_loss.stop_loss import StopLossManager  # noqa: E402
from risk_management.risk import RiskConfig, size_position  # noqa: E402
from atr_module.atr_module import compute_atr  # noqa: E402
from RSI.momentum_confirmation_rsi import (  # noqa: E402
    RSIEngine, RSIMomentumFilter, RSIMomentumMode,
)
from swing_high_low_detection.swing_high_low_detection import swing_highs_lows_online  # noqa: E402

# --- strategy config (mirrors code/main.py Initialize) ---
SL_FIXED_PCT = 0.01
TP_MULT = 3.0
PARTIAL_FRAC = 0.5
PARTIAL_AT_R = 1.0
K_TRAIL = 2.0
ATR_PERIOD = 14
K_BUFFER = 2.0
RSI_PERIOD = 14
RSI_LONG_TH = 55.0
RSI_SHORT_TH = 45.0
N_CANDIDATES = [5, 10, 20]
N_CONFIRMATION = 3
MIN_BARS_BETWEEN_SWINGS = 3
EQUITY = 100_000.0
RISK_PCT = 0.01
MAX_HOLD = 100          # bars; caps holding so actual ~ simulated stay comparable
SNAPSHOT_BARS = 256     # recent closes saved per trade for the MC market_state (bootstrap pool)
WARMUP = 320            # bars to skip at the start (swing lookback + indicator warmup)

# Cache of precomputed indicators (bars/ATR/RSI/swings) keyed on swing+indicator params,
# so a parameter grid that only varies SL/TP/buffer/RSI-thresholds reuses them (big speedup).
_IND_CACHE = {}


def load_bars(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Normalise column names so different vendors load the same:
    # 'Open time'/'Open'/'Volume' -> 'open_time'/'open'/'volume'.
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
    df = df.sort_values("open_time").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    bars = [Bar(open=float(o), high=float(h), low=float(l), close=float(c), time=str(t))
            for o, h, l, c, t in zip(df.open, df.high, df.low, df.close, df.open_time)]
    return df, bars


def _signed_r(price, entry, r_unit, is_long):
    return ((price - entry) if is_long else (entry - price)) / r_unit


def _prepare(csv_path, n_cand, n_conf, min_bars):
    """Load bars + precompute ATR/RSI/swings once (cached) for the given swing/indicator params."""
    key = (str(csv_path), tuple(n_cand), int(n_conf), int(min_bars))
    if key in _IND_CACHE:
        return _IND_CACHE[key]
    df, bars = load_bars(csv_path)
    n = len(bars)
    closes = df["close"].to_numpy()
    atr_arr = compute_atr(df["high"], df["low"], df["close"], ATR_PERIOD).to_numpy()
    eng = RSIEngine(length=RSI_PERIOD)
    rsi_vals = np.array([eng.update(float(c)) if c == c else None for c in closes], dtype=object)
    ohlc = df[["close", "high", "low"]].copy()
    ohlc.index = range(n)
    swings = swing_highs_lows_online(ohlc, N_candidates=n_cand, N_confirmation=n_conf,
                                     min_bars_between_swings=min_bars)
    confirmed = swings.dropna(subset=["HighLow", "Level"]).sort_index()
    confirm_at = {}
    for idx, row in confirmed.iterrows():
        confirm_at.setdefault(int(idx) + n_conf, []).append((float(row["HighLow"]), float(row["Level"])))
    val = (df, bars, n, closes, atr_arr, rsi_vals, confirm_at)
    _IND_CACHE[key] = val
    return val


def run_backtest(csv_path: Path, params=None, max_hold=None, verbose: bool = True):
    """Replay the strategy over a candle file. `params` (dict) overrides any tunable below;
    omitted keys fall back to the code/main.py defaults. Indicators are cached across calls."""
    g = globals()
    p = params or {}
    sl_mode = p.get("sl_mode", "fixed")
    k_sl = p.get("k_sl", 2.0)
    buffer_pct = p.get("buffer_pct", 0.0)
    SL_FIXED_PCT = p.get("fixed_pct", g["SL_FIXED_PCT"])
    TP_MULT = p.get("tp_mult", g["TP_MULT"])
    K_BUFFER = p.get("k_buffer", g["K_BUFFER"])
    K_TRAIL = p.get("k_trail", g["K_TRAIL"])
    PARTIAL_FRAC = p.get("partial_frac", g["PARTIAL_FRAC"])
    PARTIAL_AT_R = p.get("partial_at_r", g["PARTIAL_AT_R"])
    N_CANDIDATES = p.get("n_candidates", g["N_CANDIDATES"])
    N_CONFIRMATION = p.get("n_confirmation", g["N_CONFIRMATION"])
    MIN_BARS_BETWEEN_SWINGS = p.get("min_bars_between_swings", g["MIN_BARS_BETWEEN_SWINGS"])
    rsi_enabled = p.get("rsi_enabled", True)
    rsi_mode = p.get("rsi_mode", "THRESHOLD")
    rsi_long = p.get("rsi_long", g["RSI_LONG_TH"])
    rsi_short = p.get("rsi_short", g["RSI_SHORT_TH"])
    rsi_cross_long = p.get("rsi_cross_long", 50.0)
    rsi_cross_short = p.get("rsi_cross_short", 50.0)
    max_hold = max_hold if max_hold is not None else p.get("max_hold", g["MAX_HOLD"])

    df, bars, n, closes, atr_arr, rsi_vals, confirm_at = _prepare(
        csv_path, N_CANDIDATES, N_CONFIRMATION, MIN_BARS_BETWEEN_SWINGS)

    if sl_mode == "atr":
        slm = StopLossManager(mode="atr", k_sl=k_sl, atr_period=ATR_PERIOD)
    elif sl_mode in ("structural", "bos"):
        slm = StopLossManager(mode=sl_mode, fixed_pct=0.01, buffer_pct=buffer_pct)
    else:
        slm = StopLossManager(mode="fixed", fixed_pct=SL_FIXED_PCT, buffer_pct=0.0)
    rsi_filter = RSIMomentumFilter(mode=RSIMomentumMode(rsi_mode), long_threshold=rsi_long,
                                   short_threshold=rsi_short, cross_level_long=rsi_cross_long,
                                   cross_level_short=rsi_cross_short)
    risk_config = RiskConfig(risk_pct=RISK_PCT)

    swing_levels = SwingLevels()
    trades = []
    j_resume = 0  # next bar index free to evaluate a new signal (after a trade closes)

    for c in range(1, n):
        # apply swings that become confirmed exactly at bar c (no lookahead)
        for hl, lvl in confirm_at.get(c, []):
            swing_levels = update_last_swing_levels(swing_levels, highlow_flag=hl, level=lvl)

        if c < WARMUP or c < j_resume:
            continue

        signal_t = c - 1
        atr_sig = atr_arr[signal_t]
        if not np.isfinite(atr_sig) or atr_sig <= 0:
            continue

        sig = detect_bos_signal(bars=bars, t=signal_t, swing_levels=swing_levels,
                                k_buffer=K_BUFFER, atr=float(atr_sig))
        if sig is None:
            continue

        # RSI gate (optional; CROSS uses the previous bar's RSI too)
        if rsi_enabled:
            rsi_now = rsi_vals[signal_t]
            rsi_prev = rsi_vals[signal_t - 1] if signal_t > 0 else None
            dec = rsi_filter.allow_entry(direction=sig.direction, rsi_now=rsi_now, rsi_prev=rsi_prev)
            if not dec.allowed:
                continue

        slm.reset()
        try:
            plan = plan_trade_from_signal(
                bars=bars, bos_signal=sig, swing_levels=swing_levels, stop_loss_manager=slm,
                tp_mode=TakeProfitMode.RR_BASED, tp_mult=TP_MULT, risk_config=risk_config,
                equity=EQUITY, position_sizer=size_position,
            )
        except Exception:
            continue

        is_long = (plan.direction == PositionDirection.LONG)
        entry = plan.entry_price                 # = open[signal_t + 1] = open[c]
        sl = plan.sl_price
        tp = plan.tp_price
        r_unit = abs(entry - sl)
        if r_unit <= 0:
            continue

        # ---- replay exits bar-by-bar starting the bar AFTER entry ----
        partial_done = False
        trailing = None
        far_tp = entry + 1e12 if is_long else 0.0
        pnl_r = None
        reason = None
        exit_idx = None
        for j in range(c + 1, min(c + max_hold + 1, n)):
            bar = bars[j]
            if not partial_done:
                ev = check_exit_rules(bar=bar, direction=plan.direction, sl_price=sl,
                                      tp_price=tp, same_bar_rule=SameBarSlTpRule.WORST_CASE)
                if ev is not None:
                    pnl_r = _signed_r(ev.exit_price, entry, r_unit, is_long)
                    reason = ev.exit_reason.value  # 'SL' or 'TP'
                    exit_idx = j
                    break
                if check_partial_1r_reached(bar=bar, direction=plan.direction,
                                            entry_price=entry, sl_price=sl, r_mult=PARTIAL_AT_R):
                    partial_done = True
                    trailing = sl  # QC sets trailing_stop = active_sl on partial fill
                continue
            # trailing phase: update stop on close, then check on this bar (matches QC)
            atr_j = atr_arr[j]
            trailing = compute_trailing_stop(close=bar.close, atr=float(atr_j) if np.isfinite(atr_j) else 0.0,
                                             direction=plan.direction, old_stop=trailing, k_trail=K_TRAIL)
            ev = check_exit_rules(bar=bar, direction=plan.direction, sl_price=trailing,
                                  tp_price=far_tp, same_bar_rule=SameBarSlTpRule.WORST_CASE)
            if ev is not None:
                leg1 = PARTIAL_FRAC * PARTIAL_AT_R                       # +1R on the partial half
                leg2 = (1 - PARTIAL_FRAC) * _signed_r(ev.exit_price, entry, r_unit, is_long)
                pnl_r = leg1 + leg2
                reason = "PARTIAL_TP_THEN_TRAILING_SL"
                exit_idx = j
                break

        if pnl_r is None:  # hit time cap
            last = min(c + max_hold, n - 1)
            if partial_done:
                leg1 = PARTIAL_FRAC * PARTIAL_AT_R
                leg2 = (1 - PARTIAL_FRAC) * _signed_r(closes[last], entry, r_unit, is_long)
                pnl_r = leg1 + leg2
                reason = "PARTIAL_THEN_MAX_BARS"
            else:
                pnl_r = _signed_r(closes[last], entry, r_unit, is_long)
                reason = "MAX_BARS"
            exit_idx = last

        # ---- MC candidate (entry anchored to S0 = close[signal_t]) ----
        s0 = float(closes[signal_t])
        snap = closes[max(0, signal_t - SNAPSHOT_BARS + 1): signal_t + 1].astype(float)
        rets = np.diff(np.log(snap)).tolist()
        stop_frac = r_unit / entry  # actual stop distance as a fraction of entry (any SL mode)
        if is_long:
            cand_sl = s0 * (1 - stop_frac)
            cand_ptp = s0 * (1 + stop_frac * PARTIAL_AT_R)
        else:
            cand_sl = s0 * (1 + stop_frac)
            cand_ptp = s0 * (1 - stop_frac * PARTIAL_AT_R)

        trades.append({
            "direction": plan.direction.value,
            "entry_price": s0,
            "stop_loss": cand_sl,
            "partial_tp_price": cand_ptp,
            "partial_close_fraction": PARTIAL_FRAC,
            "trailing_mode": "ATR_BASED",
            "atr": float(atr_sig),
            "trailing_atr_multiple": K_TRAIL,
            "max_holding_bars": max_hold,
            "planned_size": float(plan.quantity),
            "risk_pct": RISK_PCT,
            "regime": "NEUTRAL",
            "actual_pnl_r": float(pnl_r),
            "actual_exit_reason": reason,
            "trade_timestamp": bars[signal_t].time,
            "recent_close_prices": json.dumps([round(x, 2) for x in snap.tolist()]),
            "recent_returns": json.dumps([round(x, 8) for x in rets]),
            "sigma": 0.0,
            "drift": 0.0,
        })
        j_resume = (exit_idx or c) + 1  # go flat, look for next signal after exit

    out = pd.DataFrame(trades)
    if verbose:
        print(f"bars={n:,}  trades={len(out)}")
        if len(out):
            print(f"win_rate={ (out.actual_pnl_r > 0).mean():.3f}  "
                  f"mean_R={out.actual_pnl_r.mean():.3f}  "
                  f"median_R={out.actual_pnl_r.median():.3f}  "
                  f"sum_R={out.actual_pnl_r.sum():.1f}")
            print("exit reasons:", out.actual_exit_reason.value_counts().to_dict())
            print("by direction:", out.direction.value_counts().to_dict())
    return out


if __name__ == "__main__":
    csv = ROOT / "data" / "processed" / "btc_15m.csv"
    out = run_backtest(csv)
    dest = ROOT / "data" / "processed" / "btc_15m_trades.csv"
    out.to_csv(dest, index=False)
    print("wrote", dest)

"""Re-tune the V2 BOS strategy for btc_15m, following research/hyperparameter_tuning_guidance.md
and the gridsearch methodology (rank by ROBUSTNESS across yearly windows, not just total return).

Made fee-aware: ranking uses worst-year NET expectancy at a maker fee, because the BTC/ETH study
showed the gross edge is eaten by costs. Sequential coordinate-descent in the guidance's order:
stop-loss -> take-profit -> breakout buffer -> trailing -> RSI gate. (Swing params and indicator
periods frozen per the guidance; atr-mode SL skipped — StopLossManager recomputes ATR per entry,
which is O(n) per trade.)
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import mc_backtest_harness as h

BTC = H.parent / "data" / "processed" / "btc_15m.csv"
FEE_SIDE = 0.02      # % per side (maker) — headline net assumption
MIN_TOTAL = 300      # ignore configs that barely trade
MIN_PER_WIN = 30     # a window must have this many trades to count


def evaluate(params):
    df = h.run_backtest(BTC, params=params, verbose=False)
    if len(df) < MIN_TOTAL:
        return None
    yr = pd.to_datetime(df.trade_timestamp).dt.year.to_numpy()
    gross = df.actual_pnl_r.to_numpy()
    stop_frac = (np.abs(df.entry_price - df.stop_loss) / df.entry_price).to_numpy()
    cost_r = 2 * FEE_SIDE / (stop_frac * 100)        # round-trip cost in R
    net = gross - cost_r
    rows = []
    for y in np.unique(yr):
        m = yr == y
        if m.sum() < MIN_PER_WIN:
            continue
        rows.append((int(y), int(m.sum()), float(gross[m].mean()), float(net[m].mean())))
    if not rows:
        return None
    wdf = pd.DataFrame(rows, columns=["year", "n", "gross_mean_R", "net_mean_R"])
    return dict(n=len(df), win=float((gross > 0).mean()), gross_mean=float(gross.mean()),
                net_mean=float(net.mean()), net_sum=float(net.sum()),
                worst_net=float(wdf.net_mean_R.min()), median_net=float(wdf.net_mean_R.median()),
                per_year=wdf)


def score(r):  # robustness first (worst window), then median, then total
    if r is None:
        return (-1e9, -1e9, -1e9)
    return (round(r["worst_net"], 4), round(r["median_net"], 4), round(r["net_sum"], 1))


def run_step(title, base, options):
    print(f"\n=== {title} ===")
    best_lbl, best_params, best_r, best_sc = None, None, None, (-1e18,)
    for lbl, override in options:
        cfg = {**base, **override}
        r = evaluate(cfg)
        sc = score(r)
        tag = "" if r is None else (f"n={r['n']:4d} win={r['win']:.3f} gross={r['gross_mean']:+.4f} "
                                    f"net={r['net_mean']:+.4f} worst_yr_net={r['worst_net']:+.4f}")
        print(f"  {lbl:24s} {tag}")
        if sc > best_sc:
            best_sc, best_lbl, best_params, best_r = sc, lbl, override, r
    print(f"  -> picked: {best_lbl}")
    return {**base, **best_params}, best_r


t_start = time.time()
base = {}
b0 = evaluate(base)
print("BASELINE (current main.py params):",
      f"n={b0['n']} gross={b0['gross_mean']:+.4f} net={b0['net_mean']:+.4f} worst_yr_net={b0['worst_net']:+.4f}")

# Step 1 — stop-loss family + distance
base, _ = run_step("Stop-loss (mode + distance)", base, [
    ("fixed 0.50%", {"sl_mode": "fixed", "fixed_pct": 0.005}),
    ("fixed 0.75%", {"sl_mode": "fixed", "fixed_pct": 0.0075}),
    ("fixed 1.00%", {"sl_mode": "fixed", "fixed_pct": 0.01}),
    ("fixed 1.25%", {"sl_mode": "fixed", "fixed_pct": 0.0125}),
    ("fixed 1.50%", {"sl_mode": "fixed", "fixed_pct": 0.015}),
    ("fixed 2.00%", {"sl_mode": "fixed", "fixed_pct": 0.02}),
    ("structural b0.1%", {"sl_mode": "structural", "buffer_pct": 0.001}),
    ("bos b0.1%", {"sl_mode": "bos", "buffer_pct": 0.001}),
])

# Step 2 — take-profit RR
base, _ = run_step("Take-profit (RR multiple)", base, [
    (f"tp_mult={x}", {"tp_mult": x}) for x in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
])

# Step 3 — breakout buffer (false-break filter)
base, _ = run_step("Breakout buffer (xATR)", base, [
    (f"k_buffer={x}", {"k_buffer": x}) for x in (0.0, 1.0, 2.0, 3.0)
])

# Step 4 — trailing tightness
base, _ = run_step("Trailing (xATR)", base, [
    (f"k_trail={x}", {"k_trail": x}) for x in (1.5, 2.0, 2.5, 3.0)
])

# Step 5 — RSI gate
base, _ = run_step("RSI gate", base, [
    ("OFF", {"rsi_enabled": False}),
    ("THRESHOLD 50/50", {"rsi_enabled": True, "rsi_mode": "THRESHOLD", "rsi_long": 50, "rsi_short": 50}),
    ("THRESHOLD 55/45", {"rsi_enabled": True, "rsi_mode": "THRESHOLD", "rsi_long": 55, "rsi_short": 45}),
    ("THRESHOLD 60/40", {"rsi_enabled": True, "rsi_mode": "THRESHOLD", "rsi_long": 60, "rsi_short": 40}),
    ("CROSS 50/50", {"rsi_enabled": True, "rsi_mode": "CROSS", "rsi_cross_long": 50, "rsi_cross_short": 50}),
])

final = evaluate(base)
print(f"\n{'='*60}\nFINAL TUNED BTC PARAMS  (took {time.time()-t_start:.0f}s)\n{'='*60}")
print(json.dumps(base, indent=2))
print(f"\nbaseline -> tuned:  gross {b0['gross_mean']:+.4f} -> {final['gross_mean']:+.4f} | "
      f"net {b0['net_mean']:+.4f} -> {final['net_mean']:+.4f} | "
      f"worst_yr_net {b0['worst_net']:+.4f} -> {final['worst_net']:+.4f}")
print("\nper-year (tuned):")
print(final["per_year"].to_string(index=False))

# resolve to full explicit config (fill defaults) for the saved file
g = h.__dict__
resolved = dict(
    sl_mode=base.get("sl_mode", "fixed"),
    fixed_pct=base.get("fixed_pct", g["SL_FIXED_PCT"]),
    buffer_pct=base.get("buffer_pct", 0.0),
    tp_mode="RR_BASED", tp_mult=base.get("tp_mult", g["TP_MULT"]),
    k_buffer=base.get("k_buffer", g["K_BUFFER"]),
    k_trail=base.get("k_trail", g["K_TRAIL"]),
    partial_close_fraction=g["PARTIAL_FRAC"], partial_at_r=g["PARTIAL_AT_R"],
    rsi_enabled=base.get("rsi_enabled", True), rsi_mode=base.get("rsi_mode", "THRESHOLD"),
    rsi_long=base.get("rsi_long", g["RSI_LONG_TH"]), rsi_short=base.get("rsi_short", g["RSI_SHORT_TH"]),
    rsi_cross_long=base.get("rsi_cross_long", 50.0), rsi_cross_short=base.get("rsi_cross_short", 50.0),
    n_candidates=g["N_CANDIDATES"], n_confirmation=g["N_CONFIRMATION"],
    min_bars_between_swings=g["MIN_BARS_BETWEEN_SWINGS"],
    risk_pct=g["RISK_PCT"], atr_period=g["ATR_PERIOD"], rsi_period=g["RSI_PERIOD"],
)
out = {"search_params": base, "resolved_config": resolved,
       "fee_side_pct": FEE_SIDE,
       "metrics": {k: final[k] for k in ("n", "win", "gross_mean", "net_mean", "net_sum", "worst_net", "median_net")},
       "per_year": final["per_year"].to_dict(orient="records")}
(H.parent / "data" / "processed" / "btc_tuned_params.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote data/processed/btc_tuned_params.json")

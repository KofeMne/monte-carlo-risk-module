# BTC 15m — Re-tuned Strategy Parameters

**Date:** 2026-05-27 · **Data:** `data/processed/btc_15m.csv` (BTC 15m, 2020→2025-03)
**Method:** the project's own grid-search methodology (`research/hyperparameter_tuning_guidance.md` +
`research_bos_breakout_gridsearch.ipynb`) — **ranked by robustness across yearly windows** (worst-year
first), made **fee-aware** (net expectancy at a 0.02%/side maker fee). Sequential coordinate-descent in
the guidance's recommended order: stop-loss → take-profit → breakout buffer → trailing → RSI gate.
Frozen per guidance: `risk_pct=1%`, indicator periods=14, swing params `[5,10,20]/3/3`, same-bar=WORST_CASE.
(`atr`-mode SL skipped — it recomputes ATR per entry, O(n) per trade.)

## Final values (apply these in `code/main.py`)

| Parameter | Old (ETH-tuned) | **New (BTC-tuned)** |
|---|---|---|
| `sl_mode` | `fixed` | `fixed` |
| `fixed_pct` | `0.0100` | **`0.0150`** |
| `tp_mode` | `RR_BASED` | `RR_BASED` |
| `tp_mult` | `3.0` | **`1.5`** |
| `k_buffer` (ATR breakout buffer) | `2.0` | **`3.0`** ← biggest lever |
| `k_trail` (ATR trailing) | `2.0` | **`2.5`** |
| `rsi_enabled` | `True` | **`False`** (gate is inert here) |
| `partial_exit_pct` / `partial_exit_at_r` | `0.5` / `1.0` | unchanged |
| swing `N_candidates`/`N_confirmation` | `[5,10,20]`/`3` | unchanged |
| `risk_pct` | `0.01` | unchanged (policy) |

### Exact edits in `code/main.py` → `Initialize()` (change these 5 lines, leave everything else)

```python
self.fixed_pct   = 0.0150   # was 0.0100   (stop distance)
self.tp_mult     = 1.5      # was 3.0      (take-profit RR)
self.k_buffer    = 3.0      # was 2.0      (ATR breakout buffer — biggest lever)
self.k_trail     = 2.5      # was 2.0      (ATR trailing multiple)
self.rsi_enabled = False    # was True     (RSI gate is inert on BTC)
```

## Result (vs the ETH-default config on BTC)

| | gross mean_R | net mean_R @0.02%/side | worst-year net |
|---|---|---|---|
| Baseline (ETH defaults) | +0.0031 | −0.0369 | −0.0992 |
| **Tuned (BTC)** | **+0.0935** | **+0.0668** | **−0.0100** |

Per-year net (1018 trades total):

| year | n | gross mean_R | net mean_R |
|---|---|---|---|
| 2020 | 190 | +0.026 | −0.001 |
| 2021 | 166 | +0.126 | **+0.099** |
| 2022 | 194 | +0.191 | **+0.164** |
| 2023 | 208 | +0.017 | −0.010 |
| 2024 | 221 | +0.114 | **+0.087** |
| 2025 | 39 | +0.094 | +0.067 |

→ **net-positive overall and in 4 of 6 years; worst year ≈ breakeven.**

## Why it changed

- **`k_buffer=3.0` is the dominant driver.** A stronger ATR breakout buffer cuts trades from ~2,300 to
  ~1,018 — it keeps only high-quality breakouts. This is the single biggest quality lever on BTC.
- **`tp_mult=1.5`** (vs 3.0): BTC breakouts follow through less than ETH's; a quicker target harvests more.
- **`fixed_pct=1.5%`** (vs 1%): a slightly wider stop is more robust across years and lowers cost-per-R.
- **RSI gate adds nothing** on BTC (identical results on/off) → disabled.

## Fee sensitivity

Net stays positive up to ~**0.07%/side** (was ~0.02%/side for the ETH-default config). Viable at maker /
low-fee execution; at full spot-taker (0.10%/side) it is marginal-to-negative.

## Caveats

- Sequential (greedy) search — matches the guidance's order but may miss parameter interactions; a joint
  grid could refine further.
- Net assumes a flat per-trade cost from the stop distance; real fees scale with notional/legs.
- In-sample over 2020–2025; the per-year breakdown is the robustness check, but a strict walk-forward
  (train→unseen) re-tune would be the final confirmation before live use.

Machine-readable values: `data/processed/btc_tuned_params.json`.

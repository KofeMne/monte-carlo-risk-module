# Monte Carlo Risk Module — Audit & Tuning Report

**Date:** 2026-05-27 · **Data:** `data/processed/btc_15m.csv` (BTC 15m, 2020-01 → 2025-03, ~182k bars)
**Scope:** audit the `monte_carlo/` module and tune it. *(The trading strategy itself was out of scope.)*

## TL;DR
- The MC module **works**: audited, **all 7 fixes applied**, **143 tests pass (0 fail)**.
- **Update — BTC strategy re-tune (§7):** the ETH-default config was wrong for BTC. Re-tuning per the
  project's robustness methodology (`k_buffer 2→3`, `RR 3→1.5`, `SL 1%→1.5%`, RSI off) makes BTC
  **net-positive** (+0.094R gross, +0.067R net @maker, 4/6 years positive). On the re-tuned strategy the
  MC filter is positive out-of-sample across **all** folds for the first time — but it doesn't clearly
  beat taking all trades on mean return; the win is the strategy params, not the filter. Keep MC passive.
- `MCConfig` was **calibrated on real BTC 15m** (path method, sigma window, sim count, horizon, regime multipliers).
- A local backtest harness produced **2,544 real BTC trades**; the base strategy is **~breakeven** (mean +0.003R, before fees).
- Walk-forward tuning of `DecisionConfig` **ran and produced thresholds**, but **out-of-sample the filter does not reliably beat doing nothing** — because a breakeven strategy gives the filter nothing to select. This is a *finding*, not a malfunction.
- **Recommendation:** keep `passive_mode=True`; do not deploy the filter until the base strategy has real edge and out-of-sample `val_sharpe` is positive across all folds.

---

## 1. Audit & fixes

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | Bootstrap ignored `regime_sigma_multipliers` (regime widening only worked under GBM) | medium | ✅ **applied** (scales pool deviations by regime_mult) |
| 2 | Bootstrap didn't honor `drift_mode='zero'` | medium | ✅ **applied** (demeans pool on zero; pass `'historical'` to keep the trend) |
| 3 | `main.py` crashed on non-UTF-8 consoles (cp1251) | low/blocking | ✅ **applied** (`sys.stdout.reconfigure`) |
| 4 | README's `from monte_carlo.engine import …` path broke; `models.py` dead (relative imports) | medium (integration) | ✅ **applied** (`__init__.py` package shim; `models.py` now imports) |
| 5 | `decision.py` reject reason hardcoded "5%" instead of the configured VaR level | cosmetic | ✅ **applied** (reworded) |
| 6 | `min_expected_pnl_r` not in optimizer search space | low | ✅ **applied** (now tuned, range −1.0…0.1) |
| 7 | `MC_test_optimizer::test_permissive_config_returns_finite_sharpe` failed (test bug: accept-all config + entry≠S0) | medium | ✅ **applied** (anchored entry to S0; permissive EV gate) |

Test suite: **143 passed, 13 skipped, 0 failed** (all 7 fixes applied; #7 now passes; skips are env-conditional, e.g. missing ETH parquet).
Engine pipeline verified end-to-end on synthetic and real data; `from monte_carlo.engine import …`, `from monte_carlo import …`, and `from monte_carlo.models import …` all import.

**Interpretation caveats found (handled in notebooks, not code):**
- `VaR_r`/`CVaR_r` gates are nearly inert for SL-dominated BTC trades (pinned ≈ −1R) — lean on `prob_loss`/`EV`/`Kelly`.
- `prob_sl_hit` counts *winning* trailing exits as "stops".
- `horizon_bars=500` default wastes memory vs `max_holding_bars` (≈1.7× wall time, not 10×).

## 2. Recommended `MCConfig` for BTC 15m  (`mc_audit_and_calibration.ipynb`)

| Param | Default | Recommended | Reason |
|---|---|---|---|
| `path_method` | `GBM` | **`BOOTSTRAP`** | Captures BTC fat tails/skew **and carries the trend** a breakout strategy needs; zero-drift GBM mis-describes it |
| `bootstrap_lookback` | 252 | **256–500** | Enough recent context for stable tails |
| `num_simulations` | 10000 | **10000** | Seed-to-seed `prob_loss` noise drops from ±0.018 (500) to ±0.0035 (10k) |
| `num_simulations_for_opt` | 500 | **1000** | Less per-trial noise inside the optimizer |
| `rolling_window` | 20 | **20** (10–15 if more reactive) | ~5h context; full-history median σ ≈ 0.23%/bar |
| `horizon_bars` | 500 | **`max_holding_bars + ~10`** | Paths beyond the cap are never replayed |
| `drift_mode` | `zero` | **see note** | Zero-drift is right for mean-reverting risk screening; **wrong for breakouts** (deletes the trend) → use Bootstrap or `historical` |
| `regime_sigma_multipliers` | `{}` | **`{'HIGH_VOL':2.0,'CRASH':3.5}`** | BTC p90/p50≈2.2×, p99/p50≈4.7× — defaults (1.25/1.5) too low *(now active under GBM **and** Bootstrap, fix #1)* |

## 3. Trade generation (`mc_backtest_harness.py`)

Reuses the real `code/` modules (BOS, fixed 1% SL, RR3, partial-0.5 @1R, ATR×2 trail, RSI 55/45, 2×ATR buffer)
with the config hardcoded in `code/main.py`. Output → `data/processed/btc_15m_trades.csv`.

```
2,544 trades | win 47.1% | mean_R +0.0031 | per-trade Sharpe +0.0028 | sum_R +8.0
exits: SL 1242 | PARTIAL_TP_THEN_TRAILING_SL 1190 | MAX_BARS 57 | TP 53
```
The base strategy is **breakeven gross** (i.e. net-negative after fees).

## 4. Validation results (`mc_trade_validation.ipynb`)

**Calibration** — Pearson corr of simulated metric vs real `pnl_r` (want prob_loss<0, EV>0, kelly>0):

| config | prob_loss | EV | kelly |
|---|---|---|---|
| GBM zero-drift | −0.022 | +0.031 | +0.016 |
| GBM historical | −0.012 | −0.002 | +0.011 |
| Bootstrap | **−0.047** | **+0.023** | **+0.030** |

All weak (|r|<0.05). Bootstrap discriminates best; zero-drift GBM is weakest — as expected for a momentum strategy.

**Walk-forward out-of-sample Sharpe** (per fold, chronological):

| config | fold1 | fold2 | fold3 | fold4 | mean(finite) |
|---|---|---|---|---|---|
| GBM zero | −0.025 | 0.022 | −inf | 0.070 | +0.022 |
| GBM hist | −0.052 | 0.019 | 0.023 | 0.046 | +0.009 |
| Bootstrap | −0.072 | −0.001 | 0.058 | 0.064 | +0.012 |

Every config has a **negative first fold**; one zero-drift fold is degenerate. Shipped-optimizer cross-check (§5): **−0.026**.
→ The filter is **not robustly positive out-of-sample** on this trade set.

## 5. Validity

**Valid:** the engine, the candle-based `MCConfig` calibration, the harness (faithful, causal/no-lookahead), the 2,544-trade sample, and the walk-forward methodology.

**Not yet production-valid (threats):**
1. **ETH-tuned config applied to BTC** — `code/main.py` targets ETHUSD; its params (1% SL, RR3, RSI 55/45, 2×ATR buffer, swing N=[5,10,20]) are the ETH "best params". The BTC trade log is a **proxy**.
2. **No fees / slippage / spread** — absolute PnL is optimistic; a breakeven gross strategy is net-losing.
3. **MC-candidate ≠ exact executed trade** — candidate is anchored to S0=`close[signal_t]` with proportional SL/partial and a 100-bar cap; the real trade entered at `open[t+1]` uncapped. This blurs the calibration correlation.

**Bottom line:** methodology ≈ valid; the specific tuned threshold numbers ≈ illustrative, not deployable.

## 6. Recommendations / next steps
1. Get the **base strategy to a positive, stable expectancy** (and on the **target asset**) — the filter can only select, not create edge.
2. Run the MC with **Bootstrap** (or `drift_mode='historical'`) for this breakout strategy — not zero-drift GBM.
3. Regenerate the trade log from the **correct asset + asset-tuned params + fees**, then re-run `mc_trade_validation.ipynb`.
4. Only set `passive_mode=False` once out-of-sample `val_sharpe` is positive across **all** folds.
5. All 7 audit fixes are applied — the engine itself is clean; the remaining work is strategy edge + correct data.

## 7. BTC strategy re-tune (update)

Following `research/hyperparameter_tuning_guidance.md` + the gridsearch methodology (rank by worst-year
robustness), made fee-aware. Sequential search over the high-priority buckets; full write-up in
**`notebooks/BTC_TUNED_PARAMS.md`** (+ `data/processed/btc_tuned_params.json`).

**Tuned BTC params:** `fixed_pct 1%→1.5%`, `tp_mult 3.0→1.5`, `k_buffer 2.0→3.0` (dominant lever),
`k_trail 2.0→2.5`, RSI gate off (inert). → 1,018 trades, gross +0.094R, **net +0.067R @0.02%/side**,
net-positive in 4/6 years (worst ≈ breakeven). Net stays positive up to ~0.07%/side.

**MC filter on the re-tuned strategy** (net walk-forward, OOS): mean `val_sharpe` = +0.109 (0% fee),
+0.086 (0.02%), +0.047 (0.05%) — **positive across all folds for the first time**. But calibration is
still weak (corr ≈ 0.03) and the filtered subset's mean isn't clearly above taking all trades — the edge
is the strategy re-tune, not the filter. Keep MC `passive_mode=True` until a fair filtered-vs-unfiltered
OOS comparison shows added value.

## Artifacts
- `monte_carlo/` — audited; **all 7 fixes applied** (`__init__.py` added)
- `notebooks/mc_audit_and_calibration.ipynb` — MCConfig calibration on BTC
- `notebooks/mc_backtest_harness.py` — parameterized strategy replay → trade log (reusable)
- `notebooks/tune_btc_params.py` — re-tuner (robustness grid) producing the BTC params
- `notebooks/mc_trade_validation.ipynb` — calibration + walk-forward (pre-re-tune ETH-config trade log)
- `notebooks/BTC_TUNED_PARAMS.md` + `data/processed/btc_tuned_params.json` — final BTC params
- `data/processed/btc_15m_trades.csv` — **now the re-tuned BTC trade log** (1,018 trades)
- `data/processed/eth_15m_trades.csv` — ETH trade log (for the asset-comparison)

## Reproduce
```powershell
$py = "C:\Users\vgvoz\anaconda3\python.exe"; $env:PYTHONIOENCODING="utf-8"
& $py notebooks\mc_backtest_harness.py                 # regenerate the trade log
& $py -m jupyter nbconvert --to notebook --execute --inplace notebooks\mc_audit_and_calibration.ipynb
& $py -m jupyter nbconvert --to notebook --execute --inplace notebooks\mc_trade_validation.ipynb
& $py -m pytest tests/MC_*.py -q                        # 132+ pass (test #7 known-red)
```

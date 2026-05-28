# Monte Carlo Risk Module

A pre-trade risk filter for the V2 BOS breakout strategy. Before a trade is placed it simulates
thousands of possible futures for that specific trade and turns the outcome distribution into an
**ACCEPT / REDUCE / REJECT** decision.

This repository is the **Monte Carlo work split out from the main trading repo** — it contains the MC
engine, its tests, the research notebooks, and the tuning reports. **The trading algorithm itself
(`code/`) is intentionally NOT included.**

```
V2 signal → TradeCandidate + MarketState → Monte Carlo engine → ACCEPT / REDUCE / REJECT
```

## Layout

```
monte_carlo/                  the engine (self-contained: numpy + pandas only)
  engine.py                   run_monte_carlo_analysis() — top-level API
  config.py                   MCConfig (simulation settings)
  decision.py                 DecisionConfig thresholds + make_trade_decision()
  stats.py / path_generator.py / trade_replay.py / metrics.py
  optimizer.py                walk-forward + Optuna threshold tuning (needs `optuna`)
  market_state.py / trade_candidate.py / models.py / __init__.py
  README.md                   detailed module docs
tests/                        MC_test_*.py — 143 pass, 0 fail
notebooks/
  mc_audit_and_calibration.ipynb   audit + MCConfig calibration on BTC candles
  mc_trade_validation.ipynb        calibration + walk-forward filter validation (on a trade log)
  mc_backtest_harness.py           ⚠ replays the strategy → trade log (REQUIRES the strategy code, not included)
  tune_btc_params.py               ⚠ strategy re-tuner (REQUIRES the strategy code, not included)
  MC_TUNING_RESULTS.md             ← start here: full audit + tuning report
  BTC_TUNED_PARAMS.md              final BTC strategy parameters
data/processed/
  btc_tuned_params.json        machine-readable tuned params (committed)
  # CSV data (btc_15m, btc_15m_trades, eth_15m_trades) is NOT committed — see note below.
```

> **Data:** the market/trade CSVs are archived in **`../monte_carlo_data.zip`** (kept out of git to keep
> the repo light). To run the notebooks, unzip its contents into `data/processed/`.

## What runs standalone (no strategy code)

- **The engine + tests:** `pytest tests/ -q` → 143 pass.
- **`mc_audit_and_calibration.ipynb`** — needs only `monte_carlo/` + `data/processed/btc_15m.csv`.
- **`mc_trade_validation.ipynb`** — needs only `monte_carlo/` + a trade-log CSV (`btc_15m_trades.csv`).

The trade logs are the clean boundary: the MC module **consumes** them. How they are produced (the BOS
strategy) lives in the main repo. `mc_backtest_harness.py` / `tune_btc_params.py` are included for
reference but import the strategy modules (`code/`), so they only run if you add that code back.

## Quick start

```bash
pip install -r requirements.txt

# engine sanity (no data needed)
cd monte_carlo && python main.py

# tests
pytest tests/ -q

# notebooks
jupyter notebook notebooks/mc_audit_and_calibration.ipynb
```

> Import note (fix #4): `monte_carlo/__init__.py` puts the package dir on `sys.path`, so both
> `from monte_carlo.engine import run_monte_carlo_analysis` and `from monte_carlo import ...` work.

## Status

Audited, all 7 known issues fixed, 143 tests passing. See `notebooks/MC_TUNING_RESULTS.md` for the full
audit + tuning report and the honest verdict on the filter's value.

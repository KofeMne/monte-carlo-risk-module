"""
Walk-forward parameter optimisation for Monte Carlo decision thresholds.

## Why optimise? And why walk-forward?

The DecisionConfig thresholds (reject_prob_loss, reject_var_r, etc.) determine
when the MC filter rejects trades. Setting them too tight filters too many trades
(including good ones); too loose and the filter adds no value.

The right thresholds depend on the characteristics of the V2 strategy's trades:
the typical win rate, R-multiple distribution, and tail behaviour. These can be
estimated from historical trades — but ONLY if we avoid overfitting.

### The overfitting problem

If we optimise thresholds on ALL historical data and then use those thresholds
going forward, we are likely overfitting to noise. The resulting Sharpe ratio
on the historical data will look great, but real performance will disappoint.

### Walk-forward optimisation as the solution

Walk-forward testing is the standard anti-overfitting technique for trading systems:

    Step 1: Split historical trades chronologically into N windows.
    Step 2: For each window k (k=1..N-1):
              - TRAIN: run Optuna optimisation on trades[0:k] (growing window)
              - VALIDATE: evaluate best params on trades[k:k+1] (unseen window)
    Step 3: Report the validation Sharpe (out-of-sample performance).
            Select params from the fold with best out-of-sample Sharpe.

The validation Sharpe is the honest estimate of real performance. If it is
significantly lower than the in-sample optimised Sharpe, the strategy is
overfitting and the thresholds should not be used live.

### Why Optuna (TPE)?

Optuna uses Tree-structured Parzen Estimation — a Bayesian-style algorithm that
learns from previous trials. It focuses sampling on promising regions of the
parameter space. For 5-6 continuous parameters, it typically converges in
100-200 trials vs thousands needed by grid/random search.

TPE handles constraints well (e.g. reduce < reject threshold) via the
suggest_float(...) API with conditional ranges.

### Optimisation objective: Sharpe ratio

Sharpe = mean(filtered_pnl_r) / std(filtered_pnl_r)

Where filtered_pnl_r is the actual R-multiple outcome of each trade
that the MC filter ACCEPTED (REDUCE trades get their pnl_r scaled by
size_factor; REJECT trades are excluded).

Sharpe is preferred over raw PnL because it penalises both low returns
AND high variance. A strategy that makes 2R per trade with 0.5R std is
better than one making 2R per trade with 3R std.

A floor of min_trade_fraction=0.30 prevents the optimiser from gaming the
objective by rejecting 90% of trades. Any config that keeps fewer than 30%
of trades receives -inf Sharpe and is eliminated immediately.

## Data requirements

The optimiser needs historical trades as HistoricalTrade objects. Each object
must contain:
  - The trade candidate (as it was at signal time)
  - The market state (as it was at signal time)
  - The ACTUAL outcome (actual_pnl_r, actual_exit_reason)
  - A timestamp for chronological ordering

You can generate this data by:
  A) Exporting backtest results from QuantConnect and converting them.
  B) Running the V2 strategy in passive_mode=True for 30-50+ trades live,
     then using the logged trade data + actual outcomes.
  C) Using historical backtesting code to generate signals + replay against OHLCV.

Minimum recommended dataset: 50 trades, with at least 15 per fold.
Fewer than 50 trades produces unreliable Sharpe estimates.
"""

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from config import MCConfig
from decision import DecisionConfig, make_trade_decision
from engine import run_monte_carlo_analysis
from market_state import MarketState
from metrics import calculate_risk_metrics
from path_generator import generate_paths
from stats import compute_sigma_and_drift
from trade_candidate import TradeCandidate
from trade_replay import replay_trade

logger = logging.getLogger(__name__)


@dataclass
class HistoricalTrade:
    """A completed historical trade with its MC inputs and actual outcome.

    Used as the dataset for walk-forward optimisation. Each record links
    the information available at trade entry time (what the MC would have seen)
    with the actual outcome (what the MC filter is trying to predict / avoid).

    Attributes:
        candidate:           TradeCandidate as constructed at signal time.
        market_state:        MarketState snapshot at signal time (recent_close_prices,
                             recent_returns, atr, sigma, drift, regime, timestamp).
        actual_pnl_r:        Actual trade outcome in R multiples (positive = profit).
        actual_exit_reason:  Actual exit: 'SL', 'TP', 'TRAILING_SL', 'MAX_BARS', etc.
        trade_timestamp:     ISO timestamp string for chronological ordering.
                             Required for walk-forward splitting.
    """
    candidate: TradeCandidate
    market_state: MarketState
    actual_pnl_r: float
    actual_exit_reason: str
    trade_timestamp: Optional[str] = None


def load_trades_from_csv(path: str) -> list:
    """Load historical trades from a CSV file.

    Expected columns:
        direction, entry_price, stop_loss, partial_tp_price, partial_close_fraction,
        trailing_mode, atr, trailing_atr_multiple, max_holding_bars, planned_size,
        risk_pct, regime, actual_pnl_r, actual_exit_reason, trade_timestamp,
        recent_close_prices (JSON array string, e.g. "[100.1, 100.3, ...]"),
        recent_returns (JSON array string), sigma, drift.

    Returns:
        List of HistoricalTrade objects sorted by trade_timestamp.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If the CSV path does not exist.
    """
    import csv

    required_cols = {
        'direction', 'entry_price', 'stop_loss', 'trailing_mode', 'atr',
        'trailing_atr_multiple', 'max_holding_bars', 'planned_size', 'risk_pct',
        'regime', 'actual_pnl_r', 'actual_exit_reason', 'recent_close_prices',
    }

    trades = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(reader.fieldnames or [])}."
            )

        for i, row in enumerate(reader):
            try:
                recent_closes = json.loads(row['recent_close_prices'])
                recent_returns = json.loads(row.get('recent_returns', '[]'))

                candidate = TradeCandidate(
                    direction=row['direction'],
                    entry_price=float(row['entry_price']),
                    stop_loss=float(row['stop_loss']),
                    partial_tp_price=float(row['partial_tp_price']) if row.get('partial_tp_price') else None,
                    partial_close_fraction=float(row['partial_close_fraction']) if row.get('partial_close_fraction') else None,
                    trailing_mode=row['trailing_mode'],
                    atr=float(row['atr']),
                    trailing_atr_multiple=float(row['trailing_atr_multiple']),
                    max_holding_bars=int(row['max_holding_bars']),
                    planned_size=float(row['planned_size']),
                    risk_pct=float(row['risk_pct']),
                    regime=row['regime'],
                )
                market_state = MarketState(
                    recent_close_prices=recent_closes,
                    recent_returns=recent_returns,
                    atr=float(row['atr']),
                    sigma=float(row.get('sigma', 0.0)),
                    drift=float(row.get('drift', 0.0)),
                    regime=row['regime'],
                    timestamp=row.get('trade_timestamp', ''),
                )
                trade = HistoricalTrade(
                    candidate=candidate,
                    market_state=market_state,
                    actual_pnl_r=float(row['actual_pnl_r']),
                    actual_exit_reason=row['actual_exit_reason'],
                    trade_timestamp=row.get('trade_timestamp'),
                )
                trades.append(trade)
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"Error parsing row {i + 1}: {e}") from e

    # Sort chronologically so walk-forward splitting is correct.
    trades.sort(key=lambda t: t.trade_timestamp or '')
    logger.info("Loaded %d historical trades from %s", len(trades), path)
    return trades


def run_walk_forward_optimization(
    historical_trades: list,
    mc_config: Optional[MCConfig] = None,
    n_trials: int = 200,
    n_folds: int = 5,
    min_trade_fraction: float = 0.30,
    verbose: bool = False,
) -> tuple:
    """Optimise DecisionConfig thresholds using Optuna + walk-forward validation.

    ## Process

    1. Split historical_trades into n_folds chronological windows.
    2. For each fold k (k=1 to n_folds-1):
         - Train on: trades[0 : split_points[k]]  (growing window)
         - Validate on: trades[split_points[k] : split_points[k+1]]
         - Run Optuna (n_trials) on train window, record best params.
         - Evaluate best params on the unseen validation window.
    3. Return the params with the best out-of-sample validation Sharpe.

    Args:
        historical_trades:  List of HistoricalTrade objects, sorted chronologically.
        mc_config:          MCConfig for simulation. Uses num_simulations_for_opt
                            (default 500) for speed during optimisation.
        n_trials:           Number of Optuna trials per fold. 200 is enough for 5-6 params.
        n_folds:            Number of walk-forward folds. Each fold adds one validation window.
        min_trade_fraction: Minimum fraction of trades that must be accepted for a config
                            to receive a finite Sharpe. Prevents degenerate over-filtering.
        verbose:            If True, suppress Optuna's per-trial output.

    Returns:
        Tuple of (best_decision_config: DecisionConfig, report: dict).
        report contains:
          - 'fold_results': list of per-fold dicts (params, train_sharpe, val_sharpe, n_val_trades)
          - 'best_fold_index': which fold produced the best validation Sharpe
          - 'best_val_sharpe': out-of-sample Sharpe of the returned config
          - 'n_total_trades': total trades used

    Raises:
        ImportError: If optuna is not installed.
        ValueError: If fewer than 20 total trades or fewer than 2 folds are possible.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )
    except ImportError as e:
        raise ImportError(
            "optuna is required for walk-forward optimisation. "
            "Install it with: pip install optuna"
        ) from e

    if len(historical_trades) < 20:
        raise ValueError(
            f"Need at least 20 historical trades for optimisation, got {len(historical_trades)}. "
            "Collect more trade history before running the optimiser."
        )
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 (at least one train fold and one validation fold).")

    if mc_config is None:
        mc_config = MCConfig()

    # Use a fast config for optimisation (fewer sims for speed).
    # The full num_simulations is used in production; here we trade accuracy for speed.
    fast_config = deepcopy(mc_config)
    fast_config.num_simulations = mc_config.num_simulations_for_opt

    n = len(historical_trades)
    # Create fold split points: [0, n/N, 2n/N, ..., n]
    split_points = [int(n * k / n_folds) for k in range(n_folds + 1)]
    logger.info(
        "Walk-forward optimisation: %d trades, %d folds, %d trials/fold, "
        "%d sims/trade (optimisation).",
        n, n_folds, n_trials, fast_config.num_simulations,
    )

    fold_results = []

    for fold_k in range(1, n_folds):
        train_trades = historical_trades[: split_points[fold_k]]
        val_trades = historical_trades[split_points[fold_k]: split_points[fold_k + 1]]

        if len(train_trades) < 5 or len(val_trades) < 3:
            logger.warning(
                "Fold %d: too few trades (train=%d, val=%d) — skipping.",
                fold_k, len(train_trades), len(val_trades),
            )
            continue

        logger.info(
            "Fold %d/%d: train=%d trades, val=%d trades.",
            fold_k, n_folds - 1, len(train_trades), len(val_trades),
        )

        # --- Optuna study for this fold ---
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            lambda trial: _objective(trial, train_trades, fast_config, min_trade_fraction),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        train_sharpe = study.best_value
        best_config = _params_to_decision_config(best_params)

        # Evaluate on the unseen validation window.
        val_sharpe = _evaluate_sharpe(val_trades, best_config, fast_config, min_trade_fraction)

        fold_results.append({
            'fold': fold_k,
            'params': best_params,
            'train_sharpe': train_sharpe,
            'val_sharpe': val_sharpe,
            'n_train_trades': len(train_trades),
            'n_val_trades': len(val_trades),
        })
        logger.info(
            "Fold %d: train_sharpe=%.3f, val_sharpe=%.3f — params: %s",
            fold_k, train_sharpe, val_sharpe, best_params,
        )

    if not fold_results:
        raise ValueError("No valid folds completed — dataset may be too small.")

    # Select the fold whose validation Sharpe was highest.
    best_fold = max(fold_results, key=lambda x: x['val_sharpe'])
    best_decision_config = _params_to_decision_config(best_fold['params'])

    report = {
        'fold_results': fold_results,
        'best_fold_index': best_fold['fold'],
        'best_val_sharpe': best_fold['val_sharpe'],
        'best_params': best_fold['params'],
        'n_total_trades': n,
    }
    logger.info(
        "Optimisation complete. Best fold: %d, val_sharpe=%.3f. Config: %s",
        best_fold['fold'], best_fold['val_sharpe'], best_fold['params'],
    )
    return best_decision_config, report


def _objective(
    trial,
    trades: list,
    mc_config: MCConfig,
    min_trade_fraction: float,
) -> float:
    """Optuna objective: Sharpe ratio of filtered trade outcomes on the training set.

    Suggests DecisionConfig parameters and evaluates them against actual trade outcomes.
    Returns -inf if fewer than min_trade_fraction of trades are accepted (degenerate config).
    """
    # TPE learns the conditional structure: reduce thresholds must be inside reject thresholds.
    reject_prob_loss = trial.suggest_float('reject_prob_loss', 0.55, 0.85)
    # reduce_prob_loss must be strictly below reject_prob_loss.
    reduce_prob_loss = trial.suggest_float('reduce_prob_loss', 0.40, max(0.41, reject_prob_loss - 0.05))

    reject_var_r = trial.suggest_float('reject_var_r', -5.0, -1.5)
    # reduce_var_r must be less negative (closer to zero) than reject_var_r.
    reduce_var_r = trial.suggest_float('reduce_var_r', reject_var_r + 0.25, -0.5)

    min_kelly = trial.suggest_float('min_kelly_fraction', 0.0, 0.15)
    reduce_size_factor = trial.suggest_float('reduce_size_factor', 0.25, 0.75)
    # Fix #6: tune the deeply-negative-EV reject gate too (was pinned at the -0.5 default).
    min_expected_pnl_r = trial.suggest_float('min_expected_pnl_r', -1.0, 0.1)

    try:
        decision_config = DecisionConfig(
            reject_prob_loss=reject_prob_loss,
            reduce_prob_loss=reduce_prob_loss,
            reject_var_r=reject_var_r,
            reduce_var_r=reduce_var_r,
            min_kelly_fraction=min_kelly,
            reduce_size_factor=reduce_size_factor,
            min_expected_pnl_r=min_expected_pnl_r,
        )
    except ValueError:
        # Invalid threshold combination (violates DecisionConfig constraints).
        return float('-inf')

    return _evaluate_sharpe(trades, decision_config, mc_config, min_trade_fraction)


def _evaluate_sharpe(
    trades: list,
    decision_config: DecisionConfig,
    mc_config: MCConfig,
    min_trade_fraction: float,
) -> float:
    """Compute Sharpe ratio of actual pnl_r outcomes filtered by MC decision.

    For each trade:
      - Run MC simulation (using fast_config sims).
      - Apply decision_config to get ACCEPT / REDUCE / REJECT.
      - ACCEPT: include actual_pnl_r in the series unchanged.
      - REDUCE: include actual_pnl_r * decision.size_factor.
      - REJECT: exclude from the series.

    Returns -inf if fewer than min_trade_fraction of trades pass the filter.
    """
    filtered_pnl = []

    for trade in trades:
        # Deep copy market_state so the engine's in-place sigma update
        # doesn't corrupt the original data for the next trial.
        ms_copy = deepcopy(trade.market_state)
        try:
            decision = run_monte_carlo_analysis(
                candidate=trade.candidate,
                market_state=ms_copy,
                config=mc_config,
                decision_config=decision_config,
                passive_mode=False,  # we want real decisions, not passive mode
            )
        except Exception:
            # If a single trade fails (e.g. insufficient history), skip it.
            continue

        if decision.action == 'ACCEPT':
            filtered_pnl.append(trade.actual_pnl_r)
        elif decision.action == 'REDUCE':
            filtered_pnl.append(trade.actual_pnl_r * decision.size_factor)
        # REJECT: excluded

    n_total = len(trades)
    n_kept = len(filtered_pnl)

    # Guard: if the config filters too aggressively, discard it.
    if n_kept < max(1, int(min_trade_fraction * n_total)):
        return float('-inf')

    if n_kept == 0:
        return float('-inf')

    import numpy as np
    arr = np.array(filtered_pnl)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    if std > 0:
        return mean / std
    elif mean > 0:
        # Perfect consistency: all wins, no variance. High but finite reward.
        return mean * 10.0
    else:
        return float('-inf')


def _params_to_decision_config(params: dict) -> DecisionConfig:
    """Convert an Optuna params dict to a DecisionConfig object."""
    return DecisionConfig(
        reject_prob_loss=params.get('reject_prob_loss', 0.65),
        reduce_prob_loss=params.get('reduce_prob_loss', 0.55),
        reject_var_r=params.get('reject_var_r', -3.0),
        reduce_var_r=params.get('reduce_var_r', -2.0),
        min_kelly_fraction=params.get('min_kelly_fraction', 0.05),
        reduce_size_factor=params.get('reduce_size_factor', 0.5),
        min_expected_pnl_r=params.get('min_expected_pnl_r', -0.5),
    )

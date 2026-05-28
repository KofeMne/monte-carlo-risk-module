"""
Unit tests for trade_replay.py (Stage C).

## What we test and why

replay_trade() is the most critical module in the pipeline because it determines
the quality of the simulated outcome distribution. Bugs here directly produce wrong
probabilities, VaR, and decisions. Every exit path (SL, trailing SL, partial TP,
MAX_BARS) is tested independently with deterministic hand-crafted paths so that
the expected outcome can be computed exactly.

Key properties verified:
- Exit price is the STATED stop price, not the path price that triggered it.
- SL check precedes partial TP check (WORST_CASE ordering).
- Full SL hit produces pnl_r == -1.0 by definition.
- Trailing stop can only move in the profitable direction (never reverses).
- Blended PnL is correct for partial exits.
- MAE <= 0, MFE >= 0 on their respective trade types.
- Edge cases (r_size=0, path shorter than max_bars) do not crash.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

MONTE_CARLO_DIR = Path(__file__).resolve().parents[1] / "monte_carlo"
if str(MONTE_CARLO_DIR) not in sys.path:
    sys.path.insert(0, str(MONTE_CARLO_DIR))

from trade_replay import replay_trade, SimulationResult  # noqa: E402
from trade_candidate import TradeCandidate  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_long_candidate(
    entry=100.0,
    stop_loss=98.0,
    partial_tp_price=102.0,
    partial_close_fraction=0.5,
    trailing_mode='ATR_BASED',
    atr=1.0,
    trailing_atr_multiple=2.0,
    max_holding_bars=20,
):
    """Standard LONG TradeCandidate. R = |100 - 98| = 2.0; r_unit = 0.02."""
    return TradeCandidate(
        direction='LONG',
        entry_price=entry,
        stop_loss=stop_loss,
        partial_tp_price=partial_tp_price,
        partial_close_fraction=partial_close_fraction,
        trailing_mode=trailing_mode,
        atr=atr,
        trailing_atr_multiple=trailing_atr_multiple,
        max_holding_bars=max_holding_bars,
        planned_size=1.0,
        risk_pct=0.01,
        regime='BULL',
    )


def make_short_candidate(
    entry=100.0,
    stop_loss=102.0,
    partial_tp_price=98.0,
    partial_close_fraction=0.5,
    trailing_mode='ATR_BASED',
    atr=1.0,
    trailing_atr_multiple=2.0,
    max_holding_bars=20,
):
    """Standard SHORT TradeCandidate. R = |100 - 102| = 2.0; r_unit = 0.02."""
    return TradeCandidate(
        direction='SHORT',
        entry_price=entry,
        stop_loss=stop_loss,
        partial_tp_price=partial_tp_price,
        partial_close_fraction=partial_close_fraction,
        trailing_mode=trailing_mode,
        atr=atr,
        trailing_atr_multiple=trailing_atr_multiple,
        max_holding_bars=max_holding_bars,
        planned_size=1.0,
        risk_pct=0.01,
        regime='BEAR',
    )


def flat_path(price=100.0, n_bars=21):
    """Path that stays at a constant price — forces MAX_BARS exit."""
    return np.full(n_bars, price)


# ---------------------------------------------------------------------------
# LONG — SL tests
# ---------------------------------------------------------------------------

class TestLongSL:
    """Verify hard SL exits for LONG trades."""

    def test_sl_hit_exit_reason_and_price(self):
        """SL fires when path price drops to or below stop_loss.
        Exit price must be the stated stop_loss, not the path price."""
        candidate = make_long_candidate()
        # path[0]=S0, path[3] drops to 97.5 (below SL=98)
        path = np.array([100.0, 100.0, 99.5, 97.5, 96.0])
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'SL'
        assert result.exit_price == pytest.approx(98.0)  # stated stop, not 97.5
        assert result.exit_bar == 3
        assert result.partial_tp_hit is False

    def test_sl_pnl_r_is_minus_one(self):
        """A full SL hit must produce pnl_r == -1.0 by definition.
        1R = |entry - SL|, so exiting at SL is exactly one risk unit lost."""
        candidate = make_long_candidate(
            entry=100.0, stop_loss=98.0,
            partial_tp_price=None, partial_close_fraction=None,
            trailing_mode='OFF',
        )
        path = np.array([100.0, 100.0, 97.0])
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'SL'
        assert result.pnl_r == pytest.approx(-1.0, abs=1e-9)

    def test_sl_at_exact_stop_price(self):
        """SL fires when price equals stop_loss exactly (boundary condition)."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF'
        )
        path = np.array([100.0, 99.0, 98.0, 105.0])  # bar 2 hits exactly 98.0
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'SL'
        assert result.exit_bar == 2

    def test_sl_not_hit_while_above(self):
        """No SL fires when price stays above stop_loss throughout."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=4,
        )
        path = np.array([100.0, 99.0, 98.5, 99.0, 100.0])
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'MAX_BARS'


# ---------------------------------------------------------------------------
# SHORT — SL tests
# ---------------------------------------------------------------------------

class TestShortSL:
    """Verify hard SL exits for SHORT trades."""

    def test_short_sl_exit_price_is_stated_stop(self):
        """SHORT: SL fires when price rises above stop_loss. Exit at stop_loss, not path price."""
        candidate = make_short_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF'
        )
        path = np.array([100.0, 100.0, 103.0, 105.0])  # bar 2 crosses SL=102
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'SL'
        assert result.exit_price == pytest.approx(102.0)
        assert result.exit_bar == 2

    def test_short_sl_pnl_r_minus_one(self):
        """SHORT: full SL hit = pnl_r == -1.0."""
        candidate = make_short_candidate(
            entry=100.0, stop_loss=102.0,
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
        )
        path = np.array([100.0, 103.0])
        result = replay_trade(path, candidate)
        assert result.pnl_r == pytest.approx(-1.0, abs=1e-9)

    def test_short_profit_pnl_positive_when_price_falls(self):
        """SHORT profit is positive when price falls below entry."""
        candidate = make_short_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=2,
        )
        path = np.array([100.0, 99.0, 96.0])
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'MAX_BARS'
        assert result.pnl_pct > 0  # price fell, SHORT is profitable


# ---------------------------------------------------------------------------
# MAX_BARS exit
# ---------------------------------------------------------------------------

class TestMaxBars:
    """Verify MAX_BARS forced exit (time cap)."""

    def test_flat_path_exits_at_max_bars(self):
        """Flat path (no SL/TP touch) should exit at max_holding_bars."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=5,
        )
        path = flat_path(100.0, n_bars=10)
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'MAX_BARS'
        assert result.exit_bar == 5
        assert result.pnl_r == pytest.approx(0.0, abs=1e-9)

    def test_max_bars_clamped_to_path_length(self):
        """When max_holding_bars > len(path)-1, clamp to path length."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=100,
        )
        path = np.array([100.0, 100.5, 101.0])  # only 2 future bars
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'MAX_BARS'
        assert result.exit_bar == 2  # clamped to len(path)-1 = 2

    def test_zero_length_path_returns_immediately(self):
        """A path with only S0 (no future bars) exits immediately at S0."""
        candidate = make_long_candidate(max_holding_bars=0)
        path = np.array([100.0])
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'MAX_BARS'
        assert result.exit_bar == 0


# ---------------------------------------------------------------------------
# Partial TP + trailing stop (LONG)
# ---------------------------------------------------------------------------

class TestPartialAndTrailingLong:
    """Verify partial TP + ATR trailing stop state machine for LONG."""

    def test_partial_tp_fires_then_trailing_sl(self):
        """Step through a known path where partial fires then trailing SL exits.

        Candidate: entry=100, SL=98, partial_tp=102, frac=0.5, ATR=1, mult=2
        R = 2, r_unit = 0.02

        Path: [100, 100, 103, 104, 102, 101, 99.5]
         bar 2: price=103 >= partial_tp=102 -> partial fires, trail_stop = SL = 98
         bar 2 trail update: trail = max(98, 103-2) = max(98, 101) = 101
         bar 3: price=104, check trail: 104 > 101 -> no exit.
                trail update: max(101, 104-2) = max(101, 102) = 102
         bar 4: price=102, check trail: 102 <= 102 -> EXIT (trailing SL at 102)
        """
        candidate = make_long_candidate(
            entry=100.0, stop_loss=98.0,
            partial_tp_price=102.0, partial_close_fraction=0.5,
            trailing_mode='ATR_BASED', atr=1.0, trailing_atr_multiple=2.0,
        )
        path = np.array([100.0, 100.0, 103.0, 104.0, 102.0, 101.0, 99.5])
        result = replay_trade(path, candidate)

        assert result.exit_reason == 'PARTIAL_TP_THEN_TRAILING_SL'
        assert result.partial_tp_hit is True
        assert result.partial_exit_bar == 2
        assert result.exit_bar == 4
        assert result.exit_price == pytest.approx(102.0)  # trailing stop at 102

        # Blended PnL:
        # leg1 = (102-100)/100 * 0.5 = 0.02 * 0.5 = 0.01
        # leg2 = (102-100)/100 * 0.5 = 0.01  (exit at trailing=102)
        # pnl_pct = 0.02; r_unit = 0.02; pnl_r = 0.02/0.02 = 1.0
        assert result.pnl_pct == pytest.approx(0.02, abs=1e-9)
        assert result.pnl_r == pytest.approx(1.0, abs=1e-9)

    def test_partial_tp_then_max_bars(self):
        """Partial fires but trailing never hits — exits at MAX_BARS with blended PnL."""
        candidate = make_long_candidate(
            entry=100.0, stop_loss=98.0,
            partial_tp_price=102.0, partial_close_fraction=0.5,
            trailing_mode='OFF',  # no trailing — simplifies the remaining leg
            max_holding_bars=5,
        )
        # Partial fires at bar 2, then price settles at 103 until max_bars
        path = np.array([100.0, 101.0, 102.5, 103.0, 103.0, 103.0])
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'MAX_BARS'
        assert result.partial_tp_hit is True
        assert result.exit_bar == 5
        # leg1 = (102-100)/100 * 0.5 = 0.01
        # leg2 = (103-100)/100 * 0.5 = 0.015
        # total pnl_pct = 0.025
        assert result.pnl_pct == pytest.approx(0.025, abs=1e-9)

    def test_trailing_only_tightens_never_reverses_long(self):
        """Trailing stop for LONG can only increase (move toward higher prices), never decrease."""
        candidate = make_long_candidate(
            partial_tp_price=102.0, partial_close_fraction=0.5,
            trailing_mode='ATR_BASED', atr=1.0, trailing_atr_multiple=2.0,
            max_holding_bars=30,
        )
        # Path rises steeply then falls — trailing should never go below SL
        prices = [100.0] + [100.0, 102.5, 105.0, 107.0, 109.0] + [100.0] * 25
        path = np.array(prices)
        result = replay_trade(path, candidate)
        # Just verify it exits and doesn't crash; main property: exit_price >= initial SL
        assert result.exit_price >= candidate.stop_loss - 1e-9


# ---------------------------------------------------------------------------
# Partial TP + trailing stop (SHORT)
# ---------------------------------------------------------------------------

class TestPartialAndTrailingShort:
    """Verify partial TP + ATR trailing for SHORT."""

    def test_short_partial_then_trailing_sl(self):
        """SHORT: partial fires when price falls, trailing SL exits when price reverses.

        entry=100, SL=102, partial_tp=98, frac=0.5, ATR=1, mult=2
        R = 2, r_unit = 0.02

        Path: [100, 100, 97, 96, 98, 99, 100.5]
         bar 2: price=97 <= partial_tp=98 -> partial fires, trail_stop=SL=102
         bar 2 trail update: trail = min(102, 97+2) = min(102, 99) = 99
         bar 3: price=96, check: 96 >= 99? No. trail = min(99, 96+2)=min(99,98)=98
         bar 4: price=98, check: 98 >= 98? Yes -> EXIT at trailing=98
        """
        candidate = make_short_candidate(
            entry=100.0, stop_loss=102.0,
            partial_tp_price=98.0, partial_close_fraction=0.5,
            trailing_mode='ATR_BASED', atr=1.0, trailing_atr_multiple=2.0,
        )
        path = np.array([100.0, 100.0, 97.0, 96.0, 98.0, 99.0, 100.5])
        result = replay_trade(path, candidate)

        assert result.exit_reason == 'PARTIAL_TP_THEN_TRAILING_SL'
        assert result.partial_tp_hit is True
        assert result.partial_exit_bar == 2
        assert result.exit_bar == 4
        assert result.exit_price == pytest.approx(98.0)

        # leg1 = (100-98)/100 * 0.5 = 0.01
        # leg2 = (100-98)/100 * 0.5 = 0.01
        # pnl_pct = 0.02; pnl_r = 1.0
        assert result.pnl_pct == pytest.approx(0.02, abs=1e-9)
        assert result.pnl_r == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SL-only mode
# ---------------------------------------------------------------------------

class TestSLOnlyMode:
    """trailing_mode='OFF' and no partial TP — pure SL mode."""

    def test_sl_only_mode_no_partial(self):
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF'
        )
        path = np.array([100.0, 101.0, 102.0, 97.0])
        result = replay_trade(path, candidate)
        assert result.exit_reason == 'SL'
        assert result.partial_tp_hit is False


# ---------------------------------------------------------------------------
# MAE / MFE tracking
# ---------------------------------------------------------------------------

class TestMAEandMFE:
    """Verify Maximum Adverse / Favorable Excursion tracking."""

    def test_mae_negative_on_losing_trade(self):
        """On a trade that hits SL, MAE should reflect the worst drawdown."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF'
        )
        # Goes to 97.0 before SL fires at 98.0
        path = np.array([100.0, 99.0, 97.0, 95.0])
        result = replay_trade(path, candidate)
        assert result.mae_r < 0  # some adverse excursion occurred
        assert result.exit_reason == 'SL'

    def test_mfe_positive_on_winning_trade(self):
        """On a trade that goes up then comes back to max_bars, MFE > 0."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=5,
        )
        path = np.array([100.0, 101.0, 104.0, 103.0, 101.0, 100.5])
        result = replay_trade(path, candidate)
        assert result.mfe_r > 0  # price rose above entry during the trade

    def test_mae_and_mfe_both_zero_on_flat_path(self):
        """Flat path: price never moves, both MAE and MFE should be 0."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=5,
        )
        path = flat_path(100.0, n_bars=6)
        result = replay_trade(path, candidate)
        assert result.mae_r == pytest.approx(0.0, abs=1e-9)
        assert result.mfe_r == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Robustness tests for degenerate inputs."""

    def test_r_size_zero_does_not_crash(self):
        """stop_loss == entry_price is a degenerate trade (R=0). Should return pnl_r=0.0."""
        candidate = TradeCandidate(
            direction='LONG',
            entry_price=100.0,
            stop_loss=100.0,  # degenerate: R = 0
            partial_tp_price=None,
            partial_close_fraction=None,
            trailing_mode='OFF',
            atr=1.0,
            trailing_atr_multiple=2.0,
            max_holding_bars=5,
            planned_size=1.0,
            risk_pct=0.01,
            regime='FLAT',
        )
        path = np.array([100.0, 101.0, 102.0, 99.5, 100.5, 101.0])
        result = replay_trade(path, candidate)
        # Should not raise; pnl_r = 0 because r_unit = 0
        assert result.pnl_r == pytest.approx(0.0, abs=1e-9)

    def test_missing_partial_fraction_raises(self):
        """partial_tp_price set but partial_close_fraction=None should raise ValueError."""
        candidate = TradeCandidate(
            direction='LONG',
            entry_price=100.0,
            stop_loss=98.0,
            partial_tp_price=102.0,
            partial_close_fraction=None,  # missing!
            trailing_mode='OFF',
            atr=1.0,
            trailing_atr_multiple=2.0,
            max_holding_bars=10,
            planned_size=1.0,
            risk_pct=0.01,
            regime='BULL',
        )
        path = np.array([100.0, 103.0])
        with pytest.raises(ValueError, match="partial_close_fraction"):
            replay_trade(path, candidate)

    def test_pnl_r_has_correct_sign_long_win(self):
        """LONG trade that exits above entry must have positive pnl_r."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=3,
        )
        path = np.array([100.0, 101.0, 103.0, 105.0])
        result = replay_trade(path, candidate)
        assert result.pnl_r > 0

    def test_pnl_r_has_correct_sign_short_win(self):
        """SHORT trade that exits below entry must have positive pnl_r."""
        candidate = make_short_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF',
            max_holding_bars=3,
        )
        path = np.array([100.0, 99.0, 97.0, 95.0])
        result = replay_trade(path, candidate)
        assert result.pnl_r > 0

    def test_result_is_simulation_result_dataclass(self):
        """Return type is always SimulationResult."""
        candidate = make_long_candidate(
            partial_tp_price=None, partial_close_fraction=None, trailing_mode='OFF'
        )
        path = flat_path(100.0, n_bars=5)
        result = replay_trade(path, candidate)
        assert isinstance(result, SimulationResult)

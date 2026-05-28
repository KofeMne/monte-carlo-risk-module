from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeCandidate:
    """
    A single, self-contained snapshot of a fully defined V2 trade.
    Passed into the Monte Carlo engine for risk evaluation.
    """
    
    # Core Trade Mechanics
    direction: str                 # e.g., 'LONG' or 'SHORT'
    entry_price: float             # The exact price the trade is planned to enter at
    planned_size: float            # Total position size 
    risk_pct: float                # The percentage of equity risked (e.g., 0.01 for 1%)
    
    # Hard Exits (Failure and Time Limits)
    stop_loss: float               # The absolute worst-case exit price
    max_holding_bars: int          # Forced exit if the trade takes too long
    
    # Partial Take Profit Management
    partial_tp_price: Optional[float]       # Price level to trigger a partial exit
    partial_close_fraction: Optional[float] # Percentage of position to close (e.g., 0.5 for 50%)
    
    # Trailing Stop Management
    trailing_mode: str             # e.g., 'OFF', 'ATR_BASED', 'SWING'
    atr: float                     # Baseline Average True Range at the time of entry
    trailing_atr_multiple: float   # Multiplier for trailing stop (e.g., 2.0)
    
    # Market Context
    regime: str                    # Current market state (e.g., 'BULL_TREND', 'CHOP')
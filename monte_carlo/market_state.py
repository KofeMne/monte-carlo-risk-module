from dataclasses import dataclass
from typing import List

@dataclass
class MarketState:
    """
    A snapshot of current market conditions used to calibrate 
    the Monte Carlo path generator.
    """
    
    # Raw Data for calibration
    recent_close_prices: List[float]  # The last N bars (needed for lookbacks)
    recent_returns: List[float]       # Percentage changes of recent bars
    
    # Statistical parameters (The 'Engine' settings)
    atr: float                        # Current Average True Range
    sigma: float                      # Current volatility (standard deviation of returns)
    drift: float                      # Average price direction (usually 0 for Monte Carlo)
    
    # Qualitative Data
    regime: str                       # e.g., 'TRENDING_UP', 'RANGE', 'CRASH'
    
    # Metadata
    timestamp: str                    # When this snapshot was taken
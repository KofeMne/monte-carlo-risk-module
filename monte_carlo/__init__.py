"""
Monte Carlo risk module — package entry point.

## Import mechanism (fix #4)

The internal modules use *absolute* imports (e.g. ``from config import MCConfig``) so that
the standalone runner (``python main.py``) and the test suite (which put this directory on
``sys.path``) work. For that same reason, importing the package as ``monte_carlo`` used to
fail (``from config import ...`` couldn't be resolved, and ``models.py``'s relative imports
had no package context).

This ``__init__`` puts the package directory on ``sys.path`` on import, so BOTH styles work:

    from monte_carlo import run_monte_carlo_analysis, TradeCandidate, MarketState   # convenience
    from monte_carlo.engine import run_monte_carlo_analysis                         # explicit

Note: because the submodules are also importable as top-level modules (``config``, ``engine``,
…), avoid mixing ``isinstance`` checks across ``monte_carlo.config.MCConfig`` and the top-level
``config.MCConfig``. The engine is duck-typed, so normal usage is unaffected.
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(__file__)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from engine import run_monte_carlo_analysis  # noqa: E402,F401
from config import MCConfig, default_config  # noqa: E402,F401
from decision import DecisionConfig, TradeDecision, make_trade_decision  # noqa: E402,F401
from market_state import MarketState  # noqa: E402,F401
from trade_candidate import TradeCandidate  # noqa: E402,F401

__all__ = [
    "run_monte_carlo_analysis",
    "MCConfig",
    "default_config",
    "DecisionConfig",
    "TradeDecision",
    "make_trade_decision",
    "MarketState",
    "TradeCandidate",
]

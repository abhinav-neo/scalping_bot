"""
Shared model config adapter.

This lives in its own module on purpose. It used to be defined inside train.py,
which meant that when training ran as `python -m app.train`, any pickled object
holding a reference to it recorded the module as `__main__`. Loading those models
from run_bot.py then failed with:

    AttributeError: Can't get attribute '_Cfg' on <module 'app.run_bot'>

Anything that may end up inside a pickled artifact must live at a stable, importable
module path -- never in a module that gets executed as __main__.
"""


class ModelCfg:
    """Adapter so the shared feature/label/regime modules see the fields they expect."""

    def __init__(self, s):
        self.market_symbol = s.market_symbol
        self.context_symbols = []
        self.vol_span = s.vol_span
        self.fracdiff_d = s.fracdiff_d
        self.fracdiff_thresh = 1e-4
        self.horizon_bars = s.horizon_bars
        self.pt_mult = s.pt_mult
        self.sl_mult = s.sl_mult
        self.hmm_states = s.hmm_states
        self.hmm_covariance = "diag"
        self.seed = 7


# Backwards-compatible alias for any artifact trained before the move.
_Cfg = ModelCfg

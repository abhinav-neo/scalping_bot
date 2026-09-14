"""
HMM regime layer.

We do NOT ask the HMM to predict direction. We ask it 'what regime are we in?'
and hand the posterior state probabilities to the ML model as features. That is
the clean fusion of Markov structure + ML: the boosting model gets the regime
context for free and learns nonlinear interactions the HMM can't express.

Fit is done ONLY on training data inside each walk-forward fold (see model.py) to
avoid look-ahead. This module exposes fit / transform separately for that reason.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


REGIME_COLS = ["ret_1", "vol", "vwap_dist", "mom_z"]


class RegimeModel:
    def __init__(self, cfg):
        # Store only plain scalars -- never a reference to the cfg object itself.
        # This class gets pickled, and holding a cfg instance drags that class's
        # module path into the pickle (which broke loading across entrypoints).
        self.n_states = cfg.hmm_states
        self.hmm = GaussianHMM(n_components=cfg.hmm_states,
                               covariance_type=cfg.hmm_covariance,
                               n_iter=200, random_state=cfg.seed)
        self.cols = [c for c in REGIME_COLS]
        self.mu = None
        self.sd = None

    def _prep(self, feats: pd.DataFrame):
        # Only use columns that exist. The daily/swing feature set has no
        # vwap_dist (an intraday concept), and silently raising here caused every
        # walk-forward fold to be skipped.
        if self.cols is None or not set(self.cols).issubset(feats.columns):
            self.cols = [c for c in REGIME_COLS if c in feats.columns]
            if not self.cols:
                self.cols = [c for c in ("ret_1", "vol") if c in feats.columns]
        X = feats[self.cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        return X

    def fit(self, feats: pd.DataFrame):
        X = self._prep(feats)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.hmm.fit((X - self.mu) / self.sd)
        return self

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = self._prep(feats)
        Xs = (X - self.mu) / self.sd
        post = self.hmm.predict_proba(Xs)          # posterior P(state | obs)
        state = self.hmm.predict(Xs)
        out = pd.DataFrame(post, index=feats.index,
                           columns=[f"regime_p{i}" for i in range(self.n_states)])
        out["regime"] = state
        return out

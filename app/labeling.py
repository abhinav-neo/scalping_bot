"""
Triple-barrier labeling (Lopez de Prado).

For each bar we set a profit-take barrier and a stop barrier, both scaled by the
local volatility, plus a vertical (time) barrier. The label is the sign of the
first barrier touched. This encodes the scalp's own exit logic into the target,
which is far more useful than a naive 'sign of next bar'.

Also produces the raw material for meta-labeling: given a primary side, did the
trade actually win?
"""
import numpy as np
import pandas as pd


def triple_barrier_labels(close: pd.Series, vol: pd.Series, cfg):
    """
    Returns a DataFrame with:
        ret   : realized log-return from entry to the touched barrier
        label : +1 / -1 / 0 (0 = vertical barrier hit with negligible move)
        t_exit: index location of the barrier touch (for overlap/embargo handling)
    """
    n = len(close)
    px = close.values.astype(float)
    v = vol.fillna(vol.median()).values
    H = cfg.horizon_bars
    out_ret = np.zeros(n)
    out_lab = np.zeros(n)
    out_texit = np.arange(n)

    for i in range(n - 1):
        entry = px[i]
        up = entry * (1 + cfg.pt_mult * v[i])
        dn = entry * (1 - cfg.sl_mult * v[i])
        end = min(i + H, n - 1)
        hit = end
        lab = 0
        for j in range(i + 1, end + 1):
            if px[j] >= up:
                hit, lab = j, 1
                break
            if px[j] <= dn:
                hit, lab = j, -1
                break
        r = np.log(px[hit] / entry)
        if lab == 0:  # vertical barrier: label by realized move vs a small deadband
            lab = int(np.sign(r)) if abs(r) > 0.2 * v[i] else 0
        out_ret[i], out_lab[i], out_texit[i] = r, lab, hit

    return pd.DataFrame({"ret": out_ret, "label": out_lab, "t_exit": out_texit},
                        index=close.index)


def make_meta_labels(primary_side: np.ndarray, tb_label: np.ndarray) -> np.ndarray:
    """
    Meta-label = 1 if acting on the primary side would have won, else 0.
    primary_side in {-1,+1}; tb_label in {-1,0,+1}.
    """
    win = (np.sign(primary_side) == np.sign(tb_label)) & (tb_label != 0)
    return win.astype(int)

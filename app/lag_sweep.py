"""
Lag sensitivity sweep.

backtest_v2 applies a full 1-bar (5 min) signal delay. Reality is somewhere
between 0 and 5 minutes: the bot polls every 20s, so it acts as soon as the bar
lands, but the bar itself is published 4-5 minutes after its close.

This brackets where the truth sits, and decomposes the two v1->v2 corrections
so we can see which one actually destroyed the edge:

    lag_min   0 / 1.25 / 2.5 / 5.0   (entry price linearly interpolated)
    intrabar  False (v1: barriers checked at closes only)
              True  (v2: barriers checked against intrabar high/low)

If edge is positive only at ~0 lag, the strategy needs real-time (SIP) data.
If edge is negative across the whole range, 5-minute horizons do not work here.

    python -m app.lag_sweep
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_ab import walk_forward, TREND_BARS
from .backtest_v2 import stats, COST_PER_SIDE

log = logging.getLogger("lag_sweep")

LAGS_MIN = [0.0, 1.25, 2.5, 5.0]
CONFIGS = [
    ("h3 asym",      3, 0.7, 1.0, 0.58),
    ("h6 symmetric", 6, 1.0, 1.0, 0.55),
    ("h12 wide",    12, 1.5, 1.0, 0.55),
]


def simulate_lag(sig, capital=5000.0, risk=0.005, pt=1.0, sl=1.0,
                 horizon=6, thresh=0.55, cost=COST_PER_SIDE,
                 lag_min=0.0, bar_min=5, intrabar=True):
    """
    Entry price is interpolated between the signal bar's close and the next bar's
    close, in proportion to the lag. lag_min=0 reproduces v1 entry timing;
    lag_min=bar_min reproduces v2's full one-bar delay.
    """
    idx = sig.index
    close = sig["close"].values
    high = sig["high"].values
    low = sig["low"].values
    vol = np.nan_to_num(sig["vol"].values, nan=np.nanmedian(sig["vol"].values))
    side_a = sig["side"].values
    mp = sig["meta_p"].values
    minute = idx.hour * 60 + idx.minute
    days = idx.normalize()
    frac = min(max(lag_min / bar_min, 0.0), 1.0)

    eq = capital
    pos = 0
    entry = qty = tp = slv = 0.0
    held = 0
    trades = []
    curve = np.empty(len(idx))
    cur_day, day_start, locked = None, eq, False

    n = len(idx)
    for i in range(n):
        if days[i] != cur_day:
            cur_day, day_start, locked = days[i], eq, False

        if pos != 0:
            held += 1
            if intrabar:
                hit_tp = (pos == 1 and high[i] >= tp) or (pos == -1 and low[i] <= tp)
                hit_sl = (pos == 1 and low[i] <= slv) or (pos == -1 and high[i] >= slv)
            else:
                hit_tp = (pos == 1 and close[i] >= tp) or (pos == -1 and close[i] <= tp)
                hit_sl = (pos == 1 and close[i] <= slv) or (pos == -1 and close[i] >= slv)
            eod = minute[i] >= 955
            timeout = held >= horizon
            if hit_tp or hit_sl or timeout or eod:
                if hit_sl:
                    px, reason = slv, "sl"
                elif hit_tp:
                    px, reason = tp, "tp"
                else:
                    px, reason = close[i], ("eod" if eod else "time")
                gross = pos * (px - entry) * qty
                fees = (entry + px) * qty * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": pos,
                               "bars": held, "reason": reason})
                pos, qty, held = 0, 0.0, 0

        if not locked and eq <= day_start * 0.97:
            locked = True

        if (pos == 0 and not locked and minute[i] < 940 and mp[i] >= thresh
                and i + 1 < n):
            s = int(side_a[i])
            v = max(vol[i], 1e-4)
            # entry price drifts toward the next bar as lag increases
            px_in = close[i] + frac * (close[i + 1] - close[i])
            stop_dist = sl * v * px_in
            q = (capital * risk) / max(stop_dist, 1e-6)
            q = min(q, (2.0 * capital) / px_in)
            if q > 0:
                pos, entry, qty, held = s, px_in, q, 0
                tp = entry * (1 + s * pt * v)
                slv = entry * (1 - s * sl * v)
        curve[i] = eq

    return pd.DataFrame(trades), pd.Series(curve, index=idx)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("fetching history")
    hist = fetch_history(S, need, S.train_years)
    mkt = hist[S.market_symbol][["close"]].rename(
        columns={"close": f"{S.market_symbol}_close"})

    rows = []
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg)
        mc = df[f"{S.market_symbol}_close"]
        feats["mkt_trend_bps"] = (mc / mc.shift(TREND_BARS) - 1) * 1e4
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 5000:
            continue

        trained = {}
        for name, hz, pt, sl, thr in CONFIGS:
            if hz not in trained:
                c = ModelCfg(S)
                c.horizon_bars = hz
                trained[hz] = walk_forward(df, feats, c, False)
                log.info("%s trained horizon=%d", sym, hz)
            sg = trained[hz]
            if sg is None:
                continue
            for lag in LAGS_MIN:
                for ib in (False, True):
                    t, c_ = simulate_lag(sg, pt=pt, sl=sl, horizon=hz, thresh=thr,
                                         lag_min=lag, intrabar=ib)
                    st = stats(t, c_)
                    if st["trades"] < 50:
                        continue
                    rows.append({"config": name, "lag_min": lag, "intrabar": ib,
                                 "symbol": sym, **st})
            log.info("  %s done", name)

    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return 1
    d.to_json("/app/state/lag_sweep_raw.json", orient="records")

    g = (d.groupby(["config", "intrabar", "lag_min"])
           .agg(ret=("ret_pct", "mean"), win=("win_pct", "mean"),
                be=("breakeven_pct", "mean"), pf=("pf", "mean"),
                sharpe=("sharpe", "mean"), dd=("maxdd_pct", "mean"),
                trades=("trades", "mean"), n=("symbol", "count"))
           .reset_index())
    g["edge"] = (g["win"] - g["be"]).round(2)
    for c in ("ret", "win", "be", "pf", "sharpe", "dd"):
        g[c] = g[c].round(2)
    g["trades"] = g["trades"].round(0)

    print("\n" + "=" * 104)
    print("LAG SENSITIVITY -- where does the edge die?")
    print("  lag_min 0 = v1 timing (act at bar close) | 5.0 = v2 timing (full bar delay)")
    print("  intrabar False = barriers at closes only | True = intrabar high/low")
    print("  edge = win rate minus breakeven; > 0 means the geometry is viable")
    print("=" * 104)
    for name, _, _, _, _ in CONFIGS:
        sub = g[g.config == name]
        if sub.empty:
            continue
        print(f"\n--- {name} ---")
        print(sub[["intrabar", "lag_min", "ret", "win", "be", "edge",
                   "pf", "sharpe", "dd", "trades"]].to_string(index=False))

    print("\n" + "=" * 104)
    print("DECOMPOSITION: which correction destroyed the edge?")
    print("=" * 104)
    for name, _, _, _, _ in CONFIGS:
        sub = g[g.config == name]
        if sub.empty:
            continue

        def pick(ib, lag):
            r = sub[(sub.intrabar == ib) & (np.isclose(sub.lag_min, lag))]
            return float(r["ret"].iloc[0]) if len(r) else float("nan")

        base = pick(False, 0.0)      # v1
        lag_only = pick(False, 5.0)  # lag alone
        ib_only = pick(True, 0.0)    # intrabar alone
        both = pick(True, 5.0)       # v2
        print(f"\n  {name}")
        print(f"    v1 baseline (no lag, close-only)  : {base:>9.1f}%")
        print(f"    + signal lag only                 : {lag_only:>9.1f}%  "
              f"({lag_only - base:>+8.1f} pp)")
        print(f"    + intrabar exits only             : {ib_only:>9.1f}%  "
              f"({ib_only - base:>+8.1f} pp)")
        print(f"    + both (v2)                       : {both:>9.1f}%  "
              f"({both - base:>+8.1f} pp)")

    g.to_json("/app/state/lag_sweep_summary.json", orient="records")
    print("\nsaved /app/state/lag_sweep_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Parameter sweep: maximise trade count subject to holding Sharpe.

Levers that change trade frequency:
    meta_threshold  - lower = more signals pass the confidence gate
    horizon_bars    - shorter = faster turnover (requires retraining: it changes
                      the triple-barrier labels)
    pt / sl mult    - tighter barriers = quicker exits = more trades
    market filter   - off = ~20% more trades

Training is the expensive part, so we train once per (horizon, balanced) pair and
then simulate every cheap parameter combination on those cached signals.

    python -m app.sweep
"""
import itertools
import json
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_ab import walk_forward, simulate, stats, TREND_BARS

log = logging.getLogger("sweep")

HORIZONS = [3, 6]
THRESHOLDS = [0.50, 0.53, 0.55, 0.58]
BARRIERS = [(1.0, 1.0), (0.7, 1.0), (0.7, 0.7)]
FILTERS = [True, False]
BALANCED = False          # A/B showed class balancing is a wash; keep it off


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("fetching %.1f years of %dmin bars", S.train_years, S.bar_minutes)
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

        for horizon in HORIZONS:
            c = ModelCfg(S)
            c.horizon_bars = horizon
            sig = walk_forward(df, feats, c, BALANCED)
            if sig is None:
                continue
            log.info("%s trained horizon=%d (%d oos bars)", sym, horizon, len(sig))

            for thr, (pt, sl), use_f in itertools.product(THRESHOLDS, BARRIERS, FILTERS):
                tr, curve = simulate(sig, use_f, pt=pt, sl=sl,
                                     horizon=horizon, thresh=thr)
                st = stats(tr, curve)
                if st["trades"] < 50:
                    continue
                rows.append({"symbol": sym, "horizon": horizon, "thresh": thr,
                             "pt": pt, "sl": sl, "filter": use_f, **st})

    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return 1
    d.to_json("/app/state/sweep_raw.json", orient="records")

    # average across symbols for each parameter combination
    g = (d.groupby(["horizon", "thresh", "pt", "sl", "filter"])
           .agg(trades=("trades", "mean"),
                ret=("ret_pct", "mean"),
                sharpe=("sharpe", "mean"),
                win=("win_pct", "mean"),
                pf=("pf", "mean"),
                maxdd=("maxdd_pct", "mean"),
                per_trade=("net_per_trade", "mean"),
                n=("symbol", "count"))
           .reset_index())
    g = g[g["n"] >= 3]          # keep combos that worked on most symbols
    g["trades"] = g["trades"].round(0)
    for c in ("ret", "sharpe", "win", "pf", "maxdd", "per_trade"):
        g[c] = g[c].round(2)

    # current live config for reference
    base = g[(g.horizon == 6) & (g.thresh == 0.55) & (g.pt == 1.0) &
             (g.sl == 1.0) & (g["filter"])]
    base_sharpe = float(base["sharpe"].iloc[0]) if len(base) else None
    base_trades = float(base["trades"].iloc[0]) if len(base) else None

    print("\n" + "=" * 104)
    print("SWEEP: more trades at equal-or-better Sharpe")
    if base_sharpe:
        print(f"current live config -> trades={base_trades:.0f}  sharpe={base_sharpe:.2f}")
    print("=" * 104)

    if base_sharpe:
        cand = g[(g["sharpe"] >= base_sharpe * 0.98) &
                 (g["trades"] > base_trades)].sort_values("trades", ascending=False)
        print("\nCONFIGS WITH MORE TRADES AND SHARPE WITHIN 2% OF CURRENT:")
        print(cand.head(15).to_string(index=False) if len(cand)
              else "  none -- trade count cannot be raised without losing Sharpe")

    print("\nTOP 12 BY SHARPE:")
    print(g.sort_values("sharpe", ascending=False).head(12).to_string(index=False))

    print("\nTOP 12 BY TRADE COUNT:")
    print(g.sort_values("trades", ascending=False).head(12).to_string(index=False))

    g.to_json("/app/state/sweep_summary.json", orient="records")
    print("\nsaved /app/state/sweep_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Threshold sweep: does cutting trade count rescue the strategy?

Reasoning: transaction costs scale linearly with the number of trades, while the
edge per trade does not. At layer 4 of the reality stack (all execution effects,
no costs) the strategy returns ~25%/yr; costs then take it to -23%/yr. If we trade
only the highest-conviction signals, fee drag falls proportionally -- so the
question is whether edge per trade holds up, improves, or collapses as we become
more selective.

Three possibilities, and only the data decides:
  * edge per trade rises with conviction  -> fewer, better trades win
  * edge per trade is flat                -> cutting trades cuts costs, net helps
  * edge per trade falls                  -> the meta model cannot rank, done

Everything here runs the FULL reality stack: signal lag, intrabar stops, stop
slippage, adverse-selected limit fills, and costs.

    python -m app.threshold_sweep
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_v3 import walk_forward, edge_of
from .reality_stack import simulate, deflated_sharpe, N_TRIALS

log = logging.getLogger("thresh")

CAPITAL = 5000.0
HORIZON = 6
PT, SL = 1.5, 1.0

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

FULL = dict(lag=1, intrabar=True, stop_slip_R=0.10, limit_fills=True,
            cost=0.00015)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("fetching history")
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})

    spy_d = spy["close"].resample("1D").last().dropna()
    days = (spy_d.index[-1] - spy_d.index[0]).days
    spy_ann = ((spy_d.iloc[-1] / spy_d.iloc[0]) ** (365 / max(days, 1)) - 1) * 100
    spy_daily = spy_d.pct_change().dropna()
    spy_sharpe = float(spy_daily.mean() / spy_daily.std() * np.sqrt(252))

    sigs = {}
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg0)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 5000:
            continue
        c = ModelCfg(S)
        c.horizon_bars = HORIZON
        sg = walk_forward(df, feats, c)
        if sg is not None:
            sigs[sym] = sg
            log.info("%s trained (meta_p p50=%.3f p90=%.3f p99=%.3f)", sym,
                     sg["meta_p"].quantile(0.5), sg["meta_p"].quantile(0.9),
                     sg["meta_p"].quantile(0.99))

    rows = []
    for thr in THRESHOLDS:
        per, dl = [], []
        for sym, sg in sigs.items():
            t, curve, meta = simulate(sg, thresh=thr, pt=PT, sl=SL,
                                      horizon=HORIZON, **FULL)
            if t.empty or len(t) < 20:
                continue
            daily = curve.resample("1D").last().dropna().pct_change().dropna()
            win, be, edge = edge_of(t)
            total = float(curve.iloc[-1] / CAPITAL - 1) * 100
            ann = ((1 + total / 100) ** (365 / max(days, 1)) - 1) * 100
            roll = curve.cummax()
            per.append({"trades": len(t), "ret": total, "ann": ann,
                        "edge": edge, "win": win, "be": be,
                        "dd": float(((curve - roll) / roll).min()) * 100,
                        "pnl_per_trade": float(t["pnl"].mean()),
                        "fill": meta.get("fill_rate")})
            dl.append(daily)
        if not per:
            continue
        d = pd.concat(dl, axis=1).mean(axis=1).dropna()
        sh = float(d.mean() / (d.std() + 1e-12) * np.sqrt(252)) if len(d) > 2 else np.nan
        dsr = deflated_sharpe(d.values, sh, N_TRIALS)
        rows.append({
            "thresh": thr,
            "trades": int(np.mean([p["trades"] for p in per])),
            "annual_%": round(float(np.mean([p["ann"] for p in per])), 1),
            "sharpe": round(sh, 2) if np.isfinite(sh) else None,
            "edge": round(float(np.mean([p["edge"] for p in per
                                         if p["edge"] is not None])), 2),
            "win_%": round(float(np.mean([p["win"] for p in per])), 1),
            "$/trade": round(float(np.mean([p["pnl_per_trade"] for p in per])), 3),
            "maxDD_%": round(float(np.mean([p["dd"] for p in per])), 1),
            "DSR": round(dsr, 3) if dsr is not None else None,
        })
        log.info("thr %.2f -> %d trades, %.1f%%/yr, edge %.2f",
                 thr, rows[-1]["trades"], rows[-1]["annual_%"], rows[-1]["edge"])

    out = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP -- full reality stack (lag, intrabar, slip, limit fills, costs)")
    print("  does trading less, but better, survive costs?")
    print("=" * 100)
    print(out.to_string(index=False))
    print(f"\nSPY buy and hold: {spy_ann:.1f}%/yr, Sharpe {spy_sharpe:.2f}")

    if len(out):
        best = out.loc[out["annual_%"].idxmax()]
        print("\nBEST BY RETURN:")
        print(f"  threshold {best['thresh']}: {best['annual_%']:.1f}%/yr, "
              f"{best['trades']} trades, edge {best['edge']:+.2f}, "
              f"Sharpe {best['sharpe']}, DSR {best['DSR']}")
        print(f"  vs SPY {spy_ann:.1f}%/yr -> excess {best['annual_%'] - spy_ann:+.1f} pp")
        if best["annual_%"] <= spy_ann:
            print("\n  VERDICT: no threshold beats simply holding the index.")
        elif best["DSR"] is not None and best["DSR"] < 0.90:
            print(f"\n  VERDICT: beats the index on paper, but DSR {best['DSR']} < 0.90 --")
            print("           not statistically separable from luck after the search.")
        else:
            print("\n  VERDICT: beats the index and survives the multiple-testing penalty.")

    out.to_json("/app/state/threshold_sweep.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

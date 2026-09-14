"""
Horizon sweep -- the last structural lever.

Diagnosis from the validation gate: per-trade edge before costs is ~0.83 bps while
round-trip cost is ~3.6 bps. Costs are 4.3x gross edge. No threshold fixes that,
because selectivity cannot outrun a cost that large relative to the edge.

But barriers scale with sqrt(horizon): hold 4x longer and the target move is 2x
bigger, while the cost per trade is unchanged. So edge-to-cost improves with the
square root of holding time. The question is whether it can cross 1.0 while still
closing every position the same day.

Ceiling: a US session is 78 five-minute bars. Holding all day is the maximum
compatible with the no-overnight rule.

    python -m app.horizon_sweep
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
from .reality_stack import simulate, deflated_sharpe

log = logging.getLogger("horizon")

CAPITAL = 5000.0
PT, SL = 1.5, 1.0
THRESH = 0.85
N_TRIALS = 80

# bars, label
HORIZONS = [6, 12, 24, 39, 78]      # 30m, 1h, 2h, 3.25h, full session
NOCOST = dict(lag=1, intrabar=True, stop_slip_R=0.10, limit_fills=True)
FULL = dict(lag=1, intrabar=True, stop_slip_R=0.10, limit_fills=True, cost=0.00015)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("horizon sweep | %d symbols", len(S.symbols))
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})
    spy_d = spy["close"].resample("1D").last().dropna()
    days = (spy_d.index[-1] - spy_d.index[0]).days or 1
    spy_ann = ((spy_d.iloc[-1] / spy_d.iloc[0]) ** (365 / days) - 1) * 100

    prepared = {}
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg0)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) >= 4000:
            prepared[sym] = (df, feats)
    log.info("prepared %d symbols", len(prepared))

    rows = []
    for hz in HORIZONS:
        per_nc, per_f, curves = [], [], []
        for sym, (df, feats) in prepared.items():
            c = ModelCfg(S)
            c.horizon_bars = hz
            sg = walk_forward(df, feats, c)
            if sg is None:
                continue
            tn, cn, _ = simulate(sg, thresh=THRESH, pt=PT, sl=SL, horizon=hz,
                                 **NOCOST)
            tf, cf, _ = simulate(sg, thresh=THRESH, pt=PT, sl=SL, horizon=hz,
                                 **FULL)
            if tf.empty or len(tf) < 10 or tn.empty:
                continue
            _, _, e_nc = edge_of(tn)
            _, wf_be, e_f = edge_of(tf)
            roll = cf.cummax()
            per_nc.append({"ret": float(cn.iloc[-1] / CAPITAL - 1) * 100,
                           "trades": len(tn), "edge": e_nc,
                           "ppt": float(tn["pnl"].mean())})
            per_f.append({"ret": float(cf.iloc[-1] / CAPITAL - 1) * 100,
                          "trades": len(tf), "edge": e_f,
                          "ppt": float(tf["pnl"].mean()),
                          "hold": float(tf["bars"].mean()) * 5,
                          "dd": float(((cf - roll) / roll).min()) * 100})
            curves.append(cf.resample("1D").last().dropna().pct_change().dropna())
        if not per_f:
            continue
        dl = pd.concat(curves, axis=1).mean(axis=1).dropna()
        sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252))
        dsr = deflated_sharpe(dl.values, sh, N_TRIALS) or 0

        # gross edge per trade in bps of capital, versus round-trip cost
        gross_bps = float(np.mean([p["ppt"] for p in per_nc])) / CAPITAL * 1e4
        net_bps = float(np.mean([p["ppt"] for p in per_f])) / CAPITAL * 1e4
        cost_bps = gross_bps - net_bps
        tot = int(np.sum([p["trades"] for p in per_f]))
        rows.append({
            "bars": hz, "hold_min": round(float(np.mean([p["hold"] for p in per_f])), 0),
            "trades_day": round(tot / days, 1),
            "gross_bps": round(gross_bps, 2),
            "cost_bps": round(cost_bps, 2),
            "ratio": round(gross_bps / cost_bps, 2) if cost_bps else None,
            "annual_%": round(float(np.mean([p["ret"] for p in per_f])) * 365 / days, 1),
            "sharpe": round(sh, 2),
            "edge": round(float(np.mean([p["edge"] for p in per_f
                                         if p["edge"] is not None])), 2),
            "maxDD_%": round(float(np.mean([p["dd"] for p in per_f])), 1),
            "DSR": round(dsr, 3)})
        log.info("h%-3d hold %.0fm gross %.2fbps cost %.2fbps ratio %.2f -> %.1f%%/yr",
                 hz, rows[-1]["hold_min"], gross_bps, cost_bps,
                 rows[-1]["ratio"] or 0, rows[-1]["annual_%"])

    d = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f"HORIZON SWEEP -- {len(prepared)} symbols, threshold {THRESH}, "
          "full reality stack")
    print("  gross_bps = edge per trade BEFORE costs | cost_bps = round-trip cost")
    print("  ratio > 1.0 is the requirement: the trade must earn more than it costs")
    print("=" * 100)
    print(d.to_string(index=False))
    print(f"\nSPY buy and hold: {spy_ann:.1f}%/yr")

    if len(d):
        best = d.loc[d["annual_%"].idxmax()]
        print(f"\nBEST: horizon {int(best['bars'])} bars ({best['hold_min']:.0f} min "
              f"hold), {best['annual_%']:.1f}%/yr, ratio {best['ratio']}")
        if (d["ratio"] > 1.0).any():
            print("  -> at least one horizon earns more per trade than it costs")
        else:
            print("  -> NO horizon clears its own transaction cost. The per-trade")
            print("     edge available on 5-minute signals is structurally smaller")
            print("     than retail round-trip cost, at every holding period that")
            print("     still closes the position the same day.")
    d.to_json("/app/state/horizon_sweep.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

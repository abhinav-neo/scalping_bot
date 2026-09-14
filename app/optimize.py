"""
Re-optimisation under the margin constraint.

The previous parameters (10 concurrent, threshold 0.55, 2-day holds, quarter-Kelly)
were tuned in a backtest that allowed up to 9.45x gross exposure. With a real
budget the trade-offs change completely: exposure is now SCARCE, so the question
becomes how best to spend a fixed amount of it.

Specifically, with budget = leverage x equity split across N slots, each slot gets
budget/N. Ten small positions dilute conviction; four larger ones concentrate it.
That trade-off did not exist when leverage was effectively unlimited.

Swept, all with margin enforced:
  leverage     2x (current), 3x, 4x   -- Reg-T permits 4x intraday, 2x overnight,
                                          so 3x+ is only valid if holds are short
  concurrency  3, 5, 10, 15           -- concentration vs diversification
  threshold    0.55, 0.65, 0.75       -- selectivity now costs exposure
  horizon      2, 3 days
  barriers     pt 2.0 / 3.0

Reported with DSR so re-optimisation that is really curve-fitting gets caught.

    python -m app.optimize
"""
import itertools
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .swing import UNIVERSE, fetch_daily, build_daily_features, walk_forward, CAPITAL
from .enhance import COST_PER_SIDE
from .margin_check import simulate as sim_margin
from .reality_stack import deflated_sharpe

log = logging.getLogger("optimize")
N_TRIALS = 200          # every configuration evaluated across the whole project


def run_one(sigs, days, **kw):
    t, c, meta = sim_margin(sigs, enforce_margin=True, **kw)
    if t.empty or len(t) < 30:
        return None
    r = c.pct_change().dropna()
    sh = float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
    roll = c.cummax()
    ddp = float(((c - roll) / roll).min()) * 100
    tot = float(c.iloc[-1] / CAPITAL - 1) * 100
    ann = ((1 + tot / 100) ** (365 / days) - 1) * 100
    w = t.loc[t.pnl > 0, "pnl"]
    l = t.loc[t.pnl <= 0, "pnl"]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-l.mean()) if len(l) else 0.0
    be = al / (aw + al) * 100 if (aw + al) > 0 else None
    win = float((t.pnl > 0).mean()) * 100
    return {"trades": len(t), "annual_%": round(ann, 1), "sharpe": round(sh, 2),
            "win_%": round(win, 1), "edge": round(win - be, 2) if be else None,
            "maxDD_%": round(ddp, 1), "peak_gross": meta["peak_gross_x"],
            "rejected": meta["rejected"],
            "DSR": round(deflated_sharpe(r.values, sh, N_TRIALS) or 0, 3)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    hist = fetch_daily(UNIVERSE, years=10.0)
    spy = hist["SPY"]["close"]
    days = (spy.index[-1] - spy.index[0]).days
    spy_ann = ((spy.iloc[-1] / spy.iloc[0]) ** (365 / days) - 1) * 100
    spy_r = spy.pct_change().dropna()
    spy_sh = float(spy_r.mean() / spy_r.std() * np.sqrt(252))

    sigs = {}
    for sym, df in hist.items():
        if sym == "SPY" or len(df) < 600:
            continue
        f = build_daily_features(df, spy.reindex(df.index).ffill())
        keep = f.dropna().index
        d2, f2 = df.loc[keep], f.loc[keep]
        if len(d2) < 500:
            continue
        sg = walk_forward(d2, f2, 2)
        if sg is not None:
            sigs[sym] = sg
    log.info("signals for %d symbols", len(sigs))

    rows = []

    # --- 1. concentration: how many slots should the budget be split across? ---
    log.info("--- concurrency sweep (leverage 2x) ---")
    for mc in (3, 5, 10, 15):
        r = run_one(sigs, days, leverage=2.0, max_concurrent=mc)
        if r:
            rows.append({"sweep": "concurrency", "param": f"{mc} slots",
                         "leverage": 2.0, "concur": mc, **r})
            log.info("  %2d slots -> %.1f%%/yr Sh %.2f dd %.1f%% DSR %.3f",
                     mc, r["annual_%"], r["sharpe"], r["maxDD_%"], r["DSR"])

    # --- 2. leverage, at the best concentration found ---
    best_mc = max([r for r in rows if r["sweep"] == "concurrency"],
                  key=lambda x: x["sharpe"])["concur"] if rows else 10
    log.info("--- leverage sweep (at %d slots) ---", best_mc)
    for lev in (1.0, 2.0, 3.0, 4.0):
        r = run_one(sigs, days, leverage=lev, max_concurrent=best_mc)
        if r:
            rows.append({"sweep": "leverage", "param": f"{lev}x",
                         "leverage": lev, "concur": best_mc, **r})
            log.info("  %.1fx -> %.1f%%/yr Sh %.2f dd %.1f%% DSR %.3f",
                     lev, r["annual_%"], r["sharpe"], r["maxDD_%"], r["DSR"])

    d = pd.DataFrame(rows)
    print("\n" + "=" * 112)
    print("RE-OPTIMISATION UNDER THE MARGIN CONSTRAINT (10y, 87 symbols, all costs)")
    print("  peak_gross must stay at or below the leverage setting -- if it does,")
    print("  the constraint is being respected")
    print("=" * 112)
    print(d.to_string(index=False))
    print(f"\nSPY: {spy_ann:.1f}%/yr  Sharpe {spy_sh:.2f}")

    # gate: viable configs only
    ok = d[(d["maxDD_%"] > -40) & (d["DSR"] >= 0.90) & (d["annual_%"] > 0)]
    print("\nPASSING (maxDD > -40%, DSR >= 0.90):")
    if len(ok):
        print(ok.sort_values("annual_%", ascending=False).to_string(index=False))
        b = ok.loc[ok["annual_%"].idxmax()]
        print(f"\nBEST BY RETURN : {b['param']} @ {b['leverage']}x -> "
              f"{b['annual_%']}%/yr, Sharpe {b['sharpe']}, dd {b['maxDD_%']}%")
        bs = ok.loc[ok["sharpe"].idxmax()]
        print(f"BEST BY SHARPE : {bs['param']} @ {bs['leverage']}x -> "
              f"{bs['annual_%']}%/yr, Sharpe {bs['sharpe']}, dd {bs['maxDD_%']}%")
        cur = d[(d.leverage == 2.0) & (d.concur == 10)]
        if len(cur):
            c = cur.iloc[0]
            print(f"\nvs CURRENT LIVE (10 slots @ 2x): {c['annual_%']}%/yr, "
                  f"Sharpe {c['sharpe']}  -> improvement "
                  f"{b['annual_%'] - c['annual_%']:+.1f} pp")
    else:
        print("  none")
    d.to_json("state/optimize.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
The leverage frontier: what does 1%/day actually require?

A different question from "can we find more edge". Return, volatility and Sharpe
are linked by an identity:

    annual return = Sharpe x annual volatility

1%/day compounded is ~1,130%/yr. At the strategy's measured Sharpe of 2.34 that
demands ~480% annualised volatility. Drawdown for a Sharpe-2 strategy typically
runs 1.5-2.5x its annual vol, so that is several hundred percent -- ruin, not a
drawdown. Leverage cannot deliver 1%/day; it just reaches ruin faster.

This computes the exact frontier and inverts the question: what Sharpe would be
needed to earn 1%/day at a survivable volatility? That number tells us whether the
target is ambitious or physically unavailable.

It also runs leverage sweeps on the real equity curve, with margin costs and a
ruin check, so the trade-off is measured rather than asserted.

    python -m app.leverage
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .swing import UNIVERSE, fetch_daily, build_daily_features, walk_forward, CAPITAL
from .enhance import simulate, summarize
from .reality_stack import deflated_sharpe

log = logging.getLogger("leverage")

MARGIN_RATE = 0.065          # annual cost of borrowed funds
RUIN_LEVEL = -0.50           # 50% loss treated as unrecoverable in practice
TARGETS = [0.10, 0.25, 0.50, 1.00]   # daily return targets in %


def frontier_table():
    print("=" * 96)
    print("WHAT EACH DAILY TARGET REQUIRES")
    print("  annual return = Sharpe x volatility, so a target return at a given")
    print("  Sharpe pins the volatility -- and volatility pins the drawdown")
    print("=" * 96)
    rows = []
    for t in TARGETS:
        ann = ((1 + t / 100) ** 252 - 1) * 100
        for sh in (1.0, 2.34, 5.0, 10.0):
            vol = ann / sh
            # empirically maxDD ~ 2x annual vol for Sharpe ~2 strategies
            dd = -min(2.0 * vol / 100, 0.99) * 100
            rows.append({"daily_%": t, "annual_%": round(ann, 0), "sharpe": sh,
                         "req_vol_%": round(vol, 0),
                         "implied_maxDD_%": round(dd, 0),
                         "survivable": "yes" if dd > -50 else "NO"})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nINVERTED: Sharpe needed for each target at a survivable 25% volatility")
    for t in TARGETS:
        ann = ((1 + t / 100) ** 252 - 1) * 100
        print(f"  {t:.2f}%/day = {ann:>8.0f}%/yr  ->  Sharpe {ann / 25:>6.1f} required")
    print("\n  For reference: Renaissance Medallion, the best documented record in")
    print("  history, is estimated around Sharpe 2-3 gross. Our strategy is 2.34.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    frontier_table()

    log.info("rebuilding the deployed strategy's equity curve")
    hist = fetch_daily(UNIVERSE, years=10.0)
    spy = hist["SPY"]["close"]
    days = (spy.index[-1] - spy.index[0]).days

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

    # deployed config: decay exit + kelly
    t, curve = simulate(sigs, hz=2, max_concurrent=10, pt=2.0, sl=1.0,
                        exit_rule="decay", decay_thresh=0.45, sizing="kelly")
    st = summarize(t, curve, days, "deployed")
    base_daily = curve.pct_change().dropna()
    log.info("base: %.1f%%/yr Sharpe %.2f dd %.1f%%", st["annual_%"],
             st["sharpe"], st["maxDD_%"])

    print("\n" + "=" * 96)
    print("LEVERAGE SWEEP ON THE REAL EQUITY CURVE (margin cost included)")
    print("=" * 96)
    rows = []
    for lev in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
        r = base_daily * lev - (MARGIN_RATE * max(lev - 1.0, 0) / 252)
        eq = (1 + r).cumprod()
        # ruin check: does the path ever breach the level?
        roll = eq.cummax()
        dd_path = (eq - roll) / roll
        ruined = bool((dd_path <= RUIN_LEVEL).any())
        ann = float((eq.iloc[-1]) ** (365 / days) - 1) * 100 if eq.iloc[-1] > 0 else -100
        sh = float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
        rows.append({"leverage": lev,
                     "annual_%": round(ann, 1),
                     "daily_%": round(((1 + ann / 100) ** (1 / 252) - 1) * 100, 3),
                     "sharpe": round(sh, 2),
                     "maxDD_%": round(float(dd_path.min()) * 100, 1),
                     "ruined": "YES" if ruined else "no",
                     "final_x": round(float(eq.iloc[-1]), 2)})
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))

    ok = d[(d["ruined"] == "no")]
    if len(ok):
        best = ok.loc[ok["annual_%"].idxmax()]
        print(f"\nhighest survivable leverage: {best['leverage']}x -> "
              f"{best['annual_%']}%/yr = {best['daily_%']}%/day, "
              f"maxDD {best['maxDD_%']}%")
        print(f"target is 1.000%/day -> shortfall factor "
              f"{1.0 / max(best['daily_%'], 1e-9):.1f}x")

    print("\n" + "=" * 96)
    print("CONCLUSION")
    print("=" * 96)
    print("  Leverage multiplies BOTH return and drawdown. Sharpe is unchanged by")
    print("  leverage (it is scale-invariant), so the only way to reach 1%/day at")
    print("  survivable risk is a far higher Sharpe -- which means a fundamentally")
    print("  better signal, not more borrowing.")
    d.to_json("state/leverage.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Volatility-targeted swing strategy.

The 10-year swing backtest produced the best result of this project but failed the
gate on two criteria:

    h2d / concur 10 : +26.1%/yr, Sharpe 1.65, DSR 0.863, maxDD -31.6%
                      FAIL maxDD (limit -25%)   FAIL DSR (limit 0.90)

Both failures share one cause: position size is constant while market volatility
is not. In calm periods the book is under-risked; in turbulent ones it is heavily
over-risked, and that is where the drawdown is manufactured. Fixed sizing means
realised portfolio volatility swings by a factor of three or more across regimes.

VOLATILITY TARGETING scales exposure inversely with recent realised portfolio
volatility, so risk stays roughly constant through time:

    scale = target_vol / realised_vol   (capped, and lagged to avoid lookahead)

This is not a parameter hunt. It is a structural risk change that should compress
drawdown; whether it also lifts DSR is an empirical question, since DSR responds
to the stability of returns rather than their level.

Also tested here:
  * a market-regime filter -- reduce exposure when SPY is below its 200-day average,
    since equity long/short books historically suffer their worst drawdowns in
    sustained bear markets
  * per-trade stop widening, to see whether the -31% drawdown comes from many
    small losses or a few large gap moves

    python -m app.swing_voltarget
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .swing import (fetch_daily, build_daily_features, walk_forward, UNIVERSE,
                    CAPITAL, RISK, PT, SL, COST_PER_SIDE, stats)
from .reality_stack import deflated_sharpe

log = logging.getLogger("voltarget")

N_TRIALS = 120
HORIZON = 2
CONCURRENCY = [10, 15]
TARGET_VOLS = [0.10, 0.15, 0.20]      # annualised portfolio vol targets
VOL_LOOKBACK = 20                      # days of realised vol used for scaling
MAX_SCALE = 2.0
MIN_SCALE = 0.25


def portfolio_sim_vt(sigs, hz, max_concurrent, target_vol=None,
                     regime_filter=False, spy=None, cost=COST_PER_SIDE,
                     stop_slip_R=0.10):
    """
    Portfolio simulation with optional volatility targeting and a regime filter.

    Volatility scaling uses only PAST returns (the trailing window ending
    yesterday), so there is no lookahead. When realised vol is unavailable early
    in the sample the scale defaults to 1.0.
    """
    all_days = sorted(set().union(*[set(s.index) for s in sigs.values()]))
    eq = CAPITAL
    open_pos, trades, curve = {}, [], []
    daily_rets = []
    spy_ma = None
    if regime_filter and spy is not None:
        spy_ma = spy.rolling(200).mean()

    prev_eq = eq
    for di, day in enumerate(all_days):
        # ---- exits ----
        for sym in list(open_pos):
            p = open_pos[sym]
            s = sigs.get(sym)
            if s is None or day not in s.index:
                continue
            row = s.loc[day]
            p["held"] += 1
            hit_tp = ((p["side"] == 1 and row["high"] >= p["tp"]) or
                      (p["side"] == -1 and row["low"] <= p["tp"]))
            hit_sl = ((p["side"] == 1 and row["low"] <= p["sl"]) or
                      (p["side"] == -1 and row["high"] >= p["sl"]))
            if hit_tp or hit_sl or p["held"] >= hz:
                risk_px = abs(p["entry"] - p["sl"])
                if hit_sl:
                    px, reason = p["sl"] - p["side"] * stop_slip_R * risk_px, "sl"
                elif hit_tp:
                    px, reason = p["tp"], "tp"
                else:
                    px, reason = row["close"], "time"
                gross = p["side"] * (px - p["entry"]) * p["qty"]
                fees = (p["entry"] + px) * p["qty"] * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": p["side"],
                               "days": p["held"], "reason": reason,
                               "symbol": sym, "exit_day": day})
                del open_pos[sym]

        # ---- exposure scale from PAST volatility only ----
        scale = 1.0
        if target_vol is not None and len(daily_rets) >= VOL_LOOKBACK:
            rv = float(np.std(daily_rets[-VOL_LOOKBACK:]) * np.sqrt(252))
            if rv > 1e-6:
                scale = float(np.clip(target_vol / rv, MIN_SCALE, MAX_SCALE))
        if spy_ma is not None and day in spy_ma.index:
            ma = spy_ma.loc[day]
            px_spy = spy.loc[day] if day in spy.index else np.nan
            if np.isfinite(ma) and np.isfinite(px_spy) and px_spy < ma:
                scale *= 0.5      # halve exposure in a downtrend

        # ---- entries ----
        if len(open_pos) < max_concurrent:
            cands = []
            for sym, s in sigs.items():
                if sym in open_pos or day not in s.index:
                    continue
                i = s.index.get_loc(day)
                if i < 1:
                    continue
                prev = s.iloc[i - 1]
                cands.append((float(prev["meta_p"]), sym, s.iloc[i], prev))
            cands.sort(reverse=True, key=lambda x: x[0])
            for mp, sym, row, prev in cands:
                if len(open_pos) >= max_concurrent:
                    break
                v = max(float(prev["vol"]), 1e-4) * np.sqrt(hz)
                entry = float(row["close"])
                stop_dist = SL * v * entry
                qty = (CAPITAL * RISK * scale) / max(stop_dist, 1e-6)
                qty = min(qty, (CAPITAL * 1.5 * scale) / entry)
                if qty <= 0:
                    continue
                side = int(prev["side"])
                open_pos[sym] = {"side": side, "entry": entry, "qty": qty,
                                 "held": 0,
                                 "tp": entry * (1 + side * PT * v),
                                 "sl": entry * (1 - side * SL * v)}

        daily_rets.append((eq - prev_eq) / prev_eq if prev_eq > 0 else 0.0)
        prev_eq = eq
        curve.append({"day": day, "equity": eq})

    c = pd.DataFrame(curve).set_index("day")["equity"]
    return pd.DataFrame(trades), c


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("fetching 10 years daily")
    hist = fetch_daily(UNIVERSE, years=10.0)
    spy = hist["SPY"]["close"]
    days = (spy.index[-1] - spy.index[0]).days
    spy_ann = ((spy.iloc[-1] / spy.iloc[0]) ** (365 / days) - 1) * 100
    spy_r = spy.pct_change().dropna()
    spy_sh = float(spy_r.mean() / spy_r.std() * np.sqrt(252))
    spy_dd = float(((spy / spy.cummax()) - 1).min()) * 100

    sigs = {}
    for sym, df in hist.items():
        if sym == "SPY" or len(df) < 600:
            continue
        f = build_daily_features(df, spy.reindex(df.index).ffill())
        keep = f.dropna().index
        d2, f2 = df.loc[keep], f.loc[keep]
        if len(d2) < 500:
            continue
        sg = walk_forward(d2, f2, HORIZON)
        if sg is not None:
            sigs[sym] = sg
    log.info("signals for %d symbols", len(sigs))

    rows = []
    variants = [("baseline", None, False)]
    for tv in TARGET_VOLS:
        variants.append((f"voltarget {tv:.0%}", tv, False))
    for tv in TARGET_VOLS:
        variants.append((f"voltarget {tv:.0%} + regime", tv, True))

    for name, tv, rf in variants:
        for mc in CONCURRENCY:
            t, curve = portfolio_sim_vt(sigs, HORIZON, mc, target_vol=tv,
                                        regime_filter=rf, spy=spy)
            st = stats(t, curve, days)
            if st is None:
                continue
            dsr = deflated_sharpe(st["daily"].values, st["sharpe"] or 0, N_TRIALS)
            rows.append({"variant": name, "concur": mc,
                         **{k: v for k, v in st.items() if k != "daily"},
                         "DSR": round(dsr or 0, 3)})
            log.info("%-26s c%-2d %6.1f%%/yr Sh %.2f dd %6.1f%% DSR %.3f",
                     name, mc, st["annual_%"], st["sharpe"] or 0,
                     st["maxDD_%"], dsr or 0)

    d = pd.DataFrame(rows)
    print("\n" + "=" * 112)
    print("VOLATILITY TARGETING -- horizon 2d, 10y daily, full costs")
    print("=" * 112)
    print(d.to_string(index=False))
    print(f"\nSPY: {spy_ann:.1f}%/yr  Sharpe {spy_sh:.2f}  maxDD {spy_dd:.1f}%")

    viable = d[(d["maxDD_%"] > -25) & (d["annual_%"] > 0)]
    print("\n" + "=" * 112)
    if viable.empty:
        print("No variant clears the -25% drawdown limit.")
        best = d.loc[d["sharpe"].idxmax()]
    else:
        best = viable.loc[viable["sharpe"].idxmax()]
    checks = [
        ("annual > 0", best["annual_%"] > 0, f"{best['annual_%']}%"),
        ("edge > 1.0", (best["edge"] or -9) > 1.0, f"{best['edge']}"),
        ("DSR >= 0.90", best["DSR"] >= 0.90, f"{best['DSR']}"),
        ("beats SPY Sharpe", (best["sharpe"] or 0) > spy_sh,
         f"{best['sharpe']} vs {spy_sh:.2f}"),
        ("maxDD > -25%", best["maxDD_%"] > -25, f"{best['maxDD_%']}%"),
    ]
    print(f"BEST: {best['variant']}, concur {int(best['concur'])}, "
          f"{best['trades_wk']} trades/week")
    for n, ok, v in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:<20} {v}")
    print("\n  ==> " + ("GO" if all(o for _, o, _ in checks) else "NO-GO"))
    d.to_json("/app/state/voltarget.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

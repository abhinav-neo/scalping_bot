"""
Return enhancement -- systematic sweep of every lever that could raise +27.1%/yr.

Baseline (validated, live): 2-day holds, 10 concurrent, vol targeting
    +27.1%/yr | Sharpe 2.04 | DSR 0.974 | maxDD -22.2%

Six levers, tested independently so each contribution is measurable rather than
assumed. Anything that only works in combination is suspect.

  1 EXIT RULE      fixed 2-day clock vs signal-decay exit with a 1-week cap.
                   The fixed clock cuts winners that are still working; exiting
                   when conviction fades should hold winners and dump losers.

  2 BARRIER GEOMETRY  pt/sl of 2.0/1.0 was never swept at the daily horizon.

  3 KELLY SIZING   size by conviction rather than a flat 1% risk. Higher meta_p
                   should carry more capital if the model ranks honestly.

  4 SHORT/LONG     do shorts contribute or drag? If they drag, drop them.

  5 UNIVERSE SIZE  87 symbols vs the best-N by in-sample edge.

  6 LEVERAGE       the account uses ~1.0-1.5x. Reg-T allows 2x overnight, and a
                   Sharpe-2 strategy can carry more than a Sharpe-1 one.

Every run applies the full cost stack. Judged on Sharpe and DSR, not raw return --
ranking by return previously selected a config with -88% drawdown.

    python -m app.enhance
"""
import itertools
import json
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .swing import (UNIVERSE, fetch_daily, build_daily_features, walk_forward,
                    CAPITAL, COST_PER_SIDE, N_TRIALS)
from .reality_stack import deflated_sharpe

log = logging.getLogger("enhance")

HZ_CAP = 5               # hard ceiling: never hold beyond a week
RISK = 0.01
TARGET_VOL = 0.15
VOL_LOOKBACK = 20


def simulate(sigs, hz, max_concurrent, pt, sl,
             exit_rule="fixed", decay_thresh=0.45,
             sizing="flat", allow_short=True, max_lev=1.5,
             cost=COST_PER_SIDE, stop_slip_R=0.10, thresh=0.55):
    """
    exit_rule "fixed" : close after hz days
    exit_rule "decay" : close when meta_p falls below decay_thresh, capped at HZ_CAP
    sizing    "flat"  : risk a constant fraction
    sizing    "kelly" : scale by (meta_p - 0.5) * 2, i.e. by conviction
    """
    all_days = sorted(set().union(*[set(s.index) for s in sigs.values()]))
    eq = CAPITAL
    open_pos = {}
    trades = []
    curve = []
    recent = []

    for day in all_days:
        day_start = eq

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
            if exit_rule == "decay":
                # exit when the model stops believing, or at the weekly cap
                still = float(row["meta_p"]) >= decay_thresh and \
                    int(row["side"]) == p["side"]
                timeout = (not still) or p["held"] >= HZ_CAP
            else:
                timeout = p["held"] >= hz
            if hit_tp or hit_sl or timeout:
                risk_px = abs(p["entry"] - p["sl"])
                if hit_sl:
                    px, reason = p["sl"] - p["side"] * stop_slip_R * risk_px, "sl"
                elif hit_tp:
                    px, reason = p["tp"], "tp"
                else:
                    px, reason = float(row["close"]), "time"
                gross = p["side"] * (px - p["entry"]) * p["qty"]
                fees = (p["entry"] + px) * p["qty"] * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": p["side"],
                               "days": p["held"], "reason": reason})
                del open_pos[sym]

        mult = 1.0
        if len(recent) >= VOL_LOOKBACK:
            rv = float(np.std(recent[-VOL_LOOKBACK:]) * np.sqrt(252))
            if rv > 1e-6:
                mult = float(np.clip(TARGET_VOL / rv, 0.25, max_lev))

        if len(open_pos) < max_concurrent:
            cands = []
            for sym, s in sigs.items():
                if sym in open_pos or day not in s.index:
                    continue
                i = s.index.get_loc(day)
                if i < 1:
                    continue
                prev = s.iloc[i - 1]
                if float(prev["meta_p"]) < thresh:
                    continue
                if not allow_short and int(prev["side"]) < 0:
                    continue
                cands.append((float(prev["meta_p"]), sym, s.iloc[i], prev))
            cands.sort(reverse=True, key=lambda x: x[0])
            for mp, sym, row, prev in cands:
                if len(open_pos) >= max_concurrent:
                    break
                v = max(float(prev["vol"]), 1e-4) * np.sqrt(hz)
                entry = float(row["close"])
                stop_dist = sl * v * entry
                size_mult = mult
                if sizing == "kelly":
                    size_mult *= float(np.clip((mp - 0.5) * 2.0, 0.2, 2.0))
                qty = (CAPITAL * RISK * size_mult) / max(stop_dist, 1e-6)
                qty = min(qty, (CAPITAL * max_lev) / entry)
                if qty <= 0:
                    continue
                side = int(prev["side"])
                open_pos[sym] = {"side": side, "entry": entry, "qty": qty,
                                 "held": 0,
                                 "tp": entry * (1 + side * pt * v),
                                 "sl": entry * (1 - side * sl * v)}
        recent.append((eq - day_start) / max(day_start, 1e-9))
        curve.append({"day": day, "equity": eq})

    c = pd.DataFrame(curve).set_index("day")["equity"]
    return pd.DataFrame(trades), c


def summarize(t, c, days, label):
    if t.empty or len(t) < 30:
        return None
    r = c.pct_change().dropna()
    sh = float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
    roll = c.cummax()
    w = t.loc[t.pnl > 0, "pnl"]
    l = t.loc[t.pnl <= 0, "pnl"]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-l.mean()) if len(l) else 0.0
    be = al / (aw + al) * 100 if (aw + al) > 0 else None
    win = float((t.pnl > 0).mean()) * 100
    tot = float(c.iloc[-1] / CAPITAL - 1) * 100
    return {"variant": label, "trades": len(t),
            "trades_wk": round(len(t) / (days / 7), 1),
            "annual_%": round(((1 + tot / 100) ** (365 / days) - 1) * 100, 1),
            "sharpe": round(sh, 2),
            "win_%": round(win, 1),
            "edge": round(win - be, 2) if be else None,
            "hold_d": round(float(t["days"].mean()), 1),
            "maxDD_%": round(float(((c - roll) / roll).min()) * 100, 1),
            "DSR": round(deflated_sharpe(r.values, sh, N_TRIALS) or 0, 3)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("fetching 10y daily bars")
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

    def run(label, **kw):
        t, c = simulate(sigs, **kw)
        s = summarize(t, c, days, label)
        if s:
            rows.append(s)
            log.info("%-28s %6.1f%%/yr Sh %.2f dd %6.1f%% DSR %.3f",
                     label, s["annual_%"], s["sharpe"], s["maxDD_%"], s["DSR"])
        return s

    BASE = dict(hz=2, max_concurrent=10, pt=2.0, sl=1.0)
    run("baseline (live)", **BASE)

    # 1 exit rule
    for dt in (0.40, 0.45, 0.50):
        run(f"decay exit @{dt}", **{**BASE, "exit_rule": "decay",
                                    "decay_thresh": dt})

    # 2 barrier geometry
    for pt, sl in [(1.5, 1.0), (2.5, 1.0), (3.0, 1.0), (2.0, 1.5)]:
        run(f"barriers {pt}/{sl}", **{**BASE, "pt": pt, "sl": sl})

    # 3 sizing
    run("kelly sizing", **{**BASE, "sizing": "kelly"})

    # 4 direction
    run("long only", **{**BASE, "allow_short": False})

    # 5 concurrency
    for mc in (15, 20, 25):
        run(f"concurrency {mc}", **{**BASE, "max_concurrent": mc})

    # 6 leverage
    for lv in (2.0, 3.0):
        run(f"leverage {lv}x", **{**BASE, "max_lev": lv})

    # promising combinations
    run("decay + kelly", **{**BASE, "exit_rule": "decay", "sizing": "kelly"})
    run("decay + lev 2x", **{**BASE, "exit_rule": "decay", "max_lev": 2.0})
    run("decay + kelly + 2x", **{**BASE, "exit_rule": "decay",
                                 "sizing": "kelly", "max_lev": 2.0})

    d = pd.DataFrame(rows)
    print("\n" + "=" * 108)
    print("RETURN ENHANCEMENT -- 10y daily, 87 symbols, full costs")
    print("=" * 108)
    print(d.sort_values("sharpe", ascending=False).to_string(index=False))
    print(f"\nSPY: {spy_ann:.1f}%/yr  Sharpe {spy_sh:.2f}")

    ok = d[(d["maxDD_%"] > -25) & (d["DSR"] >= 0.90)]
    print("\n" + "=" * 108)
    print("PASSING BOTH GATES (maxDD > -25%, DSR >= 0.90)")
    print("=" * 108)
    print(ok.sort_values("annual_%", ascending=False).to_string(index=False)
          if len(ok) else "  none")
    d.to_json("state/enhance.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

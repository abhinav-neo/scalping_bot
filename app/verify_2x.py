"""
Backtest the DEPLOYED 2x configuration, including the circuit breaker.

Why this exists: the 2x deployment was justified by scaling an existing equity
curve by 2 and checking for ruin. That is a mathematical projection, not a
backtest of the system that is actually running. It ignores:

  * the drawdown circuit breaker (written and deployed the same hour, never tested)
  * vol targeting at 0.30 interacting with the per-position leverage cap
  * margin limits on how many positions can actually be held

The breaker could plausibly HURT returns by de-risking into recoveries, or help by
avoiding the deep drawdown. Only a run settles it.

Deployed config under test:
    2-day holds, 10 concurrent, decay exit @0.45, quarter-Kelly sizing,
    TARGET_VOL 0.30, MAX_LEVERAGE 3.0 per position,
    breaker: -20% -> 0.75x size, -30% -> 0.5x, -40% -> halt entries

    python -m app.verify_2x
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .swing import UNIVERSE, fetch_daily, build_daily_features, walk_forward, CAPITAL
from .enhance import COST_PER_SIDE, summarize
from .reality_stack import deflated_sharpe

log = logging.getLogger("verify2x")

HZ, HZ_CAP = 2, 5
PT, SL = 2.0, 1.0
RISK = 0.01
THRESH = 0.55
VOL_LOOKBACK = 20
N_TRIALS = 140          # every configuration tried across the whole project


def simulate(sigs, *, target_vol=0.30, max_lev=3.0, max_concurrent=10,
             breaker=True, decay_thresh=0.45, kelly=True,
             cost=COST_PER_SIDE, stop_slip_R=0.10):
    all_days = sorted(set().union(*[set(s.index) for s in sigs.values()]))
    eq = CAPITAL
    peak = CAPITAL
    open_pos = {}
    trades = []
    curve = []
    recent = []
    halted_days = 0

    for day in all_days:
        day_start = eq

        # ---- manage / exit ----
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
            decayed = (float(row["meta_p"]) < decay_thresh
                       or int(row["side"]) != p["side"])
            timeout = p["held"] >= HZ_CAP
            if hit_tp or hit_sl or decayed or timeout:
                risk_px = abs(p["entry"] - p["sl"])
                if hit_sl:
                    px, reason = p["sl"] - p["side"] * stop_slip_R * risk_px, "sl"
                elif hit_tp:
                    px, reason = p["tp"], "tp"
                else:
                    px, reason = float(row["close"]), ("decay" if decayed else "time")
                gross = p["side"] * (px - p["entry"]) * p["qty"]
                fees = (p["entry"] + px) * p["qty"] * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": p["side"],
                               "days": p["held"], "reason": reason})
                del open_pos[sym]

        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak else 0.0

        # ---- sizing ----
        mult = 1.0
        if len(recent) >= VOL_LOOKBACK:
            rv = float(np.std(recent[-VOL_LOOKBACK:]) * np.sqrt(252))
            if rv > 1e-6:
                mult = float(np.clip(target_vol / rv, 0.25, max_lev))

        blocked = False
        if breaker:
            if dd <= -0.40:
                blocked = True
                halted_days += 1
            elif dd <= -0.30:
                mult *= 0.5
            elif dd <= -0.20:
                mult *= 0.75

        # ---- entries ----
        if not blocked and len(open_pos) < max_concurrent:
            cands = []
            for sym, s in sigs.items():
                if sym in open_pos or day not in s.index:
                    continue
                i = s.index.get_loc(day)
                if i < 1:
                    continue
                prev = s.iloc[i - 1]
                if prev["meta_p"] >= THRESH:
                    cands.append((float(prev["meta_p"]), sym, s.iloc[i], prev))
            cands.sort(reverse=True, key=lambda x: x[0])
            for mp, sym, row, prev in cands:
                if len(open_pos) >= max_concurrent:
                    break
                v = max(float(prev["vol"]), 1e-4) * np.sqrt(HZ)
                entry = float(row["close"])
                stop_dist = SL * v * entry
                r = RISK
                if kelly:
                    b = PT / SL
                    pw = float(np.clip(mp, 0.30, 0.80))
                    k = max((b * pw - (1 - pw)) / b, 0.0)
                    r *= float(np.clip(k * 0.25 / RISK, 0.5, 2.0))
                qty = (eq * r * mult) / max(stop_dist, 1e-6)
                qty = min(qty, (eq * max_lev) / entry)
                if qty <= 0:
                    continue
                side = int(prev["side"])
                open_pos[sym] = {"side": side, "entry": entry, "qty": qty,
                                 "held": 0,
                                 "tp": entry * (1 + side * PT * v),
                                 "sl": entry * (1 - side * SL * v)}

        recent.append((eq - day_start) / max(day_start, 1e-9))
        curve.append({"day": day, "equity": eq})
        if eq <= CAPITAL * 0.05:
            log.warning("account destroyed, stopping")
            break

    return (pd.DataFrame(trades),
            pd.DataFrame(curve).set_index("day")["equity"], halted_days)


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
        sg = walk_forward(d2, f2, HZ)
        if sg is not None:
            sigs[sym] = sg
    log.info("signals for %d symbols", len(sigs))

    VARIANTS = [
        ("1x validated (tv .15)",  dict(target_vol=0.15, max_lev=1.5, breaker=False)),
        ("2x no breaker",          dict(target_vol=0.30, max_lev=3.0, breaker=False)),
        ("2x WITH breaker (LIVE)", dict(target_vol=0.30, max_lev=3.0, breaker=True)),
        ("1.5x with breaker",      dict(target_vol=0.225, max_lev=2.25, breaker=True)),
        ("2.5x with breaker",      dict(target_vol=0.375, max_lev=3.75, breaker=True)),
    ]

    rows = []
    for name, kw in VARIANTS:
        t, c, halted = simulate(sigs, **kw)
        if t.empty:
            continue
        r = c.pct_change().dropna()
        sh = float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
        roll = c.cummax()
        ddp = (c - roll) / roll
        tot = float(c.iloc[-1] / CAPITAL - 1) * 100
        ann = ((1 + tot / 100) ** (365 / days) - 1) * 100
        w = t.loc[t.pnl > 0, "pnl"]
        l = t.loc[t.pnl <= 0, "pnl"]
        aw = float(w.mean()) if len(w) else 0.0
        al = float(-l.mean()) if len(l) else 0.0
        be = al / (aw + al) * 100 if (aw + al) > 0 else None
        win = float((t.pnl > 0).mean()) * 100
        dsr = deflated_sharpe(r.values, sh, N_TRIALS) or 0
        rows.append({"config": name, "trades": len(t),
                     "annual_%": round(ann, 1),
                     "daily_%": round(((1 + ann / 100) ** (1 / 252) - 1) * 100, 3),
                     "sharpe": round(sh, 2),
                     "edge": round(win - be, 2) if be else None,
                     "maxDD_%": round(float(ddp.min()) * 100, 1),
                     "ruin": "YES" if float(ddp.min()) <= -0.50 else "no",
                     "halt_days": halted,
                     "DSR": round(dsr, 3),
                     "final_x": round(float(c.iloc[-1] / CAPITAL), 1)})
        log.info("%-24s %.1f%%/yr Sh %.2f dd %.1f%% DSR %.3f",
                 name, ann, sh, float(ddp.min()) * 100, dsr)

    d = pd.DataFrame(rows)
    print("\n" + "=" * 116)
    print("DEPLOYED 2x CONFIG -- full backtest, 10y, 87 symbols, all costs")
    print("=" * 116)
    print(d.to_string(index=False))
    print(f"\nSPY: {spy_ann:.1f}%/yr  Sharpe {spy_sh:.2f}")

    live = d[d["config"].str.contains("LIVE")]
    if len(live):
        r = live.iloc[0]
        checks = [
            ("annual > 0", r["annual_%"] > 0, f"{r['annual_%']}%"),
            ("edge > 1.0", (r["edge"] or -9) > 1.0, f"{r['edge']}"),
            ("DSR >= 0.90", r["DSR"] >= 0.90, f"{r['DSR']}"),
            ("beats SPY Sharpe", r["sharpe"] > spy_sh,
             f"{r['sharpe']} vs {spy_sh:.2f}"),
            ("no ruin", r["ruin"] == "no", r["ruin"]),
        ]
        print("\nGATE ON THE LIVE CONFIG:")
        for n, ok, v in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {n:<20} {v}")
        print("\n  ==> " + ("GO" if all(o for _, o, _ in checks) else "NO-GO"))
    d.to_json("state/verify_2x.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

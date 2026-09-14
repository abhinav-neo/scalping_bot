"""
Entry-timing comparison: close vs next open vs midday.

The signal is ready at yesterday's close, but the live engine waits until 15:40 ET
today to act. That matches the backtest -- but waiting a full session is a real
cost if the move happens intraday. Worth measuring rather than assuming.

Three timings, same signals, same costs, same 2-day hold:

  CLOSE     enter at today's close  (what the backtest and live engine do)
  OPEN      enter at today's open   (acts ~6.5h earlier, captures the day-1 move,
                                     but eats opening-auction noise)
  MIDDAY    enter around midday     (approximated as the day's (open+close)/2)

The hold is measured in sessions either way, so entering at the open adds most of
a day of exposure per trade -- more return if the edge is real, more risk either
way. The comparison has to be risk-adjusted, not just return.

    python -m app.entry_timing
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .swing import UNIVERSE, fetch_daily, build_daily_features, walk_forward, CAPITAL
from .enhance import COST_PER_SIDE
from .reality_stack import deflated_sharpe

log = logging.getLogger("entry_timing")

HZ, HZ_CAP = 2, 5
PT, SL = 2.0, 1.0
RISK = 0.01
THRESH = 0.55
TARGET_VOL = 0.30
MAX_LEV = 3.0
VOL_LOOKBACK = 20
N_TRIALS = 150


def simulate(sigs, opens, timing="close", max_concurrent=10,
             cost=COST_PER_SIDE, stop_slip_R=0.10, breaker=True):
    all_days = sorted(set().union(*[set(s.index) for s in sigs.values()]))
    eq = CAPITAL
    peak = CAPITAL
    open_pos = {}
    trades = []
    curve = []
    recent = []

    for day in all_days:
        day_start = eq

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
            decayed = (float(row["meta_p"]) < 0.45
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
                trades.append({"pnl": gross - fees, "days": p["held"],
                               "reason": reason, "side": p["side"]})
                del open_pos[sym]

        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak else 0.0

        mult = 1.0
        if len(recent) >= VOL_LOOKBACK:
            rv = float(np.std(recent[-VOL_LOOKBACK:]) * np.sqrt(252))
            if rv > 1e-6:
                mult = float(np.clip(TARGET_VOL / rv, 0.25, MAX_LEV))
        blocked = False
        if breaker:
            if dd <= -0.40:
                blocked = True
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
                prev = s.iloc[i - 1]          # signal from the prior close
                if prev["meta_p"] >= THRESH:
                    cands.append((float(prev["meta_p"]), sym, s.iloc[i], prev))
            cands.sort(reverse=True, key=lambda x: x[0])
            for mp, sym, row, prev in cands:
                if len(open_pos) >= max_concurrent:
                    break
                c_px = float(row["close"])
                o_px = float(opens[sym].get(day, c_px))
                if timing == "open":
                    entry = o_px
                elif timing == "midday":
                    entry = (o_px + c_px) / 2.0
                else:
                    entry = c_px
                v = max(float(prev["vol"]), 1e-4) * np.sqrt(HZ)
                stop_dist = SL * v * entry
                b = PT / SL
                pw = float(np.clip(mp, 0.30, 0.80))
                k = max((b * pw - (1 - pw)) / b, 0.0)
                r = RISK * float(np.clip(k * 0.25 / RISK, 0.5, 2.0))
                qty = (eq * r * mult) / max(stop_dist, 1e-6)
                qty = min(qty, (eq * MAX_LEV) / entry)
                if qty <= 0:
                    continue
                side = int(prev["side"])
                pos = {"side": side, "entry": entry, "qty": qty, "held": 0,
                       "tp": entry * (1 + side * PT * v),
                       "sl": entry * (1 - side * SL * v)}

                # If we entered before the close, the position is exposed to the
                # REST OF TODAY. The exit loop only starts checking tomorrow, so
                # without this the open/midday variants get the day-1 move with no
                # day-1 stop risk -- a free option that inflated them 4.5x.
                if timing in ("open", "midday"):
                    hi, lo = float(row["high"]), float(row["low"])
                    hit_sl = ((side == 1 and lo <= pos["sl"]) or
                              (side == -1 and hi >= pos["sl"]))
                    hit_tp = ((side == 1 and hi >= pos["tp"]) or
                              (side == -1 and lo <= pos["tp"]))
                    if hit_sl or hit_tp:
                        risk_px = abs(entry - pos["sl"])
                        if hit_sl:      # assume the stop fills first, worst case
                            px, reason = pos["sl"] - side * stop_slip_R * risk_px, "sl"
                        else:
                            px, reason = pos["tp"], "tp"
                        gross = side * (px - entry) * qty
                        fees = (entry + px) * qty * cost
                        eq += gross - fees
                        trades.append({"pnl": gross - fees, "days": 0,
                                       "reason": reason, "side": side})
                        continue
                open_pos[sym] = pos

        recent.append((eq - day_start) / max(day_start, 1e-9))
        curve.append({"day": day, "equity": eq})
        if eq <= CAPITAL * 0.05:
            break

    return pd.DataFrame(trades), pd.DataFrame(curve).set_index("day")["equity"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    hist = fetch_daily(UNIVERSE, years=10.0)
    spy = hist["SPY"]["close"]
    days = (spy.index[-1] - spy.index[0]).days

    sigs, opens = {}, {}
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
            opens[sym] = d2["open"]
    log.info("signals for %d symbols", len(sigs))

    rows = []
    for timing in ("close", "open", "midday"):
        t, c = simulate(sigs, opens, timing=timing)
        if t.empty:
            continue
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
        rows.append({"timing": timing, "trades": len(t),
                     "annual_%": round(ann, 1),
                     "sharpe": round(sh, 2),
                     "win_%": round(win, 1),
                     "edge": round(win - be, 2) if be else None,
                     "maxDD_%": round(ddp, 1),
                     "return_per_risk": round(ann / abs(ddp), 2) if ddp else None,
                     "DSR": round(deflated_sharpe(r.values, sh, N_TRIALS) or 0, 3)})
        log.info("%-7s %.1f%%/yr Sh %.2f dd %.1f%% DSR %.3f",
                 timing, ann, sh, ddp, rows[-1]["DSR"])

    d = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("ENTRY TIMING -- same signals, same costs, 10y, full reality stack")
    print("  close = what the live engine does (waits until 15:40 ET)")
    print("=" * 100)
    print(d.to_string(index=False))

    if len(d) > 1:
        base = d[d.timing == "close"].iloc[0]
        print("\nVERSUS WAITING FOR THE CLOSE:")
        for _, r in d[d.timing != "close"].iterrows():
            print(f"  {r['timing']:<8} return {r['annual_%'] - base['annual_%']:+6.1f} pp | "
                  f"Sharpe {r['sharpe'] - base['sharpe']:+.2f} | "
                  f"drawdown {r['maxDD_%'] - base['maxDD_%']:+.1f} pp")
        best = d.loc[d["sharpe"].idxmax()]
        print(f"\n  best risk-adjusted: {best['timing']} "
              f"({best['annual_%']}%/yr, Sharpe {best['sharpe']})")
    d.to_json("state/entry_timing.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

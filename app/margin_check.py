"""
Does the margin constraint invalidate the backtest?

The live engine capped each position at 3x equity instead of capping the
portfolio, so gross exposure reached 3.7x on four positions and the broker
rejected every new order. The backtest contains the SAME per-position cap and
never enforces margin, so it simulated sizes Alpaca will not accept.

The question this answers: how much of the 143.8%/yr depended on leverage that
does not exist?

  UNCAPPED   what the original backtest did (per-position cap only)
  CAPPED     gross exposure across all open positions <= LEVERAGE x equity,
             which is what the broker actually enforces

If the capped result is materially lower, the headline figure was inflated and
should be replaced. Reported for 1x and 2x so the effect is visible at both.

    python -m app.margin_check
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .swing import UNIVERSE, fetch_daily, build_daily_features, walk_forward, CAPITAL
from .enhance import COST_PER_SIDE
from .reality_stack import deflated_sharpe

log = logging.getLogger("margin_check")

HZ, HZ_CAP = 2, 5
PT, SL = 2.0, 1.0
RISK = 0.01
THRESH = 0.55
VOL_LOOKBACK = 20
N_TRIALS = 160


def simulate(sigs, *, leverage=2.0, enforce_margin=True, max_concurrent=10,
             per_pos_cap=3.0, cost=COST_PER_SIDE, stop_slip_R=0.10):
    """
    enforce_margin=False reproduces the original backtest: each position capped at
    per_pos_cap x equity, with no portfolio limit.
    enforce_margin=True  applies the real constraint: total gross exposure across
    open positions cannot exceed leverage x equity.
    """
    all_days = sorted(set().union(*[set(s.index) for s in sigs.values()]))
    eq = CAPITAL
    peak = CAPITAL
    open_pos = {}
    trades = []
    curve = []
    recent = []
    rejected = 0
    gross_peak = 0.0

    for day in all_days:
        day_start = eq

        for sym in list(open_pos):
            p = open_pos[sym]
            s = sigs.get(sym)
            if s is None or day not in s.index:
                continue
            row = s.loc[day]
            p["held"] += 1
            p["last"] = float(row["close"])
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
                mult = float(np.clip((leverage * 0.15) / rv, 0.25, per_pos_cap))
        blocked = False
        if dd <= -0.40:
            blocked = True
        elif dd <= -0.30:
            mult *= 0.5
        elif dd <= -0.20:
            mult *= 0.75

        # current gross exposure, marked to the latest close
        gross_now = sum(abs(p["qty"] * p.get("last", p["entry"]))
                        for p in open_pos.values())
        gross_peak = max(gross_peak, gross_now / max(eq, 1e-9))

        if not blocked and len(open_pos) < max_concurrent:
            slots = max_concurrent - len(open_pos)
            budget = max(leverage * eq - gross_now, 0.0)
            per_slot = budget / max(slots, 1)

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
                b = PT / SL
                pw = float(np.clip(mp, 0.30, 0.80))
                k = max((b * pw - (1 - pw)) / b, 0.0)
                r = RISK * float(np.clip(k * 0.25 / RISK, 0.5, 2.0))
                qty = (eq * r * mult) / max(stop_dist, 1e-6)

                if enforce_margin:
                    room = max(leverage * eq - gross_now, 0.0)
                    if room <= 0:
                        rejected += 1
                        continue
                    qty = min(qty, min(per_slot, room) / entry)
                else:
                    qty = min(qty, (eq * per_pos_cap) / entry)

                qty = int(qty)
                if qty <= 0:
                    if enforce_margin:
                        rejected += 1
                    continue
                side = int(prev["side"])
                open_pos[sym] = {"side": side, "entry": entry, "qty": qty,
                                 "held": 0, "last": entry,
                                 "tp": entry * (1 + side * PT * v),
                                 "sl": entry * (1 - side * SL * v)}
                gross_now += qty * entry

        recent.append((eq - day_start) / max(day_start, 1e-9))
        curve.append({"day": day, "equity": eq})
        if eq <= CAPITAL * 0.05:
            break

    return (pd.DataFrame(trades),
            pd.DataFrame(curve).set_index("day")["equity"],
            {"rejected": rejected, "peak_gross_x": round(gross_peak, 2)})


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

    rows = []
    for lev in (1.0, 2.0):
        for enforce in (False, True):
            t, c, meta = simulate(sigs, leverage=lev, enforce_margin=enforce)
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
            rows.append({
                "leverage": lev,
                "margin": "enforced" if enforce else "IGNORED (old)",
                "trades": len(t),
                "annual_%": round(ann, 1),
                "sharpe": round(sh, 2),
                "win_%": round(win, 1),
                "edge": round(win - be, 2) if be else None,
                "maxDD_%": round(ddp, 1),
                "peak_gross_x": meta["peak_gross_x"],
                "rejected": meta["rejected"],
                "DSR": round(deflated_sharpe(r.values, sh, N_TRIALS) or 0, 3)})
            log.info("lev %.1f margin=%-8s -> %.1f%%/yr Sh %.2f peak_gross %.2fx",
                     lev, "on" if enforce else "off", ann, sh, meta["peak_gross_x"])

    d = pd.DataFrame(rows)
    print("\n" + "=" * 108)
    print("MARGIN CONSTRAINT -- does it invalidate the backtest?")
    print("  IGNORED = the original method (per-position cap only, no portfolio limit)")
    print("  enforced = gross exposure <= leverage x equity, what the broker requires")
    print("=" * 108)
    print(d.to_string(index=False))
    print(f"\nSPY: {spy_ann:.1f}%/yr  Sharpe {spy_sh:.2f}")

    for lev in (1.0, 2.0):
        old = d[(d.leverage == lev) & (d.margin.str.startswith("IGNORED"))]
        new = d[(d.leverage == lev) & (d.margin == "enforced")]
        if len(old) and len(new):
            o, n = old.iloc[0], new.iloc[0]
            print(f"\n  {lev:.0f}x leverage: {o['annual_%']}%/yr -> {n['annual_%']}%/yr "
                  f"({n['annual_%'] - o['annual_%']:+.1f} pp), "
                  f"Sharpe {o['sharpe']} -> {n['sharpe']}, "
                  f"peak gross {o['peak_gross_x']}x -> {n['peak_gross_x']}x")
    print("\n  Peak gross above the leverage setting in the IGNORED rows is exactly")
    print("  the exposure the broker would have refused.")
    d.to_json("state/margin_check.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

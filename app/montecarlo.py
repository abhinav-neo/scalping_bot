"""
Monte Carlo simulation of the live strategy.

Win rate is the decision variable. With a profit target at pt*vol and a stop at
sl*vol, breakeven sits at sl/(pt+sl) -- 58.8% at the current 0.7/1.0 geometry.
The backtest projected ~58.6%; the first live session delivered 51.7%. This
simulates what each of those worlds actually looks like over time.

Unlike a naive random walk, this models the machinery that truncates bad outcomes:
  * the -3% daily kill switch (stops trading for the rest of that day)
  * the -20% total drawdown halt (stops the bot entirely)
  * the same-day-flat constraint (no overnight gaps)

    python -m app.montecarlo
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S

log = logging.getLogger("montecarlo")

N_PATHS = 5000
TRADES_PER_DAY = 25          # horizon-3 projection, ~6/symbol across 4 names
DAYS = {"1 month": 21, "3 months": 63, "1 year": 252}

# Trade economics measured live on 2026-07-30 (29 closed trades), as a fraction
# of equity so results compound rather than assuming fixed dollar stakes.
AVG_WIN_FRAC = 0.0026        # +$12.55 on ~$4,833
AVG_LOSS_FRAC = 0.0029       # -$14.07 on ~$4,833
WIN_SD = 0.4                 # dispersion around the average, as a fraction of it


def simulate(win_rate, days, n_paths, rng,
             start=5000.0, tpd=TRADES_PER_DAY,
             daily_kill=None, halt=None):
    daily_kill = S.daily_loss_kill if daily_kill is None else daily_kill
    halt = S.max_total_drawdown if halt is None else halt

    eq = np.full(n_paths, start)
    peak = eq.copy()
    halted = np.zeros(n_paths, dtype=bool)
    max_dd = np.zeros(n_paths)
    days_traded = np.zeros(n_paths)
    kill_days = np.zeros(n_paths)

    for _ in range(days):
        day_start = eq.copy()
        locked = halted.copy()
        for _ in range(tpd):
            live = ~locked
            if not live.any():
                break
            wins = rng.random(n_paths) < win_rate
            # lognormal-ish dispersion so occasional trades are much larger
            mag = np.abs(rng.normal(1.0, WIN_SD, n_paths))
            pnl = np.where(wins, AVG_WIN_FRAC * mag, -AVG_LOSS_FRAC * mag)
            eq = np.where(live, eq * (1 + pnl), eq)

            # daily kill switch: latches for the remainder of the session
            hit = live & (eq <= day_start * (1 - daily_kill))
            locked |= hit

            peak = np.maximum(peak, eq)
            dd = (eq - peak) / peak
            max_dd = np.minimum(max_dd, dd)
            newly_halted = (~halted) & (dd <= -halt)
            halted |= newly_halted
            locked |= halted

        kill_days += (locked & ~halted).astype(int)
        days_traded += (~halted).astype(int)

    return {"equity": eq, "max_dd": max_dd, "halted": halted,
            "kill_days": kill_days, "days_traded": days_traded}


def summarize(res, start, label, win_rate, days):
    eq = res["equity"]
    ret = (eq / start - 1) * 100
    p = np.percentile(ret, [5, 25, 50, 75, 95])
    return {
        "win_rate": f"{win_rate*100:.1f}%",
        "horizon": label,
        "median_%": round(float(p[2]), 1),
        "p5_%": round(float(p[0]), 1),
        "p25_%": round(float(p[1]), 1),
        "p75_%": round(float(p[3]), 1),
        "p95_%": round(float(p[4]), 1),
        "P(profit)": f"{100*float((ret>0).mean()):.0f}%",
        "P(halt)": f"{100*float(res['halted'].mean()):.0f}%",
        "med_maxDD_%": round(float(np.median(res["max_dd"]))*100, 1),
        "med_equity": round(float(np.median(eq))),
        "kill_days": round(float(np.median(res["kill_days"])), 1),
    }


def main():
    global AVG_WIN_FRAC
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(42)
    start = S.starting_capital
    pt, sl = S.pt_mult, S.sl_mult
    breakeven = sl / (pt + sl)

    print("=" * 96)
    print("MONTE CARLO -- current live config")
    print(f"  capital ${start:,.0f} | pt/sl {pt}/{sl} | breakeven win rate "
          f"{breakeven*100:.1f}% | {TRADES_PER_DAY} trades/day")
    print(f"  avg win {AVG_WIN_FRAC*100:.3f}% of equity | avg loss {AVG_LOSS_FRAC*100:.3f}% "
          f"| {N_PATHS:,} paths")
    print(f"  guards: -{S.daily_loss_kill*100:.0f}% daily kill, "
          f"-{S.max_total_drawdown*100:.0f}% halt")
    print("=" * 96)

    scenarios = [
        (0.517, "live day 1"),
        (0.540, "modest"),
        (0.570, "near breakeven"),
        (breakeven, "breakeven"),
        (0.586, "backtest"),
        (0.610, "good"),
        (0.640, "excellent"),
    ]

    rows = []
    for wr, tag in scenarios:
        for label, d in DAYS.items():
            res = simulate(wr, d, N_PATHS, rng, start=start)
            r = summarize(res, start, label, wr, d)
            r["scenario"] = tag
            rows.append(r)

    df = pd.DataFrame(rows)
    for label in DAYS:
        sub = df[df.horizon == label][
            ["scenario", "win_rate", "median_%", "p5_%", "p95_%",
             "P(profit)", "P(halt)", "med_maxDD_%", "med_equity"]]
        print(f"\n--- {label.upper()} ---")
        print(sub.to_string(index=False))

    # --- how sensitive is the outcome to win rate? ---
    print("\n" + "=" * 96)
    print("SENSITIVITY: 1-year median return vs win rate (1 pp steps)")
    print("=" * 96)
    sens = []
    for wr in np.arange(0.50, 0.66, 0.01):
        res = simulate(wr, 252, 2000, rng, start=start)
        ret = (res["equity"] / start - 1) * 100
        sens.append({"win_rate": f"{wr*100:.0f}%",
                     "median_%": round(float(np.median(ret)), 1),
                     "P(profit)": f"{100*float((ret>0).mean()):.0f}%",
                     "P(halt)": f"{100*float(res['halted'].mean()):.0f}%"})
    print(pd.DataFrame(sens).to_string(index=False))

    # --- symmetric barriers for comparison (pt=sl=1.0 -> breakeven 50%) ---
    print("\n" + "=" * 96)
    print("COMPARISON: symmetric barriers (pt=sl=1.0, breakeven 50%)")
    print("  same win rate, but wins and losses are equal size")
    print("=" * 96)
    saved = AVG_WIN_FRAC
    AVG_WIN_FRAC = AVG_LOSS_FRAC          # symmetric payoff
    sym = []
    for wr in (0.500, 0.517, 0.540, 0.570):
        res = simulate(wr, 252, 2000, rng, start=start)
        ret = (res["equity"] / start - 1) * 100
        sym.append({"win_rate": f"{wr*100:.1f}%",
                    "median_%": round(float(np.median(ret)), 1),
                    "P(profit)": f"{100*float((ret>0).mean()):.0f}%",
                    "P(halt)": f"{100*float(res['halted'].mean()):.0f}%"})
    AVG_WIN_FRAC = saved
    print(pd.DataFrame(sym).to_string(index=False))
    print("\nNOTE: fewer trades result from symmetric barriers (wider profit target),")
    print("      so absolute returns are not directly comparable -- the useful signal")
    print("      is how much less sensitive the outcome is to win rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

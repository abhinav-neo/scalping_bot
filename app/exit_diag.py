"""
Exit-quality diagnostic.

Question: with pt=1.5 and sl=1.0 the average WIN should be 1.5x the average LOSS.
Live it is the other way round. This measures where the asymmetry actually comes
from instead of guessing.

Method: reconstruct round trips from broker fills, then express every trade as an
R-multiple (profit divided by the intended risk, i.e. the stop distance). A trade
that exits exactly at its target is +1.5R; one that exits exactly at its stop is
-1.0R. Deviations tell us:

  * stops averaging worse than -1.0R      -> exit slippage on stops
  * targets averaging under +1.5R          -> exit slippage on targets
  * many trades exiting at the TIME barrier -> the target is too far to reach
    inside the horizon, so winners get truncated while losers still take the
    full stop. That alone inverts the payoff ratio.

    python -m app.exit_diag
"""
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from .settings import S
from .analytics import Analytics

log = logging.getLogger("exit_diag")


def load_intents(state_dir):
    """entry events carry the tp/sl we intended, keyed by symbol in time order."""
    path = os.path.join(state_dir, "trades.jsonl")
    ev = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    ev.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return {}
    by = defaultdict(list)
    for e in ev:
        if e.get("event") == "entry" and e.get("tp") and e.get("sl"):
            by[e["symbol"]].append(e)
    return by


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = Analytics(S)
    trips = a.round_trips(days=7)
    if not trips:
        print("no closed trades yet")
        return 0

    intents = load_intents(S.state_dir)
    rows = []
    for t in trips:
        sym = t["symbol"]
        cand = intents.get(sym, [])
        # match the intent whose entry price is closest to the realised entry
        best, bestd = None, 1e9
        for e in cand:
            d = abs(float(e.get("price", 0)) - t["entry"])
            if d < bestd:
                best, bestd = e, d
        if best is None or bestd / max(t["entry"], 1e-9) > 0.02:
            continue
        side = t["side"]
        entry = t["entry"]
        tp, sl = float(best["tp"]), float(best["sl"])
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        target = abs(tp - entry)
        realised = (t["exit"] - entry) * side
        rows.append({
            "symbol": sym, "side": side,
            "entry": entry, "exit": t["exit"], "tp": tp, "sl": sl,
            "risk_pct": risk / entry * 100,
            "target_pct": target / entry * 100,
            "target_R": target / risk,
            "realised_R": realised / risk,
            "pnl": t["pnl"],
            "held_min": t["held_min"],
        })

    if not rows:
        print("could not match trades to intents")
        return 0

    d = pd.DataFrame(rows)

    # classify how each trade ended relative to its barriers
    def cls(r):
        tol = 0.15
        if r["realised_R"] >= r["target_R"] - tol:
            return "hit_target"
        if r["realised_R"] <= -1 + tol:
            return "hit_stop"
        return "time_exit"
    d["outcome"] = d.apply(cls, axis=1)

    print("=" * 88)
    print("EXIT QUALITY -- everything in R (1R = the intended stop distance)")
    print(f"  intended target = {S.pt_mult / S.sl_mult:.2f}R   "
          f"intended stop = -1.00R   n = {len(d)}")
    print("=" * 88)

    g = d.groupby("outcome").agg(
        n=("realised_R", "size"),
        mean_R=("realised_R", "mean"),
        median_R=("realised_R", "median"),
        mean_pnl=("pnl", "mean"),
        mean_hold=("held_min", "mean")).round(3)
    g["share_%"] = (g["n"] / len(d) * 100).round(1)
    print("\nBY OUTCOME:")
    print(g.to_string())

    wins = d[d.pnl > 0]
    losses = d[d.pnl <= 0]
    print(f"\nPAYOFF:")
    print(f"  avg win   {wins['realised_R'].mean():+.3f}R  (${wins['pnl'].mean():.2f})"
          f"  n={len(wins)}")
    print(f"  avg loss  {losses['realised_R'].mean():+.3f}R  (${losses['pnl'].mean():.2f})"
          f"  n={len(losses)}")
    ratio = abs(wins['pnl'].mean() / losses['pnl'].mean()) if len(losses) else 0
    print(f"  realised win/loss ratio {ratio:.2f}  "
          f"(intended {S.pt_mult / S.sl_mult:.2f})")
    be = abs(losses['pnl'].mean()) / (wins['pnl'].mean() + abs(losses['pnl'].mean())) * 100 \
        if len(wins) and len(losses) else None
    if be:
        print(f"  => breakeven win rate {be:.1f}%   actual {100*len(wins)/len(d):.1f}%")

    # --- slippage: how far past the barrier did we actually exit? ---
    st = d[d.outcome == "hit_stop"]
    tg = d[d.outcome == "hit_target"]
    print("\nSLIPPAGE vs the barrier price:")
    if len(st):
        print(f"  stops   : mean {st['realised_R'].mean():+.3f}R vs -1.000R intended"
              f"  -> {(st['realised_R'].mean() + 1) * 100:+.1f}% of 1R")
    if len(tg):
        print(f"  targets : mean {tg['realised_R'].mean():+.3f}R vs "
              f"{d['target_R'].mean():+.3f}R intended"
              f"  -> {(tg['realised_R'].mean() - d['target_R'].mean()) * 100:+.1f}% of 1R")

    # --- the key question: are winners being truncated by the clock? ---
    te = d[d.outcome == "time_exit"]
    print(f"\nTIME-BARRIER EXITS: {len(te)} ({100*len(te)/len(d):.0f}% of trades)")
    if len(te):
        pos = te[te.realised_R > 0]
        neg = te[te.realised_R <= 0]
        print(f"  positive at timeout: {len(pos)} mean {pos['realised_R'].mean():+.3f}R"
              if len(pos) else "  positive at timeout: 0")
        print(f"  negative at timeout: {len(neg)} mean {neg['realised_R'].mean():+.3f}R"
              if len(neg) else "  negative at timeout: 0")
        print("  -> if most trades end here, the target is too far to reach inside")
        print("     the horizon: winners get cut short while losers still pay a full stop.")

    print("\nBY SYMBOL:")
    bs = d.groupby("symbol").agg(n=("pnl", "size"), mean_R=("realised_R", "mean"),
                                 net=("pnl", "sum"),
                                 win_pct=("pnl", lambda x: (x > 0).mean() * 100),
                                 hold=("held_min", "mean")).round(2)
    print(bs.to_string())

    print("\nBY SIDE:")
    bside = d.groupby("side").agg(n=("pnl", "size"), mean_R=("realised_R", "mean"),
                                  net=("pnl", "sum"),
                                  win_pct=("pnl", lambda x: (x > 0).mean() * 100)).round(2)
    print(bside.to_string())

    d.to_json(os.path.join(S.state_dir, "exit_diag.json"), orient="records")
    print(f"\nsaved {S.state_dir}/exit_diag.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

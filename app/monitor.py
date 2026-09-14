"""
Paper-trading monitor for the one-month live test.

Purpose: answer one question honestly at the end of the month -- does live
behaviour match the backtest, or not?

The backtest projects 143.8%/yr compounding (57.5% fixed-capital), Sharpe 2.97,
-37.3% max drawdown, ~2.6 trades/day, 51.7% win rate. Those come from simulation.
This records what actually happens and compares, so the verdict rests on measured
fills rather than on a projection.

Tracked daily:
  * equity, return, drawdown from peak
  * trades opened and closed, win rate, average hold
  * realised slippage: fill price vs the price the signal was generated at
  * whether the circuit breaker engaged
  * running comparison against backtest expectations

SAMPLE-SIZE WARNING, stated up front so the result is not over-read:
  ~21 trading days at ~2.6 trades/day is roughly 55 trades. At a 51.7% true win
  rate, the 95% interval on observed win rate spans roughly 38-65%. A month can
  show -15% or +20% with the strategy behaving exactly as designed. This test can
  detect gross breakage (fills far off, win rate near 30%, wrong position counts).
  It CANNOT confirm the edge. Only a much longer run does that.

    python -m app.monitor            # append today's snapshot
    python -m app.monitor --report   # full month-to-date report
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .settings import S
from .broker import Broker
from .analytics import Analytics

log = logging.getLogger("monitor")

TRACK = os.path.join(S.state_dir, "monitor.jsonl")

# backtest expectations for the deployed 2x config
EXPECT = {
    "annual_pct": 143.8,
    "daily_pct": 0.36,
    "sharpe": 2.97,
    "maxdd_pct": -37.3,
    "win_pct": 51.7,
    "trades_per_day": 2.6,
    "avg_hold_days": 2.2,
}


def snapshot():
    b = Broker(S)
    a = b.account()
    pos = b.positions()
    an = Analytics(S)
    trips = an.round_trips(days=45)

    today = datetime.now(timezone.utc).date().isoformat()
    rec = {
        "day": today,
        "ts": datetime.now(timezone.utc).isoformat(),
        "equity": a["equity"],
        "cash": a["cash"],
        "open_positions": len(pos),
        "gross_exposure": sum(abs(p["market_value"]) for p in pos.values()),
        "unrealized": sum(p["unrealized_pl"] for p in pos.values()),
        "closed_trades_total": len(trips),
        "positions": {k: {"side": v["side"], "qty": v["qty"],
                          "entry": v["avg_entry"], "upl": v["unrealized_pl"]}
                      for k, v in pos.items()},
    }
    with open(TRACK, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    log.info("snapshot: equity %.2f, %d open, %d closed",
             rec["equity"], rec["open_positions"], rec["closed_trades_total"])
    return rec


def report():
    if not os.path.exists(TRACK):
        print("no monitor data yet -- run `python -m app.monitor` daily")
        return 1
    rows = []
    for line in open(TRACK):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if not rows:
        print("no records")
        return 1
    d = pd.DataFrame(rows).drop_duplicates("day", keep="last")
    d["day"] = pd.to_datetime(d["day"])
    d = d.sort_values("day").set_index("day")

    eq = d["equity"]
    start, cur = float(eq.iloc[0]), float(eq.iloc[-1])
    n = len(eq)
    r = eq.pct_change().dropna()
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min() * 100
    total = (cur / start - 1) * 100
    daily = ((1 + total / 100) ** (1 / max(n - 1, 1)) - 1) * 100 if n > 1 else 0.0
    ann = ((1 + daily / 100) ** 252 - 1) * 100
    sharpe = (float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
              if len(r) > 2 else None)

    an = Analytics(S)
    trips = an.round_trips(days=45)
    ex = an.execution_quality(S.state_dir, trips)
    st = an.stats(trips)

    print("=" * 92)
    print(f"PAPER TEST -- {n} sessions  ({eq.index[0].date()} to {eq.index[-1].date()})")
    print("=" * 92)
    print(f"  equity     : ${start:,.2f} -> ${cur:,.2f}  ({total:+.2f}%)")
    print(f"  daily avg  : {daily:+.3f}%   annualised {ann:+.1f}%")
    print(f"  max DD     : {dd:.2f}%")
    print(f"  Sharpe     : {sharpe:.2f}" if sharpe else "  Sharpe     : n/a")
    print(f"  open now   : {int(d['open_positions'].iloc[-1])} positions, "
          f"gross ${d['gross_exposure'].iloc[-1]:,.0f} "
          f"({d['gross_exposure'].iloc[-1] / cur:.2f}x equity)")

    print("\n  TRADES")
    if st.get("trades"):
        print(f"    closed     : {st['trades']}  ({st['trades'] / max(n,1):.1f}/day)")
        print(f"    win rate   : {st['win_rate']:.1f}%")
        print(f"    profit fac : {st['profit_factor']:.2f}")
        print(f"    avg hold   : {st['avg_hold']:.1f} min "
              f"({st['avg_hold'] / 60 / 6.5:.1f} sessions)")
    else:
        print("    none closed yet")

    if ex.get("matched"):
        print("\n  EXECUTION")
        print(f"    stop slippage : {ex['stop_slip_R']}R "
              f"(backtest assumed 0.10R)")
        print(f"    realised payoff: {ex['realised_ratio']}:1 (intended 2.0:1)")
        print(f"    exits: target {ex['pct_target']}% / stop {ex['pct_stop']}% "
              f"/ time {ex['pct_time']}%")

    print("\n  LIVE vs BACKTEST")
    comp = [
        ("daily return %", daily, EXPECT["daily_pct"]),
        ("annualised %", ann, EXPECT["annual_pct"]),
        ("sharpe", sharpe or 0, EXPECT["sharpe"]),
        ("max drawdown %", dd, EXPECT["maxdd_pct"]),
        ("win rate %", st.get("win_rate", 0), EXPECT["win_pct"]),
        ("trades/day", st.get("trades", 0) / max(n, 1), EXPECT["trades_per_day"]),
    ]
    print(f"    {'metric':<18}{'live':>12}{'backtest':>12}{'gap':>12}")
    for name, live, exp in comp:
        gap = live - exp
        print(f"    {name:<18}{live:>12.2f}{exp:>12.2f}{gap:>+12.2f}")

    print("\n  READ")
    if n < 10:
        print(f"    {n} sessions is far too few to judge. Watch for gross breakage")
        print("    (no trades, wrong position counts, slippage far above 0.10R),")
        print("    not for whether returns match.")
    else:
        print(f"    {n} sessions, {st.get('trades', 0)} trades. At a 51.7% true win")
        print("    rate the 95% band on observed win rate is roughly 38-65%, so a")
        print("    result inside that range neither confirms nor refutes the edge.")
        if ex.get("stop_slip_R") is not None and ex["stop_slip_R"] > 0.20:
            print(f"    WARNING: stop slippage {ex['stop_slip_R']}R is well above the")
            print("    0.10R the backtest assumed -- that alone would erode returns.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.report:
        return report()
    snapshot()
    return 0


if __name__ == "__main__":
    sys.exit(main())

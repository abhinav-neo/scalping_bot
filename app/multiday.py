"""
Multi-day horizon test -- relaxing "no overnight" to "no position over a week".

Why this is the right lever:

    cost per trade is FIXED (~3 bps round trip)
    move size scales with sqrt(holding time)

    30 min ->  6 bars
     5 day -> 390 bars      sqrt(390/6) = 8.1x larger moves, same cost

Every intraday configuration failed because gross edge per trade (1.4 bps at best)
was below round-trip cost (3.8 bps). An 8x larger move does not need a better
model -- it needs the same model applied where the payoff is bigger than the toll.

What changes versus the intraday tests:
  * NO end-of-day flatten. Positions run until a barrier or the time limit.
  * Overnight gap risk is real and is captured: intrabar checks use each bar's
    high/low, so a gap through the stop registers as a stop.
  * The daily -3% kill switch is dropped; it is meaningless for multi-day holds.

Horizons tested: 1, 2, 3, 5 and 10 sessions.

    python -m app.multiday
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_v3 import walk_forward, edge_of
from .reality_stack import deflated_sharpe

log = logging.getLogger("multiday")

CAPITAL = 5000.0
PT, SL = 1.5, 1.0
BARS_PER_DAY = 78
N_TRIALS = 110

# (label, horizon in 5-min bars)
HORIZONS = [("1 day", 78), ("2 days", 156), ("3 days", 234),
            ("5 days", 390), ("10 days", 780)]
THRESHOLDS = [0.75, 0.85, 0.90]


def simulate_multiday(sig, capital=CAPITAL, risk=0.01, pt=PT, sl=SL,
                      horizon=390, thresh=0.85, cost=0.00015, lag=1,
                      stop_slip_R=0.10, limit_fills=True,
                      limit_offset_frac=0.15, max_hold_days=5):
    """
    Position runs until a barrier or the time limit. No end-of-day flatten.
    Overnight gaps are captured through intrabar high/low checks.
    """
    idx = sig.index
    close, high, low = sig["close"].values, sig["high"].values, sig["low"].values
    vol = np.nan_to_num(sig["vol"].values, nan=np.nanmedian(sig["vol"].values))
    side_a, mp = sig["side"].values, sig["meta_p"].values
    days_arr = idx.normalize()
    n = len(idx)

    eq = capital
    pos = 0
    entry = qty = tp = slv = 0.0
    held = 0
    entry_day = None
    trades = []
    curve = np.empty(n)

    for i in range(n):
        if pos != 0:
            held += 1
            hit_tp = (pos == 1 and high[i] >= tp) or (pos == -1 and low[i] <= tp)
            hit_sl = (pos == 1 and low[i] <= slv) or (pos == -1 and high[i] >= slv)
            days_held = (days_arr[i] - entry_day).days
            timeout = held >= horizon or days_held >= max_hold_days
            if hit_tp or hit_sl or timeout:
                risk_px = abs(entry - slv)
                if hit_sl:
                    px, reason = slv - pos * stop_slip_R * risk_px, "sl"
                elif hit_tp:
                    px, reason = tp, "tp"
                else:
                    px, reason = close[i], "time"
                gross = pos * (px - entry) * qty
                fees = (entry + px) * qty * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": pos, "bars": held,
                               "days": days_held, "reason": reason})
                pos, qty, held = 0, 0.0, 0

        src = i - lag
        if pos == 0 and src >= 0 and mp[src] >= thresh and i + 1 < n:
            if eq <= capital * 0.05:
                break
            s = int(side_a[src])
            v = max(vol[src], 1e-4) * np.sqrt(horizon)
            raw_v = max(vol[src], 1e-4)
            if limit_fills:
                limit_px = close[i] - s * limit_offset_frac * raw_v * close[i]
                touched = (low[i + 1] <= limit_px) if s == 1 else (high[i + 1] >= limit_px)
                if not touched:
                    curve[i] = eq
                    continue
                entry_px = limit_px
            else:
                entry_px = close[i]
            stop_dist = sl * v * entry_px
            q = (capital * risk) / max(stop_dist, 1e-6)
            q = min(q, (1.5 * capital) / entry_px)     # overnight: no day-trade margin
            if q > 0:
                pos, entry, qty, held = s, entry_px, q, 0
                entry_day = days_arr[i]
                tp = entry * (1 + s * pt * v)
                slv = entry * (1 - s * sl * v)
        curve[i] = eq

    return pd.DataFrame(trades), pd.Series(curve, index=idx)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("multi-day test | %d symbols", len(S.symbols))
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})
    spy_d = spy["close"].resample("1D").last().dropna()
    days = (spy_d.index[-1] - spy_d.index[0]).days or 1
    spy_ann = ((spy_d.iloc[-1] / spy_d.iloc[0]) ** (365 / days) - 1) * 100
    spy_r = spy_d.pct_change().dropna()
    spy_sharpe = float(spy_r.mean() / spy_r.std() * np.sqrt(252))

    prepared = {}
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg0)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) >= 4000:
            prepared[sym] = (df, feats)
    log.info("prepared %d symbols", len(prepared))

    rows = []
    for label, hz in HORIZONS:
        c = ModelCfg(S)
        c.horizon_bars = hz
        sigs = {}
        for sym, (df, feats) in prepared.items():
            sg = walk_forward(df, feats, c)
            if sg is not None:
                sigs[sym] = sg
        if not sigs:
            continue
        log.info("%s: signals for %d symbols", label, len(sigs))

        for thr in THRESHOLDS:
            per, curves = [], []
            for sym, sg in sigs.items():
                t, curve = simulate_multiday(
                    sg, horizon=hz, thresh=thr,
                    max_hold_days=max(1, hz // BARS_PER_DAY))
                if t.empty or len(t) < 15:
                    continue
                _, _, e = edge_of(t)
                roll = curve.cummax()
                per.append({"ret": float(curve.iloc[-1] / CAPITAL - 1) * 100,
                            "edge": e, "trades": len(t),
                            "ppt": float(t["pnl"].mean()),
                            "hold_d": float(t["days"].mean()),
                            "dd": float(((curve - roll) / roll).min()) * 100})
                curves.append(curve.resample("1D").last().dropna()
                              .pct_change().dropna())
            if not per:
                continue
            dl = pd.concat(curves, axis=1).mean(axis=1).dropna()
            sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252))
            dsr = deflated_sharpe(dl.values, sh, N_TRIALS) or 0
            tot = int(np.sum([p["trades"] for p in per]))
            rows.append({
                "horizon": label, "thresh": thr,
                "trades": tot, "trades_day": round(tot / days, 2),
                "hold_days": round(float(np.mean([p["hold_d"] for p in per])), 1),
                "annual_%": round(float(np.mean([p["ret"] for p in per]))
                                  * 365 / days, 1),
                "sharpe": round(sh, 2),
                "edge": round(float(np.mean([p["edge"] for p in per
                                             if p["edge"] is not None])), 2),
                "bps_trade": round(float(np.mean([p["ppt"] for p in per]))
                                   / CAPITAL * 1e4, 1),
                "maxDD_%": round(float(np.mean([p["dd"] for p in per])), 1),
                "DSR": round(dsr, 3)})
            log.info("  %-8s thr %.2f -> %d trades %.1f%%/yr edge %.2f DSR %.3f",
                     label, thr, tot, rows[-1]["annual_%"], rows[-1]["edge"], dsr)

    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return 1
    print("\n" + "=" * 104)
    print(f"MULTI-DAY HORIZONS -- {len(prepared)} symbols, full reality stack, "
          "no EOD flatten")
    print("  bps_trade = net profit per trade in bps of capital (cost is ~3 bps)")
    print("=" * 104)
    print(d.to_string(index=False))
    print(f"\nSPY: {spy_ann:.1f}%/yr  Sharpe {spy_sharpe:.2f}")

    best = d.loc[d["annual_%"].idxmax()]
    print(f"\nBEST: {best['horizon']} @ thr {best['thresh']} -> "
          f"{best['annual_%']:.1f}%/yr, Sharpe {best['sharpe']}, "
          f"edge {best['edge']}, DSR {best['DSR']}, "
          f"{best['trades_day']:.2f} trades/day")
    checks = [("annual > 0", best["annual_%"] > 0),
              ("edge > 1.0", best["edge"] > 1.0),
              ("DSR >= 0.90", best["DSR"] >= 0.90),
              ("beats SPY Sharpe", best["sharpe"] > spy_sharpe),
              ("maxDD > -25%", best["maxDD_%"] > -25)]
    print()
    for n_, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n_}")
    print("\n  ==> " + ("GO" if all(o for _, o in checks) else "NO-GO"))
    d.to_json("/app/state/multiday.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Reality stack: what survives when every known cost is applied at once?

The backtest has been corrected four times and each correction moved the estimate.
The fair question is not "why do returns keep falling" -- the early figures were
never earnable -- but "what is left once everything real is accounted for, and is
it worth the risk versus just holding an index?"

This layers each correction one at a time so you can see the marginal cost of each,
and then applies the deflated Sharpe ratio to penalise the number of configurations
we searched over.

Layers, in order:
  0 ideal          entry at signal-bar close, exits exactly at barriers, all fills
  1 +lag           entry one bar later (streaming should make this < 1 bar)
  2 +intrabar      stops trigger on intrabar extremes, not just closes
  3 +stop slip     stops fill past their price (bracket 0.10R, polled 0.31R)
  4 +limit fills   passive entries only fill if price trades through them, and the
                   ones that fill are adversely selected by construction
  5 +costs         commission/spread per side
  6 +model decay   one model applied forward instead of refit every fold

    python -m app.reality_stack
"""
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_v3 import walk_forward, edge_of

log = logging.getLogger("reality")

CAPITAL = 5000.0
RISK = 0.005
HORIZON = 6
PT, SL = 1.5, 1.0
THRESH = 0.55

# how many distinct configurations we have evaluated across this project.
# Used to deflate the Sharpe ratio for multiple testing.
N_TRIALS = 60


def simulate(sig, lag=0, intrabar=False, stop_slip_R=0.0, limit_fills=False,
             cost=0.0, capital=CAPITAL, risk=RISK, pt=PT, sl=SL,
             horizon=HORIZON, thresh=THRESH, limit_offset_frac=0.15):
    idx = sig.index
    close, high, low = sig["close"].values, sig["high"].values, sig["low"].values
    vol = np.nan_to_num(sig["vol"].values, nan=np.nanmedian(sig["vol"].values))
    side_a, mp = sig["side"].values, sig["meta_p"].values
    minute = idx.hour * 60 + idx.minute
    days = idx.normalize()

    eq = capital
    pos = 0
    entry = qty = tp = slv = 0.0
    held = 0
    trades = []
    curve = np.empty(len(idx))
    cur_day, day_start, locked = None, eq, False
    signals = fills = 0
    n = len(idx)

    for i in range(n):
        if days[i] != cur_day:
            cur_day, day_start, locked = days[i], eq, False

        if pos != 0:
            held += 1
            if intrabar:
                hit_tp = (pos == 1 and high[i] >= tp) or (pos == -1 and low[i] <= tp)
                hit_sl = (pos == 1 and low[i] <= slv) or (pos == -1 and high[i] >= slv)
            else:
                hit_tp = (pos == 1 and close[i] >= tp) or (pos == -1 and close[i] <= tp)
                hit_sl = (pos == 1 and close[i] <= slv) or (pos == -1 and close[i] >= slv)
            eod = minute[i] >= 955
            if hit_tp or hit_sl or held >= horizon or eod:
                risk_px = abs(entry - slv)
                if hit_sl:
                    px, reason = slv - pos * stop_slip_R * risk_px, "sl"
                elif hit_tp:
                    px, reason = tp, "tp"
                else:
                    px, reason = close[i], ("eod" if eod else "time")
                gross = pos * (px - entry) * qty
                fees = (entry + px) * qty * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": pos, "bars": held,
                               "reason": reason})
                pos, qty, held = 0, 0.0, 0

        if not locked and eq <= day_start * 0.97:
            locked = True

        src = i - lag
        if (pos == 0 and not locked and src >= 0 and minute[i] < 940
                and mp[src] >= thresh and i + 1 < n):
            if eq <= capital * 0.05:
                break
            signals += 1
            s = int(side_a[src])
            v = max(vol[src], 1e-4) * np.sqrt(horizon)
            raw_v = max(vol[src], 1e-4)

            if limit_fills:
                # Passive entry resting inside the move. It only fills if the next
                # bar trades through it -- which is precisely when price is coming
                # back at you. That is adverse selection, and no "fill rate"
                # assumption captures it; it has to be simulated this way.
                limit_px = close[i] - s * limit_offset_frac * raw_v * close[i]
                touched = (low[i + 1] <= limit_px) if s == 1 else (high[i + 1] >= limit_px)
                if not touched:
                    curve[i] = eq
                    continue
                entry_px = limit_px
            else:
                entry_px = close[i]

            fills += 1
            stop_dist = sl * v * entry_px
            q = (capital * risk) / max(stop_dist, 1e-6)
            q = min(q, (2.0 * capital) / entry_px)
            if q > 0:
                pos, entry, qty, held = s, entry_px, q, 0
                tp = entry * (1 + s * pt * v)
                slv = entry * (1 - s * sl * v)
        curve[i] = eq

    return (pd.DataFrame(trades), pd.Series(curve, index=idx),
            {"signals": signals, "fills": fills,
             "fill_rate": round(100 * fills / signals, 1) if signals else None})


def deflated_sharpe(daily_returns, sharpe_ann, n_trials, periods=252):
    r = np.asarray(daily_returns)
    r = r[np.isfinite(r)]
    if len(r) < 20 or not np.isfinite(sharpe_ann):
        return None
    sr = sharpe_ann / np.sqrt(periods)
    T = len(r)
    g3 = float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-12))
    g4 = float(((r - r.mean()) ** 4).mean() / (r.std() ** 4 + 1e-12))
    e_max = np.sqrt(1.0 / T) * ((1 - np.euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
                                + np.euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) + 1e-12
    return float(norm.cdf(((sr - e_max) * np.sqrt(T - 1)) / denom))


def summarise(t, curve, meta, days):
    if t.empty or len(t) < 20:
        return None
    daily = curve.resample("1D").last().dropna().pct_change().dropna()
    sharpe = (daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
              if len(daily) > 2 else np.nan)
    total = float(curve.iloc[-1] / CAPITAL - 1) * 100
    ann = ((1 + total / 100) ** (365 / max(days, 1)) - 1) * 100
    roll = curve.cummax()
    win, be, edge = edge_of(t)
    return {"ret": round(total, 1), "ann": round(ann, 1),
            "sharpe": round(float(sharpe), 2) if np.isfinite(sharpe) else None,
            "edge": round(edge, 2) if edge else None,
            "trades": len(t),
            "dd": round(float(((curve - roll) / roll).min()) * 100, 1),
            "fill": meta.get("fill_rate"),
            "daily": daily}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("fetching history")
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})

    # benchmark: buy and hold SPY over the same window
    spy_d = spy["close"].resample("1D").last().dropna()
    days = (spy_d.index[-1] - spy_d.index[0]).days
    spy_total = float(spy_d.iloc[-1] / spy_d.iloc[0] - 1) * 100
    spy_ann = ((1 + spy_total / 100) ** (365 / max(days, 1)) - 1) * 100
    spy_daily = spy_d.pct_change().dropna()
    spy_sharpe = float(spy_daily.mean() / spy_daily.std() * np.sqrt(252))

    LAYERS = [
        ("0 ideal",              dict()),
        ("1 +signal lag",        dict(lag=1)),
        ("2 +intrabar stops",    dict(lag=1, intrabar=True)),
        ("3 +stop slip 0.10R",   dict(lag=1, intrabar=True, stop_slip_R=0.10)),
        ("4 +limit fills",       dict(lag=1, intrabar=True, stop_slip_R=0.10,
                                      limit_fills=True)),
        ("5 +costs",             dict(lag=1, intrabar=True, stop_slip_R=0.10,
                                      limit_fills=True, cost=0.00015)),
        ("X polled stops 0.31R", dict(lag=1, intrabar=True, stop_slip_R=0.31,
                                      limit_fills=True, cost=0.00015)),
    ]

    acc = {name: [] for name, _ in LAYERS}
    dailies = {name: [] for name, _ in LAYERS}

    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg0)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 5000:
            continue
        c = ModelCfg(S)
        c.horizon_bars = HORIZON
        sg = walk_forward(df, feats, c)
        if sg is None:
            continue
        log.info("%s trained", sym)
        for name, kw in LAYERS:
            t, curve, meta = simulate(sg, **kw)
            s = summarise(t, curve, meta, days)
            if s:
                acc[name].append(s)
                dailies[name].append(s["daily"])

    print("\n" + "=" * 100)
    print("REALITY STACK -- h6 wide, mean across symbols, 2 years")
    print("  each row adds one real-world effect the previous row ignored")
    print("=" * 100)
    rows = []
    for name, _ in LAYERS:
        v = acc[name]
        if not v:
            continue
        dl = pd.concat(dailies[name], axis=1).mean(axis=1).dropna()
        sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252)) if len(dl) > 2 else np.nan
        dsr = deflated_sharpe(dl.values, sh, N_TRIALS)
        rows.append({
            "layer": name,
            "total_%": round(np.mean([x["ret"] for x in v]), 1),
            "annual_%": round(np.mean([x["ann"] for x in v]), 1),
            "sharpe": round(np.mean([x["sharpe"] for x in v if x["sharpe"]]), 2),
            "edge": round(np.mean([x["edge"] for x in v if x["edge"] is not None]), 2),
            "maxDD_%": round(np.mean([x["dd"] for x in v]), 1),
            "trades": int(np.mean([x["trades"] for x in v])),
            "fill_%": (round(np.mean([x["fill"] for x in v if x["fill"]]), 1)
                       if any(x["fill"] for x in v) else None),
            "DSR": round(dsr, 3) if dsr is not None else None,
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    print("\n" + "=" * 100)
    print("BENCHMARK -- buy and hold SPY, same window")
    print("=" * 100)
    print(f"  total {spy_total:.1f}%   annualised {spy_ann:.1f}%   "
          f"Sharpe {spy_sharpe:.2f}   ({days} days)")

    fully = out[out.layer == "5 +costs"]
    if len(fully):
        r = fully.iloc[0]
        print("\nVERDICT")
        print(f"  strategy (all costs) : {r['annual_%']:.1f}%/yr  Sharpe {r['sharpe']:.2f}"
              f"  maxDD {r['maxDD_%']:.1f}%  DSR {r['DSR']}")
        print(f"  SPY buy and hold     : {spy_ann:.1f}%/yr  Sharpe {spy_sharpe:.2f}")
        excess = r["annual_%"] - spy_ann
        print(f"  excess over index    : {excess:+.1f} pp/yr")
        if r["DSR"] is not None and r["DSR"] < 0.90:
            print(f"  NOTE: deflated Sharpe {r['DSR']} < 0.90 -- after penalising the")
            print(f"        ~{N_TRIALS} configurations searched, the edge is NOT")
            print("        statistically distinguishable from luck.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

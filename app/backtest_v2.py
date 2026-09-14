"""
Backtest v2 -- corrected for two flaws that made the original optimistic.

FLAW 1: no signal lag.
    v1 scored a bar and entered at that same bar's close. Live, an IEX bar arrives
    4-5 minutes AFTER it closes, so the bot acts on stale information. On a 30-min
    horizon that is ~15% of the trade's life; on a 15-min horizon it is ~30%.
    This is why horizon-3 looked good in v1 and lost money live.
    v2 enters at the NEXT bar, which is what actually happens.

FLAW 2: barriers only checked at bar closes.
    v1 compared close[j] to the barriers, so a spike that pierced the stop and
    recovered within the bar was invisible. Live checks every 20 seconds against
    the trade price, so those stops DO fire. v2 checks intrabar high/low, and when
    both barriers are touched in the same bar it assumes the STOP hit first
    (worst case -- OHLC cannot resolve intrabar sequence).

Result: lower, more honest numbers, and a different ranking of configs.

    python -m app.backtest_v2
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_ab import walk_forward, TREND_BARS

log = logging.getLogger("backtest_v2")

COST_PER_SIDE = 0.00015
SIGNAL_LAG_BARS = 1


def simulate_v2(sig, capital=5000.0, risk=0.005, pt=1.0, sl=1.0,
                horizon=6, thresh=0.55, cost=COST_PER_SIDE,
                lag=SIGNAL_LAG_BARS, intrabar=True):
    idx = sig.index
    close = sig["close"].values
    high = sig["high"].values
    low = sig["low"].values
    vol = np.nan_to_num(sig["vol"].values, nan=np.nanmedian(sig["vol"].values))
    side_a = sig["side"].values
    mp = sig["meta_p"].values
    minute = idx.hour * 60 + idx.minute
    days = idx.normalize()

    eq = capital
    pos = 0
    entry = qty = tp = slv = 0.0
    held = 0
    trades = []
    curve = np.empty(len(idx))
    cur_day, day_start, locked = None, eq, False

    for i in range(len(idx)):
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
            timeout = held >= horizon
            if hit_tp or hit_sl or timeout or eod:
                if hit_sl:
                    px, reason = slv, "sl"
                elif hit_tp:
                    px, reason = tp, "tp"
                else:
                    px, reason = close[i], ("eod" if eod else "time")
                gross = pos * (px - entry) * qty
                fees = (entry + px) * qty * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": pos,
                               "bars": held, "reason": reason})
                pos, qty, held = 0, 0.0, 0

        if not locked and eq <= day_start * 0.97:
            locked = True

        src = i - lag
        if (pos == 0 and not locked and src >= 0 and minute[i] < 940
                and mp[src] >= thresh):
            s = int(side_a[src])
            v = max(vol[src], 1e-4)
            stop_dist = sl * v * close[i]
            q = (capital * risk) / max(stop_dist, 1e-6)
            q = min(q, (2.0 * capital) / close[i])
            if q > 0:
                pos, entry, qty, held = s, close[i], q, 0
                tp = entry * (1 + s * pt * v)
                slv = entry * (1 - s * sl * v)
        curve[i] = eq

    return pd.DataFrame(trades), pd.Series(curve, index=idx)


def stats(trades, curve, capital=5000.0):
    if trades.empty:
        return {"trades": 0, "ret_pct": 0.0, "sharpe": None, "win_pct": 0.0,
                "pf": 0.0, "maxdd_pct": 0.0, "breakeven_pct": None}
    daily = curve.resample("1D").last().dropna().pct_change().dropna()
    sharpe = (daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
              if len(daily) > 2 else None)
    roll = curve.cummax()
    w = trades.loc[trades.pnl > 0, "pnl"]
    l = trades.loc[trades.pnl <= 0, "pnl"]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-l.mean()) if len(l) else 0.0
    be = (al / (aw + al) * 100) if (aw + al) > 0 else None
    return {
        "trades": int(len(trades)),
        "ret_pct": round(float(curve.iloc[-1] / capital - 1) * 100, 2),
        "sharpe": round(float(sharpe), 2) if sharpe is not None else None,
        "win_pct": round(float((trades.pnl > 0).mean()) * 100, 1),
        "pf": round(float(w.sum() / (-l.sum() + 1e-9)), 2),
        "maxdd_pct": round(float(((curve - roll) / roll).min()) * 100, 2),
        "breakeven_pct": round(be, 1) if be else None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("fetching %.1f years of %dmin bars", S.train_years, S.bar_minutes)
    hist = fetch_history(S, need, S.train_years)
    mkt = hist[S.market_symbol][["close"]].rename(
        columns={"close": f"{S.market_symbol}_close"})

    configs = [
        ("h3 asym (live, lost)", 3, 0.7, 1.0, 0.58),
        ("h3 symmetric",         3, 1.0, 1.0, 0.55),
        ("h6 asym",              6, 0.7, 1.0, 0.58),
        ("h6 symmetric",         6, 1.0, 1.0, 0.55),
        ("h12 symmetric",       12, 1.0, 1.0, 0.55),
        ("h12 wide",            12, 1.5, 1.0, 0.55),
    ]

    rows = []
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg)
        mc = df[f"{S.market_symbol}_close"]
        feats["mkt_trend_bps"] = (mc / mc.shift(TREND_BARS) - 1) * 1e4
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 5000:
            continue

        trained = {}
        for name, hz, pt, sl, thr in configs:
            if hz not in trained:
                c = ModelCfg(S)
                c.horizon_bars = hz
                trained[hz] = walk_forward(df, feats, c, False)
                log.info("%s trained horizon=%d", sym, hz)
            sg = trained[hz]
            if sg is None:
                continue
            t1, c1 = simulate_v2(sg, pt=pt, sl=sl, horizon=hz, thresh=thr,
                                 lag=0, intrabar=False)
            t2, c2 = simulate_v2(sg, pt=pt, sl=sl, horizon=hz, thresh=thr,
                                 lag=1, intrabar=True)
            s1, s2 = stats(t1, c1), stats(t2, c2)
            rows.append({"symbol": sym, "config": name,
                         "v1_ret": s1["ret_pct"], "v1_win": s1["win_pct"],
                         "ret": s2["ret_pct"], "win": s2["win_pct"],
                         "be": s2["breakeven_pct"], "pf": s2["pf"],
                         "sharpe": s2["sharpe"], "dd": s2["maxdd_pct"],
                         "trades": s2["trades"]})
            log.info("  %-22s v1 %+8.1f%% (win %.1f) -> v2 %+8.1f%% (win %.1f be %.1f)",
                     name, s1["ret_pct"], s1["win_pct"], s2["ret_pct"],
                     s2["win_pct"], s2["breakeven_pct"] or 0)

    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return 1
    d.to_json("/app/state/backtest_v2_raw.json", orient="records")

    g = (d.groupby("config")
           .agg(v1_ret=("v1_ret", "mean"), ret=("ret", "mean"),
                win=("win", "mean"), be=("be", "mean"), pf=("pf", "mean"),
                sharpe=("sharpe", "mean"), dd=("dd", "mean"),
                trades=("trades", "mean"), n=("symbol", "count"))
           .reset_index())
    g["edge"] = (g["win"] - g["be"]).round(1)
    for c in ("v1_ret", "ret", "win", "be", "pf", "sharpe", "dd"):
        g[c] = g[c].round(2)
    g["trades"] = g["trades"].round(0)
    order = [c[0] for c in configs]
    g["_o"] = g["config"].map({n: i for i, n in enumerate(order)})
    g = g.sort_values("_o").drop(columns="_o")

    print("\n" + "=" * 110)
    print("BACKTEST v2 -- 1-bar signal lag + intrabar barrier checks")
    print("  v1_ret = old optimistic method | ret = corrected")
    print("  edge = win rate minus breakeven (positive means the geometry is viable)")
    print("=" * 110)
    print(g.to_string(index=False))

    print("\nCOST OF THE CORRECTION (mean across symbols):")
    for _, r in g.iterrows():
        print(f"  {r['config']:<22} {r['v1_ret']:>9.1f}%  ->  {r['ret']:>9.1f}%")

    g.to_json("/app/state/backtest_v2_summary.json", orient="records")
    print("\nsaved /app/state/backtest_v2_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

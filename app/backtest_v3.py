"""
Equity backtest v3 -- horizon-scaled barriers + pre-market regime analysis.

TWO CHANGES FROM v2:

1. BARRIER SCALING (bug fix).
   pt/sl were multiples of PER-BAR volatility (~0.1% on a 5-min equity bar), so
   changing horizon_bars changed only the time CAP, never the target. A 60-minute
   horizon still aimed at a 0.15% move that got hit within a bar or two -- which is
   why live trades averaged 4-13 minute holds against a 15-30 minute horizon, and
   why h3/h6/h12 were effectively the same strategy tested three times.
   Barriers now scale by sqrt(horizon), so a 4x longer hold targets a 2x larger
   move. Applied to the LABELS too -- otherwise the model predicts one event while
   the simulator trades another.

2. PRE-MARKET REGIME BUCKETS.
   Classifies each session BEFORE the open using only point-in-time data:
   overnight gap, prior-day return, prior realised vol vs its own history, and
   SPY's position against its 20-day average. No news, no lookahead -- a news
   classifier cannot be backtested honestly, because the model already knows what
   happened on any historical date.

   The question this answers is NOT "which strategy per regime" but the prior one:
   is edge different across regimes at all? If edge is flat, regime switching adds
   nothing. If it is positive in some bucket, that is worth building on.

    python -m app.backtest_v3
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .labeling import triple_barrier_labels, make_meta_labels
from .regime import RegimeModel
from .backtest_ab import TREND_BARS

import lightgbm as lgb

log = logging.getLogger("backtest_v3")

COST_PER_SIDE = 0.00015
N_FOLDS = 4
EMBARGO = 24
LAG_BARS = 1          # streaming should reduce this; v2 assumption kept for now

# name, horizon_bars, pt, sl, threshold
CONFIGS = [
    ("h6  (30m)  sym",   6, 1.0, 1.0, 0.55),
    ("h6  (30m)  wide",  6, 1.5, 1.0, 0.55),
    ("h12 (60m)  sym",  12, 1.0, 1.0, 0.55),
    ("h12 (60m)  wide", 12, 1.5, 1.0, 0.55),
    ("h24 (2h)   wide", 24, 1.5, 1.0, 0.55),
]


# --------------------------------------------------------------------------- #
def classify_days(spy: pd.DataFrame) -> pd.DataFrame:
    """
    One regime label per session, computed from information available BEFORE the
    first entry (prior sessions plus today's opening print).

    Buckets:
      high_vol   - prior realised vol in the top third of its trailing history
      trend_up   - SPY above its 20-day average and prior 5-day return positive
      trend_down - below its 20-day average and prior 5-day return negative
      chop       - everything else
    """
    daily = spy.resample("1D").agg({"open": "first", "close": "last",
                                    "high": "max", "low": "min"}).dropna()
    ret = np.log(daily["close"]).diff()
    rv5 = ret.rolling(5).std()
    rv_pct = rv5.rolling(60).rank(pct=True)
    ma20 = daily["close"].rolling(20).mean()
    ret5 = np.log(daily["close"]).diff(5)
    gap = np.log(daily["open"] / daily["close"].shift(1))

    # shift(1) so today's classification uses only prior sessions; the gap is the
    # single same-day input and it is known at 09:30, before entries begin at 09:35
    out = pd.DataFrame(index=daily.index)
    out["rv_pct"] = rv_pct.shift(1)
    out["above_ma"] = (daily["close"].shift(1) > ma20.shift(1))
    out["ret5"] = ret5.shift(1)
    out["gap"] = gap

    def label(r):
        if pd.isna(r["rv_pct"]) or pd.isna(r["ret5"]):
            return "unknown"
        if r["rv_pct"] >= 0.67:
            return "high_vol"
        if r["above_ma"] and r["ret5"] > 0:
            return "trend_up"
        if (not r["above_ma"]) and r["ret5"] < 0:
            return "trend_down"
        return "chop"

    out["regime"] = out.apply(label, axis=1)
    return out[["regime", "rv_pct", "gap"]]


# --------------------------------------------------------------------------- #
def _clf(seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=60, reg_lambda=1.0,
                              random_state=seed, n_jobs=2, verbose=-1)


def walk_forward(df, feats, cfg):
    """Barriers (and therefore labels) scale with sqrt(horizon)."""
    close = df["close"]
    scaled_vol = feats["vol"] * np.sqrt(cfg.horizon_bars)
    tb = triple_barrier_labels(close, scaled_vol, cfg)
    n = len(df)
    fold = n // (N_FOLDS + 1)
    out = []
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_start = tr_end + EMBARGO
        te_end = min(tr_end + fold, n)
        if te_start >= te_end:
            continue
        tr, te = slice(0, tr_end), slice(te_start, te_end)
        reg = RegimeModel(cfg).fit(feats.iloc[tr])
        rtr, rte = reg.transform(feats.iloc[tr]), reg.transform(feats.iloc[te])
        cols = [c for c in feats.columns if c != "mkt_trend_bps"] + \
               [c for c in rtr.columns if c.startswith("regime_p")]
        Xtr = pd.concat([feats.iloc[tr], rtr], axis=1)[cols]
        Xte = pd.concat([feats.iloc[te], rte], axis=1)[cols]
        Xtr = Xtr.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        Xte = Xte.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        y = tb["label"].iloc[tr].values
        m = y != 0
        if m.sum() < 300 or len(np.unique(y[m] > 0)) < 2:
            continue
        primary = _clf(7).fit(Xtr[m], (y[m] > 0).astype(int))
        p_tr = primary.predict_proba(Xtr)[:, 1]
        side_tr = np.where(p_tr >= 0.5, 1, -1)
        my = make_meta_labels(side_tr, y)
        if len(np.unique(my)) < 2:
            continue
        meta = _clf(8).fit(np.column_stack([Xtr, p_tr]), my)
        p_te = primary.predict_proba(Xte)[:, 1]
        side_te = np.where(p_te >= 0.5, 1, -1)
        mp_te = meta.predict_proba(np.column_stack([Xte, p_te]))[:, 1]
        out.append(pd.DataFrame({
            "close": close.iloc[te].values,
            "high": df["high"].iloc[te].values,
            "low": df["low"].iloc[te].values,
            "vol": feats["vol"].iloc[te].values,
            "side": side_te, "meta_p": mp_te,
        }, index=df.index[te_start:te_end]))
    return pd.concat(out) if out else None


def simulate(sig, capital=5000.0, risk=0.005, pt=1.0, sl=1.0, horizon=6,
             thresh=0.55, cost=COST_PER_SIDE, lag=LAG_BARS,
             stop_slip_R=0.0, tp_slip_R=0.0):
    """
    stop_slip_R / tp_slip_R: how far past the barrier the fill actually lands,
    expressed in R (1R = the stop distance). Measured live at 0.31R on stops and
    ~0.00R on targets when exiting via polled market orders. Bracket orders resting
    at the exchange should cut the stop figure substantially -- run both to see how
    much of the edge depends on fixing it.
    """
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
    entry_day = None

    for i in range(len(idx)):
        if days[i] != cur_day:
            cur_day, day_start, locked = days[i], eq, False

        if pos != 0:
            held += 1
            hit_tp = (pos == 1 and high[i] >= tp) or (pos == -1 and low[i] <= tp)
            hit_sl = (pos == 1 and low[i] <= slv) or (pos == -1 and high[i] >= slv)
            eod = minute[i] >= 955
            if hit_tp or hit_sl or held >= horizon or eod:
                if hit_sl:
                    # fill lands stop_slip_R past the stop, in the losing direction
                    risk_px = abs(entry - slv)
                    px = slv - pos * stop_slip_R * risk_px
                    reason = "sl"
                elif hit_tp:
                    risk_px = abs(entry - slv)
                    px = tp - pos * tp_slip_R * risk_px
                    reason = "tp"
                else:
                    px, reason = close[i], ("eod" if eod else "time")
                gross = pos * (px - entry) * qty
                fees = (entry + px) * qty * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": pos, "bars": held,
                               "reason": reason, "day": entry_day})
                pos, qty, held = 0, 0.0, 0

        if not locked and eq <= day_start * 0.97:
            locked = True

        src = i - lag
        if (pos == 0 and not locked and src >= 0 and minute[i] < 940
                and mp[src] >= thresh):
            if eq <= capital * 0.05:
                break
            s = int(side_a[src])
            # horizon-scaled barrier: target grows with the square root of time
            v = max(vol[src], 1e-4) * np.sqrt(horizon)
            stop_dist = sl * v * close[i]
            q = (capital * risk) / max(stop_dist, 1e-6)
            q = min(q, (2.0 * capital) / close[i])
            if q > 0:
                pos, entry, qty, held = s, close[i], q, 0
                entry_day = days[i]
                tp = entry * (1 + s * pt * v)
                slv = entry * (1 - s * sl * v)
        curve[i] = eq

    return pd.DataFrame(trades), pd.Series(curve, index=idx)


def edge_of(trades):
    """win rate minus breakeven, from realised win/loss sizes"""
    if trades.empty or len(trades) < 20:
        return None, None, None
    w = trades.loc[trades.pnl > 0, "pnl"]
    l = trades.loc[trades.pnl <= 0, "pnl"]
    if not len(w) or not len(l):
        return None, None, None
    aw, al = float(w.mean()), float(-l.mean())
    be = al / (aw + al) * 100
    win = float((trades.pnl > 0).mean()) * 100
    return win, be, win - be


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("fetching %.1f years", S.train_years)
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})

    regimes = classify_days(spy)
    log.info("regime day counts:\n%s", regimes["regime"].value_counts().to_string())

    all_trades = []
    rows = []
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg0)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 5000:
            continue

        trained = {}
        for name, hz, pt, sl, thr in CONFIGS:
            if hz not in trained:
                c = ModelCfg(S)
                c.horizon_bars = hz
                trained[hz] = walk_forward(df, feats, c)
                log.info("%s trained horizon=%d", sym, hz)
            sg = trained[hz]
            if sg is None:
                continue
            # three execution scenarios:
            #   ideal   - fills exactly at the barrier (what the old backtest assumed)
            #   bracket - exchange-resident stop, modest slip
            #   polled  - measured live behaviour with 20s polling + market exits
            for scen, ssl in (("ideal", 0.0), ("bracket", 0.10), ("polled", 0.31)):
                t, curve = simulate(sg, pt=pt, sl=sl, horizon=hz, thresh=thr,
                                    stop_slip_R=ssl)
                if t.empty:
                    continue
                win, be, edge = edge_of(t)
                rows.append({"symbol": sym, "config": name, "scenario": scen,
                             "ret": round(float(curve.iloc[-1] / 5000 - 1) * 100, 2),
                             "win": round(win, 1) if win else None,
                             "be": round(be, 1) if be else None,
                             "edge": round(edge, 2) if edge else None,
                             "trades": len(t),
                             "hold_min": round(float(t["bars"].mean()) * 5, 1)})
                if scen == "bracket":
                    t2 = t.copy()
                    t2["symbol"] = sym
                    t2["config"] = name
                    all_trades.append(t2)
            log.info("  %-16s ideal/bracket/polled ret %s", name,
                     [r["ret"] for r in rows[-3:]])

    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return 1

    g = (d.groupby(["config", "scenario"])
           .agg(ret=("ret", "mean"), win=("win", "mean"),
                be=("be", "mean"), edge=("edge", "mean"),
                trades=("trades", "mean"),
                hold=("hold_min", "mean"), n=("symbol", "count"))
           .reset_index())
    for c in ("ret", "win", "be", "edge", "hold"):
        g[c] = g[c].round(2)
    g["trades"] = g["trades"].round(0)
    order = [c[0] for c in CONFIGS]
    g["_o"] = g["config"].map({n: i for i, n in enumerate(order)})
    g = g.sort_values("_o").drop(columns="_o")

    print("\n" + "=" * 96)
    print("PART 1 -- HORIZON-SCALED BARRIERS (bug fixed)")
    print("  hold = mean minutes held; should now scale with the horizon")
    print("=" * 96)
    print(g.to_string(index=False))

    # ---------------- regime analysis ----------------
    tr = pd.concat(all_trades, ignore_index=True)
    tr["day"] = pd.to_datetime(tr["day"]).dt.tz_localize(None).dt.normalize()
    rmap = regimes.copy()
    rmap.index = pd.to_datetime(rmap.index).tz_localize(None).normalize()
    tr = tr.join(rmap["regime"], on="day")
    tr["regime"] = tr["regime"].fillna("unknown")

    print("\n" + "=" * 96)
    print("PART 2 -- EDGE BY PRE-MARKET REGIME")
    print("  Is edge different across regimes? If flat, a regime selector adds nothing.")
    print("=" * 96)

    best = g.sort_values("edge", ascending=False)["config"].iloc[0]
    print(f"\n(using best config: {best})")
    sub = tr[tr.config == best]
    out = []
    for reg, grp in sub.groupby("regime"):
        win, be, edge = edge_of(grp)
        out.append({"regime": reg, "trades": len(grp),
                    "net": round(float(grp.pnl.sum()), 2),
                    "win": round(win, 1) if win else None,
                    "be": round(be, 1) if be else None,
                    "edge": round(edge, 2) if edge else None,
                    "avg_pnl": round(float(grp.pnl.mean()), 3)})
    ro = pd.DataFrame(out).sort_values("edge", ascending=False, na_position="last")
    print(ro.to_string(index=False))

    print("\nALL CONFIGS x REGIME (edge):")
    piv = []
    for (cfgn, reg), grp in tr.groupby(["config", "regime"]):
        _, _, e = edge_of(grp)
        piv.append({"config": cfgn, "regime": reg, "edge": e})
    pv = pd.DataFrame(piv).pivot(index="config", columns="regime", values="edge")
    print(pv.round(2).to_string())

    tr.to_json("/app/state/v3_trades.json", orient="records")
    g.to_json("/app/state/v3_summary.json", orient="records")
    print("\nsaved /app/state/v3_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
A/B backtest for the changes made after the 2026-07-29 loss.

Question this answers: did class balancing and the SPY trend filter actually
improve anything, or were they a reaction to a single bad session?

Four configurations, identical data and folds:
    base      - original model, no filter          (what lost 3.3%)
    balanced  - class_weight="balanced"
    filter    - original model + SPY trend veto
    both      - balanced + filter                  (what is live now)

Walk-forward with an embargo, realistic per-side costs, same barrier logic and
same end-of-day flatten as the live engine.

    python -m app.backtest_ab
"""
import itertools
import json
import logging
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .labeling import triple_barrier_labels, make_meta_labels
from .regime import RegimeModel

log = logging.getLogger("backtest_ab")

N_FOLDS = 4
EMBARGO = 12
COST_PER_SIDE = 0.00015          # 1.5 bps/side
TREND_BARS = 12
TREND_LIMIT_BPS = 15.0


def _clf(seed, balanced):
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=60,
        reg_lambda=1.0, class_weight=("balanced" if balanced else None),
        random_state=seed, n_jobs=2, verbose=-1)


def walk_forward(df, feats, cfg, balanced):
    """Out-of-sample signals across folds. HMM + models fit on train only."""
    close = df["close"]
    tb = triple_barrier_labels(close, feats["vol"], cfg)
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
        # mkt_trend_bps is used by the filter, not as a model input
        cols = [c for c in feats.columns if c != "mkt_trend_bps"] + \
               [c for c in rtr.columns if c.startswith("regime_p")]
        Xtr = pd.concat([feats.iloc[tr], rtr], axis=1)[cols]
        Xte = pd.concat([feats.iloc[te], rte], axis=1)[cols]
        Xtr = Xtr.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        Xte = Xte.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values

        y = tb["label"].iloc[tr].values
        m = y != 0
        if m.sum() < 300 or len(np.unique((y[m] > 0))) < 2:
            continue

        primary = _clf(7, balanced).fit(Xtr[m], (y[m] > 0).astype(int))
        p_tr = primary.predict_proba(Xtr)[:, 1]
        side_tr = np.where(p_tr >= 0.5, 1, -1)
        my = make_meta_labels(side_tr, y)
        if len(np.unique(my)) < 2:
            continue
        meta = _clf(8, False).fit(np.column_stack([Xtr, p_tr]), my)

        p_te = primary.predict_proba(Xte)[:, 1]
        side_te = np.where(p_te >= 0.5, 1, -1)
        mp_te = meta.predict_proba(np.column_stack([Xte, p_te]))[:, 1]

        out.append(pd.DataFrame({
            "close": close.iloc[te].values,
            "high": df["high"].iloc[te].values,
            "low": df["low"].iloc[te].values,
            "vol": feats["vol"].iloc[te].values,
            "trend": feats["mkt_trend_bps"].iloc[te].values,
            "side": side_te, "meta_p": mp_te,
        }, index=df.index[te_start:te_end]))

    return pd.concat(out) if out else None


def simulate(sig, use_filter, capital=5000.0, risk=0.005,
             pt=1.0, sl=1.0, horizon=6, thresh=0.55, cost=COST_PER_SIDE):
    """Event-driven, one position at a time, flat by 15:55."""
    idx = sig.index
    close, high, low = sig["close"].values, sig["high"].values, sig["low"].values
    vol = np.nan_to_num(sig["vol"].values, nan=np.nanmedian(sig["vol"].values))
    side, mp = sig["side"].values, sig["meta_p"].values
    trend = np.nan_to_num(sig["trend"].values)
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
            px = close[i]
            hit_tp = (pos == 1 and px >= tp) or (pos == -1 and px <= tp)
            hit_sl = (pos == 1 and px <= slv) or (pos == -1 and px >= slv)
            eod = minute[i] >= 955
            if hit_tp or hit_sl or held >= horizon or eod:
                gross = pos * (px - entry) * qty
                fees = (entry + px) * qty * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": pos, "bars": held,
                               "reason": "tp" if hit_tp else "sl" if hit_sl
                                         else "eod" if eod else "time"})
                pos, qty, held = 0, 0.0, 0

        if not locked and eq <= day_start * 0.97:
            locked = True

        if pos == 0 and not locked and minute[i] < 940 and mp[i] >= thresh:
            s = int(side[i])
            if use_filter:
                if (s > 0 and trend[i] < -TREND_LIMIT_BPS) or \
                   (s < 0 and trend[i] > TREND_LIMIT_BPS):
                    curve[i] = eq
                    continue
            v = max(vol[i], 1e-4)
            stop_dist = sl * v * close[i]
            # FIXED notional off the STARTING capital, not current equity.
            # Compounding 7,000 trades over two years produces meaningless
            # exponential returns (COIN printed 133,000,000% on the first run).
            # Fixed sizing isolates the per-trade edge, which is the thing we
            # are actually trying to compare between configurations.
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
                "pf": 0.0, "maxdd_pct": 0.0, "long_pct": None}
    daily = curve.resample("1D").last().dropna().pct_change().dropna()
    sharpe = (daily.mean() / (daily.std() + 1e-12) * np.sqrt(252)
              if len(daily) > 2 else None)
    roll = curve.cummax()
    wins = trades["pnl"] > 0
    gw = trades.loc[wins, "pnl"].sum()
    gl = -trades.loc[~wins, "pnl"].sum()
    return {
        "trades": int(len(trades)),
        "ret_pct": round(float(curve.iloc[-1] / capital - 1) * 100, 2),
        "net_per_trade": round(float(trades["pnl"].mean()), 3),
        "sharpe": round(float(sharpe), 2) if sharpe is not None else None,
        "win_pct": round(float(wins.mean()) * 100, 1),
        "pf": round(float(gw / (gl + 1e-9)), 2),
        "maxdd_pct": round(float(((curve - roll) / roll).min()) * 100, 2),
        "long_pct": round(float((trades["side"] > 0).mean()) * 100, 1),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("fetching %.1f years of %dmin bars (%s)", S.train_years,
             S.bar_minutes, "SIP" if S.use_sip else "IEX")
    hist = fetch_history(S, need, S.train_years)
    mkt = hist[S.market_symbol][["close"]].rename(
        columns={"close": f"{S.market_symbol}_close"})

    configs = [("base", False, False), ("balanced", True, False),
               ("filter", False, True), ("both", True, True)]
    results = {}

    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg)
        # trailing market trend in bps -- the same signal the live filter uses
        mc = df[f"{S.market_symbol}_close"]
        feats["mkt_trend_bps"] = (mc / mc.shift(TREND_BARS) - 1) * 1e4
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 5000:
            log.warning("%s: only %d bars, skipping", sym, len(df))
            continue

        results[sym] = {}
        for name, balanced, use_filter in configs:
            sig = walk_forward(df, feats, cfg, balanced)
            if sig is None:
                continue
            tr, curve = simulate(sig, use_filter)
            results[sym][name] = stats(tr, curve)
            log.info("%-5s %-9s %s", sym, name, results[sym][name])

    # ---- aggregate ----
    print("\n" + "=" * 88)
    print("A/B BACKTEST  (walk-forward, out-of-sample, 1.5bps/side)")
    print("=" * 88)
    rows = []
    for name, _, _ in configs:
        rets = [results[s][name]["ret_pct"] for s in results if name in results[s]]
        wins = [results[s][name]["win_pct"] for s in results if name in results[s]]
        trs = [results[s][name]["trades"] for s in results if name in results[s]]
        lng = [results[s][name]["long_pct"] for s in results if name in results[s]
               if results[s][name]["long_pct"] is not None]
        shp = [results[s][name]["sharpe"] for s in results if name in results[s]
               and results[s][name]["sharpe"] is not None]
        dds = [results[s][name]["maxdd_pct"] for s in results if name in results[s]]
        rows.append({
            "config": name,
            "mean_ret_%": round(np.mean(rets), 2) if rets else None,
            "median_ret_%": round(np.median(rets), 2) if rets else None,
            "mean_sharpe": round(np.mean(shp), 2) if shp else None,
            "mean_win_%": round(np.mean(wins), 1) if wins else None,
            "mean_maxdd_%": round(np.mean(dds), 2) if dds else None,
            "long_%": round(np.mean(lng), 1) if lng else None,
            "avg_trades": int(np.mean(trs)) if trs else 0,
            "symbols_profitable": sum(1 for r in rets if r > 0),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nPER-SYMBOL RETURN %")
    per = pd.DataFrame({s: {n: results[s][n]["ret_pct"] for n in results[s]}
                        for s in results}).T
    print(per.to_string())

    json.dump(results, open("/app/state/backtest_ab.json", "w"), indent=2)
    print("\nsaved /app/state/backtest_ab.json")


if __name__ == "__main__":
    sys.exit(main())

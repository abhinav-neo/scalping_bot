"""
Crypto backtest -- honest cost model, hours-horizon.

Two things make this a different problem from equities:

1. FEES. Alpaca crypto is 0.15% maker / 0.25% taker (tier 1). Best case round trip
   is 0.30% -- ten times the ~0.03% we assume on equities. A 5-minute scalp cannot
   clear that, which is why this tests HOURS-horizon configs instead.

2. NO SHORTING. Alpaca crypto is spot: shortable=False on all 73 pairs. Long-only
   is enforced here, so short signals are simply skipped. That is a real constraint,
   not a modelling choice.

Also, crypto trades 24/7 -- no session boundary, no end-of-day flatten, no daily
kill switch tied to a trading day. Guards here are per-trade and drawdown-based.

    python -m app.crypto_backtest
"""
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb

from .settings import S
from .modelcfg import ModelCfg
from .features import build_features
from .labeling import triple_barrier_labels, make_meta_labels
from .regime import RegimeModel

log = logging.getLogger("crypto_bt")

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
MARKET = "BTC/USD"          # BTC acts as the market factor for alts
YEARS = 1.0
N_FOLDS = 4
EMBARGO = 24

# round-trip cost as a fraction of notional, charged per side
COST_MAKER = 0.0015
COST_TAKER = 0.0025

# horizon in 5-min bars, pt, sl, threshold
CONFIGS = [
    ("h12  (1h)  wide",   12, 1.5, 1.0, 0.55),
    ("h24  (2h)  wide",   24, 1.5, 1.0, 0.55),
    ("h48  (4h)  wide",   48, 1.5, 1.0, 0.55),
    ("h96  (8h)  wide",   96, 1.5, 1.0, 0.55),
    ("h288 (24h) wide",  288, 1.5, 1.0, 0.55),
    ("h288 (24h) wider", 288, 2.5, 1.0, 0.55),
]


def fetch_crypto(symbols, years):
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    c = CryptoHistoricalDataClient(S.key_id, S.secret)
    start = datetime.now(timezone.utc) - timedelta(days=int(365 * years))
    df = c.get_crypto_bars(CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start)).df
    out = {}
    for s in symbols:
        try:
            d = df.xs(s, level="symbol").copy()
        except KeyError:
            continue
        d.index = pd.to_datetime(d.index, utc=True)
        out[s] = d[["open", "high", "low", "close", "volume"]].sort_index()
    return out


def _clf(seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=60, reg_lambda=1.0,
                              random_state=seed, n_jobs=2, verbose=-1)


def walk_forward(df, feats, cfg):
    close = df["close"]
    # Labels must use the same horizon-scaled barriers the simulator uses,
    # otherwise the model is trained to predict a different event than the one
    # being traded.
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
        cols = [c for c in feats.columns] + [c for c in rtr.columns
                                             if c.startswith("regime_p")]
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


def simulate(sig, capital=5000.0, risk=0.01, pt=1.5, sl=1.0, horizon=48,
             thresh=0.55, cost=COST_MAKER, lag=1, long_only=True):
    idx = sig.index
    close, high, low = (sig["close"].values, sig["high"].values, sig["low"].values)
    vol = np.nan_to_num(sig["vol"].values, nan=np.nanmedian(sig["vol"].values))
    side_a, mp = sig["side"].values, sig["meta_p"].values

    eq = capital
    pos = 0
    entry = qty = tp = slv = 0.0
    held = 0
    trades = []
    curve = np.empty(len(idx))
    skipped_short = 0

    for i in range(len(idx)):
        if pos != 0:
            held += 1
            hit_tp = (pos == 1 and high[i] >= tp) or (pos == -1 and low[i] <= tp)
            hit_sl = (pos == 1 and low[i] <= slv) or (pos == -1 and high[i] >= slv)
            if hit_tp or hit_sl or held >= horizon:
                if hit_sl:
                    px, reason = slv, "sl"
                elif hit_tp:
                    px, reason = tp, "tp"
                else:
                    px, reason = close[i], "time"
                gross = pos * (px - entry) * qty
                fees = (entry + px) * qty * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": pos, "bars": held,
                               "reason": reason, "fees": fees})
                pos, qty, held = 0, 0.0, 0

        src = i - lag
        if pos == 0 and src >= 0 and mp[src] >= thresh:
            if eq <= capital * 0.05:
                break                      # bankrupt: stop, do not keep trading
            s = int(side_a[src])
            if long_only and s < 0:
                skipped_short += 1
                curve[i] = eq
                continue
            # Barriers must scale with the HOLDING PERIOD, not the per-bar vol.
            # vol is a 5-min std (~0.2%); a 24h horizon with pt=1.5*vol still
            # targets 0.3%, which gets hit in minutes -- so the horizon never binds
            # and fees dominate. Scaling by sqrt(horizon) makes the target match
            # the time actually being held (random-walk scaling).
            v = max(vol[src], 1e-4) * np.sqrt(horizon)
            stop_dist = sl * v * close[i]
            q = (capital * risk) / max(stop_dist, 1e-6)
            q = min(q, capital / close[i])       # spot: no leverage
            if q > 0:
                pos, entry, qty, held = s, close[i], q, 0
                tp = entry * (1 + s * pt * v)
                slv = entry * (1 - s * sl * v)
        curve[i] = eq

    return pd.DataFrame(trades), pd.Series(curve, index=idx), skipped_short


def stats(trades, curve, capital=5000.0, days=365):
    if trades.empty:
        return {"trades": 0, "ret_pct": 0.0, "win_pct": 0.0, "be_pct": None,
                "edge": None, "pf": 0.0, "dd_pct": 0.0, "sharpe": None,
                "fees_pct": 0.0, "trades_day": 0.0, "avg_hold_h": 0.0}
    daily = curve.resample("1D").last().dropna().pct_change().dropna()
    sharpe = (daily.mean() / (daily.std() + 1e-12) * np.sqrt(365)
              if len(daily) > 2 else None)
    roll = curve.cummax()
    w = trades.loc[trades.pnl > 0, "pnl"]
    l = trades.loc[trades.pnl <= 0, "pnl"]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-l.mean()) if len(l) else 0.0
    be = (al / (aw + al) * 100) if (aw + al) > 0 else None
    win = float((trades.pnl > 0).mean()) * 100
    return {
        "trades": int(len(trades)),
        "ret_pct": round(float(curve.iloc[-1] / capital - 1) * 100, 2),
        "win_pct": round(win, 1),
        "be_pct": round(be, 1) if be else None,
        "edge": round(win - be, 2) if be else None,
        "pf": round(float(w.sum() / (-l.sum() + 1e-9)), 2),
        "dd_pct": round(float(((curve - roll) / roll).min()) * 100, 2),
        "sharpe": round(float(sharpe), 2) if sharpe is not None else None,
        "fees_pct": round(float(trades["fees"].sum()) / capital * 100, 1),
        "trades_day": round(len(trades) / max(days, 1), 2),
        "avg_hold_h": round(float(trades["bars"].mean()) * 5 / 60, 1),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("=" * 100)
    print("CRYPTO BACKTEST -- Alpaca spot, LONG ONLY, 0.15%/side maker "
          "(0.30% round trip)")
    print("  equities comparison: ~0.015%/side. Crypto costs 10x more.")
    print("=" * 100)

    log.info("fetching %.1f year of 5-min crypto bars", YEARS)
    hist = fetch_crypto(SYMBOLS, YEARS)
    for k, v in hist.items():
        log.info("  %s: %d bars %s -> %s", k, len(v), v.index[0].date(),
                 v.index[-1].date())
    if MARKET not in hist:
        print("no market data")
        return 1
    mkt = hist[MARKET][["close"]].rename(columns={"close": "SPY_close"})

    rows = []
    for sym in SYMBOLS:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        if len(df) < 10000:
            continue
        cfg = ModelCfg(S)
        cfg.market_symbol = "SPY"       # column name reuse; data is BTC
        feats = build_features(df, cfg)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]

        trained = {}
        for name, hz, pt, sl, thr in CONFIGS:
            if hz not in trained:
                c = ModelCfg(S)
                c.market_symbol = "SPY"
                c.horizon_bars = hz
                trained[hz] = walk_forward(df, feats, c)
                log.info("%s trained horizon=%d", sym, hz)
            sg = trained[hz]
            if sg is None:
                continue
            days = (sg.index[-1] - sg.index[0]).days or 1
            t, curve, skipped = simulate(sg, pt=pt, sl=sl, horizon=hz, thresh=thr)
            st = stats(t, curve, days=days)
            st.update(symbol=sym, config=name, skipped_short=skipped)
            rows.append(st)
            log.info("  %-18s ret %+8.1f%% win %.1f be %.1f edge %+.2f trades %d",
                     name, st["ret_pct"], st["win_pct"], st["be_pct"] or 0,
                     st["edge"] or 0, st["trades"])

    d = pd.DataFrame(rows)
    if d.empty:
        print("no results")
        return 1
    d.to_json("/app/state/crypto_bt_raw.json", orient="records")

    g = (d.groupby("config")
           .agg(ret=("ret_pct", "mean"), win=("win_pct", "mean"),
                be=("be_pct", "mean"), edge=("edge", "mean"), pf=("pf", "mean"),
                sharpe=("sharpe", "mean"), dd=("dd_pct", "mean"),
                fees=("fees_pct", "mean"), trades=("trades", "mean"),
                tpd=("trades_day", "mean"), hold_h=("avg_hold_h", "mean"),
                n=("symbol", "count"))
           .reset_index())
    for c in ("ret", "win", "be", "edge", "pf", "sharpe", "dd", "fees", "tpd", "hold_h"):
        g[c] = g[c].round(2)
    g["trades"] = g["trades"].round(0)
    order = [c[0] for c in CONFIGS]
    g["_o"] = g["config"].map({n: i for i, n in enumerate(order)})
    g = g.sort_values("_o").drop(columns="_o")

    print("\nBY CONFIG (mean across BTC/ETH/SOL):")
    print(g.to_string(index=False))

    print("\nPER SYMBOL RETURN %:")
    print(d.pivot_table(index="symbol", columns="config",
                        values="ret_pct").round(1).to_string())

    print("\nPER SYMBOL EDGE (win - breakeven):")
    print(d.pivot_table(index="symbol", columns="config",
                        values="edge").round(2).to_string())

    sk = int(d["skipped_short"].sum())
    print(f"\nshort signals skipped (spot is long-only): {sk:,}")
    g.to_json("/app/state/crypto_bt_summary.json", orient="records")
    print("saved /app/state/crypto_bt_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

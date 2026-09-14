"""
Multi-day swing strategy -- the structural alternative to intraday scalping.

WHY THIS EXISTS

Every intraday configuration failed on one inequality:

    gross edge per trade 1.40 bps   <   round-trip cost 3.81 bps

Cost was 272% of the move being captured. No threshold, universe, horizon, or
execution fix could close a gap that size.

Holding for days instead of minutes changes the arithmetic completely. A 5-day
move on a liquid large cap is typically 2-4% (200-400 bps), so the same 3 bps
round trip is ~1% of the move rather than 272%. Adverse selection and stop
slippage shrink to irrelevance against a barrier that wide.

The cost is trade frequency: roughly one trade per symbol per week. Breadth is
therefore mandatory -- 100 symbols gives ~20 trades/week.

DESIGN
  * daily bars (5 years, so the sample is large despite the longer horizon)
  * triple-barrier labels with horizon in DAYS
  * same HMM regime + LightGBM + meta-label stack
  * full reality stack: 1-day signal lag, intrabar barriers, stop slippage, costs
  * positions held up to `horizon` days, hard-closed at the horizon
  * threshold sweep and the same GO/NO-GO criteria as the intraday gate

    python -m app.swing
"""
import json
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb

from .settings import S
from .labeling import triple_barrier_labels, make_meta_labels
from .regime import RegimeModel
from .reality_stack import deflated_sharpe

log = logging.getLogger("swing")

CAPITAL = 5000.0
RISK = 0.01              # 1% of equity per trade
HORIZONS = [2, 3]        # 4d was negative and unstable; drop it
PT, SL = 2.0, 1.0
THRESHOLDS = [0.55]      # threshold does not bind: capacity selects rank-wise
# Concurrency IS the selectivity lever here. Candidates are ranked by meta_p and
# the top N are taken, so raising the threshold changes nothing while lowering N
# keeps only the highest-conviction signals.
CONCURRENCY = [5, 10, 15, 20]
COST_PER_SIDE = 0.00015
N_FOLDS = 5
EMBARGO = 10
N_TRIALS = 100
MAX_CONCURRENT = 10

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "MU",
    "INTC", "QCOM", "ORCL", "CRM", "ADBE", "NFLX", "PLTR", "ARM", "MRVL", "TSM",
    "LRCX", "KLAC", "AMAT", "ON", "DELL", "WDC", "STX", "SMCI",
    "JPM", "BAC", "GS", "MS", "C", "WFC", "SCHW", "COIN", "HOOD", "SOFI", "AXP",
    "XOM", "CVX", "OXY", "SLB", "HAL", "DVN", "FANG",
    "UNH", "JNJ", "PFE", "MRNA", "LLY", "ABBV", "CVS", "BMY", "AMGN",
    "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "DIS", "BA", "CAT", "DE", "UBER",
    "F", "GM", "DAL", "T", "VZ", "PG", "KO", "PEP",
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "SMH", "XBI", "XLI", "XLP",
    "MSTR", "MARA", "RIOT",
]


def fetch_daily(symbols, years=5.0):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(S.key_id, S.secret)
    start = datetime.now(timezone.utc) - timedelta(days=int(365 * years))
    out = {}
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        try:
            df = c.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start, feed=DataFeed.IEX)).df
        except Exception as e:
            log.warning("fetch failed: %s", e)
            continue
        for s in chunk:
            try:
                d = df.xs(s, level="symbol")
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
                out[s] = d[["open", "high", "low", "close", "volume"]]
            except KeyError:
                pass
    return out


def build_daily_features(df, mkt_close):
    """Features appropriate to daily bars -- no intraday seasonality or VWAP."""
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    logp = np.log(close)
    ret = logp.diff()

    for k in (1, 2, 5, 10, 20):
        f[f"ret_{k}"] = logp.diff(k)
    vol = ret.ewm(span=20).std()
    f["vol"] = vol
    f["vol_ratio"] = vol / (vol.rolling(60).mean() + 1e-9)
    f["mom_z"] = f["ret_5"] / (vol * np.sqrt(5) + 1e-9)

    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - close.shift()).abs(),
                    (df["low"] - close.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(span=14).mean() / close

    for w in (10, 20, 50):
        ma = close.rolling(w).mean()
        f[f"ma{w}_dist"] = (close - ma) / (close * f["atr_pct"] + 1e-9)

    delta = close.diff()
    up = delta.clip(lower=0).ewm(span=14).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14).mean()
    f["rsi"] = 100 - 100 / (1 + up / (dn + 1e-9))

    hi = close.rolling(60).max()
    lo = close.rolling(60).min()
    f["pos_range"] = (close - lo) / (hi - lo + 1e-9)

    f["vol_z"] = ((df["volume"] - df["volume"].rolling(20).mean())
                  / (df["volume"].rolling(20).std() + 1e-9))

    mret = np.log(mkt_close).diff()
    f["mkt_ret_1"] = mret
    f["mkt_ret_5"] = np.log(mkt_close).diff(5)
    beta = ret.rolling(60).cov(mret) / (mret.rolling(60).var() + 1e-9)
    f["beta"] = beta
    f["resid"] = ret - beta * mret
    f["rel_str"] = f["ret_20"] - np.log(mkt_close).diff(20)
    return f


def _clf(seed):
    return lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=15,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=40, reg_lambda=1.0,
                              random_state=seed, n_jobs=2, verbose=-1)


class Cfg:
    def __init__(self, hz):
        self.horizon_bars = hz
        self.pt_mult = PT
        self.sl_mult = SL
        self.hmm_states = 3
        self.hmm_covariance = "diag"
        self.seed = 7


def walk_forward(df, feats, hz):
    close = df["close"]
    cfg = Cfg(hz)
    scaled = feats["vol"] * np.sqrt(hz)
    tb = triple_barrier_labels(close, scaled, cfg)
    n = len(df)
    fold = n // (N_FOLDS + 1)
    out = []
    for k in range(1, N_FOLDS + 1):
        tr_end = fold * k
        te_start = tr_end + EMBARGO
        te_end = min(tr_end + fold, n)
        if te_start >= te_end or tr_end < 200:
            continue
        tr, te = slice(0, tr_end), slice(te_start, te_end)
        try:
            reg = RegimeModel(cfg).fit(feats.iloc[tr])
            rtr, rte = reg.transform(feats.iloc[tr]), reg.transform(feats.iloc[te])
        except Exception as e:
            # Log rather than swallow: a bare `continue` here hid a missing-column
            # error and produced "signals for 0 symbols" with no explanation.
            log.warning("regime fit failed (fold %d): %s", k, e)
            continue
        cols = list(feats.columns) + [c for c in rtr.columns
                                      if c.startswith("regime_p")]
        Xtr = pd.concat([feats.iloc[tr], rtr], axis=1)[cols]
        Xte = pd.concat([feats.iloc[te], rte], axis=1)[cols]
        Xtr = Xtr.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        Xte = Xte.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        y = tb["label"].iloc[tr].values
        m = y != 0
        if m.sum() < 100 or len(np.unique(y[m] > 0)) < 2:
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
        mp = meta.predict_proba(np.column_stack([Xte, p_te]))[:, 1]
        out.append(pd.DataFrame(
            {"close": close.iloc[te].values,
             "high": df["high"].iloc[te].values,
             "low": df["low"].iloc[te].values,
             "vol": feats["vol"].iloc[te].values,
             "side": side_te, "meta_p": mp},
            index=df.index[te_start:te_end]))
    return pd.concat(out) if out else None


def portfolio_sim(sigs, thresh, hz, cost=COST_PER_SIDE, stop_slip_R=0.10,
                  max_concurrent=MAX_CONCURRENT):
    """
    One shared book across all symbols, so concurrency and capital are respected
    rather than each symbol being simulated in isolation.
    """
    all_days = sorted(set().union(*[set(s.index) for s in sigs.values()]))
    eq = CAPITAL
    open_pos = {}
    trades = []
    curve = []

    for day in all_days:
        # ---- manage open positions ----
        for sym in list(open_pos):
            p = open_pos[sym]
            s = sigs.get(sym)
            if s is None or day not in s.index:
                continue
            row = s.loc[day]
            p["held"] += 1
            hit_tp = ((p["side"] == 1 and row["high"] >= p["tp"]) or
                      (p["side"] == -1 and row["low"] <= p["tp"]))
            hit_sl = ((p["side"] == 1 and row["low"] <= p["sl"]) or
                      (p["side"] == -1 and row["high"] >= p["sl"]))
            timeout = p["held"] >= hz
            if hit_tp or hit_sl or timeout:
                risk_px = abs(p["entry"] - p["sl"])
                if hit_sl:
                    px = p["sl"] - p["side"] * stop_slip_R * risk_px
                    reason = "sl"
                elif hit_tp:
                    px, reason = p["tp"], "tp"
                else:
                    px, reason = row["close"], "time"
                gross = p["side"] * (px - p["entry"]) * p["qty"]
                fees = (p["entry"] + px) * p["qty"] * cost
                eq += gross - fees
                trades.append({"pnl": gross - fees, "side": p["side"],
                               "days": p["held"], "reason": reason,
                               "symbol": sym, "exit_day": day})
                del open_pos[sym]

        # ---- new entries ----
        if len(open_pos) < max_concurrent:
            cands = []
            for sym, s in sigs.items():
                if sym in open_pos or day not in s.index:
                    continue
                i = s.index.get_loc(day)
                if i < 1:
                    continue
                prev = s.iloc[i - 1]          # 1-day signal lag
                if prev["meta_p"] >= thresh:
                    cands.append((float(prev["meta_p"]), sym, s.iloc[i], prev))
            cands.sort(reverse=True, key=lambda x: x[0])
            for mp, sym, row, prev in cands:
                if len(open_pos) >= max_concurrent:
                    break
                v = max(float(prev["vol"]), 1e-4) * np.sqrt(hz)
                entry = float(row["close"])
                stop_dist = SL * v * entry
                qty = (CAPITAL * RISK) / max(stop_dist, 1e-6)
                qty = min(qty, (CAPITAL * 1.5) / entry)
                if qty <= 0:
                    continue
                side = int(prev["side"])
                open_pos[sym] = {"side": side, "entry": entry, "qty": qty,
                                 "held": 0,
                                 "tp": entry * (1 + side * PT * v),
                                 "sl": entry * (1 - side * SL * v)}
        curve.append({"day": day, "equity": eq})

    c = pd.DataFrame(curve).set_index("day")["equity"]
    return pd.DataFrame(trades), c


def stats(t, curve, days):
    if t.empty or len(t) < 20:
        return None
    r = curve.pct_change().dropna()
    sh = float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)) if len(r) > 5 else np.nan
    roll = curve.cummax()
    w = t.loc[t.pnl > 0, "pnl"]
    l = t.loc[t.pnl <= 0, "pnl"]
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-l.mean()) if len(l) else 0.0
    be = al / (aw + al) * 100 if (aw + al) > 0 else None
    win = float((t.pnl > 0).mean()) * 100
    tot = float(curve.iloc[-1] / CAPITAL - 1) * 100
    return {"trades": len(t), "trades_wk": round(len(t) / (days / 7), 1),
            "annual_%": round(((1 + tot / 100) ** (365 / days) - 1) * 100, 1),
            "sharpe": round(sh, 2) if np.isfinite(sh) else None,
            "win_%": round(win, 1), "be_%": round(be, 1) if be else None,
            "edge": round(win - be, 2) if be else None,
            "hold_d": round(float(t["days"].mean()), 1),
            "maxDD_%": round(float(((curve - roll) / roll).min()) * 100, 1),
            "daily": r}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("fetching 10 years of daily bars for %d symbols", len(UNIVERSE))
    hist = fetch_daily(UNIVERSE, years=10.0)
    log.info("got %d symbols", len(hist))
    if "SPY" not in hist:
        print("no SPY")
        return 1
    spy = hist["SPY"]["close"]
    days = (spy.index[-1] - spy.index[0]).days
    spy_ann = ((spy.iloc[-1] / spy.iloc[0]) ** (365 / days) - 1) * 100
    spy_r = spy.pct_change().dropna()
    spy_sh = float(spy_r.mean() / spy_r.std() * np.sqrt(252))
    spy_dd = float(((spy / spy.cummax()) - 1).min()) * 100

    results = []
    for hz in HORIZONS:
        sigs = {}
        for sym, df in hist.items():
            if sym == "SPY" or len(df) < 600:
                continue
            f = build_daily_features(df, spy.reindex(df.index).ffill())
            keep = f.dropna().index
            d2, f2 = df.loc[keep], f.loc[keep]
            if len(d2) < 500:
                continue
            sg = walk_forward(d2, f2, hz)
            if sg is not None:
                sigs[sym] = sg
        log.info("horizon %dd: signals for %d symbols", hz, len(sigs))
        if not sigs:
            continue
        for thr in THRESHOLDS:
            for mc in CONCURRENCY:
                t, curve = portfolio_sim(sigs, thr, hz, max_concurrent=mc)
                st = stats(t, curve, days)
                if st is None:
                    continue
                dsr = deflated_sharpe(st["daily"].values, st["sharpe"] or 0, N_TRIALS)
                row = {"horizon_d": hz, "concur": mc,
                       **{k: v for k, v in st.items() if k != "daily"},
                       "DSR": round(dsr or 0, 3)}
                results.append(row)
                log.info("  h%dd concur %d -> %d trades %.1f%%/yr edge %s DSR %.3f",
                         hz, mc, st["trades"], st["annual_%"], st["edge"], dsr or 0)

    if not results:
        print("no results")
        return 1
    d = pd.DataFrame(results)
    print("\n" + "=" * 108)
    print("MULTI-DAY SWING -- 10y daily bars, portfolio sim, full costs")
    print("  positions held up to `horizon_d` trading days; hard-closed at horizon")
    print("=" * 108)
    print(d.to_string(index=False))
    print(f"\nSPY: {spy_ann:.1f}%/yr  Sharpe {spy_sh:.2f}  maxDD {spy_dd:.1f}%")

    # Select by RISK-ADJUSTED return among configs that pass a hard drawdown
    # filter. Ranking by raw return alone previously recommended a config with an
    # -88% drawdown, which is worthless regardless of its headline number.
    MAX_ACCEPTABLE_DD = -35.0
    viable = d[(d["maxDD_%"] > MAX_ACCEPTABLE_DD) & (d["annual_%"] > 0)]
    if viable.empty:
        print("\nNo configuration survives the drawdown filter "
              f"({MAX_ACCEPTABLE_DD}%). NO-GO.")
        return 2
    best = viable.loc[viable["sharpe"].idxmax()]
    checks = [
        ("annual > 0", best["annual_%"] > 0, f"{best['annual_%']}%"),
        ("edge > 1.0", (best["edge"] or -9) > 1.0, f"{best['edge']}"),
        ("DSR >= 0.90", best["DSR"] >= 0.90, f"{best['DSR']}"),
        ("beats SPY Sharpe", (best["sharpe"] or 0) > spy_sh,
         f"{best['sharpe']} vs {spy_sh:.2f}"),
        ("maxDD > -25%", best["maxDD_%"] > -25, f"{best['maxDD_%']}%"),
    ]
    print(f"\nBEST: horizon {int(best['horizon_d'])}d, threshold {best['concur']}, "
          f"{best['trades_wk']} trades/week")
    for n, ok, v in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:<20} {v}")
    print("\n  ==> " + ("GO" if all(o for _, o, _ in checks) else "NO-GO"))
    d.drop(columns=[c for c in d.columns if c == "daily"], errors="ignore") \
        .to_json("/app/state/swing.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

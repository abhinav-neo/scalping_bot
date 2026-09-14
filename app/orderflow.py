"""
Order-flow features -- the last untested information source.

Everything so far used OHLCV bars, and that ceiling is measured: ~1.40 bps gross
edge per trade against ~5.97 bps of real cost. A TCN on the same bars did worse
than LightGBM, which says the limit is the DATA, not the model.

Trade prints and quotes are a different information source. From them we can build
what bars structurally cannot express:

  ORDER FLOW IMBALANCE (OFI)  signed volume from quote revisions -- the standard
      microstructure predictor of short-horizon returns
  TRADE SIGN (Lee-Ready)      classify each print as buyer- or seller-initiated by
      comparing to the prevailing quote midpoint
  VPIN-style toxicity         volume-synchronised imbalance, a proxy for informed
      flow
  EFFECTIVE SPREAD            what we actually paid, not the quoted spread
  QUOTE INTENSITY             revisions per second: quoting activity spikes ahead
      of moves
  DEPTH IMBALANCE             bid size vs ask size at the top of book
  LARGE-TRADE RATIO           share of volume in prints above the 90th percentile,
      a crude institutional-participation proxy

Test: do these lift gross edge per trade above the ~6 bps needed to clear costs?
Judged on the same metric as everything else, so results are comparable.

    python -m app.orderflow
"""
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb

from .settings import S

log = logging.getLogger("orderflow")

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "SPY", "QQQ", "BAC", "F", "INTC"]
BAR = "5min"
HORIZON = 6          # predict 30 minutes ahead
DAYS_BACK = 12       # trading days of tick data (this is heavy)


def fetch_ticks(sym, day_start, day_end):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockTradesRequest, StockQuotesRequest
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(S.key_id, S.secret)
    tr = c.get_stock_trades(StockTradesRequest(
        symbol_or_symbols=sym, start=day_start, end=day_end,
        feed=DataFeed.IEX)).df
    qt = c.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols=sym, start=day_start, end=day_end,
        feed=DataFeed.IEX)).df
    if len(tr):
        tr = tr.reset_index().set_index("timestamp")
    if len(qt):
        qt = qt.reset_index().set_index("timestamp")
    return tr, qt


def lee_ready(trades, quotes):
    """Classify each trade as buy (+1) or sell (-1) against the prevailing quote."""
    if not len(trades) or not len(quotes):
        return pd.Series(dtype=float)
    q = quotes[["bid_price", "ask_price", "bid_size", "ask_size"]].copy()
    q["mid"] = (q["bid_price"] + q["ask_price"]) / 2
    merged = pd.merge_asof(trades[["price", "size"]].sort_index(),
                           q.sort_index(), left_index=True, right_index=True,
                           direction="backward")
    sign = np.where(merged["price"] > merged["mid"], 1.0,
                    np.where(merged["price"] < merged["mid"], -1.0, np.nan))
    # ties: fall back to the tick rule
    s = pd.Series(sign, index=merged.index).ffill().fillna(0.0)
    merged["sign"] = s
    merged["eff_spread_bps"] = (2 * (merged["price"] - merged["mid"]).abs()
                                / merged["mid"] * 1e4)
    return merged


def build_of_features(merged, quotes, bar=BAR):
    """Aggregate tick-level data into bar-level order-flow features."""
    if not len(merged):
        return pd.DataFrame()
    m = merged.copy()
    m["signed_vol"] = m["sign"] * m["size"]
    m["dollar"] = m["price"] * m["size"]
    big = m["size"].quantile(0.90)
    m["is_big"] = (m["size"] >= big).astype(float)

    g = m.resample(bar)
    f = pd.DataFrame(index=g.size().index)
    vol = g["size"].sum()
    f["volume"] = vol
    f["n_trades"] = g["size"].count()
    f["ofi"] = g["signed_vol"].sum() / (vol + 1e-9)           # normalised imbalance
    f["buy_ratio"] = g.apply(lambda x: (x["sign"] > 0).mean() if len(x) else np.nan)
    f["eff_spread"] = g["eff_spread_bps"].mean()
    f["avg_trade_size"] = vol / (f["n_trades"] + 1e-9)
    f["big_trade_share"] = g.apply(
        lambda x: (x["size"] * x["is_big"]).sum() / (x["size"].sum() + 1e-9)
        if len(x) else np.nan)
    f["signed_big"] = g.apply(
        lambda x: (x["sign"] * x["size"] * x["is_big"]).sum() / (x["size"].sum() + 1e-9)
        if len(x) else np.nan)
    # VPIN-ish: absolute imbalance over the window
    f["vpin"] = g.apply(
        lambda x: abs(x["signed_vol"].sum()) / (x["size"].sum() + 1e-9)
        if len(x) else np.nan)
    f["price"] = g["price"].last()

    if len(quotes):
        q = quotes.copy()
        q["mid"] = (q["bid_price"] + q["ask_price"]) / 2
        q["depth_imb"] = ((q["bid_size"] - q["ask_size"])
                          / (q["bid_size"] + q["ask_size"] + 1e-9))
        qg = q.resample(bar)
        f["quote_intensity"] = qg["mid"].count()
        f["depth_imb"] = qg["depth_imb"].mean()
        f["quoted_spread_bps"] = qg.apply(
            lambda x: float(((x["ask_price"] - x["bid_price"]) / x["mid"]).mean() * 1e4)
            if len(x) else np.nan)
        # OFI from quote revisions (Cont et al.)
        dq = q[["bid_price", "bid_size", "ask_price", "ask_size"]].diff()
        e = np.where(dq["bid_price"] >= 0, q["bid_size"], 0) - \
            np.where(dq["bid_price"] <= 0, q["bid_size"].shift(), 0) - \
            np.where(dq["ask_price"] <= 0, q["ask_size"], 0) + \
            np.where(dq["ask_price"] >= 0, q["ask_size"].shift(), 0)
        f["ofi_quote"] = pd.Series(e, index=q.index).resample(bar).sum()
    return f


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    end_day = datetime.now(timezone.utc).date()
    rows = []

    for sym in SYMBOLS:
        parts = []
        got = 0
        d = end_day
        while got < DAYS_BACK and (end_day - d).days < 30:
            d -= timedelta(days=1)
            if d.weekday() >= 5:
                continue
            s = datetime(d.year, d.month, d.day, 13, 30, tzinfo=timezone.utc)
            e = datetime(d.year, d.month, d.day, 20, 0, tzinfo=timezone.utc)
            try:
                tr, qt = fetch_ticks(sym, s, e)
            except Exception as ex:
                log.warning("%s %s fetch failed: %s", sym, d, str(ex)[:80])
                continue
            if not len(tr) or not len(qt):
                continue
            merged = lee_ready(tr, qt)
            f = build_of_features(merged, qt)
            if len(f):
                parts.append(f)
                got += 1
        if not parts:
            log.warning("%s: no data", sym)
            continue
        F = pd.concat(parts).sort_index().dropna()
        if len(F) < 300:
            log.warning("%s: only %d bars", sym, len(F))
            continue

        # forward return over HORIZON bars, in bps
        px = F["price"]
        fwd = (np.log(px.shift(-HORIZON) / px) * 1e4).dropna()
        F = F.loc[fwd.index]

        # baseline features from price/volume only (what bars already give us)
        base_cols = ["volume", "n_trades", "avg_trade_size"]
        of_cols = [c for c in F.columns if c not in base_cols + ["price"]]

        # add simple returns so the baseline is a fair comparison
        F["ret_1"] = np.log(px / px.shift(1)).reindex(F.index)
        F["ret_6"] = np.log(px / px.shift(6)).reindex(F.index)
        F = F.dropna()
        fwd = fwd.reindex(F.index)
        base_cols += ["ret_1", "ret_6"]

        y = (fwd > 0).astype(int)
        split = int(len(F) * 0.7)

        def run(cols, tag):
            X = F[cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
            m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03,
                                   num_leaves=15, min_child_samples=40,
                                   subsample=0.8, colsample_bytree=0.8,
                                   reg_lambda=1.0, random_state=7, n_jobs=4,
                                   verbose=-1)
            m.fit(X[:split], y.values[:split])
            p = m.predict_proba(X[split:])[:, 1]
            side = np.where(p >= 0.5, 1, -1)
            fw = fwd.values[split:]
            acc = float(((p >= 0.5) == (y.values[split:] > 0)).mean())
            edge = float(np.mean(side * fw))
            conf = np.abs(p - 0.5)
            hi = conf >= np.quantile(conf, 0.8)
            edge_hi = float(np.mean(side[hi] * fw[hi])) if hi.sum() > 10 else np.nan
            return acc, edge, edge_hi

        a_b, e_b, eh_b = run(base_cols, "base")
        a_o, e_o, eh_o = run(base_cols + of_cols, "of")
        base_rate = float(max(y.values[split:].mean(),
                              1 - y.values[split:].mean()))
        rows.append({"symbol": sym, "bars": len(F), "base_%": round(base_rate * 100, 1),
                     "acc_bars_%": round(a_b * 100, 1),
                     "acc_flow_%": round(a_o * 100, 1),
                     "edge_bars": round(e_b, 2),
                     "edge_flow": round(e_o, 2),
                     "edge_flow_top20": round(eh_o, 2)})
        log.info("%-5s bars %d | acc %.1f -> %.1f | edge %.2f -> %.2f (top20 %.2f) bps",
                 sym, len(F), a_b * 100, a_o * 100, e_b, e_o, eh_o)

    d = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("ORDER FLOW vs BARS -- gross edge in bps per trade, held-out 30%")
    print("  must exceed ~5.97 bps (measured round-trip cost) to be viable")
    print("=" * 100)
    print(d.to_string(index=False))
    if len(d):
        print(f"\nmean edge from bars      : {d['edge_bars'].mean():.2f} bps")
        print(f"mean edge with order flow: {d['edge_flow'].mean():.2f} bps")
        print(f"top-20% conviction       : {d['edge_flow_top20'].mean():.2f} bps")
        best = max(d['edge_flow'].mean(), d['edge_flow_top20'].mean())
        print("\n  -> " + ("VIABLE: order flow clears the cost hurdle"
                           if best > 5.97 else
                           f"still short: {best:.2f} vs 5.97 bps needed"))
    d.to_json("state/orderflow.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

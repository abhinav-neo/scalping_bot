"""
Two untested levers against the binding inequality:

    gross edge per trade (1.40 bps)  <  cost per trade (3.81 bps)

LEVER 1 -- is the cost assumption right?
    1.5 bps/side was picked early and never verified. Mega-cap spreads are often
    0.5-1 bp TOTAL. If real cost is ~1.5 bps round trip rather than 3.8, the
    inequality is nearly reversed without changing the strategy at all. This
    measures live quoted spreads per symbol and re-runs at measured cost.

LEVER 2 -- time of day.
    Intraday volatility is U-shaped: the first and last hour move far more than
    midday, while spreads are comparable. Same cost, bigger move, so edge-to-cost
    should be much better in those windows. Never tested.

    python -m app.cost_and_time
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
from .reality_stack import simulate, deflated_sharpe

log = logging.getLogger("cost_time")

CAPITAL = 5000.0
HORIZON = 6
PT, SL = 1.5, 1.0
THRESH = 0.85


def measure_spreads(symbols):
    """Live quoted spread per symbol, in bps of mid."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(S.key_id, S.secret)
    rows = []
    try:
        q = c.get_stock_latest_quote(StockLatestQuoteRequest(
            symbol_or_symbols=list(symbols), feed=DataFeed.IEX))
    except Exception as e:
        log.warning("quote fetch failed: %s", e)
        return pd.DataFrame()
    for s, v in q.items():
        bid, ask = float(v.bid_price or 0), float(v.ask_price or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        rows.append({"symbol": s, "bid": bid, "ask": ask,
                     "spread_bps": (ask - bid) / mid * 1e4})
    return pd.DataFrame(rows)


def simulate_hours(sig, hour_lo, hour_hi, **kw):
    """Restrict entries to a clock window by blanking meta_p outside it."""
    s2 = sig.copy()
    h = s2.index.hour + s2.index.minute / 60.0
    mask = (h >= hour_lo) & (h < hour_hi)
    s2.loc[~mask, "meta_p"] = 0.0
    return simulate(s2, **kw)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})
    spy_d = spy["close"].resample("1D").last().dropna()
    days = (spy_d.index[-1] - spy_d.index[0]).days or 1

    # ---------------- LEVER 1: measured spreads ----------------
    print("=" * 92)
    print("LEVER 1 -- MEASURED SPREADS vs THE 1.5 bps/side ASSUMPTION")
    print("=" * 92)
    sp = measure_spreads(S.symbols)
    if len(sp):
        sp = sp.sort_values("spread_bps")
        print(sp.head(12).round(2).to_string(index=False))
        med = float(sp["spread_bps"].median())
        print(f"\n  median quoted spread : {med:.2f} bps")
        print(f"  crossing it costs    : {med/2:.2f} bps per side")
        print(f"  our assumption       : 1.50 bps per side")
        print("  NOTE: IEX quotes are one venue and print wider than the NBBO.")
        print("        Treat this as an upper bound on the true spread.")
    else:
        med = None
        print("  no live quotes (market closed)")

    # candidate per-side costs to test
    COSTS = [0.00005, 0.0001, 0.00015, 0.00025]

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

    sigs = {}
    for sym, (df, feats) in prepared.items():
        c = ModelCfg(S)
        c.horizon_bars = HORIZON
        sg = walk_forward(df, feats, c)
        if sg is not None:
            sigs[sym] = sg
    log.info("signals for %d symbols", len(sigs))

    BASE = dict(lag=1, intrabar=True, stop_slip_R=0.10, limit_fills=True)

    print("\n" + "=" * 92)
    print("COST SENSITIVITY (full reality stack, threshold %.2f)" % THRESH)
    print("=" * 92)
    rows = []
    for cost in COSTS:
        per, curves = [], []
        for sym, sg in sigs.items():
            t, curve, _ = simulate(sg, thresh=THRESH, pt=PT, sl=SL,
                                   horizon=HORIZON, cost=cost, **BASE)
            if t.empty or len(t) < 10:
                continue
            _, _, e = edge_of(t)
            roll = curve.cummax()
            per.append({"ret": float(curve.iloc[-1] / CAPITAL - 1) * 100,
                        "edge": e, "trades": len(t),
                        "dd": float(((curve - roll) / roll).min()) * 100})
            curves.append(curve.resample("1D").last().dropna().pct_change().dropna())
        if not per:
            continue
        dl = pd.concat(curves, axis=1).mean(axis=1).dropna()
        sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252))
        rows.append({"cost_bps_side": round(cost * 1e4, 2),
                     "roundtrip_bps": round(cost * 2e4, 2),
                     "annual_%": round(float(np.mean([p["ret"] for p in per])) * 365 / days, 1),
                     "sharpe": round(sh, 2),
                     "edge": round(float(np.mean([p["edge"] for p in per
                                                  if p["edge"] is not None])), 2),
                     "maxDD_%": round(float(np.mean([p["dd"] for p in per])), 1),
                     "trades": int(np.sum([p["trades"] for p in per]))})
        log.info("cost %.2fbps/side -> %.1f%%/yr edge %.2f",
                 cost * 1e4, rows[-1]["annual_%"], rows[-1]["edge"])
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------------- LEVER 2: time of day ----------------
    print("\n" + "=" * 92)
    print("LEVER 2 -- TIME OF DAY  (cost 1.0 bps/side)")
    print("=" * 92)
    WINDOWS = [("full session", 9.5, 15.75),
               ("open hour", 9.5, 10.5),
               ("open 90m", 9.5, 11.0),
               ("midday", 11.0, 14.0),
               ("close hour", 14.75, 15.75),
               ("open+close", None, None)]
    trows = []
    for name, lo, hi in WINDOWS:
        per, curves = [], []
        for sym, sg in sigs.items():
            if name == "open+close":
                s2 = sg.copy()
                h = s2.index.hour + s2.index.minute / 60.0
                m = ((h >= 9.5) & (h < 11.0)) | ((h >= 14.75) & (h < 15.75))
                s2.loc[~m, "meta_p"] = 0.0
                t, curve, _ = simulate(s2, thresh=THRESH, pt=PT, sl=SL,
                                       horizon=HORIZON, cost=0.0001, **BASE)
            else:
                t, curve, _ = simulate_hours(sg, lo, hi, thresh=THRESH, pt=PT,
                                             sl=SL, horizon=HORIZON,
                                             cost=0.0001, **BASE)
            if t.empty or len(t) < 10:
                continue
            _, _, e = edge_of(t)
            roll = curve.cummax()
            per.append({"ret": float(curve.iloc[-1] / CAPITAL - 1) * 100,
                        "edge": e, "trades": len(t),
                        "ppt": float(t["pnl"].mean()),
                        "dd": float(((curve - roll) / roll).min()) * 100})
            curves.append(curve.resample("1D").last().dropna().pct_change().dropna())
        if not per:
            continue
        dl = pd.concat(curves, axis=1).mean(axis=1).dropna()
        sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252))
        tot = int(np.sum([p["trades"] for p in per]))
        trows.append({"window": name,
                      "trades": tot, "trades_day": round(tot / days, 1),
                      "annual_%": round(float(np.mean([p["ret"] for p in per])) * 365 / days, 1),
                      "sharpe": round(sh, 2),
                      "edge": round(float(np.mean([p["edge"] for p in per
                                                   if p["edge"] is not None])), 2),
                      "bps_trade": round(float(np.mean([p["ppt"] for p in per]))
                                         / CAPITAL * 1e4, 2),
                      "maxDD_%": round(float(np.mean([p["dd"] for p in per])), 1),
                      "DSR": round(deflated_sharpe(dl.values, sh, 90) or 0, 3)})
        log.info("%-14s %d trades %.1f%%/yr edge %.2f",
                 name, tot, trows[-1]["annual_%"], trows[-1]["edge"])
    print(pd.DataFrame(trows).to_string(index=False))

    print("\n" + "=" * 92)
    print("READ")
    print("=" * 92)
    print("  If lower cost alone flips the sign, the strategy was always viable and")
    print("  the cost assumption was the error. If only certain hours are positive,")
    print("  the edge is concentrated in time and the fix is a session filter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

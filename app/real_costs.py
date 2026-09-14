"""Estimate real per-symbol transaction costs from bar data.

Every intraday test used a flat 1.5 bps/side, never measured. Real spreads vary
10x across instruments. Measured intraday gross edge was 1.40 bps/trade, so the
question is which symbols have round-trip costs below that.

Roll (1984): bid-ask bounce induces negative serial covariance in price changes.
Corwin-Schultz (2012): high/low ranges embed the spread.
Take the larger estimate, floor at one tick, add fees.
"""
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .settings import S

log = logging.getLogger("real_costs")

LIQUID = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
          "SMH", "EEM", "EFA", "TLT", "HYG", "GLD", "SLV", "VXX",
          "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC",
          "BAC", "F", "T", "PFE", "KO", "CSCO", "WFC", "XOM", "VZ", "MU", "PLTR"]

FEE_BPS_PER_SIDE = 0.02


def roll_spread_bps(close):
    dp = close.diff().dropna()
    if len(dp) < 100:
        return np.nan
    cov = float(np.cov(dp.values[:-1], dp.values[1:])[0, 1])
    if cov >= 0:
        return np.nan
    return 2.0 * np.sqrt(-cov) / float(close.mean()) * 1e4


def corwin_schultz_bps(high, low):
    if len(high) < 100:
        return np.nan
    hl = np.log(high / low) ** 2
    beta = (hl + hl.shift(1)).dropna()
    h2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = (np.log(h2 / l2) ** 2).dropna()
    n = min(len(beta), len(gamma))
    if n < 50:
        return np.nan
    beta, gamma = beta.values[-n:], gamma.values[-n:]
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(np.maximum(gamma, 0) / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) == 0:
        return np.nan
    return float(np.median(s)) * 1e4


def fetch_5m(symbols, days=45):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(S.key_id, S.secret)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    out = {}
    for i in range(0, len(symbols), 30):
        chunk = symbols[i:i + 30]
        try:
            df = c.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start, feed=DataFeed.IEX)).df
        except Exception as e:
            log.warning("fetch failed: %s", e)
            continue
        for s in chunk:
            try:
                d = df.xs(s, level="symbol")
                d.index = pd.to_datetime(d.index, utc=True).tz_convert(
                    "America/New_York")
                out[s] = d.between_time("09:30", "16:00")
            except KeyError:
                pass
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bars = fetch_5m(LIQUID)
    log.info("got %d symbols", len(bars))
    rows = []
    for s, d in bars.items():
        if len(d) < 500:
            continue
        roll = roll_spread_bps(d["close"])
        cs = corwin_schultz_bps(d["high"], d["low"])
        cand = [x for x in (roll, cs) if np.isfinite(x) and x > 0]
        if not cand:
            continue
        est = max(cand)
        px = float(d["close"].iloc[-1])
        est = max(est, 0.01 / px * 1e4)
        per_side = est / 2 + FEE_BPS_PER_SIDE
        vol5 = float(np.log(d["close"]).diff().std() * 1e4)
        rows.append({"symbol": s, "price": round(px, 2),
                     "roll": round(roll, 2) if np.isfinite(roll) else None,
                     "cs": round(cs, 2) if np.isfinite(cs) else None,
                     "spread_bps": round(est, 2),
                     "side_bps": round(per_side, 3),
                     "roundtrip_bps": round(per_side * 2, 2),
                     "vol5_bps": round(vol5, 1),
                     "edge_room": round(vol5 / (per_side * 2), 2)})

    d = pd.DataFrame(rows).sort_values("roundtrip_bps")
    print("=" * 100)
    print("ESTIMATED REAL COSTS (Roll + Corwin-Schultz, 5-min bars)")
    print("  flat assumption used so far: 1.50 bps/side = 3.00 round trip")
    print("=" * 100)
    print(d.to_string(index=False))
    med = float(d["roundtrip_bps"].median())
    cheap = d[d["roundtrip_bps"] <= 1.4]
    print(f"\nmedian round trip: {med:.2f} bps  (assumed 3.00)")
    print(f"symbols under 1.40 bps (measured gross edge): {len(cheap)}")
    if len(cheap):
        print("  " + ", ".join(cheap["symbol"].tolist()))
    d.to_json("state/real_costs.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Universe screener -- breadth expansion.

The governing identity:

    daily return = (trades per day) x (edge per trade)

At threshold 0.85 edge per trade is ~0.0276% of equity after every real cost.
1%/day therefore needs ~36 trades/day. Four symbols give 2.14. Lowering the
threshold to force volume destroys edge (measured: -1.48 at 0.55 vs +3.94 at 0.85),
so the only route is more symbols:

    0.535 trades/day/symbol  ->  ~67 symbols for 36 trades/day

Leverage is not an alternative: scaling the 4-symbol system to 1%/day needs 11.5x,
above the 4x Reg-T intraday cap, and would turn a -3% drawdown into -36%.

Screens: dollar volume, realised range, price band (whole-share rounding wastes
size on a small account), shortable + easy-to-borrow (half the signals are shorts),
and a correlation cut so the bets are independent rather than one tech factor
repeated many times.

    python -m app.universe
"""
import json
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .settings import S

log = logging.getLogger("universe")

CANDIDATES = [
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "MU",
    "INTC", "QCOM", "ORCL", "CRM", "ADBE", "NFLX", "PLTR", "SMCI", "ARM", "MRVL",
    "TSM", "LRCX", "KLAC", "AMAT", "ON", "SWKS", "TER", "DELL", "WDC", "STX",
    "JPM", "BAC", "GS", "MS", "C", "WFC", "SCHW", "COIN", "HOOD", "SOFI",
    "XOM", "CVX", "OXY", "SLB", "HAL", "DVN", "FANG", "APA",
    "UNH", "JNJ", "PFE", "MRNA", "LLY", "ABBV", "CVS", "BMY",
    "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "DIS", "BA", "CAT", "DE", "UBER",
    "F", "GM", "DAL", "AAL", "CCL", "NCLH", "T", "VZ",
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "SMH", "ARKK", "XBI",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXL", "TNA", "LABU", "TZA",
    "MSTR", "MARA", "RIOT", "CLSK",
]

MIN_DOLLAR_VOL_M = 50.0
MIN_ATR_PCT = 0.8
MAX_ATR_PCT = 12.0
MIN_PRICE = 8.0
MAX_PRICE = 600.0
MAX_CORR = 0.92
TARGET_N = 70

EDGE_PER_TRADE_PCT = 0.0276      # measured at threshold 0.85, full reality stack
TRADES_PER_SYM_DAY = 0.535       # measured at threshold 0.85


def fetch_daily(symbols, days=120):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(S.key_id, S.secret)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    out = {}
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        try:
            df = c.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start, feed=DataFeed.IEX)).df
        except Exception as e:
            log.warning("chunk failed: %s", e)
            continue
        for s in chunk:
            try:
                out[s] = df.xs(s, level="symbol")
            except KeyError:
                pass
    return out


def tradability(symbols):
    from alpaca.trading.client import TradingClient
    tc = TradingClient(S.key_id, S.secret, paper=True)
    ok = {}
    for s in symbols:
        try:
            a = tc.get_asset(s)
            ok[s] = {"tradable": bool(a.tradable), "shortable": bool(a.shortable),
                     "etb": bool(a.easy_to_borrow)}
        except Exception:
            ok[s] = {"tradable": False, "shortable": False, "etb": False}
    return ok


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info("screening %d candidates", len(CANDIDATES))
    bars = fetch_daily(CANDIDATES)
    log.info("daily bars for %d symbols", len(bars))
    trad = tradability(list(bars))

    rows, rets = [], {}
    for s, d in bars.items():
        if len(d) < 40:
            continue
        px = float(d["close"].iloc[-1])
        dv = float((d["close"] * d["volume"]).tail(30).mean()) / 1e6
        atr = float(((d["high"] - d["low"]) / d["close"] * 100).tail(30).mean())
        r = np.log(d["close"]).diff().dropna()
        rets[s] = r
        t = trad.get(s, {})
        rows.append({"symbol": s, "price": round(px, 2),
                     "dv_m": round(dv, 1), "atr_pct": round(atr, 2),
                     "ann_vol": round(float(r.std() * np.sqrt(252) * 100), 1),
                     "tradable": t.get("tradable", False),
                     "shortable": t.get("shortable", False),
                     "etb": t.get("etb", False)})

    d = pd.DataFrame(rows)
    if d.empty:
        print("no data")
        return 1

    d["pass"] = (d["tradable"] & d["shortable"] & d["etb"] &
                 (d["dv_m"] >= MIN_DOLLAR_VOL_M) &
                 (d["atr_pct"] >= MIN_ATR_PCT) & (d["atr_pct"] <= MAX_ATR_PCT) &
                 (d["price"] >= MIN_PRICE) & (d["price"] <= MAX_PRICE))
    passed = d[d["pass"]].copy()
    passed["score"] = np.log1p(passed["dv_m"]) * 0.5 + passed["atr_pct"]
    passed = passed.sort_values("score", ascending=False)

    chosen, dropped = [], []
    for s in passed["symbol"]:
        if len(chosen) >= TARGET_N:
            break
        clash = None
        for c in chosen:
            try:
                cc = float(rets[s].corr(rets[c]))
            except Exception:
                cc = 0.0
            if cc > MAX_CORR:
                clash = (c, round(cc, 2))
                break
        if clash:
            dropped.append((s, clash[0], clash[1]))
        else:
            chosen.append(s)

    print("=" * 92)
    print("UNIVERSE SCREEN")
    print(f"  {len(d)} candidates -> {len(passed)} passed -> {len(chosen)} after "
          f"de-correlation (max corr {MAX_CORR})")
    print("=" * 92)
    sel = passed[passed.symbol.isin(chosen)]
    print(sel[["symbol", "price", "dv_m", "atr_pct", "ann_vol"]].to_string(index=False))

    if dropped:
        print("\ndropped for correlation: " +
              ", ".join(f"{a}~{b}({c})" for a, b, c in dropped[:15]))
    rej = d[~d["pass"]]
    if len(rej):
        print(f"\nfailed filters ({len(rej)}): " + ", ".join(rej["symbol"].tolist()[:30]))

    n = len(chosen)
    proj_trades = n * TRADES_PER_SYM_DAY
    proj_daily = proj_trades * EDGE_PER_TRADE_PCT
    print("\n" + "=" * 92)
    print("PROJECTION AT THRESHOLD 0.85 (assumes edge per trade holds)")
    print("=" * 92)
    print(f"  symbols                : {n}")
    print(f"  trades/day             : {proj_trades:.1f}")
    print(f"  edge/trade             : {EDGE_PER_TRADE_PCT:.4f}% of equity")
    print(f"  daily return           : {proj_daily:.3f}%")
    print(f"  annualised             : {proj_daily * 252:.0f}%")
    print(f"  concurrency needed     : {proj_trades * 0.5 / 6.5:.1f} slots "
          f"(configured {S.max_concurrent_positions})")
    print(f"  symbols for 1.0%/day   : {1.0 / EDGE_PER_TRADE_PCT / TRADES_PER_SYM_DAY:.0f}")
    print("\n  Edge per trade was measured on 4 hand-picked symbols. A broad screen")
    print("  may dilute it -- the backtest settles that, not this projection.")

    with open("/app/state/universe.json", "w") as f:
        json.dump({"symbols": chosen, "n": n,
                   "generated": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(f"\nSYMBOLS={','.join(chosen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

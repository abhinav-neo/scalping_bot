"""
Intraday statistical arbitrage -- a different edge mechanism.

WHY THIS RATHER THAN MORE TUNING

Directional intraday prediction is closed on measured evidence:
    gross edge ~1.40 bps/trade  vs  measured round-trip cost ~5.97 bps
And edge appears to scale with volatility (~9% of 5-min vol is capturable), so
viability needs vol/cost > 11. The best instrument measured 6.32. No amount of
threshold, breadth, horizon or execution tuning fixes a 4x shortfall.

Pairs trading earns from a different source: the SPREAD between two cointegrated
instruments mean-reverts, and that reversion is far more predictable than
direction. The trade is market-neutral, so it does not need the direction call
that has been failing.

Arithmetic: two legs means ~12 bps round trip. Intraday spread moves on
cointegrated pairs run 20-40 bps, so the ratio is roughly 2-3x rather than 0.23x.
That is the first mechanism in this project where cost is smaller than the move.

METHOD
  * screen pairs within sectors/themes for cointegration (Engle-Granger on
    in-sample data only)
  * hedge ratio from in-sample OLS, held fixed out-of-sample
  * z-score of the spread; enter at |z| > entry, exit at |z| < exit or on stop
  * intraday only, both legs flat by the close
  * full costs on BOTH legs, using the per-symbol measured spreads

    python -m app.pairs
"""
import itertools
import json
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats

from .settings import S
from .reality_stack import deflated_sharpe

log = logging.getLogger("pairs")

CAPITAL = 5000.0
N_TRIALS = 120

# groups within which pairs are economically related -- pairs are only formed
# inside a group, never across, to avoid data-mined spurious cointegration
GROUPS = {
    "index":   ["SPY", "QQQ", "IWM", "DIA", "VOO", "IVV"],
    "semis":   ["SMH", "NVDA", "AMD", "INTC", "MU", "AMAT", "LRCX", "KLAC", "TSM"],
    "banks":   ["XLF", "JPM", "BAC", "C", "WFC", "MS", "GS"],
    "energy":  ["XLE", "XOM", "CVX", "OXY", "SLB", "HAL", "COP"],
    "tech":    ["XLK", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRM"],
    "health":  ["XLV", "JNJ", "PFE", "UNH", "ABBV", "MRK", "BMY", "LLY"],
    "metals":  ["GLD", "SLV", "GDX", "NEM"],
    "bonds":   ["TLT", "IEF", "HYG", "LQD", "AGG"],
    "staples": ["XLP", "KO", "PEP", "WMT", "COST", "PG"],
}

# measured round-trip costs in bps (from app.real_costs). Default for anything
# not measured is the median we observed.
COST_BPS = {
    "TLT": 1.25, "HYG": 1.51, "EFA": 1.94, "GLD": 2.60, "XLI": 3.41, "XLV": 3.42,
    "XLP": 3.53, "SPY": 3.87, "XLF": 3.96, "EEM": 4.32, "IWM": 4.38, "PFE": 4.69,
    "BAC": 4.97, "XOM": 5.11, "KO": 5.15, "XLE": 5.16, "DIA": 5.24, "SLV": 5.38,
    "QQQ": 5.75, "XLK": 6.19, "WFC": 6.40, "VZ": 6.40, "AMZN": 7.69,
    "GOOGL": 7.83, "MSFT": 7.85, "AAPL": 7.94, "META": 8.52, "SMH": 9.60,
    "TSLA": 10.19, "NVDA": 10.76, "PLTR": 12.66, "AMD": 15.85, "MU": 21.97,
    "INTC": 26.53,
}
DEFAULT_COST_BPS = 6.0

ENTRY_Z = [1.5, 2.0, 2.5]
EXIT_Z = 0.3
STOP_Z = 4.0
ZWIN = 60                 # bars for the rolling z-score (5 hours)
MIN_HALFLIFE = 2          # bars
MAX_HALFLIFE = 60
COINT_P = 0.05


def fetch_5m(symbols, days=59):
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
                out[s] = d.between_time("09:30", "16:00")["close"]
            except KeyError:
                pass
    return out


def halflife(spread):
    """Ornstein-Uhlenbeck half-life of mean reversion, in bars."""
    s = spread.dropna()
    if len(s) < 50:
        return np.nan
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    n = min(len(lag), len(delta))
    lag, delta = lag.iloc[-n:], delta.iloc[-n:]
    beta = np.polyfit(lag.values, delta.values, 1)[0]
    if beta >= 0:
        return np.nan
    return float(-np.log(2) / beta)


def engle_granger(y, x):
    """Cointegration test: regress y on x, then ADF the residual."""
    from statsmodels.tsa.stattools import adfuller
    beta, alpha = np.polyfit(x.values, y.values, 1)
    resid = y - (beta * x + alpha)
    try:
        p = adfuller(resid.dropna().values, maxlag=1, regression="c")[1]
    except Exception:
        return None
    return {"beta": float(beta), "alpha": float(alpha), "pvalue": float(p),
            "resid": resid}


def backtest_pair(a, b, pa, pb, beta, alpha, entry_z, cost_a, cost_b):
    """Trade the spread, flat by each session close."""
    idx = pa.index
    spread = pa - (beta * pb + alpha)
    mu = spread.rolling(ZWIN).mean()
    sd = spread.rolling(ZWIN).std()
    z = (spread - mu) / (sd + 1e-12)

    days = idx.normalize()
    minute = idx.hour * 60 + idx.minute
    rt_cost = (cost_a + cost_b) / 1e4          # both legs, round trip, as fraction

    eq = CAPITAL
    pos = 0
    entry_spread = 0.0
    notional = 0.0
    trades = []
    curve = []
    cur_day = None

    for i in range(len(idx)):
        if days[i] != cur_day:
            cur_day = days[i]
            if pos != 0:                        # force flat across sessions
                pos = 0
        zi = z.iloc[i]
        if not np.isfinite(zi):
            curve.append(eq)
            continue
        eod = minute[i] >= 950

        if pos != 0:
            move = (spread.iloc[i] - entry_spread) * pos
            hit_exit = abs(zi) <= EXIT_Z
            hit_stop = abs(zi) >= STOP_Z
            if hit_exit or hit_stop or eod:
                pnl_frac = move / max(abs(entry_spread), 1e-9) if entry_spread else 0.0
                # spread P&L relative to the notional deployed, minus both legs' costs
                gross = move / (pa.iloc[i]) * notional
                eq += gross - notional * rt_cost
                trades.append({"pnl": gross - notional * rt_cost,
                               "reason": "exit" if hit_exit else
                                         ("stop" if hit_stop else "eod"),
                               "z_entry": entry_z * np.sign(-pos)})
                pos = 0

        if pos == 0 and not eod and abs(zi) >= entry_z:
            pos = -1 if zi > 0 else 1           # fade the deviation
            entry_spread = spread.iloc[i]
            notional = eq * 0.5                 # half equity per pair leg-set
        curve.append(eq)

    return pd.DataFrame(trades), pd.Series(curve, index=idx)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    allsyms = sorted({s for g in GROUPS.values() for s in g})
    log.info("fetching 5-min bars for %d symbols", len(allsyms))
    px = fetch_5m(allsyms)
    log.info("got %d symbols", len(px))

    # in-sample / out-of-sample split
    common = None
    for s, v in px.items():
        common = v.index if common is None else common.intersection(v.index)
    log.info("common bars: %d", len(common))
    if len(common) < 800:
        print("insufficient overlapping data")
        return 1
    split = int(len(common) * 0.6)
    is_idx, oos_idx = common[:split], common[split:]

    # ---- screen pairs on IN-SAMPLE only ----
    found = []
    for gname, syms in GROUPS.items():
        avail = [s for s in syms if s in px]
        for a, b in itertools.combinations(avail, 2):
            ya, yb = px[a].reindex(is_idx).ffill(), px[b].reindex(is_idx).ffill()
            if ya.isna().any() or yb.isna().any():
                continue
            eg = engle_granger(ya, yb)
            if eg is None or eg["pvalue"] > COINT_P:
                continue
            hl = halflife(eg["resid"])
            if not np.isfinite(hl) or hl < MIN_HALFLIFE or hl > MAX_HALFLIFE:
                continue
            found.append({"group": gname, "a": a, "b": b,
                          "beta": eg["beta"], "alpha": eg["alpha"],
                          "pvalue": eg["pvalue"], "halflife": round(hl, 1)})
    log.info("cointegrated pairs found: %d", len(found))
    if not found:
        print("no cointegrated pairs")
        return 1
    print("\nTOP PAIRS (in-sample cointegration):")
    fp = pd.DataFrame(found).sort_values("pvalue")
    print(fp.head(15).to_string(index=False))

    # ---- backtest OUT-OF-SAMPLE ----
    results = []
    for ez in ENTRY_Z:
        curves, per = [], []
        for f in found:
            a, b = f["a"], f["b"]
            pa = px[a].reindex(oos_idx).ffill()
            pb = px[b].reindex(oos_idx).ffill()
            if pa.isna().any() or pb.isna().any():
                continue
            t, c = backtest_pair(a, b, pa, pb, f["beta"], f["alpha"], ez,
                                 COST_BPS.get(a, DEFAULT_COST_BPS),
                                 COST_BPS.get(b, DEFAULT_COST_BPS))
            if t.empty or len(t) < 10:
                continue
            per.append({"pair": f"{a}/{b}", "trades": len(t),
                        "net": float(t["pnl"].sum()),
                        "win": float((t.pnl > 0).mean()) * 100,
                        "ret": float(c.iloc[-1] / CAPITAL - 1) * 100})
            curves.append(c.resample("1D").last().dropna().pct_change().dropna())
        if not per:
            continue
        dl = pd.concat(curves, axis=1).mean(axis=1).dropna()
        sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252))
        tot_trades = int(np.sum([p["trades"] for p in per]))
        ndays = len(set(oos_idx.normalize()))
        results.append({
            "entry_z": ez, "pairs": len(per), "trades": tot_trades,
            "trades_day": round(tot_trades / max(ndays, 1), 1),
            "mean_ret_%": round(float(np.mean([p["ret"] for p in per])), 2),
            "win_%": round(float(np.mean([p["win"] for p in per])), 1),
            "sharpe": round(sh, 2),
            "DSR": round(deflated_sharpe(dl.values, sh, N_TRIALS) or 0, 3),
            "profitable_pairs": f"{sum(1 for p in per if p['ret'] > 0)}/{len(per)}"})
        log.info("entry_z %.1f -> %d trades %.1f/day mean %.2f%% sharpe %.2f",
                 ez, tot_trades, results[-1]["trades_day"],
                 results[-1]["mean_ret_%"], sh)

    print("\n" + "=" * 100)
    print("INTRADAY PAIRS -- out-of-sample, costs on BOTH legs, flat by close")
    print("=" * 100)
    print(pd.DataFrame(results).to_string(index=False))
    json.dump({"pairs": found, "results": results},
              open("state/pairs.json", "w"), indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())

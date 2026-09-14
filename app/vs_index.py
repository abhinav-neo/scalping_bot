"""
Is this strategy worth running instead of (or alongside) an index fund?

Return alone does not answer that. What matters is:

  1. CORRELATION to SPY. A strategy that just tracks the index at lower return is
     pointless -- you would buy the index. One that is uncorrelated has portfolio
     value even at equal return, because combining it with the index lowers total
     risk.
  2. BEHAVIOUR ON DOWN DAYS. Index investors are already fully exposed to market
     falls. A strategy that also earns when the market falls is worth far more
     than its standalone return suggests.
  3. THE BENCHMARK WINDOW. SPY returned ~20.8%/yr over our two-year sample against
     a long-run average nearer 10%. Comparing to an exceptional window overstates
     what the index normally delivers.
  4. LEVERAGE-ADJUSTED comparison. A Sharpe-4 strategy at 12% drawdown can be
     scaled; a Sharpe-1.25 index at 30%+ drawdown cannot safely.
  5. A BLENDED PORTFOLIO. Does adding the strategy to SPY actually improve the
     combination, or not?

    python -m app.vs_index
"""
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_v3 import walk_forward
from .reality_stack import simulate

log = logging.getLogger("vs_index")

CAPITAL = 5000.0
HORIZON = 6
PT, SL = 1.5, 1.0
THRESH = 0.85
FULL = dict(lag=1, intrabar=True, stop_slip_R=0.10, limit_fills=True, cost=0.00015)

# SPY long-run reference (approximate, for context only)
SPY_LONGRUN_RET = 10.0
SPY_LONGRUN_SHARPE = 0.50
SPY_LONGRUN_DD = -34.0


def ann(series_daily):
    return float(series_daily.mean() * 252 * 100)


def sharpe(series_daily):
    return float(series_daily.mean() / (series_daily.std() + 1e-12) * np.sqrt(252))


def maxdd(curve):
    roll = curve.cummax()
    return float(((curve - roll) / roll).min()) * 100


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})

    # strategy daily returns, averaged across symbols (equal weight)
    dl = []
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg0)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 5000:
            continue
        c = ModelCfg(S)
        c.horizon_bars = HORIZON
        sg = walk_forward(df, feats, c)
        if sg is None:
            continue
        t, curve, _ = simulate(sg, thresh=THRESH, pt=PT, sl=SL,
                               horizon=HORIZON, **FULL)
        d = curve.resample("1D").last().dropna().pct_change().dropna()
        dl.append(d.rename(sym))
        log.info("%s simulated (%d trades)", sym, len(t))

    strat = pd.concat(dl, axis=1).mean(axis=1).dropna()
    strat.index = pd.to_datetime(strat.index).tz_localize(None).normalize()

    spy_d = spy["close"].resample("1D").last().dropna()
    spy_r = spy_d.pct_change().dropna()
    spy_r.index = pd.to_datetime(spy_r.index).tz_localize(None).normalize()

    both = pd.concat([strat.rename("strat"), spy_r.rename("spy")], axis=1).dropna()
    if len(both) < 30:
        print("not enough overlapping days")
        return 1

    s_curve = (1 + both["strat"]).cumprod()
    m_curve = (1 + both["spy"]).cumprod()

    print("=" * 92)
    print(f"STRATEGY vs INDEX  (threshold {THRESH}, full reality stack, "
          f"{len(both)} trading days)")
    print("=" * 92)

    rows = [
        {"": "strategy", "annual_%": round(ann(both["strat"]), 1),
         "vol_%": round(float(both["strat"].std() * np.sqrt(252) * 100), 1),
         "sharpe": round(sharpe(both["strat"]), 2),
         "maxDD_%": round(maxdd(s_curve), 1)},
        {"": "SPY (this window)", "annual_%": round(ann(both["spy"]), 1),
         "vol_%": round(float(both["spy"].std() * np.sqrt(252) * 100), 1),
         "sharpe": round(sharpe(both["spy"]), 2),
         "maxDD_%": round(maxdd(m_curve), 1)},
        {"": "SPY (long-run avg)", "annual_%": SPY_LONGRUN_RET, "vol_%": 15.5,
         "sharpe": SPY_LONGRUN_SHARPE, "maxDD_%": SPY_LONGRUN_DD},
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 1. correlation ----------
    corr = float(both["strat"].corr(both["spy"]))
    beta = float(np.polyfit(both["spy"], both["strat"], 1)[0])
    alpha_d = float(both["strat"].mean() - beta * both["spy"].mean())
    print("\n" + "-" * 92)
    print("1. CORRELATION TO THE INDEX")
    print("-" * 92)
    print(f"  correlation : {corr:+.3f}")
    print(f"  beta        : {beta:+.3f}")
    print(f"  alpha       : {alpha_d * 252 * 100:+.1f}%/yr  (return not explained by SPY)")
    if abs(corr) < 0.3:
        print("  -> largely independent of the index: it is a diversifier, not a proxy")
    else:
        print("  -> substantially tracks the index: little diversification value")

    # ---------- 2. down days ----------
    down = both[both["spy"] < 0]
    up = both[both["spy"] >= 0]
    print("\n" + "-" * 92)
    print("2. BEHAVIOUR WHEN THE MARKET FALLS")
    print("-" * 92)
    print(f"  SPY down days ({len(down)}): strategy {down['strat'].mean()*100:+.3f}%/day"
          f"  vs SPY {down['spy'].mean()*100:+.3f}%/day")
    print(f"  SPY up days   ({len(up)}): strategy {up['strat'].mean()*100:+.3f}%/day"
          f"  vs SPY {up['spy'].mean()*100:+.3f}%/day")
    print(f"  strategy win rate on down days: "
          f"{100*(down['strat']>0).mean():.0f}%")
    if down["strat"].mean() > 0:
        print("  -> EARNS while the index loses. That is the case for holding it")
        print("     alongside an index fund rather than instead of one.")
    else:
        print("  -> also loses when the index loses: limited hedging value")

    # ---------- 3. blended portfolio ----------
    print("\n" + "-" * 92)
    print("3. BLENDED PORTFOLIO (rebalanced daily)")
    print("-" * 92)
    blends = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        b = w * both["strat"] + (1 - w) * both["spy"]
        c = (1 + b).cumprod()
        blends.append({"strategy_weight": f"{w:.0%}",
                       "annual_%": round(ann(b), 1),
                       "vol_%": round(float(b.std() * np.sqrt(252) * 100), 1),
                       "sharpe": round(sharpe(b), 2),
                       "maxDD_%": round(maxdd(c), 1)})
    bl = pd.DataFrame(blends)
    print(bl.to_string(index=False))
    best = bl.loc[bl["sharpe"].idxmax()]
    print(f"\n  best risk-adjusted blend: {best['strategy_weight']} strategy "
          f"-> Sharpe {best['sharpe']} vs {bl.iloc[0]['sharpe']} for SPY alone")

    # ---------- 4. leverage-matched ----------
    print("\n" + "-" * 92)
    print("4. LEVERAGE-MATCHED COMPARISON")
    print("-" * 92)
    spy_vol = float(both["spy"].std())
    st_vol = float(both["strat"].std())
    lev = spy_vol / st_vol if st_vol > 0 else 0
    lv = both["strat"] * lev
    lc = (1 + lv).cumprod()
    print(f"  scaling the strategy to SPY's volatility requires {lev:.2f}x leverage")
    print(f"  levered strategy: {ann(lv):.1f}%/yr  Sharpe {sharpe(lv):.2f}  "
          f"maxDD {maxdd(lc):.1f}%")
    print(f"  SPY this window : {ann(both['spy']):.1f}%/yr  "
          f"Sharpe {sharpe(both['spy']):.2f}  maxDD {maxdd(m_curve):.1f}%")
    print("\n  NOTE: leverage is a real cost and a real risk. Margin is not free,")
    print("        and a levered strategy fails faster when its edge decays.")

    print("\n" + "=" * 92)
    print("BOTTOM LINE")
    print("=" * 92)
    if abs(corr) < 0.3 and down["strat"].mean() > 0:
        print("  Uncorrelated and profitable when the index falls. Its value is as a")
        print("  COMPLEMENT to an index holding, not a replacement -- judge it by what")
        print("  it adds to the blend, not by its standalone return.")
    elif abs(corr) < 0.3:
        print("  Uncorrelated but does not earn on down days. Modest diversification")
        print("  value; standalone return is the fair basis for comparison.")
    else:
        print("  Tracks the index closely. If it does not beat the index on return,")
        print("  there is no case for running it instead of buying the index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

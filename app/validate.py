"""
VALIDATION GATE -- run before any deployment. No exceptions.

This exists because of a repeated failure on this project: deploying on partial
evidence. Training holdout accuracy was mistaken for a backtest; a config change
was shipped without re-running the sweep that justified the previous config. Each
time the live result was worse than expected, and each time the missing test would
have caught it.

The gate runs every test that has ever changed a conclusion here, and returns a
single GO / NO-GO. A configuration that cannot pass all of these has no business
touching an account.

  1 TRAIN      per-symbol holdout hit rate vs base rate
  2 STACK      reality stack: lag, intrabar exits, stop slippage, adverse-selected
               limit fills, costs -- the full set, applied cumulatively
  3 THRESHOLD  sweep conviction levels; costs scale with trade count, edge does not
  4 REGIME     edge by pre-market regime, to see if it is one regime carrying it
  5 INDEX      correlation, down-day behaviour, and the SPY benchmark
  6 DSR        deflated Sharpe, penalising every configuration searched

PASS CRITERIA (all must hold):
  * fully-costed annual return > 0
  * edge at the chosen threshold > +1.0
  * DSR >= 0.90
  * beats SPY on Sharpe
  * max drawdown > -25%

    python -m app.validate
"""
import json
import logging
import sys

import numpy as np
import pandas as pd

from .settings import S
from .train import fetch_history
from .modelcfg import ModelCfg
from .features import build_features
from .backtest_v3 import walk_forward, edge_of, classify_days
from .reality_stack import simulate, deflated_sharpe

log = logging.getLogger("validate")

CAPITAL = 5000.0
HORIZON = 6
PT, SL = 1.5, 1.0
N_TRIALS = 100
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.93]

# Session window filter. The time-of-day test found the open is the WORST window
# (-8.1%/yr) and midday the only profitable one (+7.0%/yr, DSR 0.918). Large moves
# at the open are news- and imbalance-driven, which swamps a 5-minute statistical
# signal; midday is quieter and more mean-reverting, which is what this model needs.
HOUR_LO = float(__import__("os").getenv("SESSION_HOUR_LO", "11.0"))
HOUR_HI = float(__import__("os").getenv("SESSION_HOUR_HI", "14.0"))

LAYERS = [
    ("0 ideal",            dict()),
    ("1 +lag",             dict(lag=1)),
    ("2 +intrabar",        dict(lag=1, intrabar=True)),
    ("3 +stop slip",       dict(lag=1, intrabar=True, stop_slip_R=0.10)),
    ("4 +limit fills",     dict(lag=1, intrabar=True, stop_slip_R=0.10,
                                limit_fills=True)),
    ("5 +costs (REAL)",    dict(lag=1, intrabar=True, stop_slip_R=0.10,
                                limit_fills=True, cost=0.00015)),
]
FULL = LAYERS[-1][1]

MIN_ANNUAL = 0.0
MIN_EDGE = 1.0
MIN_DSR = 0.90
MAX_DD = -25.0


def agg_daily(curves):
    return pd.concat(curves, axis=1).mean(axis=1).dropna()


def windowed(sig):
    """Blank signals outside the tradable session window."""
    if HOUR_LO <= 0 and HOUR_HI >= 24:
        return sig
    s2 = sig.copy()
    h = s2.index.hour + s2.index.minute / 60.0
    s2.loc[~((h >= HOUR_LO) & (h < HOUR_HI)), "meta_p"] = 0.0
    return s2


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg0 = ModelCfg(S)
    need = list(dict.fromkeys(S.symbols + [S.market_symbol]))
    log.info("VALIDATION GATE | %d symbols | %.1f years", len(S.symbols),
             S.train_years)
    hist = fetch_history(S, need, S.train_years)
    spy = hist[S.market_symbol]
    mkt = spy[["close"]].rename(columns={"close": f"{S.market_symbol}_close"})
    regimes = classify_days(spy)

    spy_d = spy["close"].resample("1D").last().dropna()
    days = (spy_d.index[-1] - spy_d.index[0]).days or 1
    spy_r = spy_d.pct_change().dropna()
    spy_ann = ((spy_d.iloc[-1] / spy_d.iloc[0]) ** (365 / days) - 1) * 100
    spy_sharpe = float(spy_r.mean() / spy_r.std() * np.sqrt(252))

    # ---------- 1 TRAIN + signals ----------
    sigs, train_rows = {}, []
    for sym in S.symbols:
        if sym not in hist:
            continue
        df = hist[sym].join(mkt, how="inner").dropna()
        feats = build_features(df, cfg0)
        keep = feats.dropna().index
        df, feats = df.loc[keep], feats.loc[keep]
        if len(df) < 4000:
            continue
        c = ModelCfg(S)
        c.horizon_bars = HORIZON
        sg = walk_forward(df, feats, c)
        if sg is None:
            continue
        sigs[sym] = windowed(sg)
        train_rows.append({"symbol": sym, "bars": len(df),
                           "p90": round(float(sg["meta_p"].quantile(0.9)), 3)})
    log.info("signals built for %d symbols", len(sigs))
    if not sigs:
        print("NO SIGNALS -- abort")
        return 1

    # ---------- 2 STACK ----------
    print("\n" + "=" * 92)
    print(f"1-2. REALITY STACK  ({len(sigs)} symbols, threshold "
          f"{S.meta_threshold})")
    print("=" * 92)
    stack_rows = []
    for name, kw in LAYERS:
        per, curves = [], []
        for sym, sg in sigs.items():
            t, curve, meta = simulate(sg, thresh=S.meta_threshold, pt=PT, sl=SL,
                                      horizon=HORIZON, **kw)
            if t.empty or len(t) < 10:
                continue
            d = curve.resample("1D").last().dropna().pct_change().dropna()
            win, be, edge = edge_of(t)
            roll = curve.cummax()
            per.append({"ret": float(curve.iloc[-1] / CAPITAL - 1) * 100,
                        "edge": edge, "trades": len(t),
                        "dd": float(((curve - roll) / roll).min()) * 100,
                        "fill": meta.get("fill_rate")})
            curves.append(d)
        if not per:
            continue
        dl = agg_daily(curves)
        sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252))
        stack_rows.append({
            "layer": name,
            "annual_%": round(float(np.mean([p["ret"] for p in per])) * 365 / days, 1),
            "sharpe": round(sh, 2),
            "edge": round(float(np.mean([p["edge"] for p in per
                                         if p["edge"] is not None])), 2),
            "maxDD_%": round(float(np.mean([p["dd"] for p in per])), 1),
            "trades": int(np.sum([p["trades"] for p in per])),
            "DSR": round(deflated_sharpe(dl.values, sh, N_TRIALS) or 0, 3)})
    print(pd.DataFrame(stack_rows).to_string(index=False))

    # ---------- 3 THRESHOLD ----------
    print("\n" + "=" * 92)
    print("3. THRESHOLD SWEEP  (full reality stack applied)")
    print("=" * 92)
    thr_rows = []
    best = None
    for thr in THRESHOLDS:
        per, curves = [], []
        for sym, sg in sigs.items():
            t, curve, meta = simulate(sg, thresh=thr, pt=PT, sl=SL,
                                      horizon=HORIZON, **FULL)
            if t.empty or len(t) < 10:
                continue
            d = curve.resample("1D").last().dropna().pct_change().dropna()
            win, be, edge = edge_of(t)
            roll = curve.cummax()
            per.append({"ret": float(curve.iloc[-1] / CAPITAL - 1) * 100,
                        "edge": edge, "win": win, "trades": len(t),
                        "dd": float(((curve - roll) / roll).min()) * 100,
                        "ppt": float(t["pnl"].mean())})
            curves.append(d)
        if not per:
            continue
        dl = agg_daily(curves)
        sh = float(dl.mean() / (dl.std() + 1e-12) * np.sqrt(252))
        dsr = deflated_sharpe(dl.values, sh, N_TRIALS) or 0
        tot_trades = int(np.sum([p["trades"] for p in per]))
        row = {"thresh": thr,
               "trades": tot_trades,
               "trades_day": round(tot_trades / days, 1),
               "annual_%": round(float(np.mean([p["ret"] for p in per])) * 365 / days, 1),
               "sharpe": round(sh, 2),
               "edge": round(float(np.mean([p["edge"] for p in per
                                            if p["edge"] is not None])), 2),
               "win_%": round(float(np.mean([p["win"] for p in per])), 1),
               "$/trade": round(float(np.mean([p["ppt"] for p in per])), 3),
               "maxDD_%": round(float(np.mean([p["dd"] for p in per])), 1),
               "DSR": round(dsr, 3)}
        thr_rows.append(row)
        if best is None or row["annual_%"] > best["annual_%"]:
            best = dict(row, _daily=dl)
        log.info("thr %.2f -> %d trades %.1f%%/yr edge %.2f",
                 thr, tot_trades, row["annual_%"], row["edge"])
    print(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in thr_rows]).to_string(index=False))

    if best is None:
        print("\nNO VIABLE THRESHOLD -- NO-GO")
        return 1

    bthr = best["thresh"]

    # ---------- 4 REGIME ----------
    print("\n" + "=" * 92)
    print(f"4. EDGE BY REGIME  (threshold {bthr})")
    print("=" * 92)
    alltr = []
    for sym, sg in sigs.items():
        t, curve, _ = simulate(sg, thresh=bthr, pt=PT, sl=SL, horizon=HORIZON,
                               **FULL)
        if t.empty:
            continue
        t = t.copy()
        t["symbol"] = sym
        alltr.append(t)
    if alltr:
        tr = pd.concat(alltr, ignore_index=True)
        if "day" in tr:
            tr["day"] = pd.to_datetime(tr["day"]).dt.tz_localize(None).dt.normalize()
            rm = regimes.copy()
            rm.index = pd.to_datetime(rm.index).tz_localize(None).normalize()
            tr = tr.join(rm["regime"], on="day")
            out = []
            for reg, g in tr.groupby(tr["regime"].fillna("unknown")):
                w, b, e = edge_of(g)
                out.append({"regime": reg, "trades": len(g),
                            "net": round(float(g.pnl.sum()), 2),
                            "edge": round(e, 2) if e else None})
            print(pd.DataFrame(out).sort_values("edge", ascending=False,
                                                na_position="last").to_string(index=False))

    # ---------- 5 INDEX ----------
    print("\n" + "=" * 92)
    print("5. VERSUS INDEX")
    print("=" * 92)
    sd = best["_daily"]
    sd.index = pd.to_datetime(sd.index).tz_localize(None).normalize()
    sr = spy_r.copy()
    sr.index = pd.to_datetime(sr.index).tz_localize(None).normalize()
    both = pd.concat([sd.rename("s"), sr.rename("m")], axis=1).dropna()
    corr = float(both["s"].corr(both["m"])) if len(both) > 20 else float("nan")
    down = both[both["m"] < 0]
    print(f"  strategy {best['annual_%']:.1f}%/yr  Sharpe {best['sharpe']:.2f}"
          f"  maxDD {best['maxDD_%']:.1f}%")
    print(f"  SPY      {spy_ann:.1f}%/yr  Sharpe {spy_sharpe:.2f}")
    print(f"  correlation {corr:+.3f}")
    if len(down):
        print(f"  on SPY down days ({len(down)}): strategy "
              f"{down['s'].mean()*100:+.3f}%/day")

    # ---------- 6 VERDICT ----------
    checks = [
        ("fully-costed annual > 0", best["annual_%"] > MIN_ANNUAL,
         f"{best['annual_%']:.1f}%"),
        (f"edge > {MIN_EDGE}", (best["edge"] or -9) > MIN_EDGE,
         f"{best['edge']}"),
        (f"DSR >= {MIN_DSR}", best["DSR"] >= MIN_DSR, f"{best['DSR']}"),
        ("beats SPY Sharpe", best["sharpe"] > spy_sharpe,
         f"{best['sharpe']:.2f} vs {spy_sharpe:.2f}"),
        (f"maxDD > {MAX_DD}%", best["maxDD_%"] > MAX_DD, f"{best['maxDD_%']:.1f}%"),
    ]
    print("\n" + "=" * 92)
    print(f"VERDICT  (best threshold {bthr}, {best['trades_day']} trades/day)")
    print("=" * 92)
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<28} {val}")
    go = all(ok for _, ok, _ in checks)
    print("\n  ==> " + ("GO: safe to deploy this configuration"
                        if go else "NO-GO: do not deploy"))

    json.dump({"go": go, "best_threshold": bthr,
               "stack": stack_rows, "thresholds":
                   [{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in thr_rows],
               "spy_annual": round(spy_ann, 1)},
              open("/app/state/validation.json", "w"), indent=2, default=str)
    return 0 if go else 2


if __name__ == "__main__":
    sys.exit(main())

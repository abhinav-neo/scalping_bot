"""
Model trainer. Pulls real history from Alpaca, trains the HMM regime model plus the
primary (direction) and meta (trade/skip) models per symbol, and persists them.

Run standalone:   python -m app.train
Or on a schedule via the scheduler in run_bot.py (RETRAIN_DAYS).
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from .settings import S
from .features import build_features
from .labeling import triple_barrier_labels, make_meta_labels
from .regime import RegimeModel
from .modelcfg import ModelCfg

log = logging.getLogger("train")

# Kept as an alias so existing references keep working.
_Cfg = ModelCfg


def fetch_history(s, symbols, years):
    client = StockHistoricalDataClient(s.key_id, s.secret)
    start = datetime.now(timezone.utc) - timedelta(days=int(365 * years))
    # Must match the feed used live. Training on SIP while trading on IEX would
    # mean the model sees different price/volume distributions than it trades on.
    feed = DataFeed.SIP if s.use_sip else DataFeed.IEX
    log.info("history feed = %s", feed.value)
    req = StockBarsRequest(symbol_or_symbols=list(symbols),
                           timeframe=TimeFrame(s.bar_minutes, TimeFrameUnit.Minute),
                           start=start,
                           feed=feed)
    df = client.get_stock_bars(req).df
    out = {}
    for sym in symbols:
        try:
            d = df.xs(sym, level="symbol").copy()
        except KeyError:
            continue
        d.index = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
        d = d.between_time("09:30", "16:00")
        out[sym] = d[["open", "high", "low", "close", "volume"]]
    return out


def _clf(seed, class_weight=None):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=60, reg_lambda=1.0,
                              class_weight=class_weight,
                              random_state=seed, n_jobs=2, verbose=-1)


def train_symbol(sym, px, mkt, cfg, s):
    df = px.join(mkt, how="inner").dropna()
    if len(df) < 3000:
        log.warning("%s: only %d bars, skipping", sym, len(df))
        return None

    feats = build_features(df, cfg)
    keep = feats.dropna().index
    df, feats = df.loc[keep], feats.loc[keep]

    # Barriers scale with sqrt(horizon), matching backtest_v3 and the live engine.
    # Without this the target stays at ~1 bar of volatility no matter how long the
    # horizon is, so the model learns to predict a move it is never actually held
    # long enough to require -- and horizon_bars becomes a time cap with no effect
    # on the trade being taken.
    scaled_vol = feats["vol"] * np.sqrt(cfg.horizon_bars)
    tb = triple_barrier_labels(df["close"], scaled_vol, cfg)
    y = tb["label"].values

    reg = RegimeModel(cfg).fit(feats)
    reg_f = reg.transform(feats)
    X = pd.concat([feats, reg_f], axis=1)
    cols = [c for c in X.columns if c != "regime"]
    Xv = X[cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values

    mask = y != 0
    if mask.sum() < 500:
        log.warning("%s: too few directional labels", sym)
        return None

    # holdout tail for an honest read on freshly trained models
    n = len(Xv)
    cut = int(n * 0.8)
    tr_mask = mask.copy(); tr_mask[cut:] = False

    # Class balancing was added on 2026-07-29 after a session where the bot went
    # 93% long. A 2-year A/B backtest later showed the base model trades ~48% long
    # over 8,000 trades -- the "bias" was a single-day artifact, and balancing made
    # no measurable difference (750% vs 736% return, Sharpe 5.89 vs 5.91).
    # Reverted to keep the model consistent with the swept configuration.
    primary = _clf(7).fit(Xv[tr_mask], (y[tr_mask] > 0).astype(int))
    p_up = primary.predict_proba(Xv)[:, 1]
    side = np.where(p_up >= 0.5, 1, -1)
    meta_y = make_meta_labels(side, y)
    meta_X = np.column_stack([Xv, p_up])
    meta = _clf(8).fit(meta_X[:cut], meta_y[:cut])

    # holdout diagnostics
    ho = slice(cut, n)
    meta_p_ho = meta.predict_proba(meta_X[ho])[:, 1]
    sel = meta_p_ho >= s.meta_threshold
    hit = float((meta_y[ho][sel]).mean()) if sel.sum() > 20 else None
    # Directional balance of the signals we would actually trade. Anything far
    # from 50% means the model is carrying a directional prior, not predicting.
    sel_long = float((side[ho][sel] > 0).mean()) if sel.sum() > 20 else None
    diag = {"bars": int(n), "holdout_bars": int(n - cut),
            "signals_selected": int(sel.sum()),
            "holdout_hit_rate": round(hit, 4) if hit is not None else None,
            "base_rate": round(float(meta_y[ho].mean()), 4),
            "long_share": round(sel_long, 4) if sel_long is not None else None}
    log.info("%s trained: %s", sym, diag)

    return {"primary": primary, "meta": meta, "regime": reg,
            "cols": cols, "diag": diag,
            "trained_at": datetime.now(timezone.utc).isoformat()}


def train_all(s=S):
    os.makedirs(s.model_dir, exist_ok=True)
    cfg = _Cfg(s)
    need = list(dict.fromkeys(s.symbols + [s.market_symbol]))
    log.info("fetching %.1f years of %dmin bars for %s", s.train_years, s.bar_minutes, need)
    hist = fetch_history(s, need, s.train_years)
    if s.market_symbol not in hist:
        raise RuntimeError("market symbol history missing")
    mkt = hist[s.market_symbol][["close"]].rename(
        columns={"close": f"{s.market_symbol}_close"})

    summary = {}
    for sym in s.symbols:
        if sym not in hist:
            log.warning("no history for %s", sym); continue
        art = train_symbol(sym, hist[sym], mkt, cfg, s)
        if art:
            joblib.dump(art, os.path.join(s.model_dir, f"{sym}.joblib"))
            summary[sym] = art["diag"]
    with open(os.path.join(s.model_dir, "summary.json"), "w") as f:
        json.dump({"trained_at": datetime.now(timezone.utc).isoformat(),
                   "symbols": summary}, f, indent=2)
    log.info("training complete: %s", list(summary))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    errs = S.validate()
    if errs:
        raise SystemExit("config errors:\n  - " + "\n  - ".join(errs))
    train_all(S)

"""
Sequence model vs gradient boosting -- can a better model class lift gross edge?

THE ONLY REMAINING LEVER

Measured intraday economics:
    gross edge  1.40 bps/trade
    real cost   5.97 bps round trip (measured, Roll + Corwin-Schultz)

Cost-side levers are exhausted. Even futures (~1.3 bps round trip) would leave
0.10 bps net. The strategy only becomes viable if GROSS EDGE rises to ~6+ bps,
which means extracting materially more signal from the same bars.

Every model so far has been LightGBM on a flat feature vector -- it sees one row
at a time and cannot represent temporal structure directly. A temporal convolution
network sees the full sequence of the last N bars and learns which lags matter.
This tests whether that representation finds signal the tabular model misses.

Judged on the metric that decides everything: hit rate above base rate on a
held-out period, and the implied gross edge in bps.

    python -m app.seqmodel
"""
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn

from .settings import S
from .modelcfg import ModelCfg


log = logging.getLogger("seqmodel")

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META",
           "JPM", "BAC", "XLF", "XLE"]
SEQ_LEN = 48          # 4 hours of 5-min bars
HORIZON = 6           # predict 30 minutes ahead
EPOCHS = 12
BATCH = 256
LR = 1e-3


class TCN(nn.Module):
    """Dilated causal convolutions: receptive field grows exponentially with depth,
    so 4 layers see 48 bars while staying cheap on CPU."""

    def __init__(self, n_feat, channels=48, layers=4, dropout=0.2):
        super().__init__()
        blocks = []
        for i in range(layers):
            d = 2 ** i
            inp = n_feat if i == 0 else channels
            blocks += [
                nn.Conv1d(inp, channels, kernel_size=3, padding=d * 2, dilation=d),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):            # x: (batch, seq, feat)
        h = self.net(x.transpose(1, 2))
        h = h[:, :, -1]              # causal: last position only
        return self.head(h).squeeze(-1)


def fetch(symbols, days=59):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    c = StockHistoricalDataClient(S.key_id, S.secret)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    out = {}
    for i in range(0, len(symbols), 30):
        chunk = symbols[i:i + 30]
        df = c.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start, feed=DataFeed.IEX)).df
        for s in chunk:
            try:
                d = df.xs(s, level="symbol")
                d.index = pd.to_datetime(d.index, utc=True).tz_convert(
                    "America/New_York")
                out[s] = d.between_time("09:30", "16:00")
            except KeyError:
                pass
    return out


def make_features(df, spy_close):
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    lp = np.log(close)
    ret = lp.diff()
    for k in (1, 2, 3, 6, 12):
        f[f"ret_{k}"] = lp.diff(k)
    vol = ret.ewm(span=50).std()
    f["vol"] = vol
    f["vol_ratio"] = vol / (vol.rolling(78).mean() + 1e-9)
    f["mom_z"] = f["ret_6"] / (vol * np.sqrt(6) + 1e-9)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - close.shift()).abs(),
                    (df["low"] - close.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(span=14).mean() / close
    day = df.index.normalize()
    tp = (df["high"] + df["low"] + close) / 3
    vwap = (tp * df["volume"]).groupby(day).cumsum() / (
        df["volume"].groupby(day).cumsum() + 1e-9)
    f["vwap_dist"] = (close - vwap) / (close * f["atr_pct"] + 1e-9)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(span=14).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14).mean()
    f["rsi"] = (100 - 100 / (1 + up / (dn + 1e-9))) / 100.0
    f["vol_z"] = ((df["volume"] - df["volume"].rolling(78).mean())
                  / (df["volume"].rolling(78).std() + 1e-9))
    mret = np.log(spy_close).diff()
    f["mkt_ret"] = mret
    f["mkt_lead"] = mret.shift(1)
    beta = ret.rolling(78).cov(mret) / (mret.rolling(78).var() + 1e-9)
    f["resid"] = ret - beta * mret
    mins = np.asarray(df.index.hour * 60 + df.index.minute, dtype=float)
    since = np.clip(mins - 570, 0, None)
    f["tod_sin"] = np.sin(2 * np.pi * since / 390)
    f["tod_cos"] = np.cos(2 * np.pi * since / 390)
    return f


def label_forward(close, horizon, vol):
    """Sign of the forward return over `horizon` bars, with a vol deadband so
    near-zero moves do not become training noise."""
    fwd = np.log(close.shift(-horizon) / close)
    dead = 0.2 * vol * np.sqrt(horizon)
    y = pd.Series(0, index=close.index, dtype=float)
    y[fwd > dead] = 1
    y[fwd < -dead] = -1
    return y, fwd


def build_sequences(X, y, seq_len):
    n, k = X.shape
    idx = np.arange(seq_len, n)
    seq = np.stack([X[i - seq_len:i] for i in idx]).astype(np.float32)
    return seq, y[idx], idx


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(7)
    np.random.seed(7)

    log.info("fetching bars")
    bars = fetch(SYMBOLS)
    if "SPY" not in bars:
        print("no SPY")
        return 1
    spy = bars["SPY"]["close"]
    log.info("got %d symbols", len(bars))

    rows = []
    for sym, df in bars.items():
        if len(df) < 2000:
            continue
        f = make_features(df, spy.reindex(df.index).ffill())
        y, fwd = label_forward(df["close"], HORIZON, f["vol"])
        keep = f.dropna().index.intersection(y.dropna().index)
        keep = keep[:-HORIZON] if len(keep) > HORIZON else keep
        f2, y2, fwd2 = f.loc[keep], y.loc[keep], fwd.loc[keep]
        mask = y2 != 0
        if mask.sum() < 1500:
            continue

        Xall = f2.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        mu, sd = Xall.mean(0), Xall.std(0) + 1e-9
        Xall = (Xall - mu) / sd
        yb = (y2.values > 0).astype(np.float32)

        split = int(len(Xall) * 0.7)

        # ---------- LightGBM baseline ----------
        m = mask.values
        tr_m = m.copy(); tr_m[split:] = False
        te_m = m.copy(); te_m[:split] = False
        gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                 num_leaves=31, subsample=0.8,
                                 colsample_bytree=0.8, min_child_samples=60,
                                 reg_lambda=1.0, random_state=7, n_jobs=4,
                                 verbose=-1)
        gbm.fit(Xall[tr_m], yb[tr_m])
        p_gbm = gbm.predict_proba(Xall[te_m])[:, 1]
        acc_gbm = float(((p_gbm >= 0.5) == (yb[te_m] > 0.5)).mean())
        base = float(max(yb[te_m].mean(), 1 - yb[te_m].mean()))

        # ---------- TCN ----------
        seq, yseq, idxs = build_sequences(Xall, yb, SEQ_LEN)
        mseq = m[idxs]
        s_split = np.searchsorted(idxs, split)
        tr_i = np.where(mseq[:s_split])[0]
        te_i = np.where(mseq[s_split:])[0] + s_split
        if len(tr_i) < 500 or len(te_i) < 200:
            continue

        model = TCN(seq.shape[2])
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        Xtr = torch.from_numpy(seq[tr_i])
        Ytr = torch.from_numpy(yseq[tr_i])
        model.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xtr))
            tot = 0.0
            for b in range(0, len(Xtr), BATCH):
                bi = perm[b:b + BATCH]
                opt.zero_grad()
                out = model(Xtr[bi])
                loss = lossf(out, Ytr[bi])
                loss.backward()
                opt.step()
                tot += float(loss) * len(bi)
        model.eval()
        with torch.no_grad():
            p_tcn = torch.sigmoid(model(torch.from_numpy(seq[te_i]))).numpy()
        acc_tcn = float(((p_tcn >= 0.5) == (yseq[te_i] > 0.5)).mean())

        # ---------- implied gross edge in bps ----------
        fwd_te = fwd2.values[idxs][te_i]
        side_tcn = np.where(p_tcn >= 0.5, 1, -1)
        side_gbm_full = np.where(gbm.predict_proba(Xall[idxs][te_i])[:, 1] >= 0.5, 1, -1)
        edge_tcn = float(np.mean(side_tcn * fwd_te) * 1e4)
        edge_gbm = float(np.mean(side_gbm_full * fwd_te) * 1e4)

        # high-conviction subset only
        conf = np.abs(p_tcn - 0.5)
        hi = conf >= np.quantile(conf, 0.9)
        edge_tcn_hi = float(np.mean(side_tcn[hi] * fwd_te[hi]) * 1e4)

        rows.append({"symbol": sym, "base_%": round(base * 100, 1),
                     "gbm_%": round(acc_gbm * 100, 1),
                     "tcn_%": round(acc_tcn * 100, 1),
                     "gbm_bps": round(edge_gbm, 2),
                     "tcn_bps": round(edge_tcn, 2),
                     "tcn_top10_bps": round(edge_tcn_hi, 2)})
        log.info("%-6s base %.1f%% gbm %.1f%% tcn %.1f%% | edge gbm %.2f tcn %.2f "
                 "top10 %.2f bps", sym, base * 100, acc_gbm * 100, acc_tcn * 100,
                 edge_gbm, edge_tcn, edge_tcn_hi)

    d = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("SEQUENCE MODEL vs GRADIENT BOOSTING -- held-out 30%")
    print("  edge columns are gross bps per trade, BEFORE costs")
    print("  measured real cost is ~5.97 bps round trip: edge must exceed that")
    print("=" * 96)
    print(d.to_string(index=False))
    if len(d):
        print(f"\nmean gross edge  GBM {d['gbm_bps'].mean():.2f} bps | "
              f"TCN {d['tcn_bps'].mean():.2f} bps | "
              f"TCN top-decile {d['tcn_top10_bps'].mean():.2f} bps")
        print(f"cost to beat     5.97 bps")
        best = max(d['tcn_bps'].mean(), d['tcn_top10_bps'].mean())
        print("\n  -> " + ("VIABLE: gross edge exceeds cost" if best > 5.97
                           else f"still short: best {best:.2f} vs 5.97 bps needed"))
    d.to_json("state/seqmodel.json", orient="records")
    return 0


if __name__ == "__main__":
    sys.exit(main())

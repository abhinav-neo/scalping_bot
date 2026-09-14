"""
Feature engineering.

The point of this layer is to give the ML model things the raw Markov state can't
see: memory-preserving stationary prices (fracdiff), cross-asset lead-lag to SPY,
intraday seasonality, and volatility structure.
"""
import numpy as np
import pandas as pd


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """Fixed-width fractional differentiation (Lopez de Prado). Stationarity with memory."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    width = len(w) - 1
    vals = series.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(width, len(vals)):
        out[i] = np.dot(w, vals[i - width:i + 1])
    return pd.Series(out, index=series.index)


def ewma_vol(ret: pd.Series, span: int) -> pd.Series:
    return ret.ewm(span=span).std()


def build_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    logp = np.log(close)
    ret = logp.diff()

    # --- memory-preserving stationary price ---
    f["fracdiff"] = frac_diff_ffd(logp, cfg.fracdiff_d, cfg.fracdiff_thresh)

    # --- returns & momentum over several horizons ---
    for k in (1, 3, 6, 12):
        f[f"ret_{k}"] = logp.diff(k)
    f["mom_z"] = f["ret_6"] / (ewma_vol(ret, cfg.vol_span) * np.sqrt(6) + 1e-9)

    # --- volatility structure ---
    vol = ewma_vol(ret, cfg.vol_span)
    f["vol"] = vol
    f["vol_ratio"] = vol / (vol.rolling(78).mean() + 1e-9)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - close.shift()).abs(),
                    (df["low"] - close.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.ewm(span=14).mean()
    f["atr_pct"] = f["atr"] / close

    # --- VWAP distance (resets each session) ---
    day = df.index.normalize()
    tp = (df["high"] + df["low"] + close) / 3
    cum_pv = (tp * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    vwap = cum_pv / (cum_v + 1e-9)
    f["vwap_dist"] = (close - vwap) / (close * f["atr_pct"] + 1e-9)

    # --- RSI ---
    delta = close.diff()
    up = delta.clip(lower=0).ewm(span=14).mean()
    dn = (-delta.clip(upper=0)).ewm(span=14).mean()
    f["rsi"] = 100 - 100 / (1 + up / (dn + 1e-9))

    # --- cross-asset lead-lag to SPY (and context names) ---
    mkt_col = f"{cfg.market_symbol}_close"
    if mkt_col in df.columns:
        mret = np.log(df[mkt_col]).diff()
        f["mkt_ret_1"] = mret
        f["mkt_ret_3"] = np.log(df[mkt_col]).diff(3)
        # lagged market move = does SPY lead the name?
        f["mkt_lead_1"] = mret.shift(1)
        # rolling beta and idiosyncratic (residual) return
        cov = ret.rolling(78).cov(mret)
        var = mret.rolling(78).var()
        beta = cov / (var + 1e-9)
        f["beta"] = beta
        f["resid_ret"] = ret - beta * mret
    for c in cfg.context_symbols:
        col = f"{c}_close"
        if col in df.columns:
            f[f"{c}_ret_1"] = np.log(df[col]).diff()

    # --- intraday seasonality (open/close effects dominate) ---
    minute_of_day = np.asarray(df.index.hour * 60 + df.index.minute, dtype=float)
    since_open = np.clip(minute_of_day - (9 * 60 + 30), 0, None)
    f["tod_sin"] = np.sin(2 * np.pi * since_open / 390)
    f["tod_cos"] = np.cos(2 * np.pi * since_open / 390)
    f["bars_since_open"] = since_open / 5
    f["bars_to_close"] = np.clip((16 * 60) - minute_of_day, 0, None) / 5

    return f

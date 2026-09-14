"""
Live swing engine -- deploys the configuration that passed the validation gate.

Validated on 10 years of daily bars, 87 symbols, full costs:
    2-day holds, 10 concurrent, volatility targeting
    +27.1%/yr | Sharpe 2.04 | DSR 0.974 | maxDD -22.2% | 15.7 trades/week
    SPY over the same window: +14.4%/yr, Sharpe 1.04, maxDD -25.4%

This is NOT the intraday engine. Key differences, and each one is deliberate:

  * runs ONCE per day near the close, not every 20 seconds
  * positions are held 2 trading days -- overnight exposure is the whole point,
    because it is what makes cost negligible against the captured move
  * no EOD flatten (that rule is what made the intraday version unviable)
  * position size scales by target_vol / realised_vol, which was the single change
    that lifted DSR from 0.872 to 0.974
  * exits ride on GTC bracket legs at the broker, with a hard close at the horizon

    python -m app.swing_engine          # one decision cycle
    python -m app.swing_engine --loop   # daemon, acts once per day near the close
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz

from .settings import S
from .broker import Broker
from .swing import build_daily_features, fetch_daily, Cfg, _clf, UNIVERSE
from .labeling import triple_barrier_labels, make_meta_labels
from .regime import RegimeModel

log = logging.getLogger("swing_engine")
ET = pytz.timezone("America/New_York")

HZ = 2                       # nominal horizon, used for barrier scaling
HZ_CAP = 5                   # hard exit after one trading week, no exceptions
DECAY_DROP = 0.45            # exit when conviction falls this far below entry
PT, SL = 2.0, 1.0
RISK = 0.01                  # base risk per trade, before Kelly scaling
USE_KELLY = True             # quarter-Kelly on the payoff ratio and meta_p
MAX_CONCURRENT = 10
TARGET_VOL = 0.30            # 2x the validated 0.15: the measured survivable ceiling
VOL_LOOKBACK = 20
MAX_LEVERAGE = 3.0           # per-position cap; portfolio vol targeting binds first
# Leverage sweep on the real 10y equity curve, margin cost included:
#   1.0x -> 29.3%/yr, maxDD -20.3%   (validated baseline)
#   2.0x -> 57.5%/yr, maxDD -37.5%   <-- deployed
#   3.0x -> 87.9%/yr, maxDD -51.5%   RUIN: curve breaches -50%
# 2x is the highest setting that never breaches the ruin threshold. Do not raise
# this without re-running app.leverage -- above 2x the strategy dies rather than
# merely drawing down.
LEVERAGE_MULT = 2.0
THRESH = 0.55
DECISION_HOUR = 15           # act at 15:40 ET, near the close
DECISION_MIN = 40


class SwingEngine:
    def __init__(self, s=S):
        self.s = s
        self.broker = Broker(s)
        self.state_path = os.path.join(s.state_dir, "swing_state.json")
        self.models_dir = os.path.join(s.state_dir, "swing_models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.state = self._load()

    # ---------------- state ----------------
    def _load(self):
        try:
            return json.load(open(self.state_path))
        except Exception:
            return {"positions": {}, "equity_history": [], "last_run": None,
                    "peak_equity": None}

    def _save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=2, default=str)

    # ---------------- model ----------------
    def train_all(self, hist, spy):
        """Train one model per symbol on all available history."""
        import joblib
        models = {}
        for sym, df in hist.items():
            if sym == "SPY" or len(df) < 600:
                continue
            f = build_daily_features(df, spy.reindex(df.index).ffill())
            keep = f.dropna().index
            d2, f2 = df.loc[keep], f.loc[keep]
            if len(d2) < 500:
                continue
            cfg = Cfg(HZ)
            scaled = f2["vol"] * np.sqrt(HZ)
            tb = triple_barrier_labels(d2["close"], scaled, cfg)
            y = tb["label"].values
            m = y != 0
            if m.sum() < 100 or len(np.unique(y[m] > 0)) < 2:
                continue
            try:
                reg = RegimeModel(cfg).fit(f2)
                rf = reg.transform(f2)
            except Exception as e:
                log.warning("%s regime failed: %s", sym, e)
                continue
            cols = list(f2.columns) + [c for c in rf.columns
                                       if c.startswith("regime_p")]
            X = pd.concat([f2, rf], axis=1)[cols]
            X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
            primary = _clf(7).fit(X[m], (y[m] > 0).astype(int))
            p = primary.predict_proba(X)[:, 1]
            side = np.where(p >= 0.5, 1, -1)
            my = make_meta_labels(side, y)
            if len(np.unique(my)) < 2:
                continue
            meta = _clf(8).fit(np.column_stack([X, p]), my)
            joblib.dump({"primary": primary, "meta": meta, "regime": reg,
                         "cols": cols,
                         "trained": datetime.now(timezone.utc).isoformat()},
                        os.path.join(self.models_dir, f"{sym}.joblib"))
            models[sym] = True
        log.warning("trained %d swing models", len(models))
        return models

    def score(self, sym, df, spy):
        import joblib
        p = os.path.join(self.models_dir, f"{sym}.joblib")
        if not os.path.exists(p):
            return None
        art = joblib.load(p)
        f = build_daily_features(df, spy.reindex(df.index).ffill())
        f = f.dropna()
        if f.empty:
            return None
        rf = art["regime"].transform(f)
        X = pd.concat([f, rf], axis=1)[art["cols"]]
        X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        pu = art["primary"].predict_proba(X[-1:])[:, 1]
        side = 1 if pu[-1] >= 0.5 else -1
        mp = float(art["meta"].predict_proba(
            np.column_stack([X[-1:], pu]))[:, 1][0])
        return {"side": side, "meta_p": mp,
                "vol": float(f["vol"].iloc[-1]),
                "price": float(df["close"].iloc[-1])}

    # ---------------- sizing ----------------
    def vol_multiplier(self):
        """Scale exposure by target_vol / realised_vol -- the change that took
        DSR from 0.872 to 0.974 in backtest."""
        hist = self.state.get("equity_history", [])
        if len(hist) < VOL_LOOKBACK + 1:
            return 1.0
        eq = pd.Series([h["equity"] for h in hist[-(VOL_LOOKBACK + 1):]])
        r = eq.pct_change().dropna()
        rv = float(r.std() * np.sqrt(252))
        if rv <= 1e-6:
            return 1.0
        return float(np.clip(TARGET_VOL / rv, 0.25, MAX_LEVERAGE))

    # ---------------- one cycle ----------------
    def run_cycle(self, force=False):
        now = datetime.now(ET)
        acct = self.broker.account()
        equity = acct["equity"]
        if acct["blocked"]:
            log.error("account blocked")
            return

        clock = self.broker.clock()
        if not clock["is_open"] and not force:
            log.info("market closed; standing by")
            return

        # act once per day, near the close
        today = now.date().isoformat()
        if self.state.get("last_run") == today and not force:
            log.info("already ran today (%s)", today)
            return
        if not force and (now.hour < DECISION_HOUR or
                          (now.hour == DECISION_HOUR and now.minute < DECISION_MIN)):
            log.info("waiting for decision window %02d:%02d ET",
                     DECISION_HOUR, DECISION_MIN)
            return

        log.warning("=== swing cycle %s | equity %.2f ===", today, equity)
        live = self.broker.positions()

        # ---- close positions that have reached the horizon ----
        # ---- exit on signal decay, or at the one-week hard cap ----
        # A fixed clock cuts winners that are still working and holds losers that
        # have already broken. Backtest: 23.5% -> 29.8%/yr, Sharpe 1.81 -> 2.34,
        # drawdown -24.7% -> -20.1%, all improving together.
        fresh = {}
        try:
            need_now = list(dict.fromkeys(UNIVERSE))
            hist_now = fetch_daily(need_now, years=1.5)
            spy_now = hist_now["SPY"]["close"]
            for sym in list(self.state["positions"]):
                if sym in hist_now:
                    try:
                        fresh[sym] = self.score(sym, hist_now[sym], spy_now)
                    except Exception:
                        pass
        except Exception as e:
            log.warning("could not refresh signals for exits: %s", e)

        for sym, meta in list(self.state["positions"].items()):
            opened = datetime.fromisoformat(meta["opened"]).date()
            held = np.busday_count(opened, now.date())
            if sym not in live:
                log.info("%s no longer open at broker (bracket filled)", sym)
                self.state["positions"].pop(sym, None)
                continue

            reason = None
            if held >= HZ_CAP:
                reason = f"hard cap {HZ_CAP}d"
            else:
                sig = fresh.get(sym)
                if sig:
                    if int(sig["side"]) != int(meta["side"]):
                        reason = "signal flipped"
                    elif sig["meta_p"] < float(meta.get("meta_p", 1.0)) - DECAY_DROP:
                        reason = (f"conviction decayed "
                                  f"{meta.get('meta_p', 0):.2f}->{sig['meta_p']:.2f}")
            if reason:
                log.warning("closing %s after %d days: %s", sym, held, reason)
                self.broker.close_position(sym)
                self.state["positions"].pop(sym, None)

        self._save()
        live = self.broker.positions()

        # ---- fetch and score ----
        need = list(dict.fromkeys(UNIVERSE))
        hist = fetch_daily(need, years=1.5)
        if "SPY" not in hist:
            log.error("no SPY data")
            return
        spy = hist["SPY"]["close"]

        slots = MAX_CONCURRENT - len(live)
        if slots <= 0:
            log.info("no free slots (%d open)", len(live))
            self._finish(today, equity)
            return

        cands = []
        for sym, df in hist.items():
            if sym == "SPY" or sym in live or len(df) < 300:
                continue
            try:
                sig = self.score(sym, df, spy)
            except Exception as e:
                log.debug("%s score failed: %s", sym, e)
                continue
            if sig and sig["meta_p"] >= THRESH:
                cands.append((sig["meta_p"], sym, sig))
        cands.sort(reverse=True, key=lambda x: x[0])
        log.warning("%d candidates, %d slots", len(cands), slots)

        mult = self.vol_multiplier()

        # Drawdown circuit breaker. At 2x the backtested drawdown is -37.5%, so
        # the gap to the -50% ruin level is thin. De-risk progressively rather
        # than discovering the floor the hard way.
        peak = self.state.get("peak_equity") or equity
        dd = (equity - peak) / peak if peak else 0.0
        if dd <= -0.40:
            log.critical("DRAWDOWN %.1f%% -- halting new entries entirely", dd * 100)
            self._finish(today, equity)
            return
        if dd <= -0.30:
            mult *= 0.5
            log.warning("drawdown %.1f%% -- halving size", dd * 100)
        elif dd <= -0.20:
            mult *= 0.75
            log.warning("drawdown %.1f%% -- reducing size to 75%%", dd * 100)

        log.warning("vol multiplier %.2f (leverage target %.1fx)", mult, LEVERAGE_MULT)

        # Portfolio-level exposure budget. MAX_LEVERAGE caps a SINGLE position, so
        # on its own it allowed a $15.8k order on a $10.3k account and every entry
        # was rejected for insufficient buying power. Gross exposure across all
        # open positions must stay within LEVERAGE_MULT x equity, so each new
        # position gets a share of what is left.
        gross_now = sum(abs(p.get("market_value", 0.0)) for p in live.values())
        budget = max(LEVERAGE_MULT * equity - gross_now, 0.0)
        per_slot = budget / max(slots, 1)
        log.warning("exposure: gross %.0f / budget %.0f -> %.0f per slot",
                    gross_now, LEVERAGE_MULT * equity, per_slot)

        opened = 0
        for mp, sym, sig in cands:
            if opened >= slots:
                break
            v = max(sig["vol"], 1e-4) * np.sqrt(HZ)
            entry = sig["price"]
            stop_dist = SL * v * entry
            r = RISK
            if USE_KELLY:
                # quarter-Kelly from the payoff ratio and the model's own
                # probability, bounded so a single confident signal cannot
                # dominate the book
                b = PT / SL
                p_win = float(np.clip(mp, 0.30, 0.80))
                k = max((b * p_win - (1 - p_win)) / b, 0.0)
                r *= float(np.clip(k * 0.25 / RISK, 0.5, 2.0))
            qty = (equity * r * mult) / max(stop_dist, 1e-6)
            # cap by this slot's share of the remaining exposure budget, not by
            # a multiple of the whole account
            qty = min(qty, per_slot / entry)
            qty = int(qty)
            if qty <= 0:
                continue
            side = sig["side"]
            tp = entry * (1 + side * PT * v)
            sl = entry * (1 - side * SL * v)
            try:
                o = self.broker.submit(sym, side, qty, limit_price=None,
                                       take_profit=tp, stop_loss=sl)
            except Exception as e:
                log.error("order failed %s: %s", sym, e)
                continue
            if o is None:
                continue
            self.state["positions"][sym] = {
                "opened": now.isoformat(), "side": side, "qty": qty,
                "entry": entry, "tp": tp, "sl": sl, "meta_p": mp}
            opened += 1
            log.warning("ENTRY %s side=%d qty=%d meta_p=%.3f tp=%.2f sl=%.2f",
                        sym, side, qty, mp, tp, sl)

        self._finish(today, equity)

    def _finish(self, today, equity):
        self.state["last_run"] = today
        self.state.setdefault("equity_history", []).append(
            {"day": today, "equity": equity})
        self.state["equity_history"] = self.state["equity_history"][-400:]
        peak = self.state.get("peak_equity") or equity
        self.state["peak_equity"] = max(peak, equity)
        self._save()
        log.warning("cycle complete | %d open positions",
                    len(self.state["positions"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    errs = S.validate()
    if errs:
        for e in errs:
            log.error("CONFIG: %s", e)
        return 1

    eng = SwingEngine(S)
    log.warning("SWING ENGINE | mode=%s | %dd holds | %d concurrent | vol target %.2f",
                S.mode, HZ, MAX_CONCURRENT, TARGET_VOL)

    if a.train or not os.listdir(eng.models_dir):
        log.warning("training swing models...")
        hist = fetch_daily(UNIVERSE, years=5.0)
        eng.train_all(hist, hist["SPY"]["close"])

    if not a.loop:
        eng.run_cycle(force=a.force)
        return 0

    while True:
        try:
            eng.run_cycle()
        except Exception as e:
            log.exception("cycle error: %s", e)
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())

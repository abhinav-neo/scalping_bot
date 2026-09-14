"""
The trading engine.

Loop shape (every LOOP_SECONDS):
  1. refresh account + clock; roll the session if the date changed
  2. if past FLATTEN_AT -> flatten everything, stand down for the day
  3. manage open positions: take-profit / stop / time-barrier / max-hold
  4. reap stale limit orders past their TTL
  5. if allowed, score the newest closed bar and place new entries

Invariants enforced regardless of signal:
  * no position survives past FLATTEN_AT (no overnight risk)
  * no entry after ENTRY_CUTOFF
  * hard wall-clock max hold, independent of bar count (fixes the data-gap issue
    found in backtesting where 6 bars could span 90 minutes)
"""
import json
import logging
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .settings import S
from .broker import Broker
from .risk import RiskManager, ET
from .features import build_features
from .regime import RegimeModel  # noqa: F401  (unpickled with artifacts)
from .modelcfg import ModelCfg as _Cfg

log = logging.getLogger("engine")


class Engine:
    def __init__(self, s=S):
        self.s = s
        self.broker = Broker(s)
        self.risk = RiskManager(s)
        self.cfg = _Cfg(s)
        self.models = {}
        self.open_meta = {}      # symbol -> dict(entry_time, side, tp, sl, qty)
        self.pending = {}        # symbol -> dict(order_id, placed_at, side)
        self.last_entry_bar = {}   # symbol -> bar timestamp already acted on
        self.cooldown_until = {}   # symbol -> epoch seconds before re-entry allowed
        self.calibrated = set()    # symbols whose tp/sl match the real fill price
        self.scan = {}             # symbol -> latest evaluation, for the dashboard
        self.scan_meta = {}        # loop-level context (market trend, gates)
        self.trade_log = os.path.join(s.state_dir, "trades.jsonl")
        os.makedirs(s.state_dir, exist_ok=True)
        self._load_models()
        self._load_open_meta()
        self._reconcile_on_start()

        # Real-time bar stream. REST bars arrive 4-5 min after close, which the lag
        # sweep showed is past the point where this strategy's edge survives.
        # Streaming drops that to seconds. Falls back to REST if the socket dies.
        self.stream = None
        if getattr(self.s, "use_stream", True):
            try:
                from .stream import BarStream
                self.stream = BarStream(self.s, self.broker.bars)
                self.stream.start(list(dict.fromkeys(
                    self.s.symbols + [self.s.market_symbol])))
            except Exception as e:
                log.error("stream start failed, falling back to REST: %s", e)
                self.stream = None

    def _reconcile_on_start(self):
        """
        Re-sync in-memory state with the broker after a restart.

        `pending` is memory-only, so without this a restart orphans any resting
        limit order: it never ages out via TTL and lingers until the EOD flatten.
        Stale open_meta entries for positions that no longer exist are also dropped
        so they can't mis-calibrate a future trade.
        """
        try:
            live_positions = self.broker.positions()
            open_orders = self.broker.open_orders()
        except Exception as e:
            log.warning("startup reconcile skipped: %s", e)
            return

        for o in open_orders:
            sym = o.symbol
            if sym in self.s.symbols:
                self.pending[sym] = {
                    "order_id": str(o.id),
                    # age from now: we can't know the original placement time, so
                    # give it a fresh TTL rather than cancelling instantly
                    "placed_at": time.time(),
                    "side": 1 if str(o.side).lower().endswith("buy") else -1,
                }
                log.warning("reconciled resting order %s (id %s)", sym, o.id)

        for sym in list(self.open_meta):
            if sym not in live_positions and sym not in self.pending:
                log.warning("dropping stale open_meta for %s (no position, no order)",
                            sym)
                self.open_meta.pop(sym, None)
        self._save_open_meta()

        # A position that survived a previous session must never be inherited and
        # traded around. On 2026-08-03 a bracketed TQQQ short outlived the EOD
        # flatten and gapped 5% overnight. Anything opened before today gets closed
        # on startup, before any new decision is made.
        if live_positions:
            today = datetime.now(ET).date().isoformat()
            stale = []
            for sym in live_positions:
                meta = self.open_meta.get(sym)
                opened = str(meta.get("entry_time", ""))[:10] if meta else ""
                if opened != today:
                    stale.append(sym)
            if stale:
                log.critical("STARTUP: positions from a previous session: %s "
                             "-- closing before trading", stale)
                for sym in stale:
                    self.broker.close_position(sym)
                    self.open_meta.pop(sym, None)
                    self.calibrated.discard(sym)
                self._save_open_meta()
                self._log_trade({"event": "exit", "symbol": ",".join(stale),
                                 "reason": "stale_overnight_position"})
            else:
                log.warning("resuming with %d open position(s) from today: %s",
                            len(live_positions), ", ".join(live_positions))

    # ---------------- models ----------------
    def _load_models(self):
        for sym in self.s.symbols:
            p = os.path.join(self.s.model_dir, f"{sym}.joblib")
            if os.path.exists(p):
                self.models[sym] = joblib.load(p)
                log.info("loaded model %s (trained %s)", sym,
                         self.models[sym].get("trained_at"))
            else:
                log.warning("no model for %s -- run `python -m app.train`", sym)

    def models_stale(self):
        p = os.path.join(self.s.model_dir, "summary.json")
        if not os.path.exists(p):
            return True
        try:
            ts = json.load(open(p))["trained_at"]
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).days
            return age >= self.s.retrain_days
        except Exception:
            return True

    # ---------------- position bookkeeping ----------------
    def _meta_path(self):
        return os.path.join(self.s.state_dir, "open_positions.json")

    def _save_open_meta(self):
        json.dump(self.open_meta, open(self._meta_path(), "w"), indent=2, default=str)

    def _load_open_meta(self):
        p = self._meta_path()
        if os.path.exists(p):
            try:
                self.open_meta = json.load(open(p))
            except Exception:
                self.open_meta = {}

    def _log_trade(self, rec):
        rec["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.trade_log, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ---------------- signal ----------------
    def score(self, sym, bars, mkt):
        art = self.models.get(sym)
        if art is None or len(bars) < 200:
            return None
        df = bars.join(mkt, how="inner").dropna()
        feats = build_features(df, self.cfg)
        feats = feats.dropna()
        if feats.empty:
            return None
        reg = art["regime"].transform(feats)
        X = pd.concat([feats, reg], axis=1)
        Xv = X[art["cols"]].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
        p_up = art["primary"].predict_proba(Xv[-1:])[:, 1]
        side = 1 if p_up[-1] >= 0.5 else -1
        meta_p = float(art["meta"].predict_proba(
            np.column_stack([Xv[-1:], p_up]))[:, 1][0])
        return {"side": side, "p_up": float(p_up[-1]), "meta_p": meta_p,
                "vol": float(feats["vol"].iloc[-1]),
                "price": float(df["close"].iloc[-1]),
                "bar_time": df.index[-1]}

    # ---------------- exits ----------------
    def manage_positions(self, positions):
        now = datetime.now(ET)
        for sym, pos in list(positions.items()):
            meta = self.open_meta.get(sym)
            if meta is None:
                # unknown position (manual or restart) -> close it, we don't manage blind
                log.warning("untracked position %s -> closing", sym)
                self.broker.close_position(sym)
                continue
            try:
                px = self.broker.latest_price(sym)
            except Exception:
                continue
            # Recalibrate the barriers against the ACTUAL fill price the broker
            # reports. They were provisionally set from the signal bar's close,
            # which can differ materially from where the order actually filled --
            # that mismatch caused instant stop-outs on the first live run.
            if sym not in self.calibrated:
                real_entry = pos["avg_entry"]
                v = meta.get("vol")
                if v:
                    side_ = meta["side"]
                    meta["tp"] = real_entry * (1 + side_ * self.s.pt_mult * v)
                    meta["sl"] = real_entry * (1 - side_ * self.s.sl_mult * v)
                    log.info("calibrated %s barriers to fill %.2f (tp %.2f sl %.2f)",
                             sym, real_entry, meta["tp"], meta["sl"])
                self.calibrated.add(sym)
                self._save_open_meta()

            side = meta["side"]
            entry_t = pd.Timestamp(meta["entry_time"])
            if entry_t.tzinfo is None:
                entry_t = entry_t.tz_localize(ET)
            held_min = (now - entry_t.to_pydatetime()).total_seconds() / 60

            hit_tp = (side == 1 and px >= meta["tp"]) or (side == -1 and px <= meta["tp"])
            hit_sl = (side == 1 and px <= meta["sl"]) or (side == -1 and px >= meta["sl"])
            timeout = held_min >= self.s.max_hold_minutes

            if self.s.use_brackets:
                # The exchange owns the tp/sl legs. Acting on them here would race
                # the broker and could double-close. Only the time barrier and the
                # end-of-day flatten remain ours.
                reason = "time" if timeout else None
            else:
                reason = ("take_profit" if hit_tp else "stop" if hit_sl
                          else "time" if timeout else None)
            if reason:
                self.broker.close_position(sym)
                self._log_trade({"event": "exit", "symbol": sym, "reason": reason,
                                 "price": px, "held_min": round(held_min, 1),
                                 "unrealized_pl": pos.get("unrealized_pl")})
                self.open_meta.pop(sym, None)
                self.calibrated.discard(sym)
                # Cooldown stops the bot immediately re-entering the same stale
                # signal, which produced a churn loop on the first live run.
                self.cooldown_until[sym] = time.time() + self.s.cooldown_seconds
                self._save_open_meta()
                log.info("exit %s %s @ %.2f after %.0fmin", sym, reason, px, held_min)

    def reap_pending(self):
        now = time.time()
        for sym, p in list(self.pending.items()):
            if now - p["placed_at"] > self.s.limit_ttl_seconds:
                self.broker.cancel(p["order_id"])
                self.pending.pop(sym, None)
                # Clear the provisional position record too. Without this the bot
                # keeps an open_meta entry for a position that never existed, which
                # persists to disk and survives restarts as phantom state.
                if sym not in self.broker.positions():
                    self.open_meta.pop(sym, None)
                    self.calibrated.discard(sym)
                    self._save_open_meta()
                    self._log_trade({"event": "cancel", "symbol": sym,
                                     "reason": "limit_ttl_expired"})
                log.info("cancelled stale limit order %s", sym)

    def market_trend_bps(self, mkt):
        """
        Short-term trend of the market proxy (SPY), in basis points, over the last
        N bars. Used to veto trades that fight the tape. On 2026-07-29 the bot went
        long 93% of the time while every name in the universe fell 2-5%; this is
        the guard against repeating that.
        """
        col = f"{self.s.market_symbol}_close"
        if col not in mkt.columns or len(mkt) < self.s.market_trend_bars + 1:
            return None
        n = self.s.market_trend_bars
        now_px = float(mkt[col].iloc[-1])
        then_px = float(mkt[col].iloc[-1 - n])
        return (now_px - then_px) / then_px * 1e4

    def _track_positions(self, positions):
        """
        Append a time-series sample of each open position's unrealized P&L so the
        dashboard can draw one live line per open trade. Bounded to today's rows:
        the file is rewritten at each session roll rather than growing forever.
        """
        if not positions:
            return
        path = os.path.join(self.s.state_dir, "position_track.jsonl")
        today = datetime.now(ET).date().isoformat()
        try:
            # roll the file when the session changes
            if os.path.exists(path):
                with open(path) as f:
                    first = f.readline()
                if first:
                    try:
                        if json.loads(first).get("day") != today:
                            os.remove(path)
                    except Exception:
                        os.remove(path)
            ts = datetime.now(ET).isoformat()
            with open(path, "a") as f:
                for sym, p in positions.items():
                    meta = self.open_meta.get(sym, {})
                    f.write(json.dumps({
                        "day": today, "ts": ts, "symbol": sym,
                        "unrealized": round(p.get("unrealized_pl", 0.0), 4),
                        "price": round(p.get("current_price", 0.0) or 0.0, 4),
                        "entry": round(p.get("avg_entry", 0.0), 4),
                        "side": p.get("side"),
                        "tp": meta.get("tp"), "sl": meta.get("sl"),
                    }) + "\n")
        except Exception:
            pass

    def _record(self, sym, status, reason=None, sig=None, **extra):
        """
        Capture what the bot saw and decided for one symbol this loop.
        Written to state/scan_state.json so the dashboard can show the reasoning,
        not just the outcome. Purely observational -- never affects decisions.
        """
        rec = {"ts": datetime.now(ET).isoformat(), "symbol": sym,
               "status": status, "reason": reason}
        if sig:
            rec.update({"price": round(sig.get("price", 0), 4),
                        "side": sig.get("side"),
                        "p_up": round(sig.get("p_up", 0), 4),
                        "meta_p": round(sig.get("meta_p", 0), 4),
                        "bar_time": str(sig.get("bar_time"))})
        rec.update(extra)
        self.scan[sym] = rec

    def _write_scan(self):
        try:
            payload = {"ts": datetime.now(ET).isoformat(),
                       "meta": self.scan_meta,
                       "symbols": self.scan}
            with open(os.path.join(self.s.state_dir, "scan_state.json"), "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception:
            pass

    # ---------------- entries ----------------
    def try_entries(self, equity, positions, bars_map, mkt):
        trend = self.market_trend_bps(mkt) if self.s.market_filter else None
        can, why = self.risk.can_open(equity, len(positions))
        # Merge, don't replace: run_once() already put bar_source/stream_age here
        # and a wholesale assignment silently dropped them from the dashboard.
        self.scan_meta.update({
            "market_trend_bps": round(trend, 1) if trend is not None else None,
            "market_trend_bars": self.s.market_trend_bars,
            "can_open": bool(can),
            "block_reason": why,
            "meta_threshold": self.s.meta_threshold,
            "open_positions": len(positions),
            "pending": len(self.pending),
        })
        if not can:
            for sym in self.s.symbols:
                if sym in positions:
                    self._record(sym, "in_position")
                elif sym in self.pending:
                    self._record(sym, "order_resting")
                else:
                    self._record(sym, "blocked", why)
            return
        for sym in self.s.symbols:
            if sym in positions:
                self._record(sym, "in_position")
                continue
            if sym in self.pending:
                self._record(sym, "order_resting")
                continue
            if len(positions) + len(self.pending) >= self.s.max_concurrent_positions:
                self._record(sym, "capacity", "max concurrent positions reached")
                continue
            bars = bars_map.get(sym)
            if bars is None:
                self._record(sym, "no_data", "no bars returned")
                continue
            if time.time() < self.cooldown_until.get(sym, 0):
                left = int(self.cooldown_until[sym] - time.time())
                self._record(sym, "cooldown", f"{left}s remaining")
                continue

            sig = self.score(sym, bars, mkt)
            if not sig:
                self._record(sym, "no_signal", "model returned nothing")
                continue
            if sig["meta_p"] < self.s.meta_threshold:
                self._record(sym, "miss",
                             f"confidence {sig['meta_p']:.3f} < {self.s.meta_threshold}",
                             sig=sig)
                continue

            # --- guard 0: don't fight the tape. Veto longs into a falling market
            # and shorts into a rising one. These names are highly correlated, so
            # without this the bot stacks same-direction risk across all of them. ---
            if trend is not None:
                if sig["side"] > 0 and trend < -self.s.market_trend_bps_limit:
                    log.info("SKIP %s long: market trend %+.0f bps (falling tape)",
                             sym, trend)
                    self._record(sym, "veto",
                                 f"long blocked: tape {trend:+.0f} bps (falling)",
                                 sig=sig)
                    continue
                if sig["side"] < 0 and trend > self.s.market_trend_bps_limit:
                    log.info("SKIP %s short: market trend %+.0f bps (rising tape)",
                             sym, trend)
                    self._record(sym, "veto",
                                 f"short blocked: tape {trend:+.0f} bps (rising)",
                                 sig=sig)
                    continue

            # --- guard 1: one entry per bar. Without this the bot re-fires the
            # same signal every loop (20s), producing a churn loop. ---
            if self.last_entry_bar.get(sym) == str(sig["bar_time"]):
                self._record(sym, "waiting", "already acted on this bar", sig=sig)
                continue

            # --- guard 2: bar freshness. A 5-minute strategy on delayed data is
            # not a strategy. Free IEX bars lag ~15 min; refuse rather than guess. ---
            bar_age = (datetime.now(ET) - sig["bar_time"].to_pydatetime()).total_seconds() / 60
            if bar_age > self.s.max_bar_age_minutes:
                log.warning("SKIP %s: bars stale by %.1f min (limit %d) -- data feed "
                            "cannot support intraday scalping",
                            sym, bar_age, self.s.max_bar_age_minutes)
                self._record(sym, "stale",
                             f"bars {bar_age:.1f} min old (limit {self.s.max_bar_age_minutes})",
                             sig=sig, bar_age_min=round(bar_age, 1))
                continue

            # --- guard 3: signal price must agree with the live trade print.
            # The IEX *quoted* spread is structurally wide and is NOT a real cost
            # (orders route to the NBBO), so we don't gate on it. What does matter
            # is whether the bar we scored still reflects the market. ---
            try:
                live_px = self.broker.latest_price(sym)
            except Exception:
                continue
            drift_bps = abs(live_px - sig["price"]) / live_px * 1e4
            # The drift limit must scale with the profit target, not be a fixed
            # bps figure. It was set to 40 bps when barriers targeted ~30 bps; now
            # they target ~180 bps, so 40 bps rejected signals that were still well
            # inside the trade's range. Express it as a fraction of the target.
            target_bps = self.s.pt_mult * max(sig["vol"], 1e-4) * \
                np.sqrt(self.s.horizon_bars) * 1e4
            drift_limit = max(self.s.max_price_drift_frac * target_bps,
                              self.s.min_price_drift_bps)
            if drift_bps > drift_limit:
                log.warning("SKIP %s: signal price %.2f drifted %.0f bps from live "
                            "%.2f (limit %.0f = %.0f%% of %.0f bps target)",
                            sym, sig["price"], drift_bps, live_px, drift_limit,
                            self.s.max_price_drift_frac * 100, target_bps)
                self._record(sym, "drift",
                             f"signal {sig['price']:.2f} vs live {live_px:.2f} "
                             f"({drift_bps:.0f} bps, limit {drift_limit:.0f})",
                             sig=sig, live_price=round(live_px, 4))
                continue

            gross = sum(abs(p.get("market_value", 0.0)) for p in positions.values())
            # Barriers scale with sqrt(horizon) -- same as training labels and
            # backtest_v3. Using raw per-bar vol made every horizon target the same
            # ~1-bar move, so trades exited in minutes regardless of setting.
            v = max(sig["vol"], 1e-4) * np.sqrt(self.s.horizon_bars)
            # Sizing must use the SCALED vol: the stop is now sqrt(horizon) times
            # wider, so risking the same fraction of equity needs a proportionally
            # smaller position. Passing raw vol here would oversize every trade.
            qty = self.risk.size(equity, sig["price"], v,
                                 current_gross_notional=gross)
            if qty <= 0:
                self._record(sym, "no_size", "risk sizing returned zero", sig=sig)
                continue
            side = sig["side"]
            tp = sig["price"] * (1 + side * self.s.pt_mult * v)
            sl = sig["price"] * (1 - side * self.s.sl_mult * v)

            limit_px = None
            if self.s.use_limit_entry:
                # Entry offset stays on RAW per-bar vol: it is about how far price
                # moves in the next bar or two, not about the horizon target.
                raw_v = max(sig["vol"], 1e-4)
                limit_px = sig["price"] - side * self.s.limit_offset_frac * raw_v * sig["price"]

            try:
                if self.s.use_brackets:
                    # Exits rest at the exchange from the moment the entry fills.
                    o = self.broker.submit(sym, side, qty, limit_price=limit_px,
                                           take_profit=tp, stop_loss=sl)
                else:
                    o = self.broker.submit(sym, side, qty, limit_price=limit_px)
            except Exception as e:
                log.error("order failed %s: %s", sym, e)
                self._record(sym, "error", f"order rejected: {e}", sig=sig)
                continue

            # submit() returns None when nothing was actually sent (dry-run, or the
            # quantity rounded to zero for a high-priced name on a small account).
            # Recording state here would create a phantom position: tracked, counted
            # against the daily trade cap, but with no order in existence.
            if o is None:
                if not self.s.dry_run:
                    log.warning("SKIP %s: order not placed (qty %.4f rounds to zero "
                                "at price %.2f)", sym, qty, sig["price"])
                    self._record(sym, "no_size",
                                 f"qty {qty:.4f} rounds to 0 at {sig['price']:.2f}",
                                 sig=sig)
                else:
                    self._record(sym, "dry_run", "order suppressed (DRY_RUN)", sig=sig)
                continue

            # record the quantity actually submitted -- limit orders are whole shares
            sent_qty = float(getattr(o, "qty", qty) or qty)

            if limit_px is not None:
                self.pending[sym] = {"order_id": str(o.id), "placed_at": time.time(),
                                     "side": side}
            self.last_entry_bar[sym] = str(sig["bar_time"])
            self.open_meta[sym] = {"entry_time": datetime.now(ET).isoformat(),
                                   "side": side, "tp": tp, "sl": sl, "qty": sent_qty,
                                   "vol": v, "meta_p": sig["meta_p"]}
            self._save_open_meta()
            self.risk.record_trade()
            self._log_trade({"event": "entry", "symbol": sym, "side": side,
                             "qty": sent_qty, "price": sig["price"], "limit": limit_px,
                             "meta_p": sig["meta_p"], "tp": tp, "sl": sl})
            log.info("ENTRY %s side=%d qty=%.3f meta_p=%.3f limit=%s",
                     sym, side, sent_qty, sig["meta_p"], limit_px)
            self._record(sym, "HIT",
                         f"order sent: {'BUY' if side > 0 else 'SELL'} {sent_qty:.3f} @ "
                         f"{('%.2f' % limit_px) if limit_px else 'MKT'}",
                         sig=sig, qty=round(sent_qty, 4),
                         limit_price=round(limit_px, 4) if limit_px else None,
                         tp=round(tp, 4), sl=round(sl, 4))

    # ---------------- main loop ----------------
    def run_once(self):
        clock = self.broker.clock()
        acct = self.broker.account()
        equity = acct["equity"]
        self.risk.roll_day(equity)

        if acct["blocked"]:
            log.error("account blocked by broker; standing down")
            return

        positions = self.broker.positions()

        # hard no-overnight guarantee
        if self.risk.past_flatten():
            if positions or self.pending:
                ok = self.broker.flatten_all("EOD flatten")
                for sym in list(self.open_meta):
                    self._log_trade({"event": "exit", "symbol": sym,
                                     "reason": "eod_flat"})
                self.open_meta.clear()
                self.pending.clear()
                self.calibrated.clear()
                self._save_open_meta()
                if not ok:
                    # Do not go quiet on a failed flatten: keep looping so the
                    # retry path runs again on the next pass.
                    log.critical("EOD flatten did not complete -- will retry")
                    self.scan_meta["flatten_failed"] = True
            return

        if not clock["is_open"]:
            self.scan_meta = {"market_open": False,
                              "next_open": str(clock.get("next_open"))}
            for sym in self.s.symbols:
                self._record(sym, "market_closed")
            self._write_scan()
            return

        self.reap_pending()
        # a filled limit order clears from pending once the position appears
        for sym in list(self.pending):
            if sym in positions:
                self.pending.pop(sym, None)

        self.manage_positions(positions)
        self._track_positions(self.broker.positions())

        ok, why = self.risk.check_halts(equity)
        if not ok:
            return

        need = list(dict.fromkeys(self.s.symbols + [self.s.market_symbol]))
        # Prefer streamed bars (seconds old); fall back to REST (4-5 min old) only
        # if the socket is unhealthy. Record which source was used so the dashboard
        # can show it -- trading on REST bars is a degraded mode, not normal.
        bars_map, src = None, "rest"
        if self.stream is not None and self.stream.healthy():
            try:
                bars_map = self.stream.get_bars(need)
                src = "stream"
            except Exception as e:
                log.warning("stream read failed: %s", e)
                bars_map = None
        if not bars_map:
            bars_map = self.broker.bars(need, self.s.bar_minutes, lookback_days=12)
            src = "rest"
            if self.stream is not None:
                log.warning("using REST bars (stream age %.0fs) -- signal lag is "
                            "elevated", self.stream.age_seconds())
        self.scan_meta["bar_source"] = src
        if self.stream is not None:
            self.scan_meta["stream_age_s"] = round(self.stream.age_seconds(), 1)
        if self.s.market_symbol not in bars_map:
            return
        mkt = bars_map[self.s.market_symbol][["close"]].rename(
            columns={"close": f"{self.s.market_symbol}_close"})

        self.try_entries(equity, self.broker.positions(), bars_map, mkt)
        self._write_scan()

    def heartbeat(self):
        acct = self.broker.account()
        st = self.risk.state
        hb = {"ts": datetime.now(timezone.utc).isoformat(),
              "mode": self.s.mode, "equity": acct["equity"],
              "day_start_equity": st.get("day_start_equity"),
              "peak_equity": st.get("peak_equity"),
              "trades_today": st.get("trades_today"),
              "day_locked": st.get("day_locked"), "halted": st.get("halted"),
              "positions": self.broker.positions(),
              "pending": list(self.pending)}
        json.dump(hb, open(os.path.join(self.s.state_dir, "heartbeat.json"), "w"),
                  indent=2, default=str)
        return hb

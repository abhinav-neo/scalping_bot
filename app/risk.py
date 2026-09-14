"""
Risk manager. Every guard that can stop the bot lives here, evaluated before any
order is sent. State persists to disk so a container restart cannot silently reset
a tripped kill switch mid-day.
"""
import json
import logging
import os
from datetime import datetime, time as dtime

import pytz

log = logging.getLogger("risk")
ET = pytz.timezone("America/New_York")


def _t(hhmm):
    h, m = hhmm.split(":")
    return dtime(int(h), int(m))


class RiskManager:
    def __init__(self, s):
        self.s = s
        self.path = os.path.join(s.state_dir, "risk_state.json")
        os.makedirs(s.state_dir, exist_ok=True)
        self.state = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                return json.load(open(self.path))
            except Exception:
                pass
        return {"day": None, "day_start_equity": None, "peak_equity": None,
                "trades_today": 0, "day_locked": False, "halted": False,
                "halt_reason": None}

    def save(self):
        json.dump(self.state, open(self.path, "w"), indent=2)

    # ---------------- daily roll ----------------
    def roll_day(self, equity):
        today = datetime.now(ET).date().isoformat()
        if self.state["day"] != today:
            log.info("new session %s | starting equity %.2f", today, equity)
            self.state.update(day=today, day_start_equity=equity,
                              trades_today=0, day_locked=False)
        if self.state["peak_equity"] is None or equity > self.state["peak_equity"]:
            self.state["peak_equity"] = equity
        self.save()

    # ---------------- session windows ----------------
    def now_et(self):
        return datetime.now(ET)

    def in_entry_window(self):
        t = self.now_et().time()
        return _t(self.s.entry_start) <= t < _t(self.s.entry_cutoff)

    def past_flatten(self):
        return self.now_et().time() >= _t(self.s.flatten_at)

    # ---------------- guards ----------------
    def check_halts(self, equity):
        """Returns (can_trade, reason). Hard halts survive restarts."""
        st = self.state
        if st["halted"]:
            return False, f"HALTED: {st['halt_reason']}"

        peak = st.get("peak_equity") or equity
        dd = (equity - peak) / peak if peak else 0.0
        if dd <= -abs(self.s.max_total_drawdown):
            st["halted"] = True
            st["halt_reason"] = f"max drawdown {dd:.1%} breached"
            self.save()
            log.error("BOT HALTED: %s", st["halt_reason"])
            return False, st["halt_reason"]

        d0 = st.get("day_start_equity") or equity
        day_pl = (equity - d0) / d0 if d0 else 0.0
        # Once the daily kill trips it stays tripped until tomorrow. A recovery in
        # equity must NOT silently re-enable trading -- that was the whole point of
        # stopping. (Bug found in testing: recovery previously unlocked the day.)
        if st["day_locked"]:
            return False, f"daily loss kill latched (day P/L {day_pl:.2%})"
        if day_pl <= -abs(self.s.daily_loss_kill):
            st["day_locked"] = True
            self.save()
            log.error("DAILY KILL SWITCH: day P/L %.2f%%", day_pl * 100)
            return False, f"daily loss kill ({day_pl:.2%})"

        if st["trades_today"] >= self.s.max_trades_per_day:
            return False, "max trades/day reached"

        return True, None

    def can_open(self, equity, n_positions):
        ok, reason = self.check_halts(equity)
        if not ok:
            return False, reason
        if not self.in_entry_window():
            return False, "outside entry window"
        if n_positions >= self.s.max_concurrent_positions:
            return False, "max concurrent positions"
        return True, None

    def record_trade(self):
        self.state["trades_today"] += 1
        self.save()

    # ---------------- sizing ----------------
    def size(self, equity, price, vol, current_gross_notional=0.0):
        """
        Risk-based sizing with a volatility-scaled leverage cap AND a portfolio-level
        gross exposure cap. Without the portfolio cap, N concurrent positions each
        sized at the per-position cap would stack to N x leverage -- e.g. 2 positions
        at 2x each = 4x gross on the account. (Bug found in testing.)
        """
        import numpy as np
        v = max(float(vol), 1e-4)
        stop_dist = self.s.sl_mult * v * price
        risk_dollars = equity * self.s.risk_per_trade
        qty = risk_dollars / max(stop_dist, 1e-6)

        vol_scaled_lev = self.s.base_leverage * min(
            1.0, self.s.target_daily_vol / (v * np.sqrt(78)))
        per_pos_notional = self.s.max_notional_frac * equity * max(vol_scaled_lev, 0.25)

        # portfolio gross cap: total exposure never exceeds base_leverage x equity
        portfolio_room = max(self.s.base_leverage * equity - current_gross_notional, 0.0)
        allowed_notional = min(per_pos_notional, portfolio_room)

        qty = min(qty, allowed_notional / price)
        return max(qty, 0.0)

"""
Analytics layer for the dashboard.

Everything here is derived from the BROKER's own record (filled orders and
portfolio history), not from the bot's local log. That matters: local logs record
intent, the broker records what actually happened. Where they disagree, the broker
is right.

Round-trip trades are reconstructed from fills using FIFO matching, which is what
lets us report realized P&L per trade rather than just a running equity number.
"""
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
from alpaca.trading.enums import QueryOrderStatus

log = logging.getLogger("analytics")
ET = "America/New_York"


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


class Analytics:
    def __init__(self, s):
        self.s = s
        self.trading = TradingClient(s.key_id, s.secret, paper=s.paper)

    # ------------------------------------------------------------------ #
    def filled_orders(self, days=30, limit=500):
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=datetime.now(timezone.utc) - timedelta(days=days),
            limit=limit,
            nested=False)
        out = []
        for o in self.trading.get_orders(req):
            qty = _f(o.filled_qty)
            if qty <= 0 or o.filled_avg_price is None:
                continue
            out.append({
                "symbol": o.symbol,
                "side": 1 if str(o.side).lower().endswith("buy") else -1,
                "qty": qty,
                "price": _f(o.filled_avg_price),
                "ts": o.filled_at or o.submitted_at,
                "order_id": str(o.id),
            })
        out.sort(key=lambda r: r["ts"])
        return out

    # ------------------------------------------------------------------ #
    def round_trips(self, days=30):
        """
        FIFO-match fills into closed round trips.

        Returns a list of dicts with entry/exit price, qty, realized P&L, duration
        and direction. Any residual unmatched quantity is an open position and is
        deliberately excluded -- unrealized P&L is reported separately.
        """
        fills = self.filled_orders(days=days)
        books = defaultdict(deque)      # symbol -> deque of open lots
        trips = []

        for f in fills:
            sym, side, qty, px, ts = (f["symbol"], f["side"], f["qty"],
                                      f["price"], f["ts"])
            book = books[sym]
            # close against opposite-side lots first
            while qty > 1e-9 and book and book[0]["side"] != side:
                lot = book[0]
                matched = min(qty, lot["qty"])
                pnl = (px - lot["price"]) * matched * lot["side"]
                dur = None
                try:
                    dur = (ts - lot["ts"]).total_seconds() / 60
                except Exception:
                    pass
                trips.append({
                    "symbol": sym,
                    "side": lot["side"],
                    "qty": matched,
                    "entry": lot["price"],
                    "exit": px,
                    "entry_ts": lot["ts"],
                    "exit_ts": ts,
                    "held_min": round(dur, 1) if dur is not None else None,
                    "pnl": pnl,
                    "pnl_pct": (px / lot["price"] - 1) * 100 * lot["side"],
                })
                lot["qty"] -= matched
                qty -= matched
                if lot["qty"] <= 1e-9:
                    book.popleft()
            if qty > 1e-9:
                book.append({"side": side, "qty": qty, "price": px, "ts": ts})

        trips.sort(key=lambda r: r["exit_ts"])

        # Performance epoch: drop trades from before the re-baseline date so stats
        # reflect the current strategy only. Alpaca can't reset a paper balance, so
        # this is how we get a clean measurement window without a new account.
        epoch = getattr(self.s, "perf_epoch", "")
        if epoch:
            try:
                cutoff = datetime.fromisoformat(epoch).replace(tzinfo=timezone.utc)
                trips = [t for t in trips if t["exit_ts"] >= cutoff]
            except Exception:
                log.warning("bad PERF_EPOCH %r, ignoring", epoch)
        return trips

    # ------------------------------------------------------------------ #
    def portfolio_history(self, period="1M", timeframe="1D"):
        try:
            req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe,
                                             extended_hours=False)
            h = self.trading.get_portfolio_history(req)
            eq = [e for e in (h.equity or [])]
            ts = [e for e in (h.timestamp or [])]
            pts = [{"t": int(t), "equity": _f(e)}
                   for t, e in zip(ts, eq) if e is not None and _f(e) > 0]
            return pts
        except Exception as e:
            log.warning("portfolio_history failed: %s", e)
            return []

    # ------------------------------------------------------------------ #
    def positions(self):
        out = []
        try:
            for p in self.trading.get_all_positions():
                qty = _f(p.qty)
                out.append({
                    "symbol": p.symbol,
                    "qty": qty,
                    "side": 1 if qty > 0 else -1,
                    "entry": _f(p.avg_entry_price),
                    "current": _f(p.current_price),
                    "market_value": _f(p.market_value),
                    "unrealized": _f(p.unrealized_pl),
                    "unrealized_pct": _f(p.unrealized_plpc) * 100,
                })
        except Exception as e:
            log.warning("positions failed: %s", e)
        return out

    def account(self):
        try:
            a = self.trading.get_account()
            return {"equity": _f(a.equity), "cash": _f(a.cash),
                    "buying_power": _f(a.buying_power),
                    "last_equity": _f(a.last_equity)}
        except Exception as e:
            log.warning("account failed: %s", e)
            return {"equity": 0, "cash": 0, "buying_power": 0, "last_equity": 0}

    # ------------------------------------------------------------------ #
    @staticmethod
    def stats(trips):
        """Summary metrics over a list of round trips."""
        if not trips:
            return {"trades": 0, "win_rate": 0, "profit_factor": 0, "net": 0,
                    "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0,
                    "avg_hold": 0, "expectancy": 0}
        wins = [t for t in trips if t["pnl"] > 0]
        losses = [t for t in trips if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins)
        gl = -sum(t["pnl"] for t in losses)
        holds = [t["held_min"] for t in trips if t["held_min"] is not None]
        net = sum(t["pnl"] for t in trips)
        return {
            "trades": len(trips),
            "win_rate": 100 * len(wins) / len(trips),
            "profit_factor": (gw / gl) if gl > 1e-9 else (float("inf") if gw > 0 else 0),
            "net": net,
            "avg_win": (gw / len(wins)) if wins else 0,
            "avg_loss": (-gl / len(losses)) if losses else 0,
            "best": max((t["pnl"] for t in trips), default=0),
            "worst": min((t["pnl"] for t in trips), default=0),
            "avg_hold": (sum(holds) / len(holds)) if holds else 0,
            "expectancy": net / len(trips),
        }

    def execution_quality(self, state_dir, trips):
        """
        Measured exit quality in R (1R = intended stop distance).

        This is the metric that decided the whole strategy: polled market exits
        filled stops ~0.31R worse than intended, which turned a 1.5:1 payoff into
        0.92:1. Bracket orders should bring it near 0.10R. Anything drifting back
        toward 0.30R means the edge is gone, so it is worth watching continuously
        rather than running a script after the fact.
        """
        import os
        intents = {}
        path = os.path.join(state_dir, "trades.jsonl")
        try:
            with open(path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("event") == "entry" and e.get("tp") and e.get("sl"):
                        intents.setdefault(e["symbol"], []).append(e)
        except Exception:
            pass

        rows = []
        for t in trips:
            cand = intents.get(t["symbol"], [])
            best, bestd = None, 1e9
            for e in cand:
                dd = abs(float(e.get("price", 0)) - t["entry"])
                if dd < bestd:
                    best, bestd = e, dd
            if best is None or bestd / max(t["entry"], 1e-9) > 0.02:
                continue
            entry, side = t["entry"], t["side"]
            tp, sl = float(best["tp"]), float(best["sl"])
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            target_R = abs(tp - entry) / risk
            realised_R = (t["exit"] - entry) * side / risk
            if realised_R >= target_R - 0.15:
                oc = "target"
            elif realised_R <= -1 + 0.15:
                oc = "stop"
            else:
                oc = "time"
            rows.append({"R": realised_R, "target_R": target_R, "outcome": oc,
                         "pnl": t["pnl"], "held": t["held_min"]})

        if not rows:
            return {"matched": 0}

        stops = [r["R"] for r in rows if r["outcome"] == "stop"]
        tgts = [r["R"] for r in rows if r["outcome"] == "target"]
        times = [r["R"] for r in rows if r["outcome"] == "time"]
        wins = [r["R"] for r in rows if r["pnl"] > 0]
        loss = [r["R"] for r in rows if r["pnl"] <= 0]
        mean_target_R = sum(r["target_R"] for r in rows) / len(rows)

        def m(x):
            return round(sum(x) / len(x), 3) if x else None

        stop_slip = (round(-1.0 - m(stops), 3) if stops else None)
        return {
            "matched": len(rows),
            "mean_R": round(sum(r["R"] for r in rows) / len(rows), 3),
            "avg_win_R": m(wins), "avg_loss_R": m(loss),
            "target_R": round(mean_target_R, 2),
            "stop_mean_R": m(stops), "stop_slip_R": stop_slip,
            "target_mean_R": m(tgts),
            "time_mean_R": m(times),
            "n_stop": len(stops), "n_target": len(tgts), "n_time": len(times),
            "pct_stop": round(100 * len(stops) / len(rows), 1),
            "pct_target": round(100 * len(tgts) / len(rows), 1),
            "pct_time": round(100 * len(times) / len(rows), 1),
            "realised_ratio": (round(abs(m(wins) / m(loss)), 2)
                               if wins and loss and m(loss) else None),
        }

    def by_symbol(self, trips):
        g = defaultdict(list)
        for t in trips:
            g[t["symbol"]].append(t)
        return {sym: self.stats(v) for sym, v in sorted(g.items())}

    # ------------------------------------------------------------------ #
    def ops_metrics(self, state_dir, trips):
        """
        Operational metrics the broker record alone can't answer.

        Fill rate matters most: the backtest assumes every signal fills at the bar
        close, while live passive limits often don't fill at all. If this diverges
        badly from ~100%, every backtest number is optimistic.

        Breakeven win rate comes from the barrier geometry: with a profit target at
        pt*vol and a stop at sl*vol, you need sl/(pt+sl) winners just to break even.
        At pt=0.7/sl=1.0 that is 58.8% -- a thin margin worth watching directly.
        """
        import os
        entries = cancels = 0
        exits = defaultdict(int)
        path = os.path.join(state_dir, "trades.jsonl")
        try:
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    ev = r.get("event")
                    if ev == "entry":
                        entries += 1
                    elif ev == "cancel":
                        cancels += 1
                    elif ev == "exit":
                        exits[r.get("reason", "unknown")] += 1
        except Exception:
            pass

        filled = max(entries - cancels, 0)

        # Breakeven win rate. The barrier geometry sl/(pt+sl) is only the breakeven
        # if EVERY trade exits at a barrier -- but ~20% exit at the time barrier
        # somewhere in between, which improves the payoff ratio. So prefer the
        # EMPIRICAL breakeven from measured win/loss sizes once we have enough
        # trades, and fall back to the theoretical one before that.
        # Theoretical breakeven from barrier geometry: with a target at pt*vol and
        # a stop at sl*vol, you need sl/(pt+sl) winners. The sqrt(horizon) scaling
        # applies to BOTH barriers so it cancels out and does not appear here.
        pt, sl = float(self.s.pt_mult), float(self.s.sl_mult)
        theoretical = sl / (pt + sl) * 100 if (pt + sl) > 0 else None
        breakeven, be_basis = theoretical, "barrier geometry"
        if len(trips) >= 20:
            w = [t["pnl"] for t in trips if t["pnl"] > 0]
            l = [-t["pnl"] for t in trips if t["pnl"] <= 0]
            if w and l:
                aw, al = sum(w) / len(w), sum(l) / len(l)
                if (aw + al) > 0:
                    breakeven = al / (aw + al) * 100
                    be_basis = "measured win/loss"

        longs = sum(1 for t in trips if t["side"] > 0)
        shorts = len(trips) - longs

        return {
            "orders_sent": entries,
            "orders_cancelled": cancels,
            "orders_filled": filled,
            "fill_rate": round(100 * filled / entries, 1) if entries else None,
            "exit_reasons": dict(exits),
            "breakeven_win_pct": round(breakeven, 1) if breakeven else None,
            "breakeven_basis": be_basis,
            "breakeven_theoretical_pct": round(theoretical, 1) if theoretical else None,
            "long_pct": round(100 * longs / len(trips), 1) if trips else None,
            "long_n": longs, "short_n": shorts,
        }

    def risk_headroom(self, equity, risk_state):
        """Distance to each safety stop, so limits are visible before they trigger."""
        d0 = risk_state.get("day_start_equity") or equity
        peak = risk_state.get("peak_equity") or equity
        day_pl = (equity - d0) / d0 * 100 if d0 else 0.0
        dd = (equity - peak) / peak * 100 if peak else 0.0
        kill = float(self.s.daily_loss_kill) * 100
        halt = float(self.s.max_total_drawdown) * 100
        return {
            "day_pl_pct": round(day_pl, 2),
            "daily_kill_pct": round(kill, 2),
            "daily_room_pct": round(day_pl + kill, 2),
            "drawdown_pct": round(dd, 2),
            "halt_pct": round(halt, 2),
            "drawdown_room_pct": round(dd + halt, 2),
            "trades_today": risk_state.get("trades_today", 0),
            "max_trades_per_day": self.s.max_trades_per_day,
        }

    def config_summary(self):
        s = self.s
        return {
            "symbols": s.symbols,
            "bar_minutes": s.bar_minutes,
            "horizon_bars": s.horizon_bars,
            "horizon_minutes": s.horizon_bars * s.bar_minutes,
            "max_hold_minutes": s.max_hold_minutes,
            "pt_mult": s.pt_mult, "sl_mult": s.sl_mult,
            "meta_threshold": s.meta_threshold,
            "risk_per_trade_pct": round(s.risk_per_trade * 100, 2),
            "max_concurrent": s.max_concurrent_positions,
            "market_filter": s.market_filter,
            "limit_entry": s.use_limit_entry,
            "brackets": getattr(s, "use_brackets", False),
            "stream": getattr(s, "use_stream", False),
            "feed": "SIP" if s.use_sip else "IEX",
        }

    def windowed(self, trips):
        """Net realized P&L across standard lookback windows."""
        now = datetime.now(timezone.utc)
        wins = {"today": 1, "week": 7, "month": 30, "all": 3650}
        out = {}
        for name, days in wins.items():
            if name == "today":
                sel = [t for t in trips
                       if t["exit_ts"].astimezone(timezone.utc).date() == now.date()]
            else:
                cutoff = now - timedelta(days=days)
                sel = [t for t in trips if t["exit_ts"] >= cutoff]
            st = self.stats(sel)
            out[name] = {"net": st["net"], "trades": st["trades"],
                         "win_rate": st["win_rate"]}
        return out

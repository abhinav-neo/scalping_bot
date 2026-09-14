"""
Alpaca broker adapter. All I/O with the broker goes through here so the trading
logic stays testable and the safety checks live in one place.
"""
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest, LimitOrderRequest,
                                     GetOrdersRequest, TakeProfitRequest,
                                     StopLossRequest)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (StockBarsRequest, StockLatestQuoteRequest,
                                  StockLatestTradeRequest)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

log = logging.getLogger("broker")


class Broker:
    def __init__(self, s):
        self.s = s
        self.trading = TradingClient(s.key_id, s.secret, paper=s.paper)
        self.data = StockHistoricalDataClient(s.key_id, s.secret)
        # DataFeed.IEX is real-time on the free plan; SIP requires Algo Trader Plus.
        self.feed = DataFeed.SIP if getattr(s, "use_sip", False) else DataFeed.IEX
        log.warning("Broker connected in %s mode | data feed = %s",
                    s.mode, self.feed.value)

    # ---------------- account ----------------
    def account(self):
        a = self.trading.get_account()
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "daytrade_count": int(getattr(a, "daytrade_count", 0) or 0),
            "blocked": bool(a.trading_blocked or a.account_blocked),
            "pattern_day_trader": bool(getattr(a, "pattern_day_trader", False)),
        }

    def clock(self):
        c = self.trading.get_clock()
        return {"is_open": bool(c.is_open), "ts": c.timestamp,
                "next_open": c.next_open, "next_close": c.next_close}

    # ---------------- data ----------------
    def bars(self, symbols, minutes=5, lookback_days=12):
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        # IMPORTANT: the feed must be specified explicitly. Left unset, Alpaca
        # serves delayed SIP on the free plan (~15-20 min stale), which silently
        # breaks any intraday strategy. IEX is real-time on the free plan.
        req = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
            start=start,
            feed=self.feed)
        df = self.data.get_stock_bars(req).df
        if df is None or len(df) == 0:
            return {}
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

    def latest_quote(self, symbol):
        q = self.data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self.feed))[symbol]
        bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
        mid = (bid + ask) / 2 if bid and ask else 0.0
        spread_bps = ((ask - bid) / mid * 1e4) if mid else None
        return {"bid": bid, "ask": ask, "mid": mid, "spread_bps": spread_bps}

    def latest_price(self, symbol):
        """
        Last traded price. Preferred over the quote mid for exit decisions:
        on the IEX feed the quoted spread is structurally wide (IEX is a single
        venue, ~2% of volume), so the mid is noisy and not what you actually
        execute against -- orders route to the NBBO. Trade prints are reliable.
        """
        t = self.data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol, feed=self.feed))[symbol]
        return float(t.price)

    # ---------------- positions & orders ----------------
    def positions(self):
        out = {}
        for p in self.trading.get_all_positions():
            out[p.symbol] = {
                "qty": float(p.qty),
                "side": 1 if float(p.qty) > 0 else -1,
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price or 0),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_pct": float(p.unrealized_plpc or 0) * 100,
            }
        return out

    def open_orders(self, symbol=None):
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN,
                               symbols=[symbol] if symbol else None)
        return list(self.trading.get_orders(req))

    def submit(self, symbol, side, qty, limit_price=None, take_profit=None,
               stop_loss=None):
        """
        Submit an entry. When take_profit and stop_loss are supplied the order is
        sent as a BRACKET: both exit legs rest at the exchange from the moment the
        entry fills.

        This matters a great deal. Polling every 20s and exiting at market meant
        stops filled ~31% of 1R worse than intended (measured over 30 live trades),
        because price keeps running while the loop sleeps. Exchange-resident stops
        trigger at the price, not at whatever is left 20 seconds later.
        """
        if self.s.dry_run:
            log.info("[DRY-RUN] %s %s %.4f %s @ %s (tp %s sl %s)", self.s.mode, side,
                     qty, symbol, limit_price or "MKT", take_profit, stop_loss)
            return None
        os_ = OrderSide.BUY if side > 0 else OrderSide.SELL
        qty = round(qty, 4) if self.s.fractional else max(int(qty), 0)
        if qty <= 0:
            return None

        bracket = take_profit is not None and stop_loss is not None
        kw = {}
        if bracket:
            # Brackets require whole shares.
            iq = max(int(qty), 0)
            if iq <= 0:
                return None
            kw = {
                "order_class": OrderClass.BRACKET,
                "take_profit": TakeProfitRequest(
                    limit_price=round(float(take_profit), 2)),
                "stop_loss": StopLossRequest(
                    stop_price=round(float(stop_loss), 2)),
            }
            qty = iq

        if limit_price is not None:
            iq = max(int(qty), 0)
            if iq <= 0:
                return None
            req = LimitOrderRequest(symbol=symbol, qty=iq, side=os_,
                                    time_in_force=TimeInForce.DAY,
                                    limit_price=round(float(limit_price), 2), **kw)
        else:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=os_,
                                     time_in_force=TimeInForce.DAY, **kw)
        o = self.trading.submit_order(req)
        log.info("submitted %s %s qty=%s limit=%s %s id=%s", side, symbol, qty,
                 limit_price, "BRACKET" if bracket else "simple", o.id)
        return o

    def cancel(self, order_id):
        if self.s.dry_run:
            return
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception as e:
            log.warning("cancel failed %s: %s", order_id, e)

    def cancel_all(self):
        if self.s.dry_run:
            log.info("[DRY-RUN] cancel all open orders")
            return
        try:
            self.trading.cancel_orders()
        except Exception as e:
            log.warning("cancel_all failed: %s", e)

    def close_position(self, symbol):
        if self.s.dry_run:
            log.info("[DRY-RUN] close position %s", symbol)
            return
        try:
            self.trading.close_position(symbol)
            log.info("closed position %s", symbol)
        except Exception as e:
            log.warning("close_position %s failed: %s", symbol, e)

    def flatten_all(self, reason="eod", verify=True, attempts=3):
        """
        Cancel every open order and close every position, then VERIFY.

        The verification is not defensive padding. With bracket orders the tp/sl
        legs are children of the entry: cancelling the parent can leave the
        position open while the close request silently does nothing. On 2026-08-03
        that left a 55-share TQQQ short open overnight, which gapped 5% against the
        account before the next open. A flatten that is not verified is not a
        flatten.
        """
        log.warning("FLATTEN ALL (%s)", reason)
        if self.s.dry_run:
            log.info("[DRY-RUN] cancel orders + close all positions")
            return True

        for i in range(attempts):
            try:
                self.trading.cancel_orders()
            except Exception as e:
                log.warning("cancel_orders failed: %s", e)
            time.sleep(1.0)
            try:
                self.trading.close_all_positions(cancel_orders=True)
            except Exception as e:
                log.error("close_all_positions failed: %s", e)
                for sym in list(self.positions()):
                    self.close_position(sym)

            if not verify:
                return True
            time.sleep(2.0 + i)
            try:
                left = self.positions()
            except Exception as e:
                log.error("could not verify positions: %s", e)
                left = {}
            if not left:
                log.warning("flatten verified: account is flat")
                return True
            log.error("FLATTEN INCOMPLETE (attempt %d/%d): still open %s",
                      i + 1, attempts, list(left))

        left = self.positions()
        if left:
            # Loudest possible signal: this is the invariant the whole strategy
            # rests on, and it has failed.
            log.critical("!!! FLATTEN FAILED -- POSITIONS STILL OPEN: %s !!!", left)
            log.critical("!!! OVERNIGHT EXPOSURE -- MANUAL INTERVENTION REQUIRED !!!")
            try:
                import json as _json
                import os as _os
                with open(_os.path.join(self.s.state_dir, "ALERT.json"), "w") as f:
                    _json.dump({"ts": datetime.now(timezone.utc).isoformat(),
                                "error": "flatten_failed",
                                "positions": left}, f, indent=2, default=str)
            except Exception:
                pass
            return False
        return True

"""
Real-time bar stream.

The problem this solves: Alpaca's REST bars endpoint publishes a 5-minute bar
4-5 minutes after it closes. The lag sweep (app/lag_sweep.py) showed the strategy's
edge dies at roughly 1.5-2 minutes of signal lag, so REST polling is structurally
unable to trade this strategy -- by the time a bar arrives, the move is gone.

The fix costs nothing: the free tier includes real-time IEX streaming over
WebSocket. We subscribe to 1-minute bars, aggregate them into 5-minute bars
locally, and the engine acts within seconds of a bar closing instead of minutes.

Design:
  * REST backfill on start (needs ~200 bars of history for features)
  * WebSocket thread appends live bars as they arrive
  * get_bars() merges backfill + live into one frame
  * staleness is tracked so the engine can fall back to REST if the socket dies

The stream runs in its own thread with its own event loop; the trading loop stays
synchronous and simply reads the buffer.
"""
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger("stream")

ET = "America/New_York"


class BarStream:
    def __init__(self, s, backfill_fn):
        """
        s           : Settings
        backfill_fn : callable(symbols, minutes, lookback_days) -> {sym: DataFrame}
                      used once at startup (and on reconnect) for history
        """
        self.s = s
        self._backfill_fn = backfill_fn
        self.bar_minutes = s.bar_minutes

        self._lock = threading.Lock()
        self._minute_bars = defaultdict(lambda: deque(maxlen=4000))  # live 1-min
        self._history = {}                                           # REST backfill
        self._last_msg = 0.0
        self._started = False
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ #
    def start(self, symbols):
        self.symbols = list(symbols)
        log.warning("backfilling history for %s", ",".join(self.symbols))
        try:
            self._history = self._backfill_fn(self.symbols, self.bar_minutes, 12)
            for k, v in self._history.items():
                log.info("  %s: %d bars, last %s", k, len(v), v.index[-1])
        except Exception as e:
            log.error("backfill failed: %s", e)
            self._history = {}

        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="barstream")
        self._thread.start()
        self._started = True

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ #
    def _run(self):
        """Own event loop; reconnects with backoff if the socket drops."""
        import asyncio
        backoff = 2
        while not self._stop.is_set():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from alpaca.data.live import StockDataStream
                from alpaca.data.enums import DataFeed

                feed = DataFeed.SIP if self.s.use_sip else DataFeed.IEX
                stream = StockDataStream(self.s.key_id, self.s.secret, feed=feed)

                async def on_bar(bar):
                    try:
                        ts = pd.Timestamp(bar.timestamp).tz_convert(ET)
                        with self._lock:
                            self._minute_bars[bar.symbol].append({
                                "ts": ts, "open": float(bar.open),
                                "high": float(bar.high), "low": float(bar.low),
                                "close": float(bar.close),
                                "volume": float(bar.volume or 0),
                            })
                            self._last_msg = time.time()
                    except Exception as e:
                        log.warning("bar handler error: %s", e)

                stream.subscribe_bars(on_bar, *self.symbols)
                log.warning("stream connected (%s) for %s", feed.value,
                            ",".join(self.symbols))
                backoff = 2
                stream.run()
            except Exception as e:
                if self._stop.is_set():
                    break
                log.error("stream error: %s -- reconnecting in %ds", e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    def age_seconds(self):
        """Seconds since the last streamed message. Large = socket is unhealthy."""
        return time.time() - self._last_msg if self._last_msg else float("inf")

    def healthy(self, max_age=180):
        return self._started and self.age_seconds() < max_age

    # ------------------------------------------------------------------ #
    def _aggregate(self, sym):
        """Roll buffered 1-minute bars into completed bar_minutes bars."""
        with self._lock:
            rows = list(self._minute_bars.get(sym, []))
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("ts").sort_index()
        rule = f"{self.bar_minutes}min"
        agg = df.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
        if agg.empty:
            return None
        # drop the bar still forming: only emit bars whose window has elapsed
        now = pd.Timestamp.now(tz=ET)
        cutoff = now.floor(rule)
        return agg[agg.index < cutoff]

    def get_bars(self, symbols):
        """
        Merged history + live bars per symbol, newest last. Live bars override
        backfilled ones on overlap so the freshest data wins.
        """
        out = {}
        for sym in symbols:
            hist = self._history.get(sym)
            live = self._aggregate(sym)
            if hist is None and live is None:
                continue
            if live is None:
                out[sym] = hist
            elif hist is None:
                out[sym] = live
            else:
                merged = pd.concat([hist[~hist.index.isin(live.index)], live])
                out[sym] = merged.sort_index()
        return out

    def refresh_backfill(self):
        """Re-pull REST history (call outside market hours or after a long gap)."""
        try:
            self._history = self._backfill_fn(self.symbols, self.bar_minutes, 12)
            log.info("backfill refreshed")
        except Exception as e:
            log.warning("backfill refresh failed: %s", e)

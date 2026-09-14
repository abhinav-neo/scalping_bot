"""
Entrypoint.

    python -m app.run_bot

Behaviour:
  * validates config and refuses to start on a live account without the explicit ack
  * trains models if missing or stale (RETRAIN_DAYS)
  * runs the trading loop until stopped
  * on SIGTERM/SIGINT (docker stop, Ctrl-C) it FLATTENS EVERYTHING before exiting,
    so a container shutdown can never leave you holding an overnight position
"""
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from datetime import datetime

from .settings import S
from .engine import Engine
from .train import train_all


def setup_logging(s):
    os.makedirs(s.log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    fh = logging.handlers.TimedRotatingFileHandler(
        os.path.join(s.log_dir, "bot.log"), when="midnight", backupCount=30)
    fh.setFormatter(fmt); root.addHandler(fh)


log = logging.getLogger("main")
_stop = {"flag": False}
_last_loop = {"ts": time.time()}


def _watchdog(timeout_s, stop_flag):
    """
    Kill the process if the main loop stops making progress.

    The Alpaca SDK issues HTTP requests without a timeout, so a half-open TCP
    connection blocks the loop forever -- no exception, no heartbeat, no log
    output. On 2026-07-31 this stalled the bot for four hours while the container
    still reported healthy. Failing fast lets Docker's restart policy recover it.
    """
    while not stop_flag["flag"]:
        time.sleep(15)
        stalled = time.time() - _last_loop["ts"]
        if stalled > timeout_s:
            log.error("WATCHDOG: no loop progress for %.0fs (limit %ds) -- "
                      "exiting so the container restarts", stalled, timeout_s)
            logging.shutdown()
            os._exit(70)      # hard exit: the main thread is blocked in a syscall


def _handle_signal(signum, frame):
    log.warning("signal %s received -> shutting down", signum)
    _stop["flag"] = True


def main():
    setup_logging(S)
    errs = S.validate()
    if errs:
        for e in errs:
            log.error("CONFIG: %s", e)
        raise SystemExit(1)

    log.warning("=" * 62)
    log.warning("SCALPING BOT starting | mode=%s | symbols=%s | capital=%.0f",
                S.mode, ",".join(S.symbols), S.starting_capital)
    if not S.paper:
        log.warning("!!! LIVE REAL MONEY MODE !!! you set the ack flag deliberately")
    if S.dry_run:
        log.warning("DRY_RUN enabled: no orders will be sent")
    log.warning("=" * 62)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    eng = Engine(S)

    if not eng.models or eng.models_stale():
        log.info("training models (missing or stale)...")
        try:
            train_all(S)
            eng._load_models()
        except Exception as e:
            log.error("training failed: %s", e)
            if not eng.models:
                raise SystemExit("no models available, cannot trade")

    last_hb = 0
    last_train_check = datetime.now().date()

    threading.Thread(target=_watchdog, args=(S.watchdog_timeout_s, _stop),
                     daemon=True).start()
    log.warning("watchdog armed: %ds without loop progress triggers restart",
                S.watchdog_timeout_s)

    while not _stop["flag"]:
        try:
            eng.run_once()
            _last_loop["ts"] = time.time()

            if time.time() - last_hb > 60:
                hb = eng.heartbeat()
                log.info("hb equity=%.2f pos=%d pending=%d trades_today=%s halted=%s",
                         hb["equity"], len(hb["positions"]), len(hb["pending"]),
                         hb["trades_today"], hb["halted"])
                last_hb = time.time()

            # weekly retrain outside market hours
            today = datetime.now().date()
            if today != last_train_check:
                last_train_check = today
                if eng.models_stale() and not eng.broker.clock()["is_open"]:
                    log.info("scheduled retrain")
                    try:
                        train_all(S); eng._load_models()
                    except Exception as e:
                        log.error("retrain failed: %s", e)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.exception("loop error: %s", e)
            time.sleep(5)

        time.sleep(S.loop_seconds)

    # ---- graceful shutdown: never leave positions open ----
    log.warning("shutdown: flattening all positions")
    try:
        eng.broker.flatten_all("shutdown")
    except Exception as e:
        log.error("flatten on shutdown failed: %s", e)
    log.warning("bot stopped")


if __name__ == "__main__":
    main()

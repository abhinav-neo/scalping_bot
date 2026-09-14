"""Supervisor -- runs the swing engine IN-PROCESS.

Why in-process: the previous version spawned the engine as a subprocess, and that
spawn failed intermittently with a mangled interpreter path
(No Python at '"C:\\Users\\...uv\\python\\...). Direct invocation always worked;
only the spawn failed, and only sometimes. Three attempts to fix the venv did not
stop it, and each failure silently cost a trading session -- once 18 days.

Importing and calling the engine removes the subprocess entirely, so the failure
mode cannot recur. A crash inside the engine is caught here and retried with
backoff instead of killing the process.
"""
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from datetime import datetime, timezone


def _load_dotenv_early():
    """Load .env BEFORE importing settings -- settings evaluates its defaults at
    import time, so a late load leaves state_dir pointing at the container path."""
    path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(path):
        return 0
    n = 0
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and not os.environ.get(k):
            os.environ[k] = v
            n += 1
    return n


_DOTENV = _load_dotenv_early()

from .settings import S            # noqa: E402
from .swing_engine import SwingEngine, fetch_daily, UNIVERSE   # noqa: E402

log = logging.getLogger("supervisor")

HEARTBEAT = os.path.join(S.state_dir, "swing_heartbeat.json")
LOCK = os.path.join(S.state_dir, "swing_supervisor.lock")
LOOP_SECONDS = 300
MAX_BACKOFF = 300


def _setup_logging():
    os.makedirs(S.log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.handlers.TimedRotatingFileHandler(
        os.path.join(S.log_dir, "swing.log"), when="midnight", backupCount=30)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _acquire_lock():
    if os.path.exists(LOCK):
        try:
            age = time.time() - os.path.getmtime(LOCK)
        except Exception:
            age = 9999
        if age < 900:          # a live supervisor refreshes this every loop
            return False
    os.makedirs(S.state_dir, exist_ok=True)
    open(LOCK, "w").write(str(os.getpid()))
    return True


def _beat(status, **extra):
    """Written EVERY loop, not just on cycle events, so a stale timestamp always
    means the process is actually dead."""
    try:
        d = {"ts": datetime.now(timezone.utc).isoformat(), "status": status,
             "pid": os.getpid()}
        d.update(extra)
        json.dump(d, open(HEARTBEAT, "w"), indent=2, default=str)
        open(LOCK, "w").write(str(os.getpid()))
    except Exception:
        pass


def main():
    _setup_logging()
    log.warning("=" * 60)
    log.warning("SWING SUPERVISOR (in-process) | mode=%s | %d vars from .env",
                S.mode, _DOTENV)
    log.warning("=" * 60)
    if not _acquire_lock():
        log.error("another supervisor holds the lock; exiting")
        return 1

    eng = None
    errors = 0
    backoff = 5

    while True:
        try:
            if eng is None:
                log.warning("initialising engine")
                eng = SwingEngine(S)
                if not os.listdir(eng.models_dir):
                    log.warning("no models -- training")
                    hist = fetch_daily(UNIVERSE, years=5.0)
                    eng.train_all(hist, hist["SPY"]["close"])
                log.warning("engine ready")

            eng.run_cycle()
            _beat("running", errors=errors,
                  positions=len(eng.state.get("positions", {})))
            errors = 0
            backoff = 5

        except KeyboardInterrupt:
            log.warning("interrupted; exiting")
            _beat("stopped")
            return 0
        except Exception as e:
            errors += 1
            log.error("cycle error (%d): %s", errors, e)
            log.error(traceback.format_exc()[-800:])
            _beat("error", errors=errors, last_error=str(e)[:200])
            if errors >= 3:
                log.warning("rebuilding engine after repeated errors")
                eng = None
                errors = 0
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    sys.exit(main())

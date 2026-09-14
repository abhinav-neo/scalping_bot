"""
GitHub Actions runner -- one decision cycle, then exit.

WHY THIS EXISTS

The local deployment died silently seven times: Docker engine failures, a
supervisor killed with its parent shell, uv relinking its managed Python out from
under the venv, and more. Each failure cost trading sessions, one of them 18 days.

The strategy only needs to act ONCE per day, which makes a scheduled CI job a much
better fit than a long-lived local process. A fresh container each run means there
is no process to die between runs, nothing to keep alive, and no local dependency.

STATE

The broker is the source of truth for positions. The only thing we must carry
between runs is the entry metadata the decay-exit rule needs (when a position was
opened, and the meta_p it was opened at). That is a few hundred bytes, committed
back to the repo after each run.

TIMING CAVEAT (important)

GitHub's cron is best-effort and can be delayed by 5-30+ minutes under load. The
workflow therefore schedules EARLY and this runner refuses to trade outside a safe
window, rather than firing at an unintended time. A missed run is preferable to a
run at the wrong price.

    python -m app.gh_runner
"""
import json
import logging
import os
import sys
from datetime import datetime

import pytz

ET = pytz.timezone("America/New_York")

# Trade only inside this window. The backtest enters at the close, so acting far
# from it would not correspond to the validated results.
WINDOW_START = (15, 15)
WINDOW_END = (15, 58)

log = logging.getLogger("gh_runner")


def in_window(now=None):
    now = now or datetime.now(ET)
    mins = now.hour * 60 + now.minute
    lo = WINDOW_START[0] * 60 + WINDOW_START[1]
    hi = WINDOW_END[0] * 60 + WINDOW_END[1]
    return lo <= mins <= hi


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from .settings import S
    from .swing_engine import SwingEngine, fetch_daily, UNIVERSE

    errs = S.validate()
    if errs:
        for e in errs:
            log.error("CONFIG: %s", e)
        return 1

    now = datetime.now(ET)
    log.warning("=" * 60)
    log.warning("GH RUNNER | %s ET | mode=%s", now.strftime("%Y-%m-%d %H:%M"), S.mode)
    log.warning("=" * 60)

    force = os.getenv("FORCE_RUN", "").lower() in ("1", "true", "yes")
    if not in_window(now) and not force:
        log.warning("outside the %02d:%02d-%02d:%02d ET window -- not trading. "
                    "A missed run beats a run at the wrong price.",
                    *WINDOW_START, *WINDOW_END)
        return 0

    eng = SwingEngine(S)

    if not os.path.isdir(eng.models_dir) or not os.listdir(eng.models_dir):
        log.warning("no models present -- training (this takes a few minutes)")
        hist = fetch_daily(UNIVERSE, years=5.0)
        eng.train_all(hist, hist["SPY"]["close"])

    eng.run_cycle(force=True)

    # summary for the workflow log
    try:
        from .broker import Broker
        b = Broker(S)
        a = b.account()
        pos = b.positions()
        log.warning("RESULT equity=%.2f positions=%d", a["equity"], len(pos))
        for sym, p in pos.items():
            log.warning("  %-6s %s qty=%.0f uPL=%+.2f", sym,
                        "LONG" if p["side"] > 0 else "SHORT",
                        abs(p["qty"]), p["unrealized_pl"])
        # append a compact history row the dashboard can read
        os.makedirs(S.state_dir, exist_ok=True)
        with open(os.path.join(S.state_dir, "monitor.jsonl"), "a") as f:
            f.write(json.dumps({
                "day": now.date().isoformat(),
                "ts": datetime.utcnow().isoformat() + "+00:00",
                "equity": a["equity"], "cash": a["cash"],
                "open_positions": len(pos),
                "gross_exposure": sum(abs(p["market_value"]) for p in pos.values()),
                "unrealized": sum(p["unrealized_pl"] for p in pos.values()),
            }) + "\n")
    except Exception as e:
        log.warning("summary failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())

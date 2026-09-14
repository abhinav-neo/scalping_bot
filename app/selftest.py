"""Pre-flight check: credentials, connectivity, data access, models, and clock."""
import os, sys, logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from .settings import S

def main():
    print("=" * 58); print("PRE-FLIGHT CHECK"); print("=" * 58)
    ok = True
    errs = S.validate()
    if errs:
        for e in errs: print("  FAIL:", e)
        return 1
    print(f"  OK   config valid | mode = {S.mode}")
    if not S.paper:
        print("  !!!! LIVE REAL MONEY MODE ENABLED !!!!")

    from .broker import Broker
    try:
        b = Broker(S)
        a = b.account()
        print(f"  OK   account reachable | equity ${a['equity']:,.2f} "
              f"buying power ${a['buying_power']:,.2f}")
        if a["blocked"]: print("  WARN account is blocked by broker"); ok = False
    except Exception as e:
        print("  FAIL broker connection:", e); return 1

    c = b.clock()
    print(f"  OK   market {'OPEN' if c['is_open'] else 'CLOSED'} | next open {c['next_open']}")

    try:
        need = S.symbols + [S.market_symbol]
        bars = b.bars(need, S.bar_minutes, lookback_days=5)
        for s in need:
            n = len(bars.get(s, []))
            print(f"  {'OK  ' if n else 'FAIL'} bars {s}: {n}")
            if not n: ok = False
    except Exception as e:
        print("  FAIL data access:", e); ok = False

    for s in S.symbols:
        p = os.path.join(S.model_dir, f"{s}.joblib")
        print(f"  {'OK  ' if os.path.exists(p) else 'MISS'} model {s}")
        if not os.path.exists(p): ok = False

    print("=" * 58)
    print("READY" if ok else "NOT READY -- resolve the items above")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

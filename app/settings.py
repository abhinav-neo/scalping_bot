"""
Runtime settings. Everything comes from environment variables (.env), so no
credential ever lives in the repo or in source control.

SAFETY INTERLOCK: the bot runs against the Alpaca PAPER endpoint unless BOTH
  ALPACA_PAPER=false  AND  I_UNDERSTAND_THIS_IS_REAL_MONEY=yes
are set. One flag alone will not do it, and the bot logs loudly either way.
"""
import os
from dataclasses import dataclass, field
from typing import List


def _b(key, default="true"):
    return os.getenv(key, default).strip().lower() in ("1", "true", "yes", "y")


def _f(key, default):
    return float(os.getenv(key, default))


def _i(key, default):
    return int(os.getenv(key, default))


@dataclass
class Settings:
    # ---- credentials (from .env, never hard-coded) ----
    key_id: str = os.getenv("ALPACA_KEY_ID", "")
    secret: str = os.getenv("ALPACA_SECRET", "")

    # ---- paper/live interlock ----
    paper: bool = _b("ALPACA_PAPER", "true")
    live_ack: str = os.getenv("I_UNDERSTAND_THIS_IS_REAL_MONEY", "")

    # ---- capital & universe ----
    starting_capital: float = _f("STARTING_CAPITAL", "5000")
    symbols: List[str] = field(default_factory=lambda: [
        s.strip().upper() for s in os.getenv("SYMBOLS", "AMD,COIN,NVDA,TQQQ").split(",")
        if s.strip()])
    market_symbol: str = os.getenv("MARKET_SYMBOL", "SPY")

    # ---- strategy ----
    bar_minutes: int = _i("BAR_MINUTES", 5)
    horizon_bars: int = _i("HORIZON_BARS", 6)
    max_hold_minutes: int = _i("MAX_HOLD_MINUTES", 35)   # hard wall-clock cap
    pt_mult: float = _f("PT_MULT", 1.0)
    sl_mult: float = _f("SL_MULT", 1.0)
    vol_span: int = _i("VOL_SPAN", 50)
    fracdiff_d: float = _f("FRACDIFF_D", 0.4)
    hmm_states: int = _i("HMM_STATES", 4)
    meta_threshold: float = _f("META_THRESHOLD", 0.55)

    # ---- data quality guards (added after live churn incident) ----
    # Refuse to trade on stale bars. Free IEX data lags ~15 min, which is fatal for
    # a 5-minute strategy -- these guards make that failure loud instead of silent.
    max_bar_age_minutes: int = _i("MAX_BAR_AGE_MINUTES", 10)
    max_spread_bps: float = _f("MAX_SPREAD_BPS", "25")
    # Signal bar close must still agree with the live trade print within this much.
    # Drift limit as a FRACTION of the profit target, so it rescales automatically
    # when horizon or pt_mult change. A fixed bps limit silently became wrong when
    # barriers widened from ~30bps to ~180bps.
    max_price_drift_frac: float = _f("MAX_PRICE_DRIFT_FRAC", "0.40")
    min_price_drift_bps: float = _f("MIN_PRICE_DRIFT_BPS", "25")
    max_price_drift_bps: float = _f("MAX_PRICE_DRIFT_BPS", "40")   # legacy, unused
    cooldown_seconds: int = _i("COOLDOWN_SECONDS", 300)
    # ---- market-regime filter (added after the 93%-long down-session loss) ----
    market_filter: bool = _b("MARKET_FILTER", "true")
    market_trend_bars: int = _i("MARKET_TREND_BARS", 12)        # 12 x 5min = 1hr
    market_trend_bps_limit: float = _f("MARKET_TREND_BPS_LIMIT", "15")
    # SIP requires the paid Algo Trader Plus plan. IEX is real-time and free.
    use_sip: bool = _b("USE_SIP", "false")
    # Performance epoch: ignore trades before this date when reporting stats.
    # Lets you re-baseline after a strategy change without deleting the broker
    # account (Alpaca no longer supports resetting paper balances). Format: YYYY-MM-DD.
    perf_epoch: str = os.getenv("PERF_EPOCH", "")

    # ---- execution ----
    use_limit_entry: bool = _b("USE_LIMIT_ENTRY", "true")
    limit_offset_frac: float = _f("LIMIT_OFFSET_FRAC", 0.15)
    limit_ttl_seconds: int = _i("LIMIT_TTL_SECONDS", 300)   # cancel unfilled after 1 bar
    fractional: bool = _b("FRACTIONAL_SHARES", "true")

    # ---- risk ----
    risk_per_trade: float = _f("RISK_PER_TRADE", 0.005)
    base_leverage: float = _f("BASE_LEVERAGE", 2.0)
    target_daily_vol: float = _f("TARGET_DAILY_VOL", 0.02)
    max_notional_frac: float = _f("MAX_NOTIONAL_FRAC", 1.0)
    max_concurrent_positions: int = _i("MAX_CONCURRENT_POSITIONS", 2)
    daily_loss_kill: float = _f("DAILY_LOSS_KILL", 0.03)
    max_total_drawdown: float = _f("MAX_TOTAL_DRAWDOWN", 0.15)  # halt bot entirely
    max_trades_per_day: int = _i("MAX_TRADES_PER_DAY", 120)

    # ---- session (US/Eastern) ----
    entry_start: str = os.getenv("ENTRY_START", "09:35")
    entry_cutoff: str = os.getenv("ENTRY_CUTOFF", "15:40")   # no new entries after
    flatten_at: str = os.getenv("FLATTEN_AT", "15:55")       # hard flatten

    # ---- ops ----
    dry_run: bool = _b("DRY_RUN", "false")   # true = log orders, send nothing
    loop_seconds: int = _i("LOOP_SECONDS", 20)
    model_dir: str = os.getenv("MODEL_DIR", "/app/models")
    state_dir: str = os.getenv("STATE_DIR", "/app/state")
    log_dir: str = os.getenv("LOG_DIR", "/app/logs")
    retrain_days: int = _i("RETRAIN_DAYS", 7)
    train_years: float = _f("TRAIN_YEARS", 2.0)
    # Hard restart if the main loop makes no progress for this long. Guards against
    # hung HTTP calls (the SDK sets no request timeout) silently stalling the bot.
    watchdog_timeout_s: int = _i("WATCHDOG_TIMEOUT_S", 180)
    # Stream real-time bars over WebSocket instead of polling REST. REST bars land
    # 4-5 min after close; the lag sweep showed the edge dies past ~1.5-2 min.
    use_stream: bool = _b("USE_STREAM", "true")
    max_stream_age_s: int = _i("MAX_STREAM_AGE_S", 180)
    # Bracket orders put tp/sl at the exchange instead of polling every 20s.
    # Measured live: polled market exits filled stops ~31% of 1R worse than the
    # intended price, which alone inverted a 1.5:1 payoff into 0.92:1.
    use_brackets: bool = _b("USE_BRACKETS", "true")

    def validate(self):
        errs = []
        if not self.key_id or not self.secret:
            errs.append("ALPACA_KEY_ID / ALPACA_SECRET missing (set them in .env)")
        if not self.paper and self.live_ack.lower() != "yes":
            errs.append(
                "Refusing to run against LIVE money: set ALPACA_PAPER=true, or if you "
                "really mean live, also set I_UNDERSTAND_THIS_IS_REAL_MONEY=yes")
        if not self.symbols:
            errs.append("SYMBOLS is empty")
        return errs

    @property
    def mode(self):
        return "PAPER" if self.paper else "LIVE-REAL-MONEY"


S = Settings()

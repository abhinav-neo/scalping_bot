# Scalping Bot — Dockerized, Alpaca Paper

Production intraday scalping bot. Same-day flat by construction, ~15 trades/day per
symbol, HMM regime detection + LightGBM + meta-label gating, full risk stack.

**Runs against Alpaca PAPER by default.** Going live requires two separate
deliberate flags — see Safety below.

---

## 1. Setup (5 minutes)

```bash
unzip scalping_bot.zip && cd scalping_bot
cp .env.example .env
```

Now open `.env` and paste in your **paper** credentials from
<https://app.alpaca.markets/paper/dashboard/overview>:

```
ALPACA_KEY_ID=PK...your_paper_key
ALPACA_SECRET=...your_paper_secret
ALPACA_PAPER=true
```

> Your keys stay in `.env` on your machine. `.gitignore` already excludes it — do
> not commit it, and don't paste keys into chat windows (including mine).

## 2. Train the models

```bash
docker compose build
docker compose run --rm bot python -m app.train
```

Pulls 2 years of 5-min bars from Alpaca and trains per-symbol artifacts into
`./models/`. Takes a few minutes. Check `models/summary.json` for holdout hit rates.

## 3. Run

```bash
docker compose up -d
docker compose logs -f bot
```

Dashboard: **<http://localhost:8080>** — equity, day P/L, open positions, recent
trades, kill-switch state. Auto-refreshes every 10s.

Stop safely (flattens all positions first):

```bash
docker compose stop        # 60s grace period for the flatten to complete
```

---

## Safety architecture

| Guard | Behaviour |
|---|---|
| **Paper interlock** | Live trading requires `ALPACA_PAPER=false` **AND** `I_UNDERSTAND_THIS_IS_REAL_MONEY=yes`. One flag alone is refused at startup. |
| **EOD flatten** | Hard flatten at 15:55 ET. No position survives the session. |
| **Entry cutoff** | No new entries after 15:40 ET. |
| **Max hold** | Wall-clock cap (35 min default), independent of bar count — fixes the data-gap case where 6 bars spanned 90 minutes in backtesting. |
| **Daily loss kill** | Halts trading for the day at −3%. **Latches** — an equity recovery does not re-enable it. |
| **Max drawdown halt** | Halts the bot entirely at −15% from peak. Persists across container restarts. |
| **Portfolio gross cap** | Total exposure capped at `BASE_LEVERAGE × equity`, so concurrent positions can't stack leverage. |
| **Shutdown flatten** | SIGTERM/SIGINT (docker stop, Ctrl-C) flattens everything before exit. |
| **Untracked positions** | Any position the engine doesn't recognize gets closed rather than managed blind. |
| **DRY_RUN** | `DRY_RUN=true` logs every intended order without sending it. |

Two bugs were found and fixed during testing, both worth knowing about:
1. The daily kill switch previously **unlatched** if equity recovered — it now stays
   locked for the rest of the session.
2. Per-position sizing allowed N concurrent positions to stack to N× leverage
   ($20k gross on a $5k account). Now capped at the portfolio level.

## Key settings (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `SYMBOLS` | `AMD,COIN,NVDA,TQQQ` | AMD/COIN had the best backtest DSR |
| `META_THRESHOLD` | `0.55` | Raise to trade less and more selectively |
| `RISK_PER_TRADE` | `0.005` | 0.5% of equity risked at the stop |
| `MAX_CONCURRENT_POSITIONS` | `2` | |
| `BASE_LEVERAGE` | `2.0` | Portfolio gross cap. Set `1.0` for no margin. |
| `DAILY_LOSS_KILL` | `0.03` | |
| `MAX_TOTAL_DRAWDOWN` | `0.15` | Full halt |
| `USE_LIMIT_ENTRY` | `true` | Passive entries — roughly halves fee drag |
| `RETRAIN_DAYS` | `7` | Weekly retrain, outside market hours |

## What to watch in the first two weeks

The paper run exists to answer questions the backtest **couldn't**:

1. **Fill rate.** Backtest assumed 78% on passive limits with no queue modeling.
   If real fills come in at 50–60%, the economics change materially. Compare
   `entry` events against actual positions in `state/trades.jsonl`.
2. **Realized spread cost.** The edge died above ~4–6 bps round trip in testing.
   Log `spread_bps` from quotes and check it against that budget.
3. **Trades/day and hold time.** Should land near ~15/day/symbol, median ~10 min.
   Large deviations mean the live feed differs from the backtest bars.
4. **Whether P/L tracks any of the modeled scenarios.** Recall the honest range:
   median outcomes spanned +94%/yr (mild decay) to −38%/yr (edge is noise), with
   the deflated-Sharpe-implied case sitting near breakeven.

Run it **at least 4 weeks** before drawing conclusions. Two good weeks is noise at
~15 trades/day; you need a few hundred trades before the numbers mean anything.

## Layout

```
app/
  settings.py   env config + paper/live interlock
  broker.py     Alpaca adapter (data, orders, positions, flatten)
  features.py   fracdiff, SPY lead-lag, vol structure, seasonality
  regime.py     Gaussian HMM -> posterior state probabilities
  labeling.py   triple-barrier + meta-labels
  train.py      pulls history, trains, persists per-symbol artifacts
  risk.py       kill switches, session windows, sizing
  engine.py     the trading loop
  run_bot.py    entrypoint, scheduler, graceful shutdown
  dashboard.py  read-only monitor on :8080
models/  state/  logs/     (persisted volumes)
```

## Troubleshooting

**"no models available"** → run `docker compose run --rm bot python -m app.train`.

**No trades appearing** → check ET time is inside 09:35–15:40, market is open, and
`meta_p` in the logs is clearing `META_THRESHOLD`. The bot logs a reason each loop.

**Halted and won't restart** → by design. Inspect `state/risk_state.json`, and if
you're satisfied it was a false alarm, set `"halted": false` and restart.

---

Paper first. This is not investment advice, and the strategy's edge remains
statistically unproven — that's exactly what this paper deployment is for.

---

## Changelog — 2026-07-29 / 07-30

### Live incidents and what they taught us

**Churn loop (07-29).** First live run fired 6 trades in 90 seconds, every exit at
"0min". Three causes, all fixed: barriers were computed from the signal bar's close
rather than the actual fill price; there was no per-bar deduplication so the same
signal re-fired every 20s loop; and models pickled `_Cfg` from `__main__`, which
failed to load under a different entrypoint.

**Stale data (07-30).** The Alpaca SDK silently serves *delayed SIP* when no feed is
specified — bars were 15–20 minutes old. Passing `feed=DataFeed.IEX` explicitly gives
real-time bars on the **free** plan. Algo Trader Plus ($99/mo) is NOT required for
this; it buys full NBBO coverage and higher rate limits, not basic freshness.
Exits now use trade prints rather than IEX quote mids, since IEX quotes show
~600-700 bps spreads (single venue, ~2% of volume) that are not real execution costs.

**Directional loss (07-29).** Lost 3.3% going long on 14 of 15 entries while the
universe fell 2–5%. Diagnosed as a structural long bias and "fixed" with class
balancing — but a later 2-year A/B backtest showed the base model trades **48.3%
long** across 8,000 trades. The bias did not exist; it was a single-day artifact of
15 trades. Class balancing was reverted after measuring no effect (750% vs 736%
return, Sharpe 5.89 vs 5.91).

### Audit fixes (07-30)

- **Phantom trades** — `submit()` returns `None` when nothing is sent (dry-run, or
  qty rounding to zero on a high-priced name). The bot recorded the position, counted
  it against the daily cap, and logged an entry anyway. Now detected and skipped.
- **Orphaned state** — expired limit orders left `open_meta` populated, writing
  phantom positions to disk that survived restarts.
- **No startup reconciliation** — `pending` was memory-only, so a restart orphaned
  resting orders (never aged out, lingered until EOD flatten). Added
  `_reconcile_on_start()`.
- **Quantity mismatch** — limit orders require whole shares, but the bot stored the
  unrounded fractional quantity.

### Config mismatches found after the horizon-3 change

Changing strategy parameters without re-checking the operational settings that
interact with them left three gaps:

- `MAX_HOLD_MINUTES` was 20 while `HORIZON_BARS=3` means 15 min — exits ran 33% long.
  Fixed to 16.
- `COOLDOWN_SECONDS=300` throttled frequency; the backtest has no cooldown. Reduced
  to 60. (Churn protection comes from per-bar dedup, not the cooldown.)
- `MAX_CONCURRENT_POSITIONS=2` vs the backtest's effective 4 — **still open**. The
  backtest simulates each symbol in isolation and models no correlation between them.
  These four names are all high-beta tech; four simultaneous same-direction positions
  is more concentrated than the backtest represents.

### Validation results

| Config | Trades | Sharpe | Return | Long % |
|---|---|---|---|---|
| base | 8,052 | 5.89 | 750% | 48.3% |
| class-balanced | 8,005 | 5.91 | 736% | 46.9% |
| + market filter | 6,413 | 6.28 | 614% | 47.3% |
| horizon 3, pt 0.7 (**live**) | ~10,500 | ~5.3 | ~694% | — |

**Caveat that matters:** two runs of the *same* config produced Sharpe 6.28 and 4.60,
because the runs fetched data ~45 minutes apart and walk-forward fold boundaries
shifted. **Sharpe differences under ~20% are run-to-run noise.** Trade-count
differences are structural and reliable. The market filter helped in one run and hurt
in the other — treat its benefit as unproven.

Backtest returns also assume fills at the bar close and 1.5 bps/side. Live fills on
passive limits have been materially worse. Relative comparisons between configs are
trustworthy; absolute return figures are not.

### Tools

    python -m app.selftest       # pre-flight: creds, data, models, clock
    python -m app.train          # retrain from Alpaca history
    python -m app.backtest_ab    # A/B a config change before deploying it
    python -m app.sweep          # parameter frontier: trades vs Sharpe

**Run `backtest_ab` before deploying any strategy change.** The class-balancing
episode is the cautionary tale: a fix was designed, deployed and defended on the
basis of 15 trades in one session, and the backtest later showed it addressed a
problem that never existed.

"""
Swing dashboard -- live view of the month-long paper test.

Serves a single self-contained page at http://localhost:8090 showing:
  * equity curve and cumulative return
  * daily return bars (green/red)
  * trade-by-trade P&L
  * win/loss breakdown and rolling win rate
  * open positions with unrealised P&L and days held
  * live-vs-backtest comparison table
  * execution quality (stop slippage vs the 0.10R the backtest assumes)

Read-only: it queries the broker and reads state files, and holds no ability to
place or cancel orders.

    python -m app.swing_dashboard
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from .settings import S
from .broker import Broker
from .analytics import Analytics

log = logging.getLogger("swing_dash")
PORT = int(os.getenv("SWING_DASH_PORT", "8090"))

EXPECT = {"cagr": 143.8, "daily": 0.36, "sharpe": 2.97, "maxdd": -37.3,
          "win": 51.7, "tpd": 2.6, "hold_d": 2.2, "slip_R": 0.10}

_B = None
_A = None


def broker():
    global _B
    if _B is None:
        _B = Broker(S)
    return _B


def analytics():
    global _A
    if _A is None:
        _A = Analytics(S)
    return _A


def _read_jsonl(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def payload():
    b, a = broker(), analytics()
    acct = b.account()
    pos = b.positions()
    trips = a.round_trips(days=90)
    ex = a.execution_quality(S.state_dir, trips)

    snaps = _read_jsonl(os.path.join(S.state_dir, "monitor.jsonl"))
    eq_series = []
    if snaps:
        d = pd.DataFrame(snaps).drop_duplicates("day", keep="last")
        d = d.sort_values("day")
        eq_series = [{"day": r["day"], "equity": float(r["equity"])}
                     for _, r in d.iterrows()]
    # ensure today's point is present
    today = datetime.now(timezone.utc).date().isoformat()
    if not eq_series or eq_series[-1]["day"] != today:
        eq_series.append({"day": today, "equity": acct["equity"]})

    start = eq_series[0]["equity"] if eq_series else acct["equity"]
    cur = acct["equity"]
    eqv = [p["equity"] for p in eq_series]
    daily_ret = []
    for i in range(1, len(eqv)):
        daily_ret.append({"day": eq_series[i]["day"],
                          "ret": (eqv[i] / eqv[i - 1] - 1) * 100})

    peak, dds = -1e18, []
    for v in eqv:
        peak = max(peak, v)
        dds.append((v - peak) / peak * 100)

    r = pd.Series([x["ret"] for x in daily_ret]) / 100 if daily_ret else pd.Series(dtype=float)
    sharpe = (float(r.mean() / (r.std() + 1e-12) * np.sqrt(252))
              if len(r) > 2 and r.std() > 0 else None)
    n_days = max(len(eqv) - 1, 1)
    total = (cur / start - 1) * 100 if start else 0.0
    daily_avg = ((1 + total / 100) ** (1 / n_days) - 1) * 100 if n_days else 0.0
    cagr = ((1 + daily_avg / 100) ** 252 - 1) * 100

    tr = []
    for t in trips:
        tr.append({"symbol": t["symbol"],
                   "side": "LONG" if t["side"] > 0 else "SHORT",
                   "entry": round(t["entry"], 2), "exit": round(t["exit"], 2),
                   "qty": round(t["qty"], 2),
                   "pnl": round(t["pnl"], 2),
                   "pnl_pct": round(t["pnl_pct"], 2),
                   "held_min": t["held_min"],
                   "exit_ts": (t["exit_ts"].isoformat()
                               if hasattr(t["exit_ts"], "isoformat")
                               else str(t["exit_ts"]))})
    wins = [x for x in tr if x["pnl"] > 0]
    losses = [x for x in tr if x["pnl"] <= 0]

    open_meta = {}
    try:
        open_meta = json.load(open(os.path.join(S.state_dir,
                                                "swing_state.json"))).get("positions", {})
    except Exception:
        pass
    positions = []
    for sym, p in pos.items():
        m = open_meta.get(sym, {})
        held = None
        if m.get("opened"):
            try:
                od = datetime.fromisoformat(m["opened"]).date()
                held = int(np.busday_count(od, datetime.now().date()))
            except Exception:
                pass
        positions.append({"symbol": sym,
                          "side": "LONG" if p["side"] > 0 else "SHORT",
                          "qty": round(abs(p["qty"]), 2),
                          "entry": round(p["avg_entry"], 2),
                          "current": round(p["current_price"], 2),
                          "upl": round(p["unrealized_pl"], 2),
                          "upl_pct": round(p["unrealized_pct"], 2),
                          "days_held": held})

    hb = {}
    try:
        hb = json.load(open(os.path.join(S.state_dir, "swing_heartbeat.json")))
    except Exception:
        pass
    hb_age = None
    if hb.get("ts"):
        try:
            hb_age = (datetime.now(timezone.utc)
                      - datetime.fromisoformat(hb["ts"])).total_seconds() / 60
        except Exception:
            pass

    return {
        "equity": cur, "start": start, "cash": acct["cash"],
        "total_pct": total, "daily_avg_pct": daily_avg, "cagr_pct": cagr,
        "sharpe": sharpe, "sessions": n_days,
        "maxdd_pct": min(dds) if dds else 0.0,
        "curve": eq_series, "daily": daily_ret, "dd": dds,
        "trades": tr[::-1], "n_trades": len(tr),
        "n_wins": len(wins), "n_losses": len(losses),
        "win_pct": (100 * len(wins) / len(tr)) if tr else 0.0,
        "avg_win": (sum(x["pnl"] for x in wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(x["pnl"] for x in losses) / len(losses)) if losses else 0.0,
        "gross_win": sum(x["pnl"] for x in wins),
        "gross_loss": sum(x["pnl"] for x in losses),
        "positions": positions,
        "exec": ex,
        "expect": EXPECT,
        "heartbeat_min": hb_age,
        "hb_status": hb.get("status"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swing Bot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b0d13;--card:#141822;--line:#232838;--fg:#e8eaf0;--dim:#8b93a7;
      --pos:#34d399;--neg:#f87171;--warn:#fbbf24}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);padding:20px;
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
h1{font-size:20px;margin:0 0 2px;font-weight:650}
.sub{color:var(--dim);font-size:12px;margin-bottom:16px}
.badge{padding:3px 9px;border-radius:5px;font-size:11px;font-weight:700}
.ok{background:#0d3d2e;color:#34d399}.bad{background:#7f1d1d;color:#fca5a5}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:11px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:13px 15px}
.k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;font-weight:600}
.v{font-size:21px;font-weight:660;margin-top:5px}
.s{font-size:11px;color:var(--dim);margin-top:2px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.warn{color:var(--warn)}.dim{color:var(--dim)}
.row{display:grid;grid-template-columns:1.6fr 1fr;gap:14px;margin-bottom:16px}
@media(max-width:980px){.row{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:15px}
.panel h3{margin:0 0 12px;font-size:12px;font-weight:640;color:var(--dim);
 text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;padding:7px 9px;color:var(--dim);font-size:10.5px;font-weight:600;
 text-transform:uppercase;border-bottom:1px solid var(--line)}
td{padding:7px 9px;border-bottom:1px solid #1a1e2a}
.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{padding:2px 7px;border-radius:4px;font-size:10.5px;font-weight:650}
.long{background:#0d3d2e;color:var(--pos)}.short{background:#4a1420;color:var(--neg)}
.empty{color:var(--dim);text-align:center;padding:22px}
canvas{max-height:240px}
</style></head><body><div id="app"><div class="sub">loading…</div></div>
<script>
const $=s=>document.querySelector(s);
const money=v=>(v<0?"-":"")+"$"+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const sg=v=>v>=0?"pos":"neg", pm=v=>v>=0?"+":"";
let charts={};
function mk(id,cfg){if(charts[id])charts[id].destroy();const e=document.getElementById(id);if(e)charts[id]=new Chart(e,cfg);}
function opts(extra){return Object.assign({responsive:true,maintainAspectRatio:false,
 plugins:{legend:{display:false}},
 scales:{x:{grid:{color:"#1c2130"},ticks:{color:"#8b93a7",maxTicksLimit:8,font:{size:10}}},
         y:{grid:{color:"#1c2130"},ticks:{color:"#8b93a7",font:{size:10}}}}},extra||{});}
function card(k,v,cls,s){return `<div class="card"><div class="k">${k}</div><div class="v ${cls||''}">${v}</div>${s?`<div class="s">${s}</div>`:""}</div>`}

function render(d){
  const e=d.expect;
  const hbOk = d.heartbeat_min!=null && d.heartbeat_min < 15;
  const pf = d.gross_loss!=0 ? Math.abs(d.gross_win/d.gross_loss) : 0;
  const cards=[
    card("Equity",money(d.equity),"",`from ${money(d.start)}`),
    card("Total return",pm(d.total_pct)+d.total_pct.toFixed(2)+"%",sg(d.total_pct),`${d.sessions} sessions`),
    card("Daily avg",pm(d.daily_avg_pct)+d.daily_avg_pct.toFixed(3)+"%",sg(d.daily_avg_pct),`backtest ${e.daily}%`),
    card("CAGR (proj)",pm(d.cagr_pct)+d.cagr_pct.toFixed(0)+"%",sg(d.cagr_pct),`backtest ${e.cagr}%`),
    card("Sharpe",d.sharpe==null?"–":d.sharpe.toFixed(2),"",`backtest ${e.sharpe}`),
    card("Max DD",d.maxdd_pct.toFixed(2)+"%",d.maxdd_pct<-20?"neg":"","limit -37%"),
    card("Trades",String(d.n_trades),"",`${(d.n_trades/Math.max(d.sessions,1)).toFixed(1)}/day · bt ${e.tpd}`),
    card("Win / Loss",`${d.n_wins} / ${d.n_losses}`,"",d.n_trades?d.win_pct.toFixed(1)+"% · bt "+e.win+"%":"none yet"),
    card("Profit factor",pf?pf.toFixed(2):"–",pf>=1?"pos":"neg",`avg ${money(d.avg_win)} / ${money(d.avg_loss)}`),
    card("Open",String(d.positions.length),"","max 10"),
  ].join("");

  const slip=d.exec&&d.exec.stop_slip_R!=null?d.exec.stop_slip_R:null;
  const slipCls=slip==null?"dim":(slip<=0.15?"pos":slip<=0.25?"warn":"neg");
  const execHtml = !d.exec||!d.exec.matched ? '<div class="empty">no matched trades yet</div>' :
   `<div class="grid" style="margin-bottom:0">
     ${card("Stop slippage",(slip>=0?"+":"")+slip.toFixed(3)+"R",slipCls,"backtest assumes 0.10R")}
     ${card("Realised payoff",(d.exec.realised_ratio||0).toFixed(2)+":1","","intended 2.0:1")}
     ${card("Exit mix",`${d.exec.pct_target}/${d.exec.pct_stop}/${d.exec.pct_time}`,"","target/stop/time %")}
    </div>`;

  const posRows=d.positions.length?d.positions.map(p=>`<tr>
    <td><b>${p.symbol}</b></td><td><span class="tag ${p.side=='LONG'?'long':'short'}">${p.side}</span></td>
    <td class="num">${p.qty}</td><td class="num">${money(p.entry)}</td><td class="num">${money(p.current)}</td>
    <td class="num ${sg(p.upl)}">${pm(p.upl)}${money(p.upl)}</td>
    <td class="num ${sg(p.upl)}">${pm(p.upl_pct)}${p.upl_pct.toFixed(2)}%</td>
    <td class="num">${p.days_held==null?"–":p.days_held+"d"}</td></tr>`).join("")
    :'<tr><td colspan="8" class="empty">flat — no open positions</td></tr>';

  const trRows=d.trades.length?d.trades.slice(0,40).map(t=>`<tr>
    <td class="dim">${(t.exit_ts||"").slice(5,10)}</td><td><b>${t.symbol}</b></td>
    <td><span class="tag ${t.side=='LONG'?'long':'short'}">${t.side[0]}</span></td>
    <td class="num">${money(t.entry)}</td><td class="num">${money(t.exit)}</td>
    <td class="num ${sg(t.pnl)}">${pm(t.pnl)}${money(t.pnl)}</td>
    <td class="num ${sg(t.pnl)}">${pm(t.pnl_pct)}${t.pnl_pct.toFixed(2)}%</td></tr>`).join("")
    :'<tr><td colspan="7" class="empty">no closed trades yet</td></tr>';

  $("#app").innerHTML=`
   <div style="display:flex;align-items:center;gap:10px"><h1>Swing Bot</h1>
    <span class="badge ${hbOk?'ok':'bad'}">${hbOk?'RUNNING':'CHECK BOT'}</span></div>
   <div class="sub">${d.sessions} sessions · heartbeat ${d.heartbeat_min==null?"?":d.heartbeat_min.toFixed(1)+"m ago"} · refresh 30s</div>
   <div class="grid">${cards}</div>
   <div class="row">
     <div class="panel"><h3>Equity curve</h3><canvas id="eq"></canvas></div>
     <div class="panel"><h3>Daily returns %</h3><canvas id="dr"></canvas></div>
   </div>
   <div class="row">
     <div class="panel"><h3>Trade P&L</h3><canvas id="tp"></canvas></div>
     <div class="panel"><h3>Drawdown %</h3><canvas id="dd"></canvas></div>
   </div>
   <div class="panel" style="margin-bottom:16px"><h3>Execution quality</h3>${execHtml}</div>
   <div class="panel" style="margin-bottom:16px"><h3>Open positions</h3>
     <table><thead><tr><th>Symbol</th><th>Side</th><th class="num">Qty</th><th class="num">Entry</th>
     <th class="num">Now</th><th class="num">Unreal.</th><th class="num">%</th><th class="num">Held</th>
     </tr></thead><tbody>${posRows}</tbody></table></div>
   <div class="panel"><h3>Closed trades</h3>
     <table><thead><tr><th>Date</th><th>Symbol</th><th>Side</th><th class="num">Entry</th>
     <th class="num">Exit</th><th class="num">P&L</th><th class="num">%</th></tr></thead>
     <tbody>${trRows}</tbody></table></div>`;

  const lab=d.curve.map(p=>p.day.slice(5));
  mk("eq",{type:"line",data:{labels:lab,datasets:[{data:d.curve.map(p=>p.equity),
    borderColor:d.total_pct>=0?"#34d399":"#f87171",backgroundColor:d.total_pct>=0?"#34d39922":"#f8717122",
    fill:true,tension:.25,pointRadius:0,borderWidth:2}]},options:opts()});
  mk("dr",{type:"bar",data:{labels:d.daily.map(x=>x.day.slice(5)),
    datasets:[{data:d.daily.map(x=>x.ret),backgroundColor:d.daily.map(x=>x.ret>=0?"#34d399cc":"#f87171cc")}]},
    options:opts()});
  const tp=d.trades.slice().reverse();
  mk("tp",{type:"bar",data:{labels:tp.map(t=>t.symbol),
    datasets:[{data:tp.map(t=>t.pnl),backgroundColor:tp.map(t=>t.pnl>=0?"#34d399cc":"#f87171cc")}]},
    options:opts()});
  mk("dd",{type:"line",data:{labels:lab,datasets:[{data:d.dd,borderColor:"#f87171",
    backgroundColor:"#f8717122",fill:true,tension:.2,pointRadius:0,borderWidth:2}]},options:opts()});
}
async function load(){try{const r=await fetch("/api");const d=await r.json();
  if(d.error){$("#app").innerHTML='<div class="panel neg">'+d.error+'</div>';return}render(d);}
  catch(e){$("#app").innerHTML='<div class="panel neg">error: '+e+'</div>'}}
load();setInterval(load,30000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api"):
            try:
                body = json.dumps(payload(), default=str).encode()
            except Exception as e:
                log.exception("api")
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(body, "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(f"swing dashboard: http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

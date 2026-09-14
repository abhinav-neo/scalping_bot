"""
Monitoring dashboard.

Read-only by construction: it queries the broker and reads the bot's state files,
but never places or cancels orders.

    http://localhost:8080          UI
    http://localhost:8080/api      JSON payload
    http://localhost:8080/health   liveness
"""
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .settings import S
from .analytics import Analytics

log = logging.getLogger("dashboard")
_A = None


def _an():
    global _A
    if _A is None:
        _A = Analytics(S)
    return _A


def _read(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


def _read_track(path, keep=400):
    """Last N samples per symbol from the position tracker."""
    try:
        rows = []
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        by = {}
        for r in rows:
            by.setdefault(r["symbol"], []).append(r)
        return {k: v[-keep:] for k, v in by.items()}
    except Exception:
        return {}


def payload():
    a = _an()
    hb = _read(os.path.join(S.state_dir, "heartbeat.json"), {}) or {}
    risk = _read(os.path.join(S.state_dir, "risk_state.json"), {}) or {}
    scan = _read(os.path.join(S.state_dir, "scan_state.json"), {}) or {}
    trips = a.round_trips(days=30)
    acct = a.account()
    return {
        "mode": hb.get("mode", S.mode),
        "account": acct,
        "risk": {
            "day_start_equity": risk.get("day_start_equity"),
            "peak_equity": risk.get("peak_equity"),
            "trades_today": risk.get("trades_today", 0),
            "day_locked": bool(risk.get("day_locked")),
            "halted": bool(risk.get("halted")),
            "halt_reason": risk.get("halt_reason"),
        },
        "heartbeat_ts": hb.get("ts"),
        "scan": scan,
        "track": _read_track(os.path.join(S.state_dir, "position_track.jsonl")),
        "positions": a.positions(),
        "trips": [
            {**t,
             "entry_ts": t["entry_ts"].isoformat() if t.get("entry_ts") else None,
             "exit_ts": t["exit_ts"].isoformat() if t.get("exit_ts") else None}
            for t in trips[-200:]
        ],
        "stats": a.stats(trips),
        "by_symbol": a.by_symbol(trips),
        "windows": a.windowed(trips),
        "ops": a.ops_metrics(S.state_dir, trips),
        "exec": a.execution_quality(S.state_dir, trips),
        "headroom": a.risk_headroom(acct.get("equity", 0), risk),
        "config": a.config_summary(),
        "equity_curve": a.portfolio_history("1M", "1D"),
        "intraday_curve": a.portfolio_history("1D", "5Min"),
        "symbols": S.symbols,
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scalping Bot</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0b0d13;--card:#141822;--line:#232838;--fg:#e8eaf0;--dim:#8b93a7;
      --pos:#34d399;--neg:#f87171;--acc:#60a5fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:20px}
.top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:4px}
h1{font-size:19px;margin:0;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12px;margin-bottom:18px}
.badge{padding:3px 9px;border-radius:5px;font-size:11px;font-weight:700;letter-spacing:.03em}
.paper{background:#123047;color:#7dd3fc}.live{background:#7f1d1d;color:#fca5a5}
.halt{background:#7f1d1d;color:#fca5a5}.lock{background:#78350f;color:#fcd34d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:11px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:13px 15px}
.k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;font-weight:600}
.v{font-size:21px;font-weight:660;margin-top:5px;letter-spacing:-.02em}
.s{font-size:11px;color:var(--dim);margin-top:2px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.dimc{color:var(--dim)}
.row{display:grid;grid-template-columns:1.55fr 1fr;gap:14px;margin-bottom:16px}
@media(max-width:980px){.row{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:15px}
.panel h3{margin:0 0 12px;font-size:12.5px;font-weight:640;color:var(--dim);
          text-transform:uppercase;letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;padding:7px 9px;color:var(--dim);font-size:10.5px;font-weight:600;
   text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line)}
td{padding:7px 9px;border-bottom:1px solid #1a1e2a}
tr:last-child td{border-bottom:none}
tbody tr:hover{background:#171c28}
.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{padding:2px 7px;border-radius:4px;font-size:10.5px;font-weight:650}
.long{background:#0d3d2e;color:var(--pos)}.short{background:#4a1420;color:var(--neg)}
.st{padding:2px 8px;border-radius:4px;font-size:10.5px;font-weight:700;letter-spacing:.03em}
.st-HIT{background:#0d3d2e;color:#34d399}
.st-miss{background:#26210f;color:#d4b95e}
.st-veto{background:#3d1f0d;color:#fb923c}
.st-stale{background:#3d1414;color:#f87171}
.st-drift{background:#3d1414;color:#f87171}
.st-cooldown{background:#1a2a3d;color:#7dd3fc}
.st-waiting{background:#1e2230;color:#8b93a7}
.st-in_position{background:#0d3d2e;color:#34d399}
.st-order_resting{background:#1a2a3d;color:#7dd3fc}
.st-blocked{background:#3d1414;color:#f87171}
.st-market_closed{background:#1e2230;color:#8b93a7}
.st-capacity{background:#1e2230;color:#8b93a7}
.st-no_signal,.st-no_data,.st-no_size{background:#1e2230;color:#8b93a7}
.bar{height:5px;background:#1e2230;border-radius:3px;overflow:hidden;margin-top:3px}
.bar>i{display:block;height:100%;border-radius:3px}
.mono{font-variant-numeric:tabular-nums;font-size:11.5px}
.warnc{color:#fbbf24}
.cfgbar{background:#12151d;border:1px solid var(--line);border-radius:8px;
        padding:8px 13px;font-size:11.5px;color:#c7ccd8;margin-bottom:16px}
.empty{color:var(--dim);text-align:center;padding:22px;font-size:12.5px}
canvas{max-height:230px}
.tabs{display:flex;gap:6px;margin-bottom:11px}
.tab{padding:5px 12px;border-radius:6px;background:#1a1e2a;color:var(--dim);
     font-size:11.5px;cursor:pointer;border:1px solid transparent;font-weight:600}
.tab.on{background:#1e3a5f;color:#93c5fd;border-color:#2563eb44}
.err{background:#3b1219;border:1px solid #7f1d1d;color:#fca5a5;padding:11px 14px;
     border-radius:9px;margin-bottom:15px;font-size:12.5px}
</style></head><body>
<div id="app"><div class="sub">loading...</div></div>
<script>
const $=(s)=>document.querySelector(s);
const money=(v)=>(v<0?"-":"")+"$"+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
const sgn=(v)=>(v>=0?"pos":"neg"), pm=(v)=>(v>=0?"+":"");
let charts={}, WIN="today";

function metric(k,v,cls,sub){return '<div class="card"><div class="k">'+k+'</div>'+
  '<div class="v '+(cls||'')+'">'+v+'</div>'+(sub?'<div class="s">'+sub+'</div>':'')+'</div>'}

function render(d){
  const a=d.account, r=d.risk, st=d.stats, w=d.windows;
  const dayStart=r.day_start_equity||a.equity||1;
  const dayPl=a.equity-dayStart, dayPct=dayStart?dayPl/dayStart*100:0;
  const pf=st.profit_factor===null?"-":(!isFinite(st.profit_factor)?"inf":st.profit_factor.toFixed(2));

  let banner="";
  if(r.halted) banner='<div class="err"><b>BOT HALTED</b> - '+(r.halt_reason||"drawdown limit")+'. No new positions until cleared.</div>';
  else if(r.day_locked) banner='<div class="err"><b>DAY LOCKED</b> - daily loss limit hit. Trading resumes next session.</div>';

  const badges='<span class="badge '+(d.mode.indexOf("PAPER")>=0?"paper":"live")+'">'+d.mode+'</span>'
    +(r.halted?'<span class="badge halt">HALTED</span>':"")
    +(r.day_locked&&!r.halted?'<span class="badge lock">DAY LOCKED</span>':"");

  const cards=[
    metric("Equity",money(a.equity),"","cash "+money(a.cash)),
    metric("Day P&L",pm(dayPl)+money(dayPl),sgn(dayPl),pm(dayPct)+dayPct.toFixed(2)+"%"),
    metric("Realized today",pm(w.today.net)+money(w.today.net),sgn(w.today.net),w.today.trades+" trades"),
    metric("Week",pm(w.week.net)+money(w.week.net),sgn(w.week.net),w.week.trades+" trades"),
    metric("Month",pm(w.month.net)+money(w.month.net),sgn(w.month.net),w.month.trades+" trades"),
    metric("All time",pm(w.all.net)+money(w.all.net),sgn(w.all.net),w.all.trades+" trades"),
    metric("Win rate",st.win_rate.toFixed(1)+"%","",st.trades+" closed"),
    metric("Profit factor",pf,st.profit_factor>=1?"pos":"neg","exp "+money(st.expectancy)+"/trade"),
    metric("Avg win / loss",money(st.avg_win)+" / "+money(st.avg_loss),"","hold "+st.avg_hold.toFixed(1)+"m"),
    metric("Open positions",String(d.positions.length),"",r.trades_today+" entries today")
  ].join("");

  var posRowsTop=d.positions.length?d.positions.map(function(p){return '<tr>'+
    '<td><b>'+p.symbol+'</b></td>'+
    '<td><span class="tag '+(p.side>0?'long':'short')+'">'+(p.side>0?'LONG':'SHORT')+'</span></td>'+
    '<td class="num mono">'+p.qty.toFixed(3)+'</td>'+
    '<td class="num mono">'+money(p.entry)+'</td>'+
    '<td class="num mono">'+money(p.current)+'</td>'+
    '<td class="num mono '+sgn(p.unrealized)+'">'+pm(p.unrealized)+money(p.unrealized)+'</td>'+
    '<td class="num mono '+sgn(p.unrealized)+'">'+pm(p.unrealized_pct)+p.unrealized_pct.toFixed(2)+'%</td></tr>'
  }).join(""):'<tr><td colspan="7" class="empty">flat &mdash; nothing open</td></tr>';

  var op=d.ops||{}, hr=d.headroom||{}, cf=d.config||{};

  // --- health strip: the three questions that decide if the strategy works ---
  var fr=op.fill_rate;
  var frCls=fr==null?"dimc":(fr>=70?"pos":fr>=40?"warnc":"neg");
  var be=op.breakeven_win_pct;
  var wr=st.win_rate;
  var wrCls=(st.trades<10||be==null)?"dimc":(wr>=be?"pos":"neg");
  var wrNote=be==null?"":(st.trades<10?"need 10+ trades":(wr>=be?"above breakeven":"BELOW breakeven")+" ("+(op.breakeven_basis||"")+")");

  function bar2(pct,col){return '<div class="bar"><i style="width:'+Math.max(0,Math.min(100,pct)).toFixed(0)+'%;background:'+col+'"></i></div>'}

  var healthCards=[
    metric("Fill rate",(fr==null?"&ndash;":fr.toFixed(0)+"%"),frCls,
           op.orders_filled+" of "+op.orders_sent+" orders"+(op.orders_cancelled?" ("+op.orders_cancelled+" expired)":"")),
    metric("Win rate vs breakeven",
           (st.trades?wr.toFixed(1)+"%":"&ndash;")+(be!=null?' <span class="dimc" style="font-size:12px">/ '+be.toFixed(1)+'%</span>':""),
           wrCls, wrNote),
    metric("Long / short",(op.long_n||0)+" / "+(op.short_n||0),
           (op.long_pct==null?"dimc":(op.long_pct>80||op.long_pct<20?"warnc":"")),
           op.long_pct==null?"no trades":op.long_pct.toFixed(0)+"% long"),
    metric("Daily loss room",(hr.daily_room_pct==null?"&ndash;":hr.daily_room_pct.toFixed(2)+"%"),
           (hr.daily_room_pct!=null&&hr.daily_room_pct<1?"neg":"pos"),
           "day "+(hr.day_pl_pct>=0?"+":"")+ (hr.day_pl_pct==null?"-":hr.day_pl_pct.toFixed(2))+"% of -"+(hr.daily_kill_pct||3)+"%"),
    metric("Drawdown room",(hr.drawdown_room_pct==null?"&ndash;":hr.drawdown_room_pct.toFixed(2)+"%"),
           (hr.drawdown_room_pct!=null&&hr.drawdown_room_pct<5?"neg":"pos"),
           "dd "+(hr.drawdown_pct==null?"-":hr.drawdown_pct.toFixed(2))+"% of -"+(hr.halt_pct||20)+"%"),
    metric("Orders today",(hr.trades_today||0)+" / "+(hr.max_trades_per_day||120),"",
           "cap "+(hr.max_trades_per_day||120)+"/day")
  ].join("");

  // --- exit reason mix ---
  var er=op.exit_reasons||{};
  var erKeys=Object.keys(er);
  var erTotal=erKeys.reduce(function(a,k){return a+er[k]},0);
  var erColors={take_profit:"#34d399",stop:"#f87171",time:"#fbbf24",
                eod_flat:"#8b93a7",kill_switch:"#fb923c"};
  var erRows=erKeys.length?erKeys.sort(function(a,b){return er[b]-er[a]}).map(function(k){
    var pctv=100*er[k]/erTotal;
    return '<tr><td>'+k.replace(/_/g," ")+'</td><td class="num mono">'+er[k]+'</td>'+
           '<td class="num mono">'+pctv.toFixed(0)+'%</td>'+
           '<td style="width:45%">'+bar2(pctv,erColors[k]||"#60a5fa")+'</td></tr>';
  }).join(""):'<tr><td colspan="4" class="empty">no exits yet</td></tr>';

  var cfgTxt='<span class="dimc">'+cf.symbols.join(" ")+'</span> &middot; '+
    cf.bar_minutes+'m bars &middot; horizon '+cf.horizon_bars+' ('+cf.horizon_minutes+'m) &middot; '+
    'max hold '+cf.max_hold_minutes+'m &middot; pt/sl '+cf.pt_mult+'/'+cf.sl_mult+' &middot; '+
    'thr '+cf.meta_threshold+' &middot; risk '+cf.risk_per_trade_pct+'% &middot; '+
    'max '+cf.max_concurrent+' pos &middot; filter '+(cf.market_filter?"on":"off")+' &middot; '+
    (cf.limit_entry?"limit":"market")+' entry &middot; '+cf.feed;

  const trips=d.trips.slice().reverse();
  const tripRows=trips.length?trips.slice(0,60).map(function(t){return '<tr>'+
    '<td class="dimc">'+(t.exit_ts||"").slice(11,19)+'</td>'+
    '<td><b>'+t.symbol+'</b></td>'+
    '<td><span class="tag '+(t.side>0?'long':'short')+'">'+(t.side>0?'L':'S')+'</span></td>'+
    '<td class="num">'+t.qty.toFixed(3)+'</td>'+
    '<td class="num">'+money(t.entry)+'</td>'+
    '<td class="num">'+money(t.exit)+'</td>'+
    '<td class="num">'+(t.held_min!=null?t.held_min.toFixed(1)+"m":"-")+'</td>'+
    '<td class="num '+sgn(t.pnl)+'">'+pm(t.pnl)+money(t.pnl)+'</td>'+
    '<td class="num '+sgn(t.pnl)+'">'+pm(t.pnl_pct)+t.pnl_pct.toFixed(2)+'%</td></tr>'
  }).join(""):'<tr><td colspan="9" class="empty">no closed trades yet</td></tr>';

  const bs=Object.keys(d.by_symbol);
  const symRows=bs.length?bs.map(function(s){var v=d.by_symbol[s];return '<tr>'+
    '<td><b>'+s+'</b></td><td class="num">'+v.trades+'</td>'+
    '<td class="num">'+v.win_rate.toFixed(0)+'%</td>'+
    '<td class="num">'+(!isFinite(v.profit_factor)?"inf":v.profit_factor.toFixed(2))+'</td>'+
    '<td class="num '+sgn(v.net)+'">'+pm(v.net)+money(v.net)+'</td></tr>'
  }).join(""):'<tr><td colspan="5" class="empty">no data</td></tr>';

  var sc=d.scan||{}, sm=sc.meta||{}, ssy=sc.symbols||{};
  var order=d.symbols.slice();
  var scanRows=order.length?order.map(function(s){
    var v=ssy[s];
    if(!v) return '<tr><td><b>'+s+'</b></td><td colspan="6" class="dimc">not evaluated yet</td></tr>';
    var conf=v.meta_p!=null?v.meta_p:null;
    var thr=sm.meta_threshold||0.55;
    var barCol=conf==null?"#8b93a7":(conf>=thr?"#34d399":"#d4b95e");
    var bar=conf==null?"":'<div class="bar"><i style="width:'+Math.min(100,conf*100).toFixed(0)+'%;background:'+barCol+'"></i></div>';
    return '<tr>'+
      '<td><b>'+s+'</b></td>'+
      '<td><span class="st st-'+v.status+'">'+v.status.replace(/_/g," ")+'</span></td>'+
      '<td>'+(v.side==null?'<span class="dimc">-</span>':'<span class="tag '+(v.side>0?'long':'short')+'">'+(v.side>0?'LONG':'SHORT')+'</span>')+'</td>'+
      '<td class="num mono">'+(v.price!=null?money(v.price):"-")+'</td>'+
      '<td class="num mono">'+(conf!=null?conf.toFixed(3):"-")+bar+'</td>'+
      '<td class="mono dimc">'+(v.bar_time?String(v.bar_time).slice(11,16):"-")+'</td>'+
      '<td class="dimc">'+(v.reason||"")+'</td></tr>';
  }).join(""):'<tr><td colspan="7" class="empty">no scan data</td></tr>';

  var trendTxt=sm.market_trend_bps==null?"n/a":(sm.market_trend_bps>=0?"+":"")+sm.market_trend_bps.toFixed(0)+" bps";
  var trendCls=sm.market_trend_bps==null?"dimc":(sm.market_trend_bps>=0?"pos":"neg");
  var gateTxt=sm.market_open===false?"market closed"
      :(sm.can_open?"accepting entries":"blocked: "+(sm.block_reason||"-"));
  var gateCls=sm.market_open===false?"dimc":(sm.can_open?"pos":"neg");

  var ex=d.exec||{};
  var execPanel;
  if(!ex.matched){
    execPanel='<div class="panel" style="margin-bottom:16px"><h3>Execution quality</h3>'+
      '<div class="empty">no matched trades yet &mdash; needs entries with recorded tp/sl</div></div>';
  } else {
    var slip=ex.stop_slip_R;
    var slipCls=slip==null?"dimc":(slip<=0.15?"pos":slip<=0.25?"warnc":"neg");
    var slipNote=slip==null?"":(slip<=0.15?"brackets working"
                  :slip<=0.25?"worse than modelled":"edge at risk");
    var ratio=ex.realised_ratio;
    var ratioCls=ratio==null?"dimc":(ratio>=1.3?"pos":ratio>=1.0?"warnc":"neg");
    execPanel='<div class="panel" style="margin-bottom:16px">'+
      '<h3>Execution quality &mdash; measured in R (1R = intended stop)</h3>'+
      '<div class="grid" style="margin-bottom:12px">'+
        metric("Stop slippage",(slip==null?"&ndash;":(slip>=0?"+":"")+slip.toFixed(3)+"R"),slipCls,
               "modelled 0.10R &middot; "+slipNote)+
        metric("Realised payoff",(ratio==null?"&ndash;":ratio.toFixed(2)+":1"),ratioCls,
               "intended "+(cf.pt_mult/cf.sl_mult).toFixed(2)+":1")+
        metric("Avg win / loss",(ex.avg_win_R==null?"&ndash;":ex.avg_win_R.toFixed(2)+"R")+
               " / "+(ex.avg_loss_R==null?"&ndash;":ex.avg_loss_R.toFixed(2)+"R"),"",
               "expectancy "+(ex.mean_R==null?"-":(ex.mean_R>=0?"+":"")+ex.mean_R.toFixed(3)+"R"))+
        metric("Matched trades",String(ex.matched),"","of "+st.trades+" closed")+
      '</div>'+
      '<table><thead><tr><th>Exit</th><th class="num">n</th><th class="num">%</th>'+
      '<th class="num">Mean R</th><th></th></tr></thead><tbody>'+
      '<tr><td>target</td><td class="num mono">'+ex.n_target+'</td>'+
        '<td class="num mono">'+ex.pct_target+'%</td>'+
        '<td class="num mono pos">'+(ex.target_mean_R==null?"-":"+"+ex.target_mean_R.toFixed(2))+'</td>'+
        '<td style="width:40%">'+bar2(ex.pct_target,"#34d399")+'</td></tr>'+
      '<tr><td>stop</td><td class="num mono">'+ex.n_stop+'</td>'+
        '<td class="num mono">'+ex.pct_stop+'%</td>'+
        '<td class="num mono neg">'+(ex.stop_mean_R==null?"-":ex.stop_mean_R.toFixed(2))+'</td>'+
        '<td>'+bar2(ex.pct_stop,"#f87171")+'</td></tr>'+
      '<tr><td>time barrier</td><td class="num mono">'+ex.n_time+'</td>'+
        '<td class="num mono">'+ex.pct_time+'%</td>'+
        '<td class="num mono">'+(ex.time_mean_R==null?"-":(ex.time_mean_R>=0?"+":"")+ex.time_mean_R.toFixed(2))+'</td>'+
        '<td>'+bar2(ex.pct_time,"#fbbf24")+'</td></tr>'+
      '</tbody></table>'+
      '<div class="s" style="margin-top:9px">Target is '+(ex.target_R||0).toFixed(2)+
      'R away. A high share of time-barrier exits means the target is too far to '+
      'reach inside the horizon &mdash; winners get truncated while losers pay in full.</div>'+
      '</div>';
  }

  $("#app").innerHTML=
   '<div class="top"><h1>Scalping Bot</h1>'+badges+'</div>'+
   '<div class="sub">updated '+(d.heartbeat_ts?d.heartbeat_ts.slice(11,19)+" UTC":"-")+' &middot; '+d.symbols.join(" &middot; ")+' &middot; refresh 15s</div>'+
   banner+
   '<div class="cfgbar">'+cfgTxt+' &middot; '+(cf.brackets?'<b>bracket exits</b>':'polled exits')+'</div>'+
   '<div class="grid">'+healthCards+'</div>'+
   execPanel+
   '<div class="row" style="margin-bottom:16px">'+
     '<div class="panel"><h3>Open positions &mdash; live P&L</h3><canvas id="opentrk"></canvas></div>'+
     '<div class="panel"><h3>Open now ('+d.positions.length+')</h3>'+
       '<table><thead><tr><th>Symbol</th><th>Side</th><th class="num">Qty</th><th class="num">Entry</th>'+
       '<th class="num">Current</th><th class="num">Unreal.</th><th class="num">%</th></tr></thead>'+
       '<tbody>'+posRowsTop+'</tbody></table></div>'+
   '</div>'+
   '<div class="panel" style="margin-bottom:16px"><h3>Live scanner &mdash; what the bot is seeing right now</h3>'+
     '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:11px;font-size:12px">'+
       '<span class="dimc">Tape ('+(sm.market_trend_bars||12)+' bars): <b class="'+trendCls+'">'+trendTxt+'</b></span>'+
       '<span class="dimc">Gate: <b class="'+gateCls+'">'+gateTxt+'</b></span>'+
       '<span class="dimc">Threshold: <b>'+(sm.meta_threshold||0.55)+'</b></span>'+
       '<span class="dimc">Scanned: <b>'+(sc.ts?String(sc.ts).slice(11,19):"-")+'</b></span>'+
     '</div>'+
     '<table><thead><tr><th>Symbol</th><th>Status</th><th>Signal</th><th class="num">Price</th>'+
     '<th class="num">Confidence</th><th>Bar</th><th>Detail</th></tr></thead><tbody>'+scanRows+'</tbody></table></div>'+
   '<div class="grid">'+cards+'</div>'+
   '<div class="row">'+
     '<div class="panel"><h3>Equity</h3>'+
       '<div class="tabs">'+
         '<div class="tab '+(WIN==="today"?"on":"")+'" onclick="setWin(\'today\')">Intraday</div>'+
         '<div class="tab '+(WIN==="month"?"on":"")+'" onclick="setWin(\'month\')">30 days</div>'+
       '</div><canvas id="eq"></canvas></div>'+
     '<div class="panel"><h3>Cumulative realized P&L</h3><canvas id="cum"></canvas></div>'+
   '</div>'+
   '<div class="row">'+
     '<div class="panel"><h3>Trade P&L - each closed trade</h3><canvas id="bars"></canvas></div>'+
     '<div class="panel"><h3>By symbol</h3>'+
       '<table><thead><tr><th>Sym</th><th class="num">Trades</th><th class="num">Win</th>'+
       '<th class="num">PF</th><th class="num">Net</th></tr></thead><tbody>'+symRows+'</tbody></table>'+
       '<h3 style="margin-top:18px">Exit reasons</h3>'+
       '<table><thead><tr><th>Reason</th><th class="num">n</th><th class="num">%</th><th></th></tr></thead>'+
       '<tbody>'+erRows+'</tbody></table></div>'+
   '</div>'+
   '<div class="panel"><h3>Closed trades - most recent 60</h3>'+
     '<table><thead><tr><th>Exit</th><th>Symbol</th><th>Side</th><th class="num">Qty</th>'+
     '<th class="num">Entry</th><th class="num">Exit</th><th class="num">Held</th>'+
     '<th class="num">P&L</th><th class="num">%</th></tr></thead><tbody>'+tripRows+'</tbody></table></div>';

  drawEquity(d); drawCum(d); drawBars(d); drawOpenTrack(d);
}

var PALETTE=["#60a5fa","#34d399","#fbbf24","#f472b6","#a78bfa","#fb923c"];

function drawOpenTrack(d){
  var track=d.track||{}, open=d.positions.map(function(p){return p.symbol});
  var syms=Object.keys(track).filter(function(s){return open.indexOf(s)>=0});
  if(!syms.length){
    mk("opentrk",{type:"line",data:{labels:[],datasets:[]},
      options:Object.assign(baseOpts(),{plugins:{legend:{display:false},
        tooltip:{enabled:false},
        title:{display:true,text:"no open positions",color:"#8b93a7",font:{size:12}}}})});
    return;
  }
  // union of timestamps across open symbols, so lines share an x-axis
  var tset={};
  syms.forEach(function(s){track[s].forEach(function(r){tset[r.ts]=1})});
  var times=Object.keys(tset).sort();
  var labels=times.map(function(t){return t.slice(11,16)});
  var datasets=syms.map(function(s,i){
    var m={}; track[s].forEach(function(r){m[r.ts]=r.unrealized});
    var last=null;
    var data=times.map(function(t){
      if(m[t]!==undefined){last=m[t];return m[t]}
      return null;   // gap before the position opened
    });
    var col=PALETTE[i%PALETTE.length];
    var side=(d.positions.filter(function(p){return p.symbol===s})[0]||{}).side;
    return {label:s+(side>0?" LONG":" SHORT"),data:data,borderColor:col,
            backgroundColor:col+"22",fill:false,tension:.2,pointRadius:0,
            borderWidth:2,spanGaps:false};
  });
  var opts=baseOpts();
  opts.plugins.legend={display:true,position:"top",
    labels:{color:"#8b93a7",boxWidth:10,boxHeight:10,font:{size:11},usePointStyle:true}};
  opts.plugins.tooltip={mode:"index",intersect:false,
    callbacks:{label:function(c){return c.dataset.label+": "+(c.parsed.y>=0?"+":"")+"$"+c.parsed.y.toFixed(2)}}};
  mk("opentrk",{type:"line",data:{labels:labels,datasets:datasets},options:opts});
}

function baseOpts(){return{responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false},tooltip:{mode:"index",intersect:false}},
  scales:{x:{grid:{color:"#1c2130"},ticks:{color:"#8b93a7",maxTicksLimit:8,font:{size:10}}},
          y:{grid:{color:"#1c2130"},ticks:{color:"#8b93a7",font:{size:10}}}}}}

function mk(id,cfg){if(charts[id])charts[id].destroy();
  var el=document.getElementById(id); if(!el)return; charts[id]=new Chart(el,cfg)}

function drawEquity(d){
  var src=WIN==="today"?d.intraday_curve:d.equity_curve;
  if(!src||!src.length){mk("eq",{type:"line",data:{labels:[],datasets:[]},options:baseOpts()});return}
  var labels=src.map(function(p){var dt=new Date(p.t*1000);
    return WIN==="today"?dt.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"})
                        :dt.toLocaleDateString([],{month:"short",day:"numeric"})});
  var vals=src.map(function(p){return p.equity}), up=vals[vals.length-1]>=vals[0];
  mk("eq",{type:"line",data:{labels:labels,datasets:[{data:vals,borderColor:up?"#34d399":"#f87171",
    backgroundColor:up?"#34d39922":"#f8717122",fill:true,tension:.25,pointRadius:0,borderWidth:2}]},
    options:baseOpts()});
}

function drawCum(d){
  var t=d.trips; if(!t.length){mk("cum",{type:"line",data:{labels:[],datasets:[]},options:baseOpts()});return}
  var c=0; var vals=t.map(function(x){c+=x.pnl;return c});
  var labels=t.map(function(x,i){return String(i+1)});
  mk("cum",{type:"line",data:{labels:labels,datasets:[{data:vals,
    borderColor:c>=0?"#34d399":"#f87171",backgroundColor:c>=0?"#34d39922":"#f8717122",
    fill:true,tension:.2,pointRadius:0,borderWidth:2}]},options:baseOpts()});
}

function drawBars(d){
  var t=d.trips.slice(-60);
  if(!t.length){mk("bars",{type:"bar",data:{labels:[],datasets:[]},options:baseOpts()});return}
  mk("bars",{type:"bar",data:{labels:t.map(function(x){return x.symbol}),
    datasets:[{data:t.map(function(x){return x.pnl}),
    backgroundColor:t.map(function(x){return x.pnl>=0?"#34d399cc":"#f87171cc"}),
    borderRadius:2}]},options:baseOpts()});
}

function setWin(w){WIN=w;load()}
async function load(){
  try{var r=await fetch("/api");var d=await r.json();
    if(d.error){$("#app").innerHTML='<div class="err">'+d.error+'</div>';return}
    render(d)}
  catch(e){$("#app").innerHTML='<div class="err">dashboard error: '+e+'</div>'}
}
load(); setInterval(load,15000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/api"):
            try:
                body = json.dumps(payload(), default=str).encode()
            except Exception as e:
                log.exception("api error")
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(body, "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    print(f"dashboard on http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()

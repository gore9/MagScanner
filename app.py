#!/usr/bin/env python3
"""
app.py — MagScanner web dashboard.

Features:
  • Symbol cards (Mag7, ETFs, AI & Infra) with live RSI + signal badges
  • Signal history with outcome tracking (Win / Loss / Pending)
  • My Positions sidebar — check a signal to track it as an open trade;
    the scanner monitors it and recommends exits automatically.

Run:  python3 app.py
      python3 app.py --force   (scan outside market hours, for testing)
"""

import os, sys, json, time, threading, argparse, webbrowser
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional

import pytz
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from scanner import (
    fetch_bars, get_symbol_status, is_market_open,
    next_market_open_seconds, ALL_SYMBOLS, MAG7, ETFS, AI_INFRA,
    SCAN_INTERVAL_SEC, MIN_CONFIDENCE_PCT, ALERT_COOLDOWN_SEC,
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ET               = pytz.timezone("America/New_York")
PORT             = 5050
SIGNALS_FILE     = os.path.join(os.path.dirname(__file__), "signals.json")
POSITIONS_FILE   = os.path.join(os.path.dirname(__file__), "positions.json")
OUTCOME_DELAY      = 30 * 60    # seconds before marking outcome
OUTCOME_MIN_PCT    = 0.15       # minimum % move to count as success
STOP_LOSS_PCT      = 3.0        # % move against position to flag stop-loss exit
AUTO_POSITION_PCT  = 90         # confidence threshold for automatic position entry

# ── Shared state ──────────────────────────────────────────────────────────────
_lock           = threading.Lock()
_symbol_status: Dict[str, dict] = {}
_signals:       List[dict]      = []
_positions:     List[dict]      = []   # open / exit-recommended positions
_last_scan:     Optional[str]   = None
_scanning:      bool            = False


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_all():
    global _signals, _positions
    for path, target in [(SIGNALS_FILE, "_signals"), (POSITIONS_FILE, "_positions")]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    val = json.load(f)
                globals()[target] = val
            except Exception:
                pass


def _save_signals():
    with open(SIGNALS_FILE, "w") as f:
        json.dump(_signals, f, indent=2, default=str)


def _save_positions():
    with open(POSITIONS_FILE, "w") as f:
        json.dump(_positions, f, indent=2, default=str)


# ── Signal helpers ────────────────────────────────────────────────────────────

def _add_signal(status: dict):
    entry = {
        "id":         f"{status['symbol']}_{int(time.time())}",
        "symbol":     status["symbol"],
        "direction":  status["direction"],
        "confidence": status["confidence"],
        "entry":      status["entry"],
        "rsi":        round(status.get("rsi", 0), 1),
        "signals":    status.get("signals", [])[:3],
        "fired_at":   datetime.now(ET).isoformat(),
        "outcome":    "pending",
        "exit":       None,
        "pct_move":   None,
        "in_positions": False,
    }
    with _lock:
        _signals.insert(0, entry)
        if len(_signals) > 200:
            _signals.pop()
        _save_signals()


def _check_outcomes():
    now_ts = time.time()
    with _lock:
        pending = [s for s in _signals if s["outcome"] == "pending"]
    if not pending:
        return

    syms = list({s["symbol"] for s in pending
                 if now_ts - _iso_to_epoch(s["fired_at"]) >= OUTCOME_DELAY})
    if not syms:
        return

    try:
        data = fetch_bars(syms, days=1)
    except Exception:
        return

    prices = {sym: float(df.iloc[-1]["close"])
              for sym, df in data.items() if not df.empty}

    with _lock:
        changed = False
        for sig in _signals:
            if sig["outcome"] != "pending":
                continue
            if now_ts - _iso_to_epoch(sig["fired_at"]) < OUTCOME_DELAY:
                continue
            price = prices.get(sig["symbol"])
            if price is None:
                continue
            pct   = (price - sig["entry"]) / sig["entry"] * 100
            if sig["direction"] == "LONG":
                outcome = "success" if pct >= OUTCOME_MIN_PCT else ("failure" if pct <= -OUTCOME_MIN_PCT else "neutral")
            else:
                outcome = "success" if pct <= -OUTCOME_MIN_PCT else ("failure" if pct >= OUTCOME_MIN_PCT else "neutral")
            sig.update({"outcome": outcome, "exit": round(price, 2), "pct_move": round(pct, 2)})
            changed = True
        if changed:
            _save_signals()


def _iso_to_epoch(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = ET.localize(dt)
        return dt.timestamp()
    except Exception:
        return 0.0


# ── Position helpers ──────────────────────────────────────────────────────────

def add_position(signal_id: str, auto_entered: bool = False) -> bool:
    """Move a signal into the open-positions list. Returns True if added."""
    with _lock:
        sig = next((s for s in _signals if s["id"] == signal_id), None)
        if sig is None:
            return False
        if any(p["signal_id"] == signal_id for p in _positions):
            return False  # already tracked
        pos = {
            "id":           f"pos_{signal_id}",
            "signal_id":    signal_id,
            "symbol":       sig["symbol"],
            "direction":    sig["direction"],
            "entry":        sig["entry"],
            "entry_time":   datetime.now(ET).isoformat(),
            "rsi_entry":    sig.get("rsi", 0),
            "status":       "open",        # open | exit_recommended | closed
            "exit_reason":  None,
            "current":      sig["entry"],  # updated each scan
            "pnl_pct":      0.0,
            "auto_entered": auto_entered,
        }
        _positions.insert(0, pos)
        sig["in_positions"] = True
        _save_positions()
        _save_signals()
    return True


def close_position(pos_id: str) -> bool:
    """
    Mark a position as closed, recording final P&L.
    Keeps the record in positions.json for performance tracking.
    """
    with _lock:
        pos = next((p for p in _positions if p["id"] == pos_id), None)
        if pos is None or pos["status"] == "closed":
            return False
        pos["status"]        = "closed"
        pos["closed_at"]     = datetime.now(ET).isoformat()
        pos["final_pnl_pct"] = round(pos.get("pnl_pct", 0.0), 2)
        pos["exit_price"]    = pos.get("current", pos["entry"])
        if pos.get("auto_entered") and pos.get("exit_reason"):
            pos["closed_by"] = "auto_exit"
        elif pos.get("exit_reason"):
            pos["closed_by"] = "exit_signal"
        else:
            pos["closed_by"] = "user"
        _save_positions()
    return True


# ── Performance analytics ─────────────────────────────────────────────────────

def _calc_performance() -> dict:
    with _lock:
        closed = [p for p in _positions if p["status"] == "closed"]

    if not closed:
        return {"empty": True}

    pnls  = [p.get("final_pnl_pct", 0.0) for p in closed]
    wins  = [x for x in pnls if x > 0]
    losses= [x for x in pnls if x < 0]
    flat  = [x for x in pnls if x == 0]

    total_win_pct  = sum(wins)
    total_loss_pct = abs(sum(losses))
    profit_factor  = (total_win_pct / total_loss_pct) if total_loss_pct > 0 else float("inf")

    # Per-symbol breakdown
    by_sym: Dict[str, list] = {}
    for p in closed:
        by_sym.setdefault(p["symbol"], []).append(p.get("final_pnl_pct", 0.0))

    sym_rows = []
    for sym, sym_pnls in sorted(by_sym.items()):
        sym_wins = [x for x in sym_pnls if x > 0]
        sym_losses = [x for x in sym_pnls if x < 0]
        sym_rows.append({
            "symbol":   sym,
            "trades":   len(sym_pnls),
            "wins":     len(sym_wins),
            "losses":   len(sym_losses),
            "win_rate": len(sym_wins) / len(sym_pnls) * 100,
            "avg_pnl":  sum(sym_pnls) / len(sym_pnls),
            "total_pnl":sum(sym_pnls),
        })
    sym_rows.sort(key=lambda r: r["total_pnl"], reverse=True)

    # Per-direction breakdown
    by_dir: Dict[str, list] = {}
    for p in closed:
        by_dir.setdefault(p["direction"], []).append(p.get("final_pnl_pct", 0.0))

    dir_rows = []
    for dire, dir_pnls in by_dir.items():
        dir_wins = [x for x in dir_pnls if x > 0]
        dir_rows.append({
            "direction": dire,
            "trades":    len(dir_pnls),
            "wins":      len(dir_wins),
            "win_rate":  len(dir_wins) / len(dir_pnls) * 100,
            "avg_pnl":   sum(dir_pnls) / len(dir_pnls),
        })

    return {
        "empty":         False,
        "total":         len(closed),
        "wins":          len(wins),
        "losses":        len(losses),
        "flat":          len(flat),
        "win_rate":      len(wins) / len(closed) * 100,
        "avg_win":       sum(wins) / len(wins) if wins else 0,
        "avg_loss":      sum(losses) / len(losses) if losses else 0,
        "best":          max(pnls),
        "worst":         min(pnls),
        "total_pnl":     sum(pnls),
        "profit_factor": profit_factor,
        "expectancy":    sum(pnls) / len(pnls),   # avg P&L per trade
        "by_symbol":     sym_rows,
        "by_direction":  dir_rows,
        "trades":        sorted(closed, key=lambda x: x.get("closed_at",""), reverse=True),
    }


def _update_positions(data: Dict[str, object]):
    """
    Called after each scan. Updates current price / P&L and checks exit signals.
    Auto-entered positions (confidence ≥ 90%) are closed automatically when an
    exit condition fires. Manually-entered positions get an exit recommendation
    displayed in the sidebar for the user to act on.
    `data` is the raw {symbol: DataFrame} map from fetch_bars.
    """
    with _lock:
        open_pos = [p for p in _positions if p["status"] in ("open", "exit_recommended")]

    if not open_pos:
        return

    changed       = False
    auto_close_ids: List[str] = []   # pos IDs to auto-close after the loop

    for pos in open_pos:
        sym = pos["symbol"]
        df5 = data.get(sym)
        if df5 is None or df5.empty:
            continue

        current = float(df5.iloc[-1]["close"])
        pnl     = (current - pos["entry"]) / pos["entry"] * 100
        if pos["direction"] == "SHORT":
            pnl = -pnl   # short profits from price drop

        pos["current"] = round(current, 2)
        pos["pnl_pct"] = round(pnl, 2)
        changed = True

        # ── Exit analysis ─────────────────────────────────────────────────────
        exit_reasons: List[str] = []

        # 1. Stop-loss
        if pnl <= -STOP_LOSS_PCT:
            exit_reasons.append(f"Stop-loss triggered ({pnl:+.1f}%)")

        # 2. Reversal signal from scanner
        st = _symbol_status.get(sym)
        if st:
            opposite = "SHORT" if pos["direction"] == "LONG" else "LONG"
            if st["direction"] == opposite and st["confidence"] >= 60:
                exit_reasons.append(
                    f"{opposite} signal {st['confidence']}% — "
                    + (st.get("signals") or ["reversal detected"])[0].split("[")[0].strip()
                )

            # 3. RSI extreme in wrong direction
            rsi = st.get("rsi", 50)
            if pos["direction"] == "LONG" and rsi >= 80:
                exit_reasons.append(f"RSI {rsi:.0f} — extreme overbought, consider taking profits")
            elif pos["direction"] == "SHORT" and rsi <= 20:
                exit_reasons.append(f"RSI {rsi:.0f} — extreme oversold, consider covering")

        with _lock:
            if exit_reasons:
                reason_str = " · ".join(exit_reasons)
                pos["exit_reason"] = reason_str
                if pos.get("auto_entered"):
                    # Queue for auto-close — executed outside the lock to avoid deadlock
                    auto_close_ids.append(pos["id"])
                elif pos["status"] != "exit_recommended":
                    pos["status"] = "exit_recommended"
                    changed = True
            elif not exit_reasons and pos["status"] == "exit_recommended":
                if not pos.get("auto_entered"):
                    # Conditions cleared — revert manually-entered position to open
                    pos["status"]      = "open"
                    pos["exit_reason"] = None
                    changed = True

    if changed:
        with _lock:
            _save_positions()

    # Auto-close positions queued above (called outside lock)
    for pos_id in auto_close_ids:
        sym_label = next((p["symbol"] for p in _positions if p["id"] == pos_id), pos_id)
        if close_position(pos_id):
            with _lock:
                pos = next((p for p in _positions if p["id"] == pos_id), None)
            pnl_str = f"{pos['final_pnl_pct']:+.2f}%" if pos else "?"
            print(f"  ⚡ AUTO-CLOSED: {sym_label} → {pnl_str} ({pos['exit_reason'] if pos else ''})")


# ── Scanner loop ──────────────────────────────────────────────────────────────

def scanner_loop(force: bool = False):
    global _last_scan, _scanning
    _load_all()
    cooldowns: Dict[str, float] = {}

    while True:
        if force or is_market_open():
            with _lock:
                _scanning = True

            print(f"[{datetime.now(ET).strftime('%H:%M:%S')}] Scanning {len(ALL_SYMBOLS)} symbols…")
            try:
                data = fetch_bars(ALL_SYMBOLS)
            except Exception as e:
                print(f"  fetch error: {e}")
                with _lock:
                    _scanning = False
                time.sleep(60)
                continue

            for sym in ALL_SYMBOLS:
                df5 = data.get(sym)
                if df5 is None or df5.empty:
                    continue
                try:
                    st = get_symbol_status(sym, df5)
                except Exception as e:
                    print(f"  {sym}: {e}")
                    continue
                if st is None:
                    continue
                with _lock:
                    _symbol_status[sym] = st

                key  = f"{sym}:{st['direction']}"
                last = cooldowns.get(key, 0)
                if (st["confidence"] >= MIN_CONFIDENCE_PCT
                        and time.time() - last >= ALERT_COOLDOWN_SEC):
                    print(f"  *** {sym} {st['direction']} {st['confidence']}% ***")
                    cooldowns[key] = time.time()
                    _add_signal(st)
                    # Auto-enter position when confidence is very high
                    if st["confidence"] >= AUTO_POSITION_PCT:
                        sig_id = _signals[0]["id"] if _signals else None
                        if sig_id and add_position(sig_id, auto_entered=True):
                            print(f"  ⚡ AUTO-POSITION: {sym} {st['direction']} @ ${st['entry']:.2f} ({st['confidence']}% conf)")

            _update_positions(data)
            _check_outcomes()

            with _lock:
                _last_scan = datetime.now(ET).strftime("%I:%M:%S %p")
                _scanning  = False

            print(f"  Done. Next scan in {SCAN_INTERVAL_SEC // 60} min.")
            time.sleep(SCAN_INTERVAL_SEC)

        else:
            wait = next_market_open_seconds()
            h, rem = divmod(wait, 3600)
            print(f"Market closed. Opens in {h}h {rem // 60}m.")
            for _ in range(min(wait // 60, 60)):
                time.sleep(60)


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _card_html(sym: str) -> str:
    with _lock:
        st = _symbol_status.get(sym)
    if st is None:
        return (f'<div class="card"><div class="sym">{sym}</div>'
                f'<div class="price">—</div>'
                f'<span class="badge neutral">scanning</span></div>')
    price, rsi = st.get("entry", 0), st.get("rsi", float("nan"))
    conf, direction = st.get("confidence", 0), st.get("direction", "—")
    tf_aligned = st.get("tf_aligned", 0)
    align_str  = f" · {tf_aligned}/4 TF" if tf_aligned >= 3 else ""
    if conf >= MIN_CONFIDENCE_PCT:
        badge = (f'<span class="badge {"long" if direction=="LONG" else "short"}">'
                 f'{"▲" if direction=="LONG" else "▼"} {direction}</span>')
        conf_str = f"<div class='rsi'>{conf}% conf{align_str}</div>"
    elif conf >= 50:
        cls  = "lean-long" if direction == "LONG" else "lean-short"
        icon = "~ long" if direction == "LONG" else "~ short"
        badge, conf_str = f'<span class="badge {cls}">{icon}</span>', ""
    else:
        badge, conf_str = '<span class="badge neutral">neutral</span>', ""
    rsi_str = f"RSI {rsi:.0f}" if rsi == rsi else "RSI —"
    return (f'<div class="card"><div class="sym">{sym}</div>'
            f'<div class="price">${price:,.2f}</div>'
            f'{badge}<div class="rsi">{rsi_str}</div>{conf_str}</div>')


def _outcome_html(outcome, pct):
    pct_str = f" ({'+' if pct and pct>0 else ''}{pct:.2f}%)" if pct is not None else ""
    if outcome == "success":  return f'<span class="out-success">✓ Win{pct_str}</span>'
    if outcome == "failure":  return f'<span class="out-failure">✗ Loss{pct_str}</span>'
    if outcome == "neutral":  return f'<span class="out-neutral">— Flat{pct_str}</span>'
    return '<span class="out-pending">⏳ Pending</span>'


def _history_table_html(signals):
    if not signals:
        return '<div class="empty">No signals yet. Scanner fires when confidence ≥ 62%.</div>'
    rows = []
    for sig in signals[:60]:
        dire  = sig["direction"]
        color = "#3fb950" if dire == "LONG" else "#f85149"
        conf  = sig["confidence"]
        bar   = f'<span class="conf-bar" style="width:{int(conf*.8)}px;background:{color}"></span>{conf}%'
        fired = sig.get("fired_at", "")[:16].replace("T", " ")
        sigs  = "; ".join(s.split("[")[0].strip() for s in sig.get("signals", [])[:2])
        in_pos = sig.get("in_positions", False)
        cb_html = (
            f'<input type="checkbox" class="pos-cb" disabled checked title="Already in positions">'
            if in_pos else
            f'<input type="checkbox" class="pos-cb" onchange="addPosition(\'{sig["id"]}\', this)">'
        )
        rows.append(
            f"<tr>"
            f"<td style='width:28px'>{cb_html}</td>"
            f"<td>{fired}</td>"
            f'<td style="font-weight:700">{sig["symbol"]}</td>'
            f'<td class="{"dir-long" if dire=="LONG" else "dir-short"}">{"▲" if dire=="LONG" else "▼"} {dire}</td>'
            f"<td>{bar}</td>"
            f"<td>${sig.get('entry',0):,.2f}</td>"
            f"<td>{_outcome_html(sig['outcome'], sig.get('pct_move'))}</td>"
            f'<td><div class="sigs">{sigs}</div></td>'
            f"</tr>"
        )
    return ("<table><thead><tr>"
            "<th></th><th>Time</th><th>Symbol</th><th>Direction</th>"
            "<th>Confidence</th><th>Entry</th><th>Outcome</th><th>Key Signals</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _positions_html(positions):
    open_pos = [p for p in positions if p["status"] != "closed"]
    if not open_pos:
        return '<div class="empty-pos">No open positions.<br>Check a signal below to start tracking.</div>'

    rows = []
    for pos in open_pos:
        dire   = pos["direction"]
        pnl    = pos.get("pnl_pct", 0.0)
        cur    = pos.get("current", pos["entry"])
        pnl_cl = "pnl-pos" if pnl >= 0 else "pnl-neg"
        pnl_s  = f"{'+' if pnl >= 0 else ''}{pnl:.2f}%"
        badge  = (f'<span class="badge {"long" if dire=="LONG" else "short"}" style="font-size:10px">'
                  f'{"▲" if dire=="LONG" else "▼"} {dire}</span>')

        # Auto badge
        auto_badge = (' <span style="font-size:9px;background:#f0a500;color:#000;padding:1px 5px;'
                      'border-radius:3px;font-weight:700;vertical-align:middle">⚡ AUTO</span>'
                      if pos.get("auto_entered") else "")

        # Time held
        secs_held = time.time() - _iso_to_epoch(pos["entry_time"])
        h, rem = divmod(int(secs_held), 3600)
        m = rem // 60
        held_str = f"{h}h {m}m" if h else f"{m}m"

        exit_block = ""
        if pos["status"] == "exit_recommended":
            reason = pos.get("exit_reason", "Reversal detected")
            auto_note = " · Auto-exit pending…" if pos.get("auto_entered") else ""
            exit_block = f'<div class="exit-flag">⚠ EXIT SIGNAL{auto_note}<div class="exit-reason">{reason}</div></div>'

        rows.append(
            f'<div class="pos-card {"pos-exit" if pos["status"]=="exit_recommended" else ""}">'
            f'  <div class="pos-header">'
            f'    <span class="pos-sym">{pos["symbol"]}</span>{badge}{auto_badge}'
            f'    <button class="pos-close" onclick="removePos(\'{pos["id"]}\')" title="Close position">×</button>'
            f'  </div>'
            f'  <div class="pos-prices">'
            f'    <div><span class="lbl">Entry</span> ${pos["entry"]:,.2f}</div>'
            f'    <div><span class="lbl">Now</span> ${cur:,.2f} '
            f'      <span class="{pnl_cl}">{pnl_s}</span></div>'
            f'  </div>'
            f'  <div class="pos-meta">Held {held_str} · RSI at entry {pos.get("rsi_entry","—")}</div>'
            f'  {exit_block}'
            f'</div>'
        )
    return "".join(rows)


def _pnl_cell(pnl: float) -> str:
    if pnl > 0:   return f'<span style="color:#3fb950;font-weight:700">+{pnl:.2f}%</span>'
    if pnl < 0:   return f'<span style="color:#f85149;font-weight:700">{pnl:.2f}%</span>'
    return f'<span style="color:#8b949e">0.00%</span>'


def _performance_html() -> str:
    p = _calc_performance()

    if p.get("empty"):
        return ('<div style="text-align:center;color:#8b949e;padding:60px 0;font-size:14px">'
                'No closed positions yet.<br>'
                'Open a position from Signal History, then close it (×) to start tracking performance.'
                '</div>')

    # ── Summary stat cards ────────────────────────────────────────────────────
    pf_str = f"{p['profit_factor']:.2f}" if p['profit_factor'] != float('inf') else "∞"
    wr_color  = "#3fb950" if p["win_rate"] >= 50 else "#f85149"
    exp_color = "#3fb950" if p["expectancy"] >= 0 else "#f85149"

    stat_cards = f"""
    <div class="perf-stats">
      <div class="pstat"><div class="pval">{p['total']}</div><div class="plbl">Closed Trades</div></div>
      <div class="pstat"><div class="pval" style="color:{wr_color}">{p['win_rate']:.1f}%</div><div class="plbl">Win Rate</div></div>
      <div class="pstat"><div class="pval green">+{p['avg_win']:.2f}%</div><div class="plbl">Avg Win</div></div>
      <div class="pstat"><div class="pval red">{p['avg_loss']:.2f}%</div><div class="plbl">Avg Loss</div></div>
      <div class="pstat"><div class="pval" style="color:{exp_color}">{'+' if p['expectancy']>=0 else ''}{p['expectancy']:.2f}%</div><div class="plbl">Expectancy</div></div>
      <div class="pstat"><div class="pval gold">{pf_str}</div><div class="plbl">Profit Factor</div></div>
      <div class="pstat"><div class="pval green">+{p['best']:.2f}%</div><div class="plbl">Best Trade</div></div>
      <div class="pstat"><div class="pval red">{p['worst']:.2f}%</div><div class="plbl">Worst Trade</div></div>
      <div class="pstat"><div class="pval" style="color:{'#3fb950' if p['total_pnl']>=0 else '#f85149'}">{'+' if p['total_pnl']>=0 else ''}{p['total_pnl']:.2f}%</div><div class="plbl">Total P&amp;L</div></div>
    </div>"""

    # ── W/L bar ───────────────────────────────────────────────────────────────
    total = p["total"]
    w_pct = p["wins"] / total * 100
    l_pct = p["losses"] / total * 100
    f_pct = p["flat"] / total * 100
    wl_bar = f"""
    <div class="wl-bar-wrap">
      <div class="wl-bar">
        <div style="width:{w_pct:.1f}%;background:#3fb950" title="{p['wins']} wins"></div>
        <div style="width:{f_pct:.1f}%;background:#444c56" title="{p['flat']} flat"></div>
        <div style="width:{l_pct:.1f}%;background:#f85149" title="{p['losses']} losses"></div>
      </div>
      <div class="wl-labels">
        <span style="color:#3fb950">▐ {p['wins']} Wins</span>
        <span style="color:#8b949e">  {p['flat']} Flat</span>
        <span style="color:#f85149">  {p['losses']} Losses</span>
      </div>
    </div>"""

    # ── By direction ──────────────────────────────────────────────────────────
    dir_rows = "".join(
        f"<tr><td><b>{'▲' if r['direction']=='LONG' else '▼'}</b> "
        f"<span class='{'dir-long' if r['direction']=='LONG' else 'dir-short'}'>{r['direction']}</span></td>"
        f"<td>{r['trades']}</td>"
        f"<td>{r['wins']}</td>"
        f"<td>{r['trades']-r['wins']}</td>"
        f"<td><b>{r['win_rate']:.1f}%</b></td>"
        f"<td>{_pnl_cell(r['avg_pnl'])}</td></tr>"
        for r in p["by_direction"]
    )
    dir_table = f"""
    <table><thead><tr>
      <th>Direction</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Avg P&amp;L</th>
    </tr></thead><tbody>{dir_rows}</tbody></table>"""

    # ── By symbol ─────────────────────────────────────────────────────────────
    sym_rows = "".join(
        f"<tr><td style='font-weight:700'>{r['symbol']}</td>"
        f"<td>{r['trades']}</td>"
        f"<td style='color:#3fb950'>{r['wins']}</td>"
        f"<td style='color:#f85149'>{r['losses']}</td>"
        f"<td><b>{r['win_rate']:.1f}%</b></td>"
        f"<td>{_pnl_cell(r['avg_pnl'])}</td>"
        f"<td>{_pnl_cell(r['total_pnl'])}</td></tr>"
        for r in p["by_symbol"]
    )
    sym_table = f"""
    <table><thead><tr>
      <th>Symbol</th><th>Trades</th><th>Wins</th><th>Losses</th>
      <th>Win Rate</th><th>Avg P&amp;L</th><th>Total P&amp;L</th>
    </tr></thead><tbody>{sym_rows}</tbody></table>"""

    # ── Closed trades log ─────────────────────────────────────────────────────
    trade_rows = []
    for t in p["trades"]:
        dire      = t["direction"]
        pnl       = t.get("final_pnl_pct", 0.0)
        entry_t   = t.get("entry_time","")[:16].replace("T"," ")
        close_t   = t.get("closed_at","")[:16].replace("T"," ")
        entry_p   = t.get("entry", 0)
        exit_p    = t.get("exit_price", entry_p)
        held_s    = _iso_to_epoch(t.get("closed_at","")) - _iso_to_epoch(t.get("entry_time",""))
        h, rem    = divmod(max(0,int(held_s)), 3600)
        held_str  = f"{h}h {rem//60}m" if h else f"{rem//60}m"
        closed_by  = t.get("closed_by", "user")
        auto_flag  = (' <span style="font-size:9px;background:#f0a500;color:#000;padding:1px 4px;'
                      'border-radius:3px;font-weight:700">⚡AUTO</span>'
                      if t.get("auto_entered") else "")
        if closed_by == "auto_exit":
            reason = t.get("exit_reason") or "Auto-exit signal"
        elif closed_by == "exit_signal":
            reason = t.get("exit_reason") or "Exit signal"
        else:
            reason = "Manual close"
        row_bg    = "background:#0d2a0d" if pnl > 0 else ("background:#2a0d0d" if pnl < 0 else "")
        trade_rows.append(
            f'<tr style="{row_bg}">'
            f"<td>{entry_t}</td>"
            f'<td style="font-weight:700">{t["symbol"]}{auto_flag}</td>'
            f'<td class="{"dir-long" if dire=="LONG" else "dir-short"}">{"▲" if dire=="LONG" else "▼"} {dire}</td>'
            f"<td>${entry_p:,.2f}</td>"
            f"<td>${exit_p:,.2f}</td>"
            f"<td>{_pnl_cell(pnl)}</td>"
            f"<td>{held_str}</td>"
            f'<td><div class="sigs">{reason[:60]}</div></td>'
            f"</tr>"
        )

    trades_table = ("<table><thead><tr>"
        "<th>Opened</th><th>Symbol</th><th>Direction</th><th>Entry</th>"
        "<th>Exit</th><th>P&amp;L</th><th>Held</th><th>Reason</th>"
        "</tr></thead><tbody>" + "".join(trade_rows) + "</tbody></table>")

    return f"""
    {stat_cards}
    {wl_bar}
    <div class="perf-grid">
      <div>
        <h2 class="sec" style="margin-bottom:10px">By Direction</h2>
        {dir_table}
      </div>
      <div>
        <h2 class="sec" style="margin-bottom:10px">By Symbol</h2>
        {sym_table}
      </div>
    </div>
    <div style="margin-top:24px">
      <h2 class="sec" style="margin-bottom:10px">Closed Trades</h2>
      {trades_table}
    </div>
    """


# ── Full page ─────────────────────────────────────────────────────────────────

PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MagScanner</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:#0d1117;color:#e6edf3;min-height:100vh}}
header{{background:#161b22;border-bottom:1px solid #30363d;
        padding:12px 24px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
header h1{{font-size:17px;font-weight:700;letter-spacing:1px}}
.tabs{{display:flex;gap:2px;margin-left:20px}}
.tab{{padding:5px 14px;border-radius:6px;font-size:12px;font-weight:600;
      cursor:pointer;border:none;background:none;color:#8b949e;letter-spacing:.3px}}
.tab:hover{{background:#21262d;color:#e6edf3}}
.tab.active{{background:#21262d;color:#58a6ff}}
.dot{{width:10px;height:10px;border-radius:50%}}
.dot.live{{background:#3fb950;box-shadow:0 0 6px #3fb950}}
.dot.closed{{background:#f85149}}
.dot.scanning{{background:#d29922;animation:pulse .8s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.meta{{font-size:12px;color:#8b949e;margin-left:auto}}

/* layout */
.layout{{display:flex;gap:0;height:calc(100vh - 50px)}}
.main{{flex:1;overflow-y:auto;padding:20px 24px;min-width:0}}
.sidebar{{width:290px;flex-shrink:0;background:#0d1117;
          border-left:1px solid #30363d;
          display:flex;flex-direction:column;overflow:hidden}}
.sidebar-header{{padding:14px 16px;border-bottom:1px solid #30363d;
                 display:flex;align-items:center;justify-content:space-between}}
.sidebar-header h2{{font-size:12px;font-weight:700;letter-spacing:.8px;
                    color:#8b949e;text-transform:uppercase}}
.pos-count{{background:#21262d;border-radius:10px;
            padding:2px 8px;font-size:11px;color:#8b949e}}
.sidebar-body{{flex:1;overflow-y:auto;padding:12px}}

/* stats */
.stats{{display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;
       padding:12px 16px;min-width:100px}}
.stat .val{{font-size:24px;font-weight:700;margin-bottom:2px}}
.stat .lbl{{font-size:10px;color:#8b949e;text-transform:uppercase}}
.green{{color:#3fb950}}.red{{color:#f85149}}.gold{{color:#d29922}}

/* cards */
h2.sec{{font-size:11px;font-weight:600;letter-spacing:.8px;color:#8b949e;
         text-transform:uppercase;margin-bottom:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));
       gap:8px;margin-bottom:20px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;
       padding:10px 12px}}
.card .sym{{font-size:13px;font-weight:700;margin-bottom:3px}}
.card .price{{font-size:12px;color:#8b949e;margin-bottom:5px}}
.card .rsi{{font-size:11px;color:#8b949e;margin-top:3px}}
.badge{{display:inline-block;font-size:10px;font-weight:700;
        padding:2px 7px;border-radius:4px}}
.badge.long{{background:#0d4a1e;color:#3fb950}}
.badge.short{{background:#4a0d0d;color:#f85149}}
.badge.lean-long{{background:#1a2e1a;color:#56a86b}}
.badge.lean-short{{background:#2e1a1a;color:#a85656}}
.badge.neutral{{background:#21262d;color:#8b949e}}

/* table */
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:7px 10px;color:#8b949e;font-size:10px;
    font-weight:600;text-transform:uppercase;letter-spacing:.5px;
    border-bottom:1px solid #30363d}}
td{{padding:9px 10px;border-bottom:1px solid #21262d;vertical-align:middle}}
tr:hover td{{background:#161b22}}
.dir-long{{color:#3fb950;font-weight:700}}
.dir-short{{color:#f85149;font-weight:700}}
.out-success{{color:#3fb950;font-weight:700}}
.out-failure{{color:#f85149;font-weight:700}}
.out-neutral{{color:#8b949e}}
.out-pending{{color:#d29922}}
.conf-bar{{display:inline-block;height:5px;border-radius:3px;
           margin-right:5px;vertical-align:middle}}
.sigs{{font-size:10px;color:#8b949e}}
.empty{{color:#8b949e;font-size:12px;padding:20px 0;text-align:center}}
.section{{margin-bottom:28px}}

/* checkbox */
.pos-cb{{width:15px;height:15px;cursor:pointer;accent-color:#58a6ff}}
.pos-cb:disabled{{opacity:.5;cursor:default}}

/* position cards */
.empty-pos{{color:#8b949e;font-size:12px;text-align:center;
            padding:24px 12px;line-height:1.6}}
.pos-card{{background:#161b22;border:1px solid #30363d;border-radius:8px;
           padding:12px;margin-bottom:10px}}
.pos-card.pos-exit{{border-color:#d29922;background:#1a1800}}
.pos-header{{display:flex;align-items:center;gap:6px;margin-bottom:8px}}
.pos-sym{{font-size:14px;font-weight:700;flex:1}}
.pos-close{{background:none;border:none;color:#8b949e;font-size:18px;
            cursor:pointer;padding:0 2px;line-height:1}}
.pos-close:hover{{color:#f85149}}
.pos-prices{{font-size:12px;display:flex;flex-direction:column;gap:3px;
             margin-bottom:6px}}
.pos-prices .lbl{{color:#8b949e;margin-right:4px}}
.pnl-pos{{color:#3fb950;font-weight:700}}
.pnl-neg{{color:#f85149;font-weight:700}}
.pos-meta{{font-size:10px;color:#8b949e;margin-bottom:6px}}
.exit-flag{{background:#2a1f00;border:1px solid #d29922;border-radius:6px;
            padding:8px;margin-top:6px}}
.exit-flag{{color:#d29922;font-size:11px;font-weight:700}}
.exit-reason{{color:#e6a817;font-size:10px;font-weight:400;margin-top:3px;line-height:1.4}}

/* performance tab */
.perf-stats{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px}}
.pstat{{background:#161b22;border:1px solid #30363d;border-radius:8px;
        padding:12px 16px;min-width:110px;flex:1}}
.pval{{font-size:22px;font-weight:700;margin-bottom:2px}}
.plbl{{font-size:10px;color:#8b949e;text-transform:uppercase}}
.wl-bar-wrap{{margin-bottom:24px}}
.wl-bar{{display:flex;height:12px;border-radius:6px;overflow:hidden;margin-bottom:6px}}
.wl-bar div{{transition:width .3s}}
.wl-labels{{font-size:12px;display:flex;gap:16px}}
.perf-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}}
@media(max-width:900px){{.perf-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<header>
  <div class="dot {dot_class}"></div>
  <h1>MAG SCANNER</h1>
  <div class="tabs">
    <button class="tab active" onclick="showTab('dashboard',this)">Dashboard</button>
    <button class="tab" onclick="showTab('performance',this)">Performance</button>
  </div>
  <span class="meta">
    {market_status} &nbsp;·&nbsp; Last scan: {last_scan} &nbsp;·&nbsp; Auto-refresh: 30s
  </span>
</header>

<div class="layout">

<!-- ── Main ── -->
<div class="main" id="main">

<!-- Dashboard tab -->
<div id="tab-dashboard">
  <div class="stats">
    <div class="stat"><div class="val">{total_signals}</div><div class="lbl">Signals</div></div>
    <div class="stat"><div class="val green">{success_count}</div><div class="lbl">Wins</div></div>
    <div class="stat"><div class="val red">{failure_count}</div><div class="lbl">Losses</div></div>
    <div class="stat"><div class="val gold">{win_rate}</div><div class="lbl">Win Rate</div></div>
    <div class="stat"><div class="val gold">{pending_count}</div><div class="lbl">Pending</div></div>
  </div>

  <div class="section">
    <h2 class="sec">Mag 7</h2>
    <div class="grid">{mag7_cards}</div>
    <h2 class="sec">ETFs</h2>
    <div class="grid">{etf_cards}</div>
    <h2 class="sec">AI &amp; Infra</h2>
    <div class="grid">{ai_cards}</div>
  </div>

  <div class="section">
    <h2 class="sec">Signal History &nbsp;<span style="color:#444c56;font-weight:400">— check ☐ to track as open position</span></h2>
    {history_table}
  </div>

</div><!-- /tab-dashboard -->

<!-- Performance tab (hidden by default) -->
<div id="tab-performance" style="display:none">
  {performance_html}
</div>

</div><!-- /main -->

<!-- ── Sidebar ── -->
<div class="sidebar">
  <div class="sidebar-header">
    <h2>My Positions</h2>
    <span class="pos-count" id="pos-count">{pos_count}</span>
  </div>
  <div class="sidebar-body" id="pos-body">
    {positions_html}
  </div>
</div>

</div><!-- /layout -->

<script>
// Auto-refresh page every 30s (preserves active tab)
let _activeTab = localStorage.getItem('mag_tab') || 'dashboard';
setTimeout(() => location.reload(), 30000);

// Restore tab on load
window.addEventListener('DOMContentLoaded', () => {{
  if (_activeTab === 'performance') {{
    document.getElementById('tab-dashboard').style.display   = 'none';
    document.getElementById('tab-performance').style.display = 'block';
    document.querySelectorAll('.tab').forEach(t => {{
      t.classList.toggle('active', t.textContent.trim().toLowerCase() === 'performance');
    }});
  }}
}});

function showTab(name, btn) {{
  document.getElementById('tab-dashboard').style.display   = name === 'dashboard'   ? 'block' : 'none';
  document.getElementById('tab-performance').style.display = name === 'performance' ? 'block' : 'none';
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  localStorage.setItem('mag_tab', name);
  _activeTab = name;
}}

function addPosition(signalId, cb) {{
  cb.disabled = true;
  fetch('/api/position/add', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id: signalId}})
  }})
  .then(r => r.json())
  .then(d => {{
    if (d.ok) {{
      cb.checked = true;
      cb.title   = 'Added to positions';
      return _refreshSidebar();
    }} else {{
      cb.disabled = false;
    }}
  }})
  .catch(() => {{ cb.disabled = false; }});
}}

function removePos(posId) {{
  fetch('/api/position/close', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id: posId}})
  }})
  .then(r => r.json())
  .then(() => _refreshSidebar());
}}

function _refreshSidebar() {{
  return fetch('/api/positions').then(r => r.json()).then(data => {{
    document.getElementById('pos-body').innerHTML    = data.html;
    document.getElementById('pos-count').textContent = data.count;
  }});
}}

// Keep sidebar live every 10s
setInterval(_refreshSidebar, 10000);
</script>

</body>
</html>
"""


def _build_page() -> str:
    with _lock:
        signals   = list(_signals)
        positions = list(_positions)
        last_scan = _last_scan or "—"
        scanning  = _scanning

    resolved   = [s for s in signals if s["outcome"] in ("success", "failure")]
    successes  = sum(1 for s in resolved if s["outcome"] == "success")
    failures   = len(resolved) - successes
    win_rate   = f"{successes / len(resolved) * 100:.0f}%" if resolved else "—"
    pending    = sum(1 for s in signals if s["outcome"] == "pending")
    open_count = sum(1 for p in positions if p["status"] != "closed")

    if scanning:
        dot_class, market_status = "scanning", "Scanning…"
    elif is_market_open():
        dot_class, market_status = "live", "Market Open"
    else:
        secs = next_market_open_seconds()
        h, rem = divmod(secs, 3600)
        dot_class = "closed"
        market_status = f"Market Closed — opens in {h}h {rem // 60}m"

    return PAGE.format(
        dot_class      = dot_class,
        market_status  = market_status,
        last_scan      = last_scan,
        total_signals  = len(signals),
        success_count  = successes,
        failure_count  = failures,
        win_rate       = win_rate,
        pending_count  = pending,
        pos_count      = open_count,
        mag7_cards      = "".join(_card_html(s) for s in MAG7),
        etf_cards       = "".join(_card_html(s) for s in ETFS),
        ai_cards        = "".join(_card_html(s) for s in AI_INFRA),
        history_table   = _history_table_html(signals),
        positions_html  = _positions_html(positions),
        performance_html= _performance_html(),
    )


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send_json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            try:
                return json.loads(self.rfile.read(length))
            except Exception:
                pass
        return {}

    def do_GET(self):
        if self.path == "/":
            body = _build_page().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/positions":
            with _lock:
                positions = list(_positions)
            count = sum(1 for p in positions if p["status"] != "closed")
            self._send_json({"html": _positions_html(positions), "count": count})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        data = self._read_body()

        if self.path == "/api/position/add":
            ok = add_position(data.get("id", ""))
            self._send_json({"ok": ok})

        elif self.path == "/api/position/close":
            ok = close_position(data.get("id", ""))
            self._send_json({"ok": ok})

        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Scan even when market is closed")
    args = parser.parse_args()

    t = threading.Thread(target=scanner_loop, args=(args.force,), daemon=True)
    t.start()

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url    = f"http://localhost:{PORT}"
    print(f"MagScanner → {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

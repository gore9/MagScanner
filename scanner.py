#!/usr/bin/env python3
"""
MagScanner — Real-time Mag7 + ETF scanner with popup alerts.

Polls Alpaca every 5 minutes during market hours.
Analyzes 5-min, 10-min, 15-min, and 30-min bars for each symbol.
Higher timeframes carry more weight — 30m confirmation is the strongest filter.
Fires popup alerts for high-confidence long/short setups.

Usage:
    python scanner.py

    Or for testing outside market hours:
    python scanner.py --force
"""

import os
import sys
import queue
import threading
import time
import argparse
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import pytz
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from indicators import compute_all, resample_bars, trend_direction, ema_cross_signal, vwap_cross_signal, rsi_score, volume_score
from patterns import detect_patterns

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

# ── Config ────────────────────────────────────────────────────────────────────

MAG7     = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
ETFS     = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF"]
AI_INFRA = ["AMD", "TSM", "MU", "MRVL", "SMCI", "ASML", "SNDK", "AVGO", "ARM", "INTC"]
ALL_SYMBOLS = MAG7 + ETFS + AI_INFRA

SCAN_INTERVAL_SEC  = 300      # 5 minutes between full scans
MIN_CONFIDENCE_PCT = 62       # minimum confidence % to fire popup
ALERT_COOLDOWN_SEC = 1800     # 30 min: won't re-alert same symbol+direction
DAYS_HISTORY       = 6        # days of 5-min bars to fetch; 6d ≈ 65 30-min bars (EMA50 warmup)
MIN_BARS_NEEDED    = 55       # need at least this many 5-min bars to analyze

ET = pytz.timezone("America/New_York")

# ── Market-hours check ────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:   # Sat/Sun
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def next_market_open_seconds() -> int:
    """Return seconds until next NYSE open."""
    now = datetime.now(ET)
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now > candidate or now.weekday() >= 5:
        # roll to next weekday
        days_ahead = 1
        while True:
            nxt = now + timedelta(days=days_ahead)
            if nxt.weekday() < 5:
                candidate = nxt.replace(hour=9, minute=30, second=0, microsecond=0)
                break
            days_ahead += 1
    return max(0, int((candidate - now).total_seconds()))


# ── Alpaca data fetch ─────────────────────────────────────────────────────────

def fetch_bars(symbols: List[str], days: int = DAYS_HISTORY) -> Dict[str, pd.DataFrame]:
    """
    Fetch `days` of 5-min bars for all symbols from Alpaca.
    Returns {symbol: DataFrame} mapping.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests   import StockBarsRequest
    from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)

    end_dt   = datetime.now(ET)
    start_dt = end_dt - timedelta(days=days + 1)   # +1 for weekends

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_dt,
        end=end_dt,
        feed="iex",
    )

    bars = client.get_stock_bars(request)
    raw  = bars.df

    if raw.empty:
        return {}

    # Flatten multi-index (symbol, timestamp) → separate DataFrames
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()
    else:
        raw = raw.reset_index()

    raw.columns = [c.lower() for c in raw.columns]

    # Normalise timestamp column
    ts_col = next((c for c in raw.columns if "time" in c), raw.columns[0])
    raw = raw.rename(columns={ts_col: "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    if raw["timestamp"].dt.tz is None:
        raw["timestamp"] = raw["timestamp"].dt.tz_localize("UTC")
    raw["timestamp"] = raw["timestamp"].dt.tz_convert(ET)

    # Filter market hours
    t = raw["timestamp"].dt.time
    mopen  = pd.Timestamp("09:30").time()
    mclose = pd.Timestamp("15:55").time()
    raw = raw[(t >= mopen) & (t <= mclose)].copy()

    result: Dict[str, pd.DataFrame] = {}
    sym_col = "symbol" if "symbol" in raw.columns else None

    for sym in symbols:
        if sym_col:
            sub = raw[raw[sym_col] == sym].drop(columns=[sym_col], errors="ignore")
        else:
            sub = raw.copy()

        if sub.empty:
            continue

        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        sub  = sub[[c for c in cols if c in sub.columns]].copy()
        for c in ["open", "high", "low", "close", "volume"]:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")

        sub = sub.dropna(subset=["open", "high", "low", "close"])
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        result[sym] = sub

    return result


# ── Signal scoring ────────────────────────────────────────────────────────────

def score_timeframe(df: pd.DataFrame, timeframe_label: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Score a single timeframe DataFrame.

    Returns:
        (bull_score, bear_score, bull_labels, bear_labels)
    """
    if len(df) < 20:
        return (0, 0, [], [])

    df = compute_all(df)
    last = df.iloc[-1]

    bull        = 0
    bear        = 0
    bull_labels: List[str] = []
    bear_labels: List[str] = []

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_val = last.get("rsi", np.nan)
    rs = rsi_score(rsi_val)
    if rs > 0:
        bull += rs
        bull_labels.append(f"RSI {rsi_val:.0f} — oversold [{timeframe_label}]")
    elif rs < 0:
        bear += abs(rs)
        bear_labels.append(f"RSI {rsi_val:.0f} — overbought [{timeframe_label}]")

    # ── VWAP ─────────────────────────────────────────────────────────────────
    vs = vwap_cross_signal(df)
    if vs > 0:
        bull += vs
        bull_labels.append(f"{'VWAP reclaim' if vs == 2 else 'Above VWAP'} [{timeframe_label}]")
    elif vs < 0:
        bear += abs(vs)
        bear_labels.append(f"{'VWAP breakdown' if vs == -2 else 'Below VWAP'} [{timeframe_label}]")

    # ── EMA Trend ─────────────────────────────────────────────────────────────
    td = trend_direction(last)
    if td == 1:
        bull += 2
        bull_labels.append(f"EMA bullish stack (9>21>50) [{timeframe_label}]")
    elif td == -1:
        bear += 2
        bear_labels.append(f"EMA bearish stack (9<21<50) [{timeframe_label}]")
    else:
        if last["ema9"] > last["ema21"]:
            bull += 1
            bull_labels.append(f"EMA9 > EMA21 [{timeframe_label}]")
        elif last["ema9"] < last["ema21"]:
            bear += 1
            bear_labels.append(f"EMA9 < EMA21 [{timeframe_label}]")
        if last["close"] > last["ema50"]:
            bull += 1
        elif last["close"] < last["ema50"]:
            bear += 1

    # ── EMA Cross ─────────────────────────────────────────────────────────────
    ec = ema_cross_signal(df)
    if ec == 2:
        bull += 3
        bull_labels.append(f"EMA 9/21 bullish cross [{timeframe_label}]")
    elif ec == -2:
        bear += 3
        bear_labels.append(f"EMA 9/21 bearish cross [{timeframe_label}]")

    # ── Relative Volume ────────────────────────────────────────────────────────
    rvol = last.get("rvol", np.nan)
    vs2 = volume_score(rvol)
    if vs2 > 0:
        vol_lbl = f"High volume {rvol:.1f}x avg [{timeframe_label}]"
        bull += vs2
        bear += vs2   # volume amplifies whichever direction
        bull_labels.append(vol_lbl)
        bear_labels.append(vol_lbl)

    # ── Price vs EMA50 (support/resistance context) ──────────────────────────
    if not pd.isna(last["ema50"]):
        atr = last.get("atr", last["close"] * 0.001)
        dist = last["close"] - last["ema50"]
        if 0 < dist < atr * 0.5:
            bull += 1
            bull_labels.append(f"Near EMA50 support [{timeframe_label}]")
        elif -atr * 0.5 < dist < 0:
            bear += 1
            bear_labels.append(f"Below EMA50 resistance [{timeframe_label}]")

    # ── Candlestick Patterns ──────────────────────────────────────────────────
    patterns = detect_patterns(df)
    for pscore, plabel in patterns:
        if pscore > 0:
            bull += pscore
            bull_labels.append(f"{plabel} [{timeframe_label}]")
        elif pscore < 0:
            bear += abs(pscore)
            bear_labels.append(f"{plabel} [{timeframe_label}]")

    return (bull, bear, bull_labels, bear_labels)


def _score_symbol(df5: pd.DataFrame) -> Dict:
    """
    Core scoring — always runs, returns full result dict regardless of confidence.
    Used by both analyze_symbol (signal) and get_symbol_status (card update).

    Timeframe weights (higher TF = more influence on final score):
        5m  × 1.0  — short-term momentum / entry trigger
        10m × 1.3  — near-term trend confirmation
        15m × 1.6  — intraday structure
        30m × 2.0  — broad intraday bias (strongest filter, reduces variability)

    Alignment bonuses:
        All 4 aligned → +4.5  (very high-conviction setup)
        3 of 4 aligned → +2.0 (good confluence, 30m must agree)

    Confidence formula recalibrated for 4-TF weight sum (5.9 vs old 3.9):
        confidence = min(96, int(42 + raw_score * 1.8))
    """
    df10 = resample_bars(df5.copy(), 10)
    df15 = resample_bars(df5.copy(), 15)
    df30 = resample_bars(df5.copy(), 30)

    b5,  br5,  blbl5,  brlbl5  = score_timeframe(df5,  "5m")
    b10, br10, blbl10, brlbl10 = score_timeframe(df10, "10m")
    b15, br15, blbl15, brlbl15 = score_timeframe(df15, "15m")
    b30, br30, blbl30, brlbl30 = score_timeframe(df30, "30m")

    bull_total = b5 * 1.0 + b10 * 1.3 + b15 * 1.6 + b30 * 2.0
    bear_total = br5 * 1.0 + br10 * 1.3 + br15 * 1.6 + br30 * 2.0

    bull_labels_all = blbl5  + blbl10  + blbl15  + blbl30
    bear_labels_all = brlbl5 + brlbl10 + brlbl15 + brlbl30

    # Alignment bonus — rewards convergence across timeframes
    bull_aligned = sum([b5 > br5, b10 > br10, b15 > br15, b30 > br30])
    bear_aligned = sum([br5 > b5, br10 > b10, br15 > b15, br30 > b30])

    if bull_aligned == 4:
        bull_total += 4.5
        bull_labels_all.insert(0, "All 4 timeframes aligned BULLISH")
    elif bull_aligned == 3:
        bull_total += 2.0
        bull_labels_all.insert(0, "3/4 timeframes aligned BULLISH")

    if bear_aligned == 4:
        bear_total += 4.5
        bear_labels_all.insert(0, "All 4 timeframes aligned BEARISH")
    elif bear_aligned == 3:
        bear_total += 2.0
        bear_labels_all.insert(0, "3/4 timeframes aligned BEARISH")

    if bull_total >= bear_total:
        direction  = "LONG"
        raw_score  = bull_total
        raw_labels = bull_labels_all
    else:
        direction  = "SHORT"
        raw_score  = bear_total
        raw_labels = bear_labels_all

    seen: set = set()
    signal_labels: List[str] = []
    for lbl in raw_labels:
        key = lbl.split("[")[0].strip()
        if key not in seen:
            seen.add(key)
            signal_labels.append(lbl)

    # Recalibrated multiplier: old weight-sum 3.9 → new 5.9, scale = 3.9/5.9 ≈ 0.66
    # Old: 42 + score * 2.7  →  New: 42 + score * 1.8
    confidence = min(96, int(42 + raw_score * 1.8))

    df5e     = compute_all(df5)
    last     = df5e.iloc[-1]
    rsi_now  = float(last.get("rsi", float("nan")))
    vwap_now = float(last.get("vwap", float("nan")))

    return {
        "direction":  direction,
        "confidence": confidence,
        "entry":      float(df5.iloc[-1]["close"]),
        "timestamp":  df5.iloc[-1]["timestamp"],
        "rsi":        rsi_now,
        "vwap":       vwap_now,
        "bull_score": round(bull_total, 1),
        "bear_score": round(bear_total, 1),
        "tf_aligned": bull_aligned if bull_total >= bear_total else bear_aligned,
        "signals":    signal_labels[:8] or ["Composite signal across timeframes"],
    }


def analyze_symbol(sym: str, df5: pd.DataFrame) -> Optional[Dict]:
    """
    Returns a signal dict if confidence >= MIN_CONFIDENCE_PCT, else None.
    Always also returns a status dict (for card updates) via second return value.
    Call get_symbol_status() for card data without the confidence gate.
    """
    if len(df5) < MIN_BARS_NEEDED:
        return None
    result = _score_symbol(df5)
    if result["confidence"] < MIN_CONFIDENCE_PCT:
        return None
    result["symbol"] = sym
    return result


def get_symbol_status(sym: str, df5: pd.DataFrame) -> Optional[Dict]:
    """Returns current state for card display — no confidence threshold."""
    if len(df5) < MIN_BARS_NEEDED:
        return None
    result = _score_symbol(df5)
    result["symbol"] = sym
    return result


# ── Popup alert window ────────────────────────────────────────────────────────

def show_alert_popup(signal: Dict, root_ref):
    """
    Create a styled Toplevel popup for a trade signal.
    Must be called from the tkinter main thread.
    """
    import tkinter as tk
    from tkinter import font as tkfont

    LONG_COLOR  = "#00C853"   # vivid green
    SHORT_COLOR = "#FF1744"   # vivid red
    BG          = "#0D1117"   # dark background
    CARD_BG     = "#161B22"   # slightly lighter card
    FG          = "#E6EDF3"   # main text
    DIM         = "#8B949E"   # muted text
    ACCENT      = LONG_COLOR if signal["direction"] == "LONG" else SHORT_COLOR

    popup = tk.Toplevel(root_ref)
    popup.title("MagScanner Alert")
    popup.configure(bg=BG)
    popup.resizable(False, False)
    popup.attributes("-topmost", True)

    # Position: stack popups near top-right corner
    popup.update_idletasks()
    popup.geometry("+%d+%d" % (
        popup.winfo_screenwidth() - 520,
        80 + (len(root_ref._active_popups) * 370) % (popup.winfo_screenheight() - 400)
    ))

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = tk.Frame(popup, bg=ACCENT, padx=20, pady=14)
    hdr.pack(fill="x")

    direction_icon = "▲ LONG" if signal["direction"] == "LONG" else "▼ SHORT"
    tk.Label(hdr, text=direction_icon, font=("Helvetica Neue", 14, "bold"),
             fg=BG, bg=ACCENT).pack(side="left")
    tk.Label(hdr, text=signal["symbol"], font=("Helvetica Neue", 22, "bold"),
             fg=BG, bg=ACCENT).pack(side="left", padx=(12, 0))

    ts_str = signal["timestamp"].strftime("%I:%M %p") if hasattr(signal["timestamp"], "strftime") else str(signal["timestamp"])
    tk.Label(hdr, text=ts_str, font=("Helvetica Neue", 11),
             fg=BG, bg=ACCENT, anchor="e").pack(side="right")

    # ── Price row ─────────────────────────────────────────────────────────────
    price_frame = tk.Frame(popup, bg=CARD_BG, padx=20, pady=10)
    price_frame.pack(fill="x")
    tk.Label(price_frame, text=f"${signal['entry']:.2f}", font=("Helvetica Neue", 28, "bold"),
             fg=FG, bg=CARD_BG).pack(side="left")
    if not pd.isna(signal["rsi"]):
        rsi_col = "#FF9800" if signal["rsi"] > 65 else ("#64B5F6" if signal["rsi"] < 35 else DIM)
        tk.Label(price_frame, text=f"RSI {signal['rsi']:.0f}",
                 font=("Helvetica Neue", 13, "bold"),
                 fg=rsi_col, bg=CARD_BG).pack(side="right", padx=8)

    # ── Confidence bar ────────────────────────────────────────────────────────
    conf_frame = tk.Frame(popup, bg=CARD_BG, padx=20, pady=6)
    conf_frame.pack(fill="x")

    tk.Label(conf_frame, text="CONFIDENCE", font=("Helvetica Neue", 9, "bold"),
             fg=DIM, bg=CARD_BG).pack(anchor="w")

    bar_outer = tk.Frame(conf_frame, bg="#30363D", height=18, width=440)
    bar_outer.pack(anchor="w", pady=(2, 0))
    bar_outer.pack_propagate(False)

    bar_fill_w = int(440 * signal["confidence"] / 100)
    bar_inner = tk.Frame(bar_outer, bg=ACCENT, height=18, width=bar_fill_w)
    bar_inner.place(x=0, y=0)

    tk.Label(conf_frame, text=f"{signal['confidence']}%",
             font=("Helvetica Neue", 16, "bold"), fg=ACCENT, bg=CARD_BG).pack(anchor="e")

    # ── Divider ───────────────────────────────────────────────────────────────
    tk.Frame(popup, bg="#30363D", height=1).pack(fill="x")

    # ── Signal list ───────────────────────────────────────────────────────────
    sig_frame = tk.Frame(popup, bg=CARD_BG, padx=20, pady=12)
    sig_frame.pack(fill="x")

    tk.Label(sig_frame, text="SIGNALS", font=("Helvetica Neue", 9, "bold"),
             fg=DIM, bg=CARD_BG).pack(anchor="w", pady=(0, 4))

    for lbl in signal["signals"]:
        row = tk.Frame(sig_frame, bg=CARD_BG)
        row.pack(anchor="w", pady=1)
        tk.Label(row, text="◆", font=("Helvetica Neue", 8),
                 fg=ACCENT, bg=CARD_BG).pack(side="left")
        tk.Label(row, text=f"  {lbl}", font=("Helvetica Neue", 11),
                 fg=FG, bg=CARD_BG, justify="left").pack(side="left")

    # ── Score detail ──────────────────────────────────────────────────────────
    score_frame = tk.Frame(popup, bg="#0D1117", padx=20, pady=6)
    score_frame.pack(fill="x")
    tk.Label(score_frame,
             text=f"Bull: {signal['bull_score']}  |  Bear: {signal['bear_score']}",
             font=("Helvetica Neue", 9), fg=DIM, bg=BG).pack(anchor="w")

    # ── Footer with timer ─────────────────────────────────────────────────────
    foot = tk.Frame(popup, bg=BG, padx=20, pady=10)
    foot.pack(fill="x")

    countdown_var = tk.StringVar(value="Auto-close: 60s")
    tk.Label(foot, textvariable=countdown_var, font=("Helvetica Neue", 9),
             fg=DIM, bg=BG).pack(side="left")

    tk.Button(foot, text="  CLOSE  ", font=("Helvetica Neue", 10, "bold"),
              bg="#30363D", fg=FG, relief="flat", padx=12, pady=4,
              command=lambda: _close_popup(popup, root_ref)).pack(side="right")

    # Auto-close countdown
    remaining = [60]
    def tick():
        if not popup.winfo_exists():
            return
        remaining[0] -= 1
        countdown_var.set(f"Auto-close: {remaining[0]}s")
        if remaining[0] <= 0:
            _close_popup(popup, root_ref)
        else:
            popup.after(1000, tick)

    popup.after(1000, tick)

    # Track active popups
    root_ref._active_popups.append(popup)
    popup.protocol("WM_DELETE_WINDOW", lambda: _close_popup(popup, root_ref))


def _close_popup(popup, root_ref):
    try:
        if popup in root_ref._active_popups:
            root_ref._active_popups.remove(popup)
        popup.destroy()
    except Exception:
        pass


# ── Scanner loop (runs in background thread) ──────────────────────────────────

class Scanner:
    def __init__(self, event_queue: queue.Queue, force_mode: bool = False):
        self.event_queue   = event_queue   # carries both {"type":"status"} and {"type":"signal"}
        self.force_mode    = force_mode
        self._cooldowns: Dict[str, float] = {}
        self._running = True

    def _in_cooldown(self, sym: str, direction: str) -> bool:
        return (time.time() - self._cooldowns.get(f"{sym}:{direction}", 0)) < ALERT_COOLDOWN_SEC

    def _set_cooldown(self, sym: str, direction: str):
        self._cooldowns[f"{sym}:{direction}"] = time.time()

    def run_once(self):
        print(f"\n[{datetime.now(ET).strftime('%H:%M:%S')}] Scanning {len(ALL_SYMBOLS)} symbols...")
        self.event_queue.put({"type": "scanning"})
        try:
            data = fetch_bars(ALL_SYMBOLS)
        except Exception as e:
            print(f"  ERROR fetching bars: {e}")
            self.event_queue.put({"type": "error", "msg": str(e)})
            return

        fired = 0
        for sym in ALL_SYMBOLS:
            df5 = data.get(sym)
            if df5 is None or df5.empty:
                print(f"  {sym}: no data")
                continue

            try:
                status = get_symbol_status(sym, df5)
            except Exception as e:
                print(f"  {sym}: analysis error — {e}")
                continue

            if status is None:
                continue

            # Always push a status update so the card stays current
            self.event_queue.put({"type": "status", **status})

            # Only fire a popup signal if confidence is high enough and not in cooldown
            if (status["confidence"] >= MIN_CONFIDENCE_PCT
                    and not self._in_cooldown(sym, status["direction"])):
                print(f"  {sym}: *** {status['direction']} {status['confidence']}% *** firing alert")
                self._set_cooldown(sym, status["direction"])
                self.event_queue.put({"type": "signal", **status})
                fired += 1
            else:
                lvl = status["confidence"]
                print(f"  {sym}: {status['direction']} {lvl}%"
                      + (" (cooldown)" if self._in_cooldown(sym, status["direction"]) else ""))

        self.event_queue.put({"type": "scan_done"})
        print(f"  Scan complete. {fired} alert(s) fired.")

    def loop(self):
        while self._running:
            if self.force_mode or is_market_open():
                self.run_once()
                print(f"  Next scan in {SCAN_INTERVAL_SEC // 60} minutes.")
                time.sleep(SCAN_INTERVAL_SEC)
            else:
                wait = next_market_open_seconds()
                if wait > 60:
                    print(f"Market closed. Next open in ~{wait // 3600}h {(wait % 3600) // 60}m.")
                    # Sleep in 60s chunks so we can catch ^C
                    for _ in range(min(wait // 60, 60)):
                        if not self._running:
                            break
                        time.sleep(60)
                else:
                    time.sleep(wait + 5)

    def stop(self):
        self._running = False


# ── Dashboard UI ─────────────────────────────────────────────────────────────

# Palette
BG       = "#0D1117"
CARD_BG  = "#161B22"
CARD_OUT = "#21262D"   # card border / outer
FG       = "#E6EDF3"
DIM      = "#8B949E"
ACCENT   = "#58A6FF"
LONG_C   = "#00C853"
SHORT_C  = "#FF4444"
NEUTRAL  = "#444C56"


def _rsi_color(rsi: float) -> str:
    if pd.isna(rsi):        return DIM
    if rsi < 30:            return "#64B5F6"   # oversold → blue
    if rsi < 45:            return "#81C784"   # approaching oversold → green
    if rsi > 70:            return "#FF7043"   # overbought → orange-red
    if rsi > 55:            return "#FFB74D"   # approaching overbought → amber
    return DIM


class SymbolCard:
    """One card in the dashboard grid, updates in-place when new data arrives."""

    COLS   = 4          # cards per row
    W, H   = 155, 115   # card pixel dimensions

    def __init__(self, parent, sym: str, row: int, col: int):
        import tkinter as tk
        self.sym = sym

        self.frame = tk.Frame(parent, bg=CARD_OUT,
                              width=self.W, height=self.H,
                              highlightthickness=2,
                              highlightbackground=CARD_OUT,
                              highlightcolor=CARD_OUT)
        self.frame.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        self.frame.pack_propagate(False)
        self.frame.grid_propagate(False)

        inner = tk.Frame(self.frame, bg=CARD_BG, padx=10, pady=8)
        inner.place(relx=0, rely=0, relwidth=1, relheight=1)

        tag_color = "#64B5F6" if sym in ETFS else "#A5D6A7"
        self._sym_lbl = tk.Label(inner, text=sym,
                                  font=("Helvetica Neue", 13, "bold"),
                                  fg=tag_color, bg=CARD_BG, anchor="w")
        self._sym_lbl.pack(anchor="w")

        self._price_var = tk.StringVar(value="—")
        tk.Label(inner, textvariable=self._price_var,
                 font=("Helvetica Neue", 18, "bold"),
                 fg=FG, bg=CARD_BG, anchor="w").pack(anchor="w", pady=(2, 0))

        row2 = tk.Frame(inner, bg=CARD_BG)
        row2.pack(anchor="w", fill="x")

        self._rsi_var = tk.StringVar(value="RSI —")
        self._rsi_lbl = tk.Label(row2, textvariable=self._rsi_var,
                                  font=("Helvetica Neue", 10),
                                  fg=DIM, bg=CARD_BG)
        self._rsi_lbl.pack(side="left")

        self._sig_var = tk.StringVar(value="SCANNING")
        self._sig_lbl = tk.Label(inner, textvariable=self._sig_var,
                                  font=("Helvetica Neue", 11, "bold"),
                                  fg=DIM, bg=CARD_BG, anchor="w")
        self._sig_lbl.pack(anchor="w", pady=(4, 0))

        self._conf_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._conf_var,
                 font=("Helvetica Neue", 9),
                 fg=DIM, bg=CARD_BG, anchor="w").pack(anchor="w")

        self._inner = inner

    def update(self, status: Dict):
        """Refresh card with latest data. Called from main thread only."""
        price      = status.get("entry", float("nan"))
        rsi        = status.get("rsi",   float("nan"))
        direction  = status.get("direction", "—")
        confidence = status.get("confidence", 0)

        self._price_var.set(f"${price:,.2f}" if not pd.isna(price) else "—")

        rsi_str = f"RSI {rsi:.0f}" if not pd.isna(rsi) else "RSI —"
        self._rsi_var.set(rsi_str)
        self._rsi_lbl.configure(fg=_rsi_color(rsi))

        if confidence >= MIN_CONFIDENCE_PCT:
            color = LONG_C if direction == "LONG" else SHORT_C
            icon  = "▲ LONG" if direction == "LONG" else "▼ SHORT"
            self._sig_var.set(icon)
            self._sig_lbl.configure(fg=color)
            self._conf_var.set(f"{confidence}% confidence")
            self.frame.configure(highlightbackground=color)
        else:
            # Lean indicator even below threshold
            if direction == "LONG" and confidence >= 50:
                self._sig_var.set("~ leaning long")
                self._sig_lbl.configure(fg="#558B2F")
            elif direction == "SHORT" and confidence >= 50:
                self._sig_var.set("~ leaning short")
                self._sig_lbl.configure(fg="#B71C1C")
            else:
                self._sig_var.set("neutral")
                self._sig_lbl.configure(fg=DIM)
            self._conf_var.set(f"{confidence}%")
            self.frame.configure(highlightbackground=CARD_OUT)


def build_dashboard(root, event_queue: queue.Queue):
    """Build the main dashboard window. Must run on tkinter main thread."""
    import tkinter as tk

    root.title("MagScanner")
    root.configure(bg=BG)
    root.resizable(False, False)
    root._active_popups = []

    # ── Top bar ───────────────────────────────────────────────────────────────
    topbar = tk.Frame(root, bg=ACCENT, padx=16, pady=10)
    topbar.pack(fill="x")

    tk.Label(topbar, text="MAG SCANNER",
             font=("Helvetica Neue", 16, "bold"), fg=BG, bg=ACCENT).pack(side="left")

    status_dot = tk.Label(topbar, text="●", font=("Helvetica Neue", 14),
                          fg=BG, bg=ACCENT)
    status_dot.pack(side="left", padx=(16, 4))

    status_var = tk.StringVar(value="Initializing")
    tk.Label(topbar, textvariable=status_var,
             font=("Helvetica Neue", 11, "bold"), fg=BG, bg=ACCENT).pack(side="left")

    next_var = tk.StringVar(value="")
    tk.Label(topbar, textvariable=next_var,
             font=("Helvetica Neue", 10), fg=BG, bg=ACCENT).pack(side="right")

    # ── Section labels ────────────────────────────────────────────────────────
    def section_label(parent, text):
        tk.Label(parent, text=text,
                 font=("Helvetica Neue", 9, "bold"),
                 fg=DIM, bg=BG).pack(anchor="w", padx=14, pady=(10, 2))

    section_label(root, "MAG 7")

    # ── Mag7 card row ─────────────────────────────────────────────────────────
    mag7_frame = tk.Frame(root, bg=BG)
    mag7_frame.pack(padx=8, fill="x")
    mag7_cards: Dict[str, SymbolCard] = {}
    for i, sym in enumerate(MAG7):
        card = SymbolCard(mag7_frame, sym, row=0, col=i)
        mag7_cards[sym] = card

    section_label(root, "ETFs")

    # ── ETF card row ──────────────────────────────────────────────────────────
    etf_frame = tk.Frame(root, bg=BG)
    etf_frame.pack(padx=8, fill="x")
    etf_cards: Dict[str, SymbolCard] = {}
    for i, sym in enumerate(ETFS):
        card = SymbolCard(etf_frame, sym, row=0, col=i)
        etf_cards[sym] = card

    all_cards = {**mag7_cards, **etf_cards}

    # ── Signal log ────────────────────────────────────────────────────────────
    tk.Frame(root, bg=CARD_OUT, height=1).pack(fill="x", padx=8, pady=(12, 0))
    section_label(root, "SIGNAL LOG")

    log_frame = tk.Frame(root, bg=CARD_BG, padx=12, pady=8)
    log_frame.pack(fill="x", padx=8, pady=(0, 10))

    log_rows: List = []
    MAX_LOG = 6
    for _ in range(MAX_LOG):
        lv = tk.StringVar(value="")
        lbl = tk.Label(log_frame, textvariable=lv,
                       font=("Courier", 10), fg=DIM, bg=CARD_BG, anchor="w")
        lbl.pack(anchor="w", fill="x")
        log_rows.append((lv, lbl))

    log_entries: List[tuple] = []   # (text, color)

    def add_log(text: str, color: str = DIM):
        log_entries.insert(0, (text, color))
        if len(log_entries) > MAX_LOG:
            log_entries.pop()
        for i, (lv, lbl) in enumerate(log_rows):
            if i < len(log_entries):
                lv.set(log_entries[i][0])
                lbl.configure(fg=log_entries[i][1])
            else:
                lv.set("")

    add_log("Waiting for first scan…", DIM)

    # ── Countdown timer ───────────────────────────────────────────────────────
    countdown = [SCAN_INTERVAL_SEC]
    is_scanning = [False]

    def poll():
        """Drain the event queue and refresh UI. Called every 500 ms."""
        try:
            while True:
                ev = event_queue.get_nowait()
                etype = ev.get("type")

                if etype == "scanning":
                    is_scanning[0] = True
                    countdown[0] = SCAN_INTERVAL_SEC
                    add_log(f"[{datetime.now(ET).strftime('%H:%M:%S')}]  Scanning…", DIM)

                elif etype == "scan_done":
                    is_scanning[0] = False

                elif etype == "error":
                    add_log(f"ERROR: {ev.get('msg', '?')}", "#FF6B6B")

                elif etype == "status":
                    sym = ev.get("symbol")
                    if sym in all_cards:
                        all_cards[sym].update(ev)

                elif etype == "signal":
                    sym  = ev.get("symbol", "?")
                    dire = ev.get("direction", "?")
                    conf = ev.get("confidence", 0)
                    ts   = ev.get("timestamp")
                    ts_s = ts.strftime("%H:%M") if hasattr(ts, "strftime") else ""
                    color = LONG_C if dire == "LONG" else SHORT_C
                    icon  = "▲" if dire == "LONG" else "▼"
                    first_sig = ev.get("signals", [""])[0]
                    add_log(f"{ts_s}  {icon} {dire:<5}  {sym:<6}  {conf}%   {first_sig}", color)
                    show_alert_popup(ev, root)

        except queue.Empty:
            pass

        # Clock + status bar
        now = datetime.now(ET)
        if is_market_open():
            status_var.set(f"LIVE  {now.strftime('%H:%M:%S')}")
            status_dot.configure(fg="#00C853")
            if not is_scanning[0]:
                countdown[0] = max(0, countdown[0] - 1)
            mins, secs = divmod(countdown[0], 60)
            next_var.set(f"Next scan  {mins}:{secs:02d}")
        else:
            status_var.set("MARKET CLOSED")
            status_dot.configure(fg="#FF4444")
            secs_until = next_market_open_seconds()
            h, rem = divmod(secs_until, 3600)
            m = rem // 60
            next_var.set(f"Opens in  {h}h {m}m")

        root.after(1000, poll)

    root.after(500, poll)


def main():
    parser = argparse.ArgumentParser(description="MagScanner — real-time trade signal alerts")
    parser.add_argument("--force", action="store_true",
                        help="Run scanner even when market is closed (for testing)")
    args = parser.parse_args()

    if not ALPACA_KEY or not ALPACA_SECRET:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        sys.exit(1)

    event_queue: queue.Queue = queue.Queue()

    scanner = Scanner(event_queue, force_mode=args.force)
    scan_thread = threading.Thread(target=scanner.loop, daemon=True, name="scanner")
    scan_thread.start()

    import tkinter as tk
    root = tk.Tk()
    build_dashboard(root, event_queue)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        scanner.stop()
        print("\nMagScanner stopped.")


if __name__ == "__main__":
    main()

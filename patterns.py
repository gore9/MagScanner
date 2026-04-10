"""
patterns.py — Candlestick pattern detection for MagScanner.

Each detector returns a tuple: (score: int, label: str)
  Positive score = bullish signal
  Negative score = bearish signal
  0              = no pattern

All detectors operate on the last 1-3 bars of a DataFrame.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


PatternResult = Tuple[int, str]


# ── Single-bar patterns ───────────────────────────────────────────────────────

def _body(row: pd.Series) -> float:
    return abs(row["close"] - row["open"])

def _range(row: pd.Series) -> float:
    r = row["high"] - row["low"]
    return r if r > 0 else np.nan

def _upper_wick(row: pd.Series) -> float:
    return row["high"] - max(row["open"], row["close"])

def _lower_wick(row: pd.Series) -> float:
    return min(row["open"], row["close"]) - row["low"]

def _is_bullish(row: pd.Series) -> bool:
    return row["close"] > row["open"]

def _is_bearish(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def hammer(row: pd.Series) -> PatternResult:
    """
    Hammer (bullish): small body at top, lower wick >= 2x body, tiny upper wick.
    Best after a downtrend.
    """
    rng = _range(row)
    if not rng or pd.isna(rng):
        return (0, "")
    body    = _body(row)
    lw      = _lower_wick(row)
    uw      = _upper_wick(row)
    body_pct = body / rng
    if body_pct < 0.35 and lw >= 2 * body and uw <= 0.15 * rng:
        score = 3 if lw >= 3 * body else 2
        return (score, "Hammer")
    return (0, "")


def shooting_star(row: pd.Series) -> PatternResult:
    """
    Shooting Star (bearish): small body at bottom, upper wick >= 2x body, tiny lower wick.
    Best after an uptrend.
    """
    rng = _range(row)
    if not rng or pd.isna(rng):
        return (0, "")
    body    = _body(row)
    uw      = _upper_wick(row)
    lw      = _lower_wick(row)
    body_pct = body / rng
    if body_pct < 0.35 and uw >= 2 * body and lw <= 0.15 * rng:
        score = -(3 if uw >= 3 * body else 2)
        return (score, "Shooting Star")
    return (0, "")


def hanging_man(row: pd.Series) -> PatternResult:
    """
    Hanging Man (bearish): same shape as hammer but appears after an uptrend.
    We tag it separately; context (trend) is checked by the scoring engine.
    """
    rng = _range(row)
    if not rng or pd.isna(rng):
        return (0, "")
    body = _body(row)
    lw   = _lower_wick(row)
    uw   = _upper_wick(row)
    body_pct = body / rng
    # Hanging man is typically bearish (body at top, long lower shadow)
    if body_pct < 0.35 and lw >= 2 * body and uw <= 0.15 * rng and _is_bearish(row):
        return (-2, "Hanging Man")
    return (0, "")


def inverted_hammer(row: pd.Series) -> PatternResult:
    """
    Inverted Hammer (bullish): small body at bottom, upper wick >= 2x body.
    """
    rng = _range(row)
    if not rng or pd.isna(rng):
        return (0, "")
    body = _body(row)
    uw   = _upper_wick(row)
    lw   = _lower_wick(row)
    body_pct = body / rng
    if body_pct < 0.35 and uw >= 2 * body and lw <= 0.15 * rng and _is_bullish(row):
        return (2, "Inverted Hammer")
    return (0, "")


def doji(row: pd.Series) -> PatternResult:
    """
    Doji: body < 5% of range. Indicates indecision — slight score in
    direction of wick (long lower wick → slight bull, long upper wick → slight bear).
    """
    rng = _range(row)
    if not rng or pd.isna(rng):
        return (0, "")
    body = _body(row)
    if body / rng > 0.07:
        return (0, "")
    lw = _lower_wick(row)
    uw = _upper_wick(row)
    if lw > uw * 1.5:
        return (1, "Bullish Doji")
    if uw > lw * 1.5:
        return (-1, "Bearish Doji")
    return (0, "Doji")


# ── Two-bar patterns ─────────────────────────────────────────────────────────

def bullish_engulfing(prev: pd.Series, curr: pd.Series) -> PatternResult:
    """
    Bullish Engulfing: bearish prev candle fully swallowed by bullish curr candle.
    """
    if not _is_bearish(prev) or not _is_bullish(curr):
        return (0, "")
    if curr["open"] < prev["close"] and curr["close"] > prev["open"]:
        body_ratio = _body(curr) / max(_body(prev), 1e-8)
        score = 4 if body_ratio >= 1.8 else 3
        return (score, "Bullish Engulfing")
    return (0, "")


def bearish_engulfing(prev: pd.Series, curr: pd.Series) -> PatternResult:
    """
    Bearish Engulfing: bullish prev candle fully swallowed by bearish curr candle.
    """
    if not _is_bullish(prev) or not _is_bearish(curr):
        return (0, "")
    if curr["open"] > prev["close"] and curr["close"] < prev["open"]:
        body_ratio = _body(curr) / max(_body(prev), 1e-8)
        score = -(4 if body_ratio >= 1.8 else 3)
        return (score, "Bearish Engulfing")
    return (0, "")


def bullish_harami(prev: pd.Series, curr: pd.Series) -> PatternResult:
    """Bullish Harami: large bearish candle contains small bullish candle."""
    if not _is_bearish(prev) or not _is_bullish(curr):
        return (0, "")
    if curr["open"] > prev["close"] and curr["close"] < prev["open"]:
        if _body(curr) < 0.5 * _body(prev):
            return (2, "Bullish Harami")
    return (0, "")


def bearish_harami(prev: pd.Series, curr: pd.Series) -> PatternResult:
    """Bearish Harami: large bullish candle contains small bearish candle."""
    if not _is_bullish(prev) or not _is_bearish(curr):
        return (0, "")
    if curr["open"] < prev["close"] and curr["close"] > prev["open"]:
        if _body(curr) < 0.5 * _body(prev):
            return (-2, "Bearish Harami")
    return (0, "")


def tweezer_bottom(prev: pd.Series, curr: pd.Series) -> PatternResult:
    """Tweezer Bottom: two candles share the same low → bullish reversal."""
    if abs(prev["low"] - curr["low"]) < 0.001 * curr["close"]:
        if _is_bearish(prev) and _is_bullish(curr):
            return (2, "Tweezer Bottom")
    return (0, "")


def tweezer_top(prev: pd.Series, curr: pd.Series) -> PatternResult:
    """Tweezer Top: two candles share the same high → bearish reversal."""
    if abs(prev["high"] - curr["high"]) < 0.001 * curr["close"]:
        if _is_bullish(prev) and _is_bearish(curr):
            return (-2, "Tweezer Top")
    return (0, "")


# ── Three-bar patterns ────────────────────────────────────────────────────────

def morning_star(c0: pd.Series, c1: pd.Series, c2: pd.Series) -> PatternResult:
    """
    Morning Star (3-bar bullish reversal):
      Bar 0: large bearish candle
      Bar 1: small body (star) — gaps down
      Bar 2: large bullish candle closing into bar 0's body
    """
    if not _is_bearish(c0) or not _is_bullish(c2):
        return (0, "")
    if _body(c1) > 0.3 * _body(c0):
        return (0, "")
    if c2["close"] > c0["open"] + (_body(c0) * 0.5):
        return (4, "Morning Star")
    if c2["close"] > c0["close"] + (_body(c0) * 0.3):
        return (3, "Morning Star")
    return (0, "")


def evening_star(c0: pd.Series, c1: pd.Series, c2: pd.Series) -> PatternResult:
    """
    Evening Star (3-bar bearish reversal):
      Bar 0: large bullish candle
      Bar 1: small body (star) — gaps up
      Bar 2: large bearish candle closing into bar 0's body
    """
    if not _is_bullish(c0) or not _is_bearish(c2):
        return (0, "")
    if _body(c1) > 0.3 * _body(c0):
        return (0, "")
    if c2["close"] < c0["open"] - (_body(c0) * 0.5):
        return (-4, "Evening Star")
    if c2["close"] < c0["close"] - (_body(c0) * 0.3):
        return (-3, "Evening Star")
    return (0, "")


def three_white_soldiers(c0: pd.Series, c1: pd.Series, c2: pd.Series) -> PatternResult:
    """Three consecutive bullish candles with higher closes."""
    if _is_bullish(c0) and _is_bullish(c1) and _is_bullish(c2):
        if c1["close"] > c0["close"] and c2["close"] > c1["close"]:
            if c1["open"] > c0["open"] and c2["open"] > c1["open"]:
                return (3, "Three White Soldiers")
    return (0, "")


def three_black_crows(c0: pd.Series, c1: pd.Series, c2: pd.Series) -> PatternResult:
    """Three consecutive bearish candles with lower closes."""
    if _is_bearish(c0) and _is_bearish(c1) and _is_bearish(c2):
        if c1["close"] < c0["close"] and c2["close"] < c1["close"]:
            if c1["open"] < c0["open"] and c2["open"] < c1["open"]:
                return (-3, "Three Black Crows")
    return (0, "")


# ── Main detector ─────────────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> List[PatternResult]:
    """
    Run all pattern detectors on the last 3 bars of `df`.
    Returns list of (score, label) tuples — filtering out zero-score results.
    """
    if len(df) < 3:
        return []

    results: List[PatternResult] = []
    c0, c1, c2 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    # Single-bar (last bar)
    for fn in [hammer, shooting_star, hanging_man, inverted_hammer, doji]:
        r = fn(c2)
        if r[0] != 0:
            results.append(r)

    # Two-bar
    for fn in [bullish_engulfing, bearish_engulfing,
               bullish_harami, bearish_harami,
               tweezer_bottom, tweezer_top]:
        r = fn(c1, c2)
        if r[0] != 0:
            results.append(r)

    # Three-bar
    for fn in [morning_star, evening_star,
               three_white_soldiers, three_black_crows]:
        r = fn(c0, c1, c2)
        if r[0] != 0:
            results.append(r)

    return results

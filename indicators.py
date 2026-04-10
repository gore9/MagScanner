"""
indicators.py — Technical indicator calculations for MagScanner.

Computes per-bar: EMA(9/21/50), VWAP, RSI(14), ATR(14), relative volume.
All functions accept a pd.DataFrame with OHLCV columns and return it enriched.
"""

import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA 9, 21, 50 columns."""
    df = df.copy()
    df["ema9"]  = _ema(df["close"], 9)
    df["ema21"] = _ema(df["close"], 21)
    df["ema50"] = _ema(df["close"], 50)
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Intraday VWAP — resets each calendar date."""
    df = df.copy()
    df["_date"] = df["timestamp"].dt.date
    df["_tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    df["_tpv"]  = df["_tp"] * df["volume"]
    df["_cvol"] = df.groupby("_date")["volume"].cumsum()
    df["_ctpv"] = df.groupby("_date")["_tpv"].cumsum()
    df["vwap"]  = df["_ctpv"] / df["_cvol"].replace(0, np.nan)
    df = df.drop(columns=["_date", "_tp", "_tpv", "_cvol", "_ctpv"])
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder RSI."""
    df = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df


def add_rel_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Relative volume vs same time-of-day average."""
    df = df.copy()
    df["_tod"] = df["timestamp"].dt.strftime("%H:%M")
    df["rvol"] = df.groupby("_tod")["volume"].transform(
        lambda x: x / max(x.mean(), 1)
    )
    df = df.drop(columns=["_tod"])
    return df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run all indicators in sequence."""
    df = add_ema(df)
    df = add_vwap(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_rel_volume(df)
    return df


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 5-min bars into N-minute bars."""
    df = df.set_index("timestamp").sort_index()
    rule = f"{minutes}min"
    agg = df.resample(rule, closed="left", label="left").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",     "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    agg = agg.reset_index()
    return agg


# ── Derived signal helpers ────────────────────────────────────────────────────

def trend_direction(row: pd.Series) -> int:
    """
    +1  = EMA9 > EMA21 > EMA50 (bullish stack)
    -1  = EMA9 < EMA21 < EMA50 (bearish stack)
     0  = mixed
    """
    if row["ema9"] > row["ema21"] and row["ema21"] > row["ema50"]:
        return 1
    if row["ema9"] < row["ema21"] and row["ema21"] < row["ema50"]:
        return -1
    return 0


def ema_cross_signal(df: pd.DataFrame, lookback: int = 3) -> int:
    """
    +2 if EMA9 crossed ABOVE EMA21 within last `lookback` bars.
    -2 if EMA9 crossed BELOW EMA21 within last `lookback` bars.
    """
    if len(df) < lookback + 1:
        return 0
    recent = df.tail(lookback + 1)
    prev_cross = recent.iloc[:-1]["ema9"] - recent.iloc[:-1]["ema21"]
    curr_cross = recent.iloc[-1]["ema9"] - recent.iloc[-1]["ema21"]
    if (prev_cross < 0).any() and curr_cross > 0:
        return 2
    if (prev_cross > 0).any() and curr_cross < 0:
        return -2
    return 0


def vwap_cross_signal(df: pd.DataFrame, lookback: int = 3) -> int:
    """
    +2 if price crossed ABOVE VWAP within last `lookback` bars.
    -2 if price crossed BELOW VWAP within last `lookback` bars.
     ±1 for just being above/below (no recent cross).
    """
    if len(df) < lookback + 1:
        return 0
    recent = df.tail(lookback + 1)
    prev_above = (recent.iloc[:-1]["close"] > recent.iloc[:-1]["vwap"])
    last       = recent.iloc[-1]
    curr_above = last["close"] > last["vwap"]
    if (not prev_above.all()) and curr_above:
        return 2   # just reclaimed VWAP
    if prev_above.all() and (not curr_above):
        return -2  # just lost VWAP
    return 1 if curr_above else -1


def rsi_score(rsi_val: float) -> int:
    """
    Oversold (< 45) → positive score.
    Overbought (> 55) → negative score.
    Neutral zone → 0.
    """
    if pd.isna(rsi_val):
        return 0
    if rsi_val < 20:  return  4
    if rsi_val < 30:  return  3
    if rsi_val < 38:  return  2
    if rsi_val < 45:  return  1
    if rsi_val > 80:  return -4
    if rsi_val > 70:  return -3
    if rsi_val > 62:  return -2
    if rsi_val > 56:  return -1
    return 0  # neutral 45-56 zone → no contribution


def volume_score(rvol_val: float) -> int:
    """High relative volume boosts confidence."""
    if pd.isna(rvol_val):
        return 0
    if rvol_val >= 3.0:  return 2
    if rvol_val >= 1.8:  return 1
    return 0

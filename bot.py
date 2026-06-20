"""
ETH Trading Signal Bot
----------------------
Fetches live ETH/USDT data from Binance and generates
BUY / SELL / HOLD signals with entry price, stop-loss, and take-profit.

Indicators used:
  - EMA 9 / EMA 21  (trend direction)
  - RSI 14          (momentum / overbought-oversold)
  - MACD            (momentum crossover)
  - ATR 14          (volatility — used to size SL/TP)
  - 1h HTF EMA trend filter (optional, used by backtest)

No API key required — uses Binance public REST endpoints.
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

# ─── Config ───────────────────────────────────────────────────────────────────
SYMBOL        = "ETHUSDT"
INTERVAL      = "15m"        # candlestick interval (1m 5m 15m 1h 4h 1d)
LIMIT         = 200          # number of candles to fetch
POLL_SECONDS  = 60           # how often to refresh (seconds)

# Risk/reward multipliers applied to ATR
SL_ATR_MULT   = 1.5          # stop-loss  = entry ± (ATR × 1.5)
TP_ATR_MULT   = 3.0          # take-profit = entry ± (ATR × 3.0)

# RSI thresholds
RSI_OVERSOLD    = 35
RSI_OVERBOUGHT  = 65

# Position sizing (used by backtest)
ACCOUNT_SIZE_USDT   = 1000.0   # change to your account size
RISK_PCT_PER_TRADE  = 0.01     # risk 1% of account per trade
MIN_NOTIONAL_USDT   = 10.0     # minimum trade size

BINANCE_URL = "https://api.binance.com/api/v3/klines"
# ──────────────────────────────────────────────────────────────────────────────


def fetch_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Pull OHLCV candles from Binance public API."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades",
        "taker_base", "taker_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # EMA
    df["ema9"]  = close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()

    return df


def get_htf_trend(df: pd.DataFrame) -> str:
    """
    Determine higher-timeframe trend from a pre-computed dataframe.
    Returns 'BULL', 'BEAR', or 'FLAT'.
    Used by bot for the optional HTF filter and by the backtester.
    """
    last = df.iloc[-1]
    if last["ema9"] > last["ema21"] and last["close"] > last["ema21"]:
        return "BULL"
    elif last["ema9"] < last["ema21"] and last["close"] < last["ema21"]:
        return "BEAR"
    else:
        return "FLAT"


def generate_signal(df: pd.DataFrame, htf_trend: str = None) -> dict:
    """
    Scoring system — each indicator votes +1 (bullish) or -1 (bearish).
    Score ≥ +2  → BUY
    Score ≤ -2  → SELL
    Otherwise   → HOLD

    Optional htf_trend ('BULL' | 'BEAR' | 'FLAT' | None):
      If provided, BUY signals are suppressed in BEAR trend and
      SELL signals are suppressed in BULL trend.
    """
    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    price = last["close"]
    atr   = last["atr"]
    rsi   = last["rsi"]

    scores  = []
    reasons = []

    # 1. EMA crossover
    if last["ema9"] > last["ema21"]:
        scores.append(1)
        reasons.append("EMA9 > EMA21 (bullish trend)")
    else:
        scores.append(-1)
        reasons.append("EMA9 < EMA21 (bearish trend)")

    # 2. Price vs EMA21
    if price > last["ema21"]:
        scores.append(1)
        reasons.append("Price above EMA21")
    else:
        scores.append(-1)
        reasons.append("Price below EMA21")

    # 3. RSI
    if rsi < RSI_OVERSOLD:
        scores.append(1)
        reasons.append(f"RSI {rsi:.1f} — oversold")
    elif rsi > RSI_OVERBOUGHT:
        scores.append(-1)
        reasons.append(f"RSI {rsi:.1f} — overbought")
    else:
        scores.append(0)
        reasons.append(f"RSI {rsi:.1f} — neutral")

    # 4. MACD histogram direction
    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
        scores.append(1)
        reasons.append("MACD histogram rising (bullish momentum)")
    elif last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]:
        scores.append(-1)
        reasons.append("MACD histogram falling (bearish momentum)")
    else:
        scores.append(0)
        reasons.append("MACD histogram flat/mixed")

    # 5. MACD line vs signal line
    if last["macd"] > last["macd_signal"]:
        scores.append(1)
        reasons.append("MACD above signal line")
    else:
        scores.append(-1)
        reasons.append("MACD below signal line")

    total = sum(scores)

    if total >= 2:
        direction = "BUY"
    elif total <= -2:
        direction = "SELL"
    else:
        direction = "HOLD"

    # Apply HTF trend filter if provided
    if htf_trend == "BEAR" and direction == "BUY":
        direction = "HOLD"
        reasons.append("BUY suppressed — 1h trend is BEAR")
    elif htf_trend == "BULL" and direction == "SELL":
        direction = "HOLD"
        reasons.append("SELL suppressed — 1h trend is BULL")

    if direction == "BUY":
        sl = round(price - atr * SL_ATR_MULT, 2)
        tp = round(price + atr * TP_ATR_MULT, 2)
    elif direction == "SELL":
        sl = round(price + atr * SL_ATR_MULT, 2)
        tp = round(price - atr * TP_ATR_MULT, 2)
    else:
        sl = None
        tp = None

    return {
        "time":      last["open_time"].strftime("%Y-%m-%d %H:%M"),
        "entry":     price,           # entry price (current close)
        "price":     price,           # alias kept for display
        "signal":    direction,
        "score":     total,
        "sl":        sl,
        "tp":        tp,
        "atr":       round(atr, 2),
        "rsi":       round(rsi, 2),
        "ema9":      round(last["ema9"], 2),
        "ema21":     round(last["ema21"], 2),
        "macd":      round(last["macd"], 4),
        "macd_sig":  round(last["macd_signal"], 4),
        "htf_trend": htf_trend,
        "reasons":   reasons,
        "scores":    scores,
    }


def color_signal(signal: str) -> str:
    if signal == "BUY":
        return Fore.GREEN + Style.BRIGHT + signal
    elif signal == "SELL":
        return Fore.RED + Style.BRIGHT + signal
    else:
        return Fore.YELLOW + Style.BRIGHT + signal


def print_report(result: dict):
    sep = "─" * 56
    signal = result["signal"]
    price  = result["price"]
    atr    = result["atr"]

    print(f"\n{Fore.CYAN}{sep}")
    print(f"  ETH/USDT Trading Signal  |  {result['time']}")
    print(f"{Fore.CYAN}{sep}")

    print(f"  Signal     : {color_signal(signal)}  (score {result['score']:+d} / 5)")
    print(f"  Entry Price: {Fore.WHITE}{Style.BRIGHT}${price:,.2f}")

    if signal != "HOLD":
        sl_color = Fore.RED   if signal == "BUY"  else Fore.GREEN
        tp_color = Fore.GREEN if signal == "BUY"  else Fore.RED
        sl_dist  = abs(result["sl"] - price) / price * 100
        tp_dist  = abs(result["tp"] - price) / price * 100
        rr       = tp_dist / sl_dist if sl_dist > 0 else 0

        print(f"  Stop-Loss  : {sl_color}${result['sl']:,.2f}  ({sl_dist:.2f}% from entry)")
        print(f"  Take-Profit: {tp_color}${result['tp']:,.2f}  ({tp_dist:.2f}% from entry)")
        print(f"  Risk/Reward: {Fore.WHITE}{rr:.1f}R")
    else:
        print(f"  Stop-Loss  : —")
        print(f"  Take-Profit: —")

    htf = result.get("htf_trend")
    if htf:
        htf_color = Fore.GREEN if htf == "BULL" else (Fore.RED if htf == "BEAR" else Fore.YELLOW)
        print(f"  1h Trend   : {htf_color}{htf}")

    print(f"\n  {Fore.CYAN}── Indicators ──────────────────────────────────")
    print(f"  ATR(14)    : {atr}")
    print(f"  RSI(14)    : {result['rsi']}")
    print(f"  EMA9/EMA21 : {result['ema9']} / {result['ema21']}")
    print(f"  MACD/Signal: {result['macd']} / {result['macd_sig']}")

    print(f"\n  {Fore.CYAN}── Reasoning ───────────────────────────────────")
    for i, (score, reason) in enumerate(zip(result["scores"], result["reasons"]), 1):
        if score > 0:
            icon = Fore.GREEN + "▲"
        elif score < 0:
            icon = Fore.RED + "▼"
        else:
            icon = Fore.YELLOW + "●"
        print(f"  {icon} {Fore.WHITE}{reason}")

    # Extra reasons (e.g. HTF suppression note) that have no score entry
    for reason in result["reasons"][len(result["scores"]):]:
        print(f"  {Fore.YELLOW}● {Fore.WHITE}{reason}")

    print(f"{Fore.CYAN}{sep}\n")


def main():
    print(Fore.CYAN + Style.BRIGHT + """
  ╔═══════════════════════════════════════╗
  ║       ETH/USDT  Trading Signal Bot    ║
  ║   Binance  |  EMA · RSI · MACD · ATR  ║
  ╚═══════════════════════════════════════╝
""")
    print(f"  Interval : {INTERVAL}  |  Refresh every {POLL_SECONDS}s")
    print(f"  SL mult  : {SL_ATR_MULT}× ATR  |  TP mult : {TP_ATR_MULT}× ATR")
    print(f"  Press Ctrl+C to stop.\n")

    while True:
        try:
            df     = fetch_candles(SYMBOL, INTERVAL, LIMIT)
            df     = compute_indicators(df)
            result = generate_signal(df)
            print_report(result)

        except requests.exceptions.RequestException as e:
            print(Fore.RED + f"  [Network error] {e}")
        except Exception as e:
            print(Fore.RED + f"  [Error] {e}")

        print(f"  Next refresh in {POLL_SECONDS}s …")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

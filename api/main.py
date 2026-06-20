"""
ETH Trading Signal Bot — FastAPI Backend
Deploy on Railway / Render (free tier).

Signal lifecycle:
  IDLE      → no active trade, scanning for next signal
  ACTIVE    → trade open, watching price vs SL/TP
  CLOSED    → last trade just closed (TP hit or SL hit), shown briefly
              then resets to IDLE to scan for the next signal
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import threading
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─── Config ───────────────────────────────────────────────────────────────────
BINANCE_URL    = "https://api.binance.com/api/v3/klines"
TICKER_URL     = "https://api.binance.com/api/v3/ticker/price"
SYMBOL         = "ETHUSDT"
INTERVAL       = "15m"
LIMIT          = 200
SL_ATR_MULT    = 1.5
TP_ATR_MULT    = 3.0
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65
POLL_SECONDS   = 60   # how often the background loop runs
# ──────────────────────────────────────────────────────────────────────────────

# ─── Shared state (protected by a lock) ───────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "status":       "IDLE",       # IDLE | ACTIVE | CLOSED
    "signal":       None,         # last generated signal dict
    "trade":        None,         # active trade dict
    "last_signal":  None,         # last BUY/SELL signal (for history)
    "history":      [],           # list of closed trades
    "last_update":  None,
    "error":        None,
}
# ──────────────────────────────────────────────────────────────────────────────


def fetch_candles():
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def get_live_price():
    resp = requests.get(TICKER_URL, params={"symbol": SYMBOL}, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])


def compute_indicators(df):
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    df["ema9"]  = close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()

    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()

    return df


def generate_signal(df):
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = last["close"]
    atr   = last["atr"]
    rsi   = last["rsi"]

    scores  = []
    reasons = []

    if last["ema9"] > last["ema21"]:
        scores.append(1);  reasons.append("EMA9 > EMA21 — bullish trend")
    else:
        scores.append(-1); reasons.append("EMA9 < EMA21 — bearish trend")

    if price > last["ema21"]:
        scores.append(1);  reasons.append("Price above EMA21")
    else:
        scores.append(-1); reasons.append("Price below EMA21")

    if rsi < RSI_OVERSOLD:
        scores.append(1);  reasons.append(f"RSI {rsi:.1f} — oversold")
    elif rsi > RSI_OVERBOUGHT:
        scores.append(-1); reasons.append(f"RSI {rsi:.1f} — overbought")
    else:
        scores.append(0);  reasons.append(f"RSI {rsi:.1f} — neutral")

    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
        scores.append(1);  reasons.append("MACD histogram rising — bullish momentum")
    elif last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]:
        scores.append(-1); reasons.append("MACD histogram falling — bearish momentum")
    else:
        scores.append(0);  reasons.append("MACD histogram flat/mixed")

    if last["macd"] > last["macd_signal"]:
        scores.append(1);  reasons.append("MACD above signal line")
    else:
        scores.append(-1); reasons.append("MACD below signal line")

    total = sum(scores)

    if total >= 2:
        direction = "BUY"
        sl = round(price - atr * SL_ATR_MULT, 2)
        tp = round(price + atr * TP_ATR_MULT, 2)
    elif total <= -2:
        direction = "SELL"
        sl = round(price + atr * SL_ATR_MULT, 2)
        tp = round(price - atr * TP_ATR_MULT, 2)
    else:
        direction = "HOLD"
        sl = None
        tp = None

    sl_dist = round(abs(sl - price) / price * 100, 2) if sl else None
    tp_dist = round(abs(tp - price) / price * 100, 2) if tp else None
    rr      = round(tp_dist / sl_dist, 1) if (sl_dist and sl_dist > 0) else None

    return {
        "symbol":      SYMBOL,
        "interval":    INTERVAL,
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "signal":      direction,
        "score":       total,
        "entry":       round(price, 2),
        "sl":          sl,
        "tp":          tp,
        "sl_pct":      sl_dist,
        "tp_pct":      tp_dist,
        "rr":          rr,
        "rsi":         round(rsi, 2),
        "atr":         round(atr, 2),
        "ema9":        round(last["ema9"], 2),
        "ema21":       round(last["ema21"], 2),
        "macd":        round(last["macd"], 4),
        "macd_signal": round(last["macd_signal"], 4),
        "reasons":     [{"text": r, "score": s} for r, s in zip(reasons, scores)],
    }


def check_trade_closed(trade: dict, live_price: float) -> str | None:
    """
    Returns 'TP', 'SL', or None depending on whether price has
    crossed the take-profit or stop-loss of the active trade.
    """
    sig = trade["signal"]
    sl  = trade["sl"]
    tp  = trade["tp"]

    if sig == "BUY":
        if live_price >= tp:
            return "TP"
        if live_price <= sl:
            return "SL"
    elif sig == "SELL":
        if live_price <= tp:
            return "TP"
        if live_price >= sl:
            return "SL"
    return None


def pnl_pct(trade: dict, exit_price: float) -> float:
    entry = trade["entry"]
    if trade["signal"] == "BUY":
        return round((exit_price - entry) / entry * 100, 2)
    else:
        return round((entry - exit_price) / entry * 100, 2)


# ─── Background loop ──────────────────────────────────────────────────────────
def background_loop():
    global _state
    while True:
        try:
            with _lock:
                status = _state["status"]

            if status == "IDLE":
                # Scan for a new signal
                df     = fetch_candles()
                df     = compute_indicators(df)
                sig    = generate_signal(df)

                with _lock:
                    _state["signal"]      = sig
                    _state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    _state["error"]       = None

                    if sig["signal"] in ("BUY", "SELL"):
                        # Open a new trade
                        _state["status"] = "ACTIVE"
                        _state["trade"]  = {
                            "signal":     sig["signal"],
                            "entry":      sig["entry"],
                            "sl":         sig["sl"],
                            "tp":         sig["tp"],
                            "sl_pct":     sig["sl_pct"],
                            "tp_pct":     sig["tp_pct"],
                            "rr":         sig["rr"],
                            "opened_at":  sig["timestamp"],
                            "live_price": sig["entry"],
                            "unrealised_pct": 0.0,
                        }

            elif status == "ACTIVE":
                # Check if SL or TP was hit
                live = get_live_price()
                trade = None
                with _lock:
                    trade = dict(_state["trade"])

                result = check_trade_closed(trade, live)

                with _lock:
                    if _state["trade"]:
                        _state["trade"]["live_price"]      = round(live, 2)
                        _state["trade"]["unrealised_pct"]  = pnl_pct(trade, live)
                    _state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                if result:
                    closed_trade = dict(trade)
                    closed_trade["exit_price"]  = round(live, 2)
                    closed_trade["exit_reason"] = result          # 'TP' or 'SL'
                    closed_trade["pnl_pct"]     = pnl_pct(trade, live)
                    closed_trade["closed_at"]   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                    with _lock:
                        _state["status"]      = "CLOSED"
                        _state["trade"]       = closed_trade
                        _state["last_signal"] = closed_trade
                        _state["history"].insert(0, closed_trade)
                        _state["history"]     = _state["history"][:20]  # keep last 20

                    # Wait 10s so frontend can show the closed result, then reset to IDLE
                    time.sleep(10)
                    with _lock:
                        _state["status"] = "IDLE"
                        _state["trade"]  = None

            elif status == "CLOSED":
                # Handled above via sleep; just wait
                pass

        except Exception as e:
            with _lock:
                _state["error"] = str(e)

        time.sleep(POLL_SECONDS)


# Start background thread on startup
_thread = threading.Thread(target=background_loop, daemon=True)
_thread.start()
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ETH Signal Bot API is running"}


@app.get("/signal")
def get_signal():
    with _lock:
        state = dict(_state)

    if state["error"]:
        return {"ok": False, "error": state["error"]}

    return {
        "ok":          True,
        "status":      state["status"],        # IDLE | ACTIVE | CLOSED
        "signal":      state["signal"],        # latest indicator snapshot
        "trade":       state["trade"],         # active or just-closed trade
        "history":     state["history"],       # last 20 closed trades
        "last_update": state["last_update"],
    }

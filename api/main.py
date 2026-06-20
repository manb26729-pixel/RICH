"""
ETH Trading Signal Bot — FastAPI Backend
No pandas/numpy — pure Python calculations.
Zero build issues, works on any Python version.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import threading
import time
import math
from datetime import datetime, timezone

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
POLL_SECONDS   = 60
# ──────────────────────────────────────────────────────────────────────────────


# ─── Pure Python indicator math ───────────────────────────────────────────────

def ema(values: list, span: int) -> list:
    """Exponential moving average."""
    k = 2.0 / (span + 1)
    result = [None] * len(values)
    for i, v in enumerate(values):
        if v is None:
            continue
        if result[i - 1] is None:
            result[i] = v
        else:
            result[i] = v * k + result[i - 1] * (1 - k)
    return result


def ema_com(values: list, com: int) -> list:
    """EMA with center-of-mass (alpha = 1/(1+com))."""
    k = 1.0 / (1 + com)
    result = [None] * len(values)
    for i, v in enumerate(values):
        if v is None:
            continue
        if result[i - 1] is None:
            result[i] = v
        else:
            result[i] = v * k + result[i - 1] * (1 - k)
    return result


def compute_indicators(candles: list) -> dict:
    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]

    # EMA 9 / 21
    ema9  = ema(closes, 9)
    ema21 = ema(closes, 21)

    # RSI 14
    gains  = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    gains  = [None] + gains
    losses = [None] + losses
    avg_gain = ema_com(gains,  13)
    avg_loss = ema_com(losses, 13)
    rsi = []
    for g, l in zip(avg_gain, avg_loss):
        if g is None or l is None:
            rsi.append(None)
        elif l == 0:
            rsi.append(100.0)
        else:
            rsi.append(100 - (100 / (1 + g / l)))

    # MACD (12, 26, 9)
    ema12  = ema(closes, 12)
    ema26  = ema(closes, 26)
    macd_line = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(ema12, ema26)
    ]
    macd_signal = ema(macd_line, 9)
    macd_hist   = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(macd_line, macd_signal)
    ]

    # ATR 14
    tr = [None]
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i]  - closes[i-1])
        tr.append(max(tr1, tr2, tr3))
    atr = ema_com(tr, 13)

    return {
        "ema9":        ema9,
        "ema21":       ema21,
        "rsi":         rsi,
        "macd":        macd_line,
        "macd_signal": macd_signal,
        "macd_hist":   macd_hist,
        "atr":         atr,
    }


def fetch_candles() -> list:
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return [
        {
            "open_time": row[0],
            "open":  float(row[1]),
            "high":  float(row[2]),
            "low":   float(row[3]),
            "close": float(row[4]),
        }
        for row in resp.json()
    ]


def get_live_price() -> float:
    resp = requests.get(TICKER_URL, params={"symbol": SYMBOL}, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])


def r2(v): return round(v, 2) if v is not None else None
def r4(v): return round(v, 4) if v is not None else None


def generate_signal(candles: list) -> dict:
    ind   = compute_indicators(candles)
    n     = len(candles) - 1     # last index
    price = candles[n]["close"]

    e9   = ind["ema9"][n]
    e21  = ind["ema21"][n]
    rsi  = ind["rsi"][n]
    macd = ind["macd"][n]
    msig = ind["macd_signal"][n]
    mhst = ind["macd_hist"][n]
    mhst_prev = ind["macd_hist"][n-1]
    atr  = ind["atr"][n]

    scores  = []
    reasons = []

    # 1. EMA crossover
    if e9 and e21 and e9 > e21:
        scores.append(1);  reasons.append("EMA9 > EMA21 — bullish trend")
    else:
        scores.append(-1); reasons.append("EMA9 < EMA21 — bearish trend")

    # 2. Price vs EMA21
    if e21 and price > e21:
        scores.append(1);  reasons.append("Price above EMA21")
    else:
        scores.append(-1); reasons.append("Price below EMA21")

    # 3. RSI
    if rsi and rsi < RSI_OVERSOLD:
        scores.append(1);  reasons.append(f"RSI {rsi:.1f} — oversold")
    elif rsi and rsi > RSI_OVERBOUGHT:
        scores.append(-1); reasons.append(f"RSI {rsi:.1f} — overbought")
    else:
        scores.append(0);  reasons.append(f"RSI {rsi:.1f if rsi else '—'} — neutral")

    # 4. MACD histogram direction
    if mhst and mhst_prev and mhst > 0 and mhst > mhst_prev:
        scores.append(1);  reasons.append("MACD histogram rising — bullish momentum")
    elif mhst and mhst_prev and mhst < 0 and mhst < mhst_prev:
        scores.append(-1); reasons.append("MACD histogram falling — bearish momentum")
    else:
        scores.append(0);  reasons.append("MACD histogram flat/mixed")

    # 5. MACD vs signal
    if macd and msig and macd > msig:
        scores.append(1);  reasons.append("MACD above signal line")
    else:
        scores.append(-1); reasons.append("MACD below signal line")

    total = sum(scores)

    if total >= 2:
        direction = "BUY"
        sl = r2(price - atr * SL_ATR_MULT) if atr else None
        tp = r2(price + atr * TP_ATR_MULT) if atr else None
    elif total <= -2:
        direction = "SELL"
        sl = r2(price + atr * SL_ATR_MULT) if atr else None
        tp = r2(price - atr * TP_ATR_MULT) if atr else None
    else:
        direction = "HOLD"
        sl = tp = None

    sl_pct = r2(abs(sl - price) / price * 100) if sl else None
    tp_pct = r2(abs(tp - price) / price * 100) if tp else None
    rr     = r2(tp_pct / sl_pct) if (sl_pct and sl_pct > 0) else None

    return {
        "symbol":      SYMBOL,
        "interval":    INTERVAL,
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "signal":      direction,
        "score":       total,
        "entry":       r2(price),
        "sl":          sl,
        "tp":          tp,
        "sl_pct":      sl_pct,
        "tp_pct":      tp_pct,
        "rr":          rr,
        "rsi":         r2(rsi),
        "atr":         r2(atr),
        "ema9":        r2(e9),
        "ema21":       r2(e21),
        "macd":        r4(macd),
        "macd_signal": r4(msig),
        "reasons":     [{"text": r, "score": s} for r, s in zip(reasons, scores)],
    }


def check_trade_closed(trade: dict, live: float):
    sig, sl, tp = trade["signal"], trade["sl"], trade["tp"]
    if sig == "BUY":
        if live >= tp: return "TP"
        if live <= sl: return "SL"
    elif sig == "SELL":
        if live <= tp: return "TP"
        if live >= sl: return "SL"
    return None


def pnl_pct(trade: dict, exit_price: float) -> float:
    entry = trade["entry"]
    if trade["signal"] == "BUY":
        return r2((exit_price - entry) / entry * 100)
    return r2((entry - exit_price) / entry * 100)


# ─── Shared state ─────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "status":      "IDLE",
    "signal":      None,
    "trade":       None,
    "history":     [],
    "last_update": None,
    "error":       None,
}


def background_loop():
    while True:
        try:
            with _lock:
                status = _state["status"]

            if status == "IDLE":
                candles = fetch_candles()
                sig     = generate_signal(candles)
                with _lock:
                    _state["signal"]      = sig
                    _state["last_update"] = sig["timestamp"]
                    _state["error"]       = None
                    if sig["signal"] in ("BUY", "SELL"):
                        _state["status"] = "ACTIVE"
                        _state["trade"]  = {
                            "signal":          sig["signal"],
                            "entry":           sig["entry"],
                            "sl":              sig["sl"],
                            "tp":              sig["tp"],
                            "sl_pct":          sig["sl_pct"],
                            "tp_pct":          sig["tp_pct"],
                            "rr":              sig["rr"],
                            "opened_at":       sig["timestamp"],
                            "live_price":      sig["entry"],
                            "unrealised_pct":  0.0,
                        }

            elif status == "ACTIVE":
                live  = get_live_price()
                with _lock:
                    trade = dict(_state["trade"])

                result = check_trade_closed(trade, live)

                with _lock:
                    if _state["trade"]:
                        _state["trade"]["live_price"]     = r2(live)
                        _state["trade"]["unrealised_pct"] = pnl_pct(trade, live)
                    _state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                if result:
                    closed = dict(trade)
                    closed["exit_price"]  = r2(live)
                    closed["exit_reason"] = result
                    closed["pnl_pct"]     = pnl_pct(trade, live)
                    closed["closed_at"]   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    with _lock:
                        _state["status"]  = "CLOSED"
                        _state["trade"]   = closed
                        _state["history"] = [closed] + _state["history"]
                        _state["history"] = _state["history"][:20]
                    time.sleep(10)
                    with _lock:
                        _state["status"] = "IDLE"
                        _state["trade"]  = None

        except Exception as e:
            with _lock:
                _state["error"] = str(e)

        time.sleep(POLL_SECONDS)


threading.Thread(target=background_loop, daemon=True).start()
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "ETH Signal Bot API running"}


@app.get("/signal")
def get_signal():
    with _lock:
        s = dict(_state)
    if s["error"]:
        return {"ok": False, "error": s["error"]}
    return {
        "ok":          True,
        "status":      s["status"],
        "signal":      s["signal"],
        "trade":       s["trade"],
        "history":     s["history"],
        "last_update": s["last_update"],
    }

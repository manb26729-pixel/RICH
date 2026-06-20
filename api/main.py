"""
ETH Trading Signal Bot — Enhanced API
Indicators:
  - EMA 9/21/50       (trend direction + regime)
  - RSI 14            (momentum)
  - MACD 12/26/9      (momentum crossover)
  - ATR 14            (volatility / SL sizing)
  - Volume MA 20      (volume confirmation)
  - ADX 14            (trend strength / regime filter)
  - Candlestick       (engulfing, hammer, shooting star)
  - Support/Resistance (swing high/low for TP targeting)

Signal fires only when score >= 4/8 AND volume confirms AND ADX > 20
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import threading
import time
from datetime import datetime, timezone

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ─── Config ───────────────────────────────────────────────────────────────────
BINANCE_URL    = "https://api.binance.us/api/v3/klines"
TICKER_URL     = "https://api.binance.us/api/v3/ticker/price"
SYMBOL         = "ETHUSD"
INTERVAL       = "15m"
LIMIT          = 200
SL_ATR_MULT    = 1.5
TP_ATR_MULT    = 3.0
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65
MIN_ADX        = 20       # below this = ranging/choppy, skip signal
MIN_SCORE      = 4        # out of 8 to fire BUY/SELL
POLL_SECONDS   = 60
ACCOUNT_SIZE_USDT   = 1000.0
RISK_PCT_PER_TRADE  = 0.01
MIN_NOTIONAL_USDT   = 10.0
# ──────────────────────────────────────────────────────────────────────────────


# ─── Pure Python math ─────────────────────────────────────────────────────────

def _ema(values, span):
    k = 2.0 / (span + 1)
    out = [None] * len(values)
    for i, v in enumerate(values):
        if v is None:
            continue
        prev = out[i - 1] if i > 0 else None
        out[i] = v if prev is None else v * k + prev * (1 - k)
    return out


def _ema_com(values, com):
    k = 1.0 / (1 + com)
    out = [None] * len(values)
    for i, v in enumerate(values):
        if v is None:
            continue
        prev = out[i - 1] if i > 0 else None
        out[i] = v if prev is None else v * k + prev * (1 - k)
    return out


def _sma(values, period):
    out = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = [v for v in values[i - period + 1:i + 1] if v is not None]
        out[i] = sum(window) / len(window) if window else None
    return out


def compute_indicators(candles):
    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]
    opens   = [c["open"]   for c in candles]

    # EMAs
    ema9  = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    ema50 = _ema(closes, 50)

    # RSI
    gains  = [None] + [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [None] + [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    ag = _ema_com(gains,  13)
    al = _ema_com(losses, 13)
    rsi = []
    for g, l in zip(ag, al):
        if g is None or l is None:
            rsi.append(None)
        elif l == 0:
            rsi.append(100.0)
        else:
            rsi.append(100 - 100 / (1 + g / l))

    # MACD
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line   = [a - b if a and b else None for a, b in zip(ema12, ema26)]
    macd_signal = _ema(macd_line, 9)
    macd_hist   = [a - b if a and b else None for a, b in zip(macd_line, macd_signal)]

    # ATR
    tr = [None]
    for i in range(1, len(closes)):
        tr.append(max(highs[i] - lows[i],
                      abs(highs[i] - closes[i-1]),
                      abs(lows[i]  - closes[i-1])))
    atr = _ema_com(tr, 13)

    # Volume MA 20
    vol_ma = _sma(volumes, 20)

    # ADX 14
    plus_dm  = [None]
    minus_dm = [None]
    for i in range(1, len(closes)):
        up   = highs[i]  - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm.append(up   if up > down and up > 0   else 0)
        minus_dm.append(down if down > up and down > 0 else 0)

    atr14     = _ema_com(tr, 13)
    plus_di   = [14 * (p / a) if p is not None and a and a > 0 else None
                 for p, a in zip(_ema_com(plus_dm, 13), atr14)]
    minus_di  = [14 * (m / a) if m is not None and a and a > 0 else None
                 for m, a in zip(_ema_com(minus_dm, 13), atr14)]
    dx = []
    for p, m in zip(plus_di, minus_di):
        if p is None or m is None or (p + m) == 0:
            dx.append(None)
        else:
            dx.append(100 * abs(p - m) / (p + m))
    adx = _ema_com(dx, 13)

    return {
        "ema9": ema9, "ema21": ema21, "ema50": ema50,
        "rsi": rsi,
        "macd": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "atr": atr,
        "vol_ma": vol_ma, "volumes": volumes,
        "adx": adx, "plus_di": plus_di, "minus_di": minus_di,
        "opens": opens, "closes": closes, "highs": highs, "lows": lows,
    }


def detect_candle_pattern(ind, i):
    """
    Returns 'BULL', 'BEAR', or None based on candlestick pattern at index i.
    Patterns: engulfing, hammer/shooting star, doji-with-direction.
    """
    o  = ind["opens"][i];   c  = ind["closes"][i]
    h  = ind["highs"][i];   l  = ind["lows"][i]
    o1 = ind["opens"][i-1]; c1 = ind["closes"][i-1]

    body     = abs(c - o)
    body1    = abs(c1 - o1)
    candle_range = h - l if h != l else 0.0001

    # Bullish engulfing
    if c1 < o1 and c > o and c > o1 and o < c1:
        return "BULL"
    # Bearish engulfing
    if c1 > o1 and c < o and c < o1 and o > c1:
        return "BEAR"

    # Hammer (bullish) — small body at top, long lower wick
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if lower_wick > 2 * body and upper_wick < body and body / candle_range < 0.35:
        return "BULL"

    # Shooting star (bearish) — small body at bottom, long upper wick
    if upper_wick > 2 * body and lower_wick < body and body / candle_range < 0.35:
        return "BEAR"

    return None


def find_sr_levels(ind, i, lookback=40):
    """
    Find nearest support (below price) and resistance (above price)
    using recent swing highs/lows.
    """
    price  = ind["closes"][i]
    start  = max(1, i - lookback)
    highs  = ind["highs"][start:i]
    lows   = ind["lows"][start:i]

    # Swing highs: local max in a 3-bar window
    swing_highs = []
    swing_lows  = []
    for j in range(1, len(highs) - 1):
        if highs[j] > highs[j-1] and highs[j] > highs[j+1]:
            swing_highs.append(highs[j])
        if lows[j]  < lows[j-1]  and lows[j]  < lows[j+1]:
            swing_lows.append(lows[j])

    resistance = min((h for h in swing_highs if h > price), default=None)
    support    = max((l for l in swing_lows  if l < price), default=None)

    return support, resistance


def fetch_candles():
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return [
        {
            "open_time": row[0],
            "open":   float(row[1]),
            "high":   float(row[2]),
            "low":    float(row[3]),
            "close":  float(row[4]),
            "volume": float(row[5]),
        }
        for row in resp.json()
    ]


def get_live_price():
    resp = requests.get(TICKER_URL, params={"symbol": SYMBOL}, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["price"])


def r2(v): return round(v, 2) if v is not None else None
def r4(v): return round(v, 4) if v is not None else None


def generate_signal(candles):
    ind   = compute_indicators(candles)
    n     = len(candles) - 1
    price = candles[n]["close"]

    e9   = ind["ema9"][n]
    e21  = ind["ema21"][n]
    e50  = ind["ema50"][n]
    rsi  = ind["rsi"][n]
    macd = ind["macd"][n]
    msig = ind["macd_signal"][n]
    mhst = ind["macd_hist"][n]
    mhst_prev = ind["macd_hist"][n-1]
    atr  = ind["atr"][n]
    adx  = ind["adx"][n]
    pdi  = ind["plus_di"][n]
    mdi  = ind["minus_di"][n]
    vol      = ind["volumes"][n]
    vol_ma   = ind["vol_ma"][n]

    scores  = []
    reasons = []
    flags   = []   # non-scoring notes

    # ── 1. EMA 9/21 crossover ────────────────────────────────────────────────
    if e9 and e21 and e9 > e21:
        scores.append(1);  reasons.append("EMA9 > EMA21 — bullish trend")
    else:
        scores.append(-1); reasons.append("EMA9 < EMA21 — bearish trend")

    # ── 2. Price vs EMA50 (macro trend) ──────────────────────────────────────
    if e50 and price > e50:
        scores.append(1);  reasons.append("Price above EMA50 — macro bullish")
    else:
        scores.append(-1); reasons.append("Price below EMA50 — macro bearish")

    # ── 3. RSI ────────────────────────────────────────────────────────────────
    if rsi and rsi < RSI_OVERSOLD:
        scores.append(1);  reasons.append(f"RSI {rsi:.1f} — oversold")
    elif rsi and rsi > RSI_OVERBOUGHT:
        scores.append(-1); reasons.append(f"RSI {rsi:.1f} — overbought")
    else:
        scores.append(0);  reasons.append(f"RSI {round(rsi,1) if rsi else '—'} — neutral")

    # ── 4. MACD histogram momentum ───────────────────────────────────────────
    if mhst and mhst_prev and mhst > 0 and mhst > mhst_prev:
        scores.append(1);  reasons.append("MACD histogram rising — bullish momentum")
    elif mhst and mhst_prev and mhst < 0 and mhst < mhst_prev:
        scores.append(-1); reasons.append("MACD histogram falling — bearish momentum")
    else:
        scores.append(0);  reasons.append("MACD histogram flat/mixed")

    # ── 5. MACD line vs signal ───────────────────────────────────────────────
    if macd and msig and macd > msig:
        scores.append(1);  reasons.append("MACD above signal line")
    else:
        scores.append(-1); reasons.append("MACD below signal line")

    # ── 6. ADX directional bias (+DI vs -DI) ─────────────────────────────────
    if pdi and mdi:
        if pdi > mdi:
            scores.append(1);  reasons.append(f"+DI {pdi:.1f} > -DI {mdi:.1f} — bulls in control")
        else:
            scores.append(-1); reasons.append(f"-DI {mdi:.1f} > +DI {pdi:.1f} — bears in control")
    else:
        scores.append(0); reasons.append("ADX DI unavailable")

    # ── 7. Volume confirmation ───────────────────────────────────────────────
    if vol and vol_ma and vol > vol_ma * 1.1:
        scores.append(1);  reasons.append(f"Volume {vol:.0f} above MA — confirmed move")
    elif vol and vol_ma and vol < vol_ma * 0.7:
        scores.append(-1); reasons.append(f"Volume {vol:.0f} below MA — weak move")
    else:
        scores.append(0);  reasons.append("Volume near average")

    # ── 8. Candlestick pattern ───────────────────────────────────────────────
    pattern = detect_candle_pattern(ind, n)
    if pattern == "BULL":
        scores.append(1);  reasons.append("Bullish candlestick pattern")
    elif pattern == "BEAR":
        scores.append(-1); reasons.append("Bearish candlestick pattern")
    else:
        scores.append(0);  reasons.append("No strong candle pattern")

    total = sum(scores)

    # ── ADX regime filter ────────────────────────────────────────────────────
    trending = adx and adx > MIN_ADX
    if not trending:
        flags.append(f"ADX {round(adx,1) if adx else '—'} < {MIN_ADX} — ranging market, signal suppressed")

    # ── Direction decision ───────────────────────────────────────────────────
    if total >= MIN_SCORE and trending:
        direction = "BUY"
    elif total <= -MIN_SCORE and trending:
        direction = "SELL"
    else:
        direction = "HOLD"

    # ── SL/TP — use S/R levels if available, else ATR fallback ───────────────
    support, resistance = find_sr_levels(ind, n)

    if direction == "BUY":
        sl = r2(support  if support    and support  > price - atr * 2 else price - atr * SL_ATR_MULT)
        tp = r2(resistance if resistance and resistance < price + atr * 5 else price + atr * TP_ATR_MULT)
    elif direction == "SELL":
        sl = r2(resistance if resistance and resistance < price + atr * 2 else price + atr * SL_ATR_MULT)
        tp = r2(support  if support    and support  > price - atr * 5 else price - atr * TP_ATR_MULT)
    else:
        sl = tp = None

    sl_pct = r2(abs(sl - price) / price * 100) if sl else None
    tp_pct = r2(abs(tp - price) / price * 100) if tp else None
    rr     = r2(tp_pct / sl_pct) if (sl_pct and sl_pct > 0) else None

    # Confidence label
    abs_score = abs(total)
    if abs_score >= 7:   confidence = "VERY HIGH"
    elif abs_score >= 5: confidence = "HIGH"
    elif abs_score >= 3: confidence = "MEDIUM"
    else:                confidence = "LOW"

    return {
        "symbol":      SYMBOL,
        "interval":    INTERVAL,
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "signal":      direction,
        "score":       total,
        "max_score":   8,
        "confidence":  confidence,
        "trending":    trending,
        "adx":         r2(adx),
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
        "ema50":       r2(e50),
        "macd":        r4(macd),
        "macd_signal": r4(msig),
        "volume":      r2(vol),
        "volume_ma":   r2(vol_ma),
        "candle":      pattern,
        "reasons":     [{"text": r, "score": s} for r, s in zip(reasons, scores)],
        "flags":       flags,
    }


def check_trade_closed(trade, live):
    sig, sl, tp = trade["signal"], trade["sl"], trade["tp"]
    if sig == "BUY":
        if live >= tp: return "TP"
        if live <= sl: return "SL"
    elif sig == "SELL":
        if live <= tp: return "TP"
        if live >= sl: return "SL"
    return None


def pnl_pct(trade, exit_price):
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
                            "signal":         sig["signal"],
                            "entry":          sig["entry"],
                            "sl":             sig["sl"],
                            "tp":             sig["tp"],
                            "sl_pct":         sig["sl_pct"],
                            "tp_pct":         sig["tp_pct"],
                            "rr":             sig["rr"],
                            "confidence":     sig["confidence"],
                            "opened_at":      sig["timestamp"],
                            "live_price":     sig["entry"],
                            "unrealised_pct": 0.0,
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
    return {"status": "ETH Signal Bot API running — enhanced v2"}


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

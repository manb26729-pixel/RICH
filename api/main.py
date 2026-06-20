"""
ETH Advanced Signal Bot — Maximum Intelligence Edition
=======================================================
10-layer scoring system (max score: 14 points):

Layer 1 — Trend (EMA 9/21/50/200)
Layer 2 — Momentum (RSI, Stochastic RSI)
Layer 3 — MACD (line + histogram)
Layer 4 — Volatility / ATR regime
Layer 5 — Volume (OBV + volume MA)
Layer 6 — ADX trend strength
Layer 7 — Candlestick patterns
Layer 8 — Multi-timeframe (1h trend must agree)
Layer 9 — Order book pressure (bid/ask imbalance)
Layer 10 — Fear & Greed Index (market sentiment)

Signal fires only when:
  - Score >= 8/14 (high conviction only)
  - ADX > 20 (trending market)
  - MTF agrees (1h EMA confirms direction)
  - Volume above average
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests, threading, time
from datetime import datetime, timezone

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

# ── Config ────────────────────────────────────────────────────────────────────
BINANCE_URL  = "https://api.binance.us/api/v3/klines"
TICKER_URL   = "https://api.binance.us/api/v3/ticker/price"
DEPTH_URL    = "https://api.binance.us/api/v3/depth"
FNG_URL      = "https://api.alternative.me/fng/?limit=1"
SYMBOL       = "ETHUSD"
INTERVAL     = "15m"
INTERVAL_HTF = "1h"
LIMIT        = 200
SL_ATR_MULT  = 1.5
TP_ATR_MULT  = 3.0
MIN_RR       = 1.8   # minimum risk/reward enforced on every signal
MIN_SCORE    = 8      # out of 14
POLL_SECONDS = 60
ACCOUNT_SIZE_USDT  = 1000.0
RISK_PCT_PER_TRADE = 0.01
MIN_NOTIONAL_USDT  = 10.0
# ─────────────────────────────────────────────────────────────────────────────


# ── Pure Python math helpers ──────────────────────────────────────────────────

def _ema(values, span):
    k = 2.0 / (span + 1)
    out = [None] * len(values)
    for i, v in enumerate(values):
        if v is None: continue
        prev = out[i-1] if i > 0 else None
        out[i] = v if prev is None else v * k + prev * (1 - k)
    return out

def _ema_com(values, com):
    k = 1.0 / (1 + com)
    out = [None] * len(values)
    for i, v in enumerate(values):
        if v is None: continue
        prev = out[i-1] if i > 0 else None
        out[i] = v if prev is None else v * k + prev * (1 - k)
    return out

def _sma(values, period):
    out = [None] * len(values)
    for i in range(period - 1, len(values)):
        w = [v for v in values[i-period+1:i+1] if v is not None]
        out[i] = sum(w) / len(w) if w else None
    return out

def r2(v): return round(v, 2) if v is not None else None
def r4(v): return round(v, 4) if v is not None else None


# ── Indicator computation ─────────────────────────────────────────────────────

def compute_indicators(candles):
    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]
    opens   = [c["open"]   for c in candles]

    ema9   = _ema(closes, 9)
    ema21  = _ema(closes, 21)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200)

    # RSI
    gains  = [None] + [max(closes[i]-closes[i-1], 0) for i in range(1, len(closes))]
    losses = [None] + [max(closes[i-1]-closes[i], 0) for i in range(1, len(closes))]
    ag = _ema_com(gains, 13);  al = _ema_com(losses, 13)
    rsi = []
    for g, l in zip(ag, al):
        if g is None or l is None: rsi.append(None)
        elif l == 0: rsi.append(100.0)
        else: rsi.append(100 - 100/(1 + g/l))

    # Stochastic RSI (RSI smoothed into stochastic)
    stoch_rsi = [None] * len(rsi)
    for i in range(14, len(rsi)):
        window = [v for v in rsi[i-13:i+1] if v is not None]
        if not window: continue
        lo, hi = min(window), max(window)
        stoch_rsi[i] = (rsi[i] - lo) / (hi - lo) * 100 if hi != lo else 50

    # MACD
    ema12 = _ema(closes, 12); ema26 = _ema(closes, 26)
    macd_line   = [a-b if a and b else None for a,b in zip(ema12, ema26)]
    macd_signal = _ema(macd_line, 9)
    macd_hist   = [a-b if a and b else None for a,b in zip(macd_line, macd_signal)]

    # ATR
    tr = [None]
    for i in range(1, len(closes)):
        tr.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    atr = _ema_com(tr, 13)

    # OBV (On Balance Volume)
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:   obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]: obv.append(obv[-1] - volumes[i])
        else:                          obv.append(obv[-1])
    obv_ema = _ema(obv, 21)

    vol_ma = _sma(volumes, 20)

    # ADX
    plus_dm  = [None] + [max(highs[i]-highs[i-1], 0) if highs[i]-highs[i-1] > lows[i-1]-lows[i] else 0 for i in range(1, len(closes))]
    minus_dm = [None] + [max(lows[i-1]-lows[i], 0) if lows[i-1]-lows[i] > highs[i]-highs[i-1] else 0 for i in range(1, len(closes))]
    atr14    = _ema_com(tr, 13)
    plus_di  = [14*(p/a) if p is not None and a and a>0 else None for p,a in zip(_ema_com(plus_dm,13), atr14)]
    minus_di = [14*(m/a) if m is not None and a and a>0 else None for m,a in zip(_ema_com(minus_dm,13), atr14)]
    dx = []
    for p,m in zip(plus_di, minus_di):
        if p is None or m is None or (p+m)==0: dx.append(None)
        else: dx.append(100 * abs(p-m)/(p+m))
    adx = _ema_com(dx, 13)

    # Bollinger Bands (20, 2)
    bb_mid = _sma(closes, 20)
    bb_std = [None] * len(closes)
    for i in range(19, len(closes)):
        w = closes[i-19:i+1]
        mean = sum(w)/len(w)
        bb_std[i] = (sum((x-mean)**2 for x in w)/len(w))**0.5
    bb_upper = [m+2*s if m and s else None for m,s in zip(bb_mid, bb_std)]
    bb_lower = [m-2*s if m and s else None for m,s in zip(bb_mid, bb_std)]

    return {
        "ema9":ema9,"ema21":ema21,"ema50":ema50,"ema200":ema200,
        "rsi":rsi,"stoch_rsi":stoch_rsi,
        "macd":macd_line,"macd_signal":macd_signal,"macd_hist":macd_hist,
        "atr":atr,"obv":obv,"obv_ema":obv_ema,
        "vol_ma":vol_ma,"volumes":volumes,
        "adx":adx,"plus_di":plus_di,"minus_di":minus_di,
        "bb_upper":bb_upper,"bb_lower":bb_lower,"bb_mid":bb_mid,
        "opens":opens,"closes":closes,"highs":highs,"lows":lows,
    }


# ── Candlestick patterns ──────────────────────────────────────────────────────

def detect_candle_pattern(ind, i):
    o=ind["opens"][i];   c=ind["closes"][i]
    h=ind["highs"][i];   l=ind["lows"][i]
    o1=ind["opens"][i-1];c1=ind["closes"][i-1]
    body=abs(c-o); body1=abs(c1-o1)
    rng=h-l if h!=l else 0.0001
    lower_wick=min(o,c)-l; upper_wick=h-max(o,c)
    # Bullish engulfing
    if c1<o1 and c>o and c>o1 and o<c1: return "BULL"
    # Bearish engulfing
    if c1>o1 and c<o and c<o1 and o>c1: return "BEAR"
    # Hammer
    if lower_wick>2*body and upper_wick<body and body/rng<0.35: return "BULL"
    # Shooting star
    if upper_wick>2*body and lower_wick<body and body/rng<0.35: return "BEAR"
    # Morning star (3-candle)
    if i >= 2:
        o2=ind["opens"][i-2]; c2=ind["closes"][i-2]
        if c2<o2 and abs(c1-o1)<body1*0.3 and c>o and c>((o2+c2)/2): return "BULL"
    # Evening star
    if i >= 2:
        o2=ind["opens"][i-2]; c2=ind["closes"][i-2]
        if c2>o2 and abs(c1-o1)<body1*0.3 and c<o and c<((o2+c2)/2): return "BEAR"
    return None


# ── Support / Resistance ──────────────────────────────────────────────────────

def find_sr_levels(ind, i, lookback=50):
    price=ind["closes"][i]
    start=max(1,i-lookback)
    highs=ind["highs"][start:i]; lows=ind["lows"][start:i]
    swing_highs=[]; swing_lows=[]
    for j in range(1, len(highs)-1):
        if highs[j]>highs[j-1] and highs[j]>highs[j+1]: swing_highs.append(highs[j])
        if lows[j]<lows[j-1]  and lows[j]<lows[j+1]:   swing_lows.append(lows[j])
    resistance=min((h for h in swing_highs if h>price), default=None)
    support   =max((l for l in swing_lows  if l<price), default=None)
    return support, resistance


# ── Data fetchers ─────────────────────────────────────────────────────────────

def fetch_candles(interval=INTERVAL, limit=LIMIT):
    params={"symbol":SYMBOL,"interval":interval,"limit":limit}
    resp=requests.get(BINANCE_URL, params=params, timeout=10); resp.raise_for_status()
    return [{"open_time":r[0],"open":float(r[1]),"high":float(r[2]),
             "low":float(r[3]),"close":float(r[4]),"volume":float(r[5])} for r in resp.json()]

def get_live_price():
    resp=requests.get(TICKER_URL, params={"symbol":SYMBOL}, timeout=5); resp.raise_for_status()
    return float(resp.json()["price"])

def get_order_book_pressure():
    """Returns bid_pct: % of top-20 depth that is bids. >55 = buy pressure, <45 = sell pressure."""
    try:
        resp=requests.get(DEPTH_URL, params={"symbol":SYMBOL,"limit":20}, timeout=5); resp.raise_for_status()
        data=resp.json()
        bid_vol=sum(float(b[1]) for b in data["bids"])
        ask_vol=sum(float(a[1]) for a in data["asks"])
        total=bid_vol+ask_vol
        return round(bid_vol/total*100, 1) if total>0 else 50.0
    except: return 50.0

def get_fear_greed():
    """Returns Fear & Greed index 0-100. <25=extreme fear(buy), >75=extreme greed(sell)."""
    try:
        resp=requests.get(FNG_URL, timeout=5); resp.raise_for_status()
        data=resp.json()
        val=int(data["data"][0]["value"])
        label=data["data"][0]["value_classification"]
        return val, label
    except: return 50, "Neutral"

def get_htf_trend():
    """Get 1h candles and return trend: BULL, BEAR, FLAT."""
    try:
        candles=fetch_candles(interval="1h", limit=100)
        ind=compute_indicators(candles)
        n=len(candles)-1
        e9=ind["ema9"][n]; e21=ind["ema21"][n]; e50=ind["ema50"][n]
        price=candles[n]["close"]
        adx=ind["adx"][n]
        if e9 and e21 and e50 and e9>e21 and price>e50: return "BULL"
        if e9 and e21 and e50 and e9<e21 and price<e50: return "BEAR"
        return "FLAT"
    except: return "FLAT"


# ── Master signal generator ───────────────────────────────────────────────────

def generate_signal(candles, htf_trend="FLAT", bid_pct=50.0, fng_val=50, fng_label="Neutral"):
    ind   = compute_indicators(candles)
    n     = len(candles) - 1
    price = candles[n]["close"]

    e9    = ind["ema9"][n];   e21   = ind["ema21"][n]
    e50   = ind["ema50"][n];  e200  = ind["ema200"][n]
    rsi   = ind["rsi"][n];    srsi  = ind["stoch_rsi"][n]
    macd  = ind["macd"][n];   msig  = ind["macd_signal"][n]
    mhst  = ind["macd_hist"][n]; mhst_prev = ind["macd_hist"][n-1]
    atr   = ind["atr"][n]
    adx   = ind["adx"][n];    pdi   = ind["plus_di"][n]; mdi = ind["minus_di"][n]
    vol   = ind["volumes"][n]; vol_ma = ind["vol_ma"][n]
    obv   = ind["obv"][n];    obv_ema = ind["obv_ema"][n]
    bbu   = ind["bb_upper"][n]; bbl = ind["bb_lower"][n]; bbm = ind["bb_mid"][n]

    scores=[]; reasons=[]

    # ── LAYER 1: EMA Stack (2 pts) ────────────────────────────────────────────
    if e9 and e21 and e9>e21:
        scores.append(1); reasons.append("EMA9 > EMA21 — short-term bullish")
    else:
        scores.append(-1); reasons.append("EMA9 < EMA21 — short-term bearish")

    if e50 and e200 and price>e50 and e50>e200:
        scores.append(1); reasons.append("Price > EMA50 > EMA200 — strong bull stack")
    elif e50 and e200 and price<e50 and e50<e200:
        scores.append(-1); reasons.append("Price < EMA50 < EMA200 — strong bear stack")
    else:
        scores.append(0); reasons.append("EMA stack mixed — no clear macro trend")

    # ── LAYER 2: RSI + Stochastic RSI (2 pts) ────────────────────────────────
    if rsi and rsi < 35:
        scores.append(1); reasons.append(f"RSI {round(rsi,1)} — oversold, reversal likely")
    elif rsi and rsi > 65:
        scores.append(-1); reasons.append(f"RSI {round(rsi,1)} — overbought, pullback likely")
    else:
        scores.append(0); reasons.append(f"RSI {round(rsi,1) if rsi else '—'} — neutral zone")

    if srsi and srsi < 20:
        scores.append(1); reasons.append(f"Stoch RSI {round(srsi,1)} — oversold momentum")
    elif srsi and srsi > 80:
        scores.append(-1); reasons.append(f"Stoch RSI {round(srsi,1)} — overbought momentum")
    else:
        scores.append(0); reasons.append(f"Stoch RSI {round(srsi,1) if srsi else '—'} — neutral")

    # ── LAYER 3: MACD (2 pts) ─────────────────────────────────────────────────
    if mhst and mhst_prev and mhst>0 and mhst>mhst_prev:
        scores.append(1); reasons.append("MACD histogram expanding bullish")
    elif mhst and mhst_prev and mhst<0 and mhst<mhst_prev:
        scores.append(-1); reasons.append("MACD histogram expanding bearish")
    else:
        scores.append(0); reasons.append("MACD histogram flat/contracting")

    if macd and msig and macd>msig:
        scores.append(1); reasons.append("MACD above signal — bullish crossover zone")
    else:
        scores.append(-1); reasons.append("MACD below signal — bearish crossover zone")

    # ── LAYER 4: Bollinger Bands (1 pt) ──────────────────────────────────────
    if bbl and bbu and bbm:
        bb_width = (bbu - bbl) / bbm * 100
        if price < bbl:
            scores.append(1); reasons.append(f"Price below lower BB — oversold squeeze")
        elif price > bbu:
            scores.append(-1); reasons.append(f"Price above upper BB — overbought extension")
        else:
            scores.append(0); reasons.append(f"Price inside Bollinger Bands (width {round(bb_width,2)}%)")
    else:
        scores.append(0); reasons.append("Bollinger Bands unavailable")

    # ── LAYER 5: OBV + Volume (2 pts) ────────────────────────────────────────
    if obv and obv_ema and obv > obv_ema:
        scores.append(1); reasons.append("OBV above EMA — accumulation (smart money buying)")
    elif obv and obv_ema and obv < obv_ema:
        scores.append(-1); reasons.append("OBV below EMA — distribution (smart money selling)")
    else:
        scores.append(0); reasons.append("OBV neutral")

    if vol and vol_ma and vol > vol_ma * 1.2:
        scores.append(1); reasons.append(f"Volume {round(vol,1)} — 20%+ above avg, strong confirmation")
    elif vol and vol_ma and vol < vol_ma * 0.7:
        scores.append(-1); reasons.append(f"Volume {round(vol,1)} — weak, move not confirmed")
    else:
        scores.append(0); reasons.append("Volume near average")

    # ── LAYER 6: ADX directional (1 pt) ──────────────────────────────────────
    if pdi and mdi and pdi > mdi:
        scores.append(1); reasons.append(f"+DI {round(pdi,1)} > -DI {round(mdi,1)} — bulls in control")
    elif pdi and mdi:
        scores.append(-1); reasons.append(f"-DI {round(mdi,1)} > +DI {round(pdi,1)} — bears in control")
    else:
        scores.append(0); reasons.append("ADX DI unavailable")

    # ── LAYER 7: Candlestick pattern (1 pt) ──────────────────────────────────
    pattern = detect_candle_pattern(ind, n)
    if pattern == "BULL":
        scores.append(1); reasons.append("Bullish candlestick pattern confirmed")
    elif pattern == "BEAR":
        scores.append(-1); reasons.append("Bearish candlestick pattern confirmed")
    else:
        scores.append(0); reasons.append("No strong candlestick pattern")

    # ── LAYER 8: Multi-timeframe 1h (1 pt) ───────────────────────────────────
    if htf_trend == "BULL":
        scores.append(1); reasons.append("1h timeframe bullish — HTF confirms direction")
    elif htf_trend == "BEAR":
        scores.append(-1); reasons.append("1h timeframe bearish — HTF confirms direction")
    else:
        scores.append(0); reasons.append("1h timeframe flat — no HTF confirmation")

    # ── LAYER 9: Order book pressure (1 pt) ──────────────────────────────────
    if bid_pct >= 55:
        scores.append(1); reasons.append(f"Order book {bid_pct}% bids — buy pressure")
    elif bid_pct <= 45:
        scores.append(-1); reasons.append(f"Order book {bid_pct}% asks — sell pressure")
    else:
        scores.append(0); reasons.append(f"Order book balanced ({bid_pct}% bids)")

    # ── LAYER 10: Fear & Greed (1 pt) ────────────────────────────────────────
    if fng_val <= 25:
        scores.append(1); reasons.append(f"Fear & Greed {fng_val} — Extreme Fear (contrarian BUY)")
    elif fng_val >= 75:
        scores.append(-1); reasons.append(f"Fear & Greed {fng_val} — Extreme Greed (contrarian SELL)")
    else:
        scores.append(0); reasons.append(f"Fear & Greed {fng_val} ({fng_label}) — neutral sentiment")

    total = sum(scores)
    max_pts = len(scores)  # 14

    # ── Regime + direction decision ───────────────────────────────────────────
    trending = adx and adx > MIN_ADX
    vol_ok   = vol and vol_ma and vol >= vol_ma * 0.8
    mtf_ok   = htf_trend != "FLAT"

    flags = []
    if not trending: flags.append(f"ADX {round(adx,1) if adx else '—'} — ranging market")
    if not vol_ok:   flags.append("Volume below average — weak confirmation")
    if not mtf_ok:   flags.append("1h trend flat — no HTF confluence")

    gate = trending  # minimum gate: must be trending

    if total >= MIN_SCORE and gate:
        direction = "BUY"
    elif total <= -MIN_SCORE and gate:
        direction = "SELL"
    else:
        direction = "HOLD"

    # ── SL/TP using S/R + ATR ─────────────────────────────────────────────────
    support, resistance = find_sr_levels(ind, n)
    MIN_RR = 1.8  # enforce minimum risk/reward ratio

    if direction == "BUY":
        sl = r2(support    if support    and support    > price - atr*2 else price - atr*SL_ATR_MULT)
        sl_dist = abs(price - sl) if sl else atr * SL_ATR_MULT
        min_tp  = price + sl_dist * MIN_RR
        tp_sr   = resistance if resistance and resistance < price + atr*6 else None
        tp = r2(tp_sr if tp_sr and tp_sr >= min_tp else min_tp)
    elif direction == "SELL":
        sl = r2(resistance if resistance and resistance < price + atr*2 else price + atr*SL_ATR_MULT)
        sl_dist = abs(sl - price) if sl else atr * SL_ATR_MULT
        min_tp  = price - sl_dist * MIN_RR
        tp_sr   = support if support and support > price - atr*6 else None
        tp = r2(tp_sr if tp_sr and tp_sr <= min_tp else min_tp)
    else:
        sl = tp = None

    sl_pct = r2(abs(sl-price)/price*100) if sl else None
    tp_pct = r2(abs(tp-price)/price*100) if tp else None
    rr     = r2(tp_pct/sl_pct) if sl_pct and sl_pct>0 else None

    score_pct = round(abs(total)/max_pts*100)
    if score_pct >= 80:   confidence = "VERY HIGH"
    elif score_pct >= 60: confidence = "HIGH"
    elif score_pct >= 40: confidence = "MEDIUM"
    else:                 confidence = "LOW"

    return {
        "symbol":SYMBOL,"interval":INTERVAL,
        "timestamp":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "signal":direction,"score":total,"max_score":max_pts,
        "score_pct":score_pct,"confidence":confidence,"trending":trending,
        "entry":r2(price),"sl":sl,"tp":tp,"sl_pct":sl_pct,"tp_pct":tp_pct,"rr":rr,
        "rsi":r2(rsi),"stoch_rsi":r2(srsi),"atr":r2(atr),
        "ema9":r2(e9),"ema21":r2(e21),"ema50":r2(e50),"ema200":r2(e200),
        "macd":r4(macd),"macd_signal":r4(msig),
        "adx":r2(adx),"plus_di":r2(pdi),"minus_di":r2(mdi),
        "volume":r2(vol),"volume_ma":r2(vol_ma),
        "bb_upper":r2(bbu),"bb_lower":r2(bbl),
        "htf_trend":htf_trend,"bid_pct":bid_pct,
        "fear_greed":fng_val,"fear_greed_label":fng_label,
        "candle":pattern,"flags":flags,
        "reasons":[{"text":r,"score":s} for r,s in zip(reasons,scores)],
    }


# ── Trade lifecycle helpers ───────────────────────────────────────────────────

def check_trade_closed(trade, live):
    sig,sl,tp = trade["signal"],trade["sl"],trade["tp"]
    if sig=="BUY":
        if live>=tp: return "TP"
        if live<=sl: return "SL"
    elif sig=="SELL":
        if live<=tp: return "TP"
        if live>=sl: return "SL"
    return None

def pnl_pct(trade, exit_price):
    entry=trade["entry"]
    if trade["signal"]=="BUY": return r2((exit_price-entry)/entry*100)
    return r2((entry-exit_price)/entry*100)


# ── Shared state ──────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_state = {"status":"IDLE","signal":None,"trade":None,"history":[],"last_update":None,"error":None}


def background_loop():
    while True:
        try:
            with _lock: status = _state["status"]

            if status == "IDLE":
                candles   = fetch_candles()
                htf       = get_htf_trend()
                bid_pct   = get_order_book_pressure()
                fng, flab = get_fear_greed()
                sig       = generate_signal(candles, htf, bid_pct, fng, flab)

                with _lock:
                    _state["signal"]      = sig
                    _state["last_update"] = sig["timestamp"]
                    _state["error"]       = None
                    if sig["signal"] in ("BUY","SELL"):
                        _state["status"] = "ACTIVE"
                        _state["trade"]  = {
                            "signal":sig["signal"],"entry":sig["entry"],
                            "sl":sig["sl"],"tp":sig["tp"],
                            "sl_pct":sig["sl_pct"],"tp_pct":sig["tp_pct"],"rr":sig["rr"],
                            "confidence":sig["confidence"],"score_pct":sig["score_pct"],
                            "opened_at":sig["timestamp"],
                            "live_price":sig["entry"],"unrealised_pct":0.0,
                        }

            elif status == "ACTIVE":
                live = get_live_price()
                with _lock: trade = dict(_state["trade"])
                result = check_trade_closed(trade, live)
                with _lock:
                    if _state["trade"]:
                        _state["trade"]["live_price"]     = r2(live)
                        _state["trade"]["unrealised_pct"] = pnl_pct(trade, live)
                    _state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if result:
                    closed = dict(trade)
                    closed.update({"exit_price":r2(live),"exit_reason":result,
                                   "pnl_pct":pnl_pct(trade,live),
                                   "closed_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})
                    with _lock:
                        _state["status"]  = "CLOSED"
                        _state["trade"]   = closed
                        _state["history"] = [closed]+_state["history"]
                        _state["history"] = _state["history"][:20]
                    time.sleep(10)
                    with _lock:
                        _state["status"] = "IDLE"
                        _state["trade"]  = None

        except Exception as e:
            with _lock: _state["error"] = str(e)

        time.sleep(POLL_SECONDS)


threading.Thread(target=background_loop, daemon=True).start()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root(): return {"status":"ETH Signal Bot — Maximum Intelligence v3"}

@app.get("/signal")
def get_signal():
    with _lock: s=dict(_state)
    if s["error"]: return {"ok":False,"error":s["error"]}
    return {"ok":True,"status":s["status"],"signal":s["signal"],
            "trade":s["trade"],"history":s["history"],"last_update":s["last_update"]}

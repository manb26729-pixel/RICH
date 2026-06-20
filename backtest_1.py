"""
ETH Bot Backtester
------------------
Replays the exact signal logic from bot.py (EMA/RSI/MACD/ATR scoring +
1h HTF trend filter) over historical Binance candles, simulates trade
fills bar-by-bar, and reports win rate, expectancy, drawdown, and more.

WHY THIS EXISTS
One profitable trade tells you almost nothing about whether a strategy
has an edge. This script runs the strategy over months of real history
so you can see win rate and expectancy across dozens/hundreds of signals
instead of one. Past performance still does not guarantee future
results — markets change regime — but it's a vastly better starting
point than N=1.

HOW TO RUN
    pip install requests pandas numpy
    python backtest.py

Binance's public klines endpoint is NOT reachable from Claude's sandbox
(network allowlist blocks it), so this must be run on YOUR machine.
No API key is required — same public endpoint bot.py already uses.

METHODOLOGY NOTES (read before trusting the numbers)
- No look-ahead: the signal at candle i is generated using only data
  up to and including candle i. The HTF trend used is from the most
  recently CLOSED 1h candle at that point in time (never a future one).
- Trade entry: assumed to fill at candle i's close (the same price the
  signal reports as "entry"). In live trading there will be slippage
  and the few seconds between signal and order placement — this is
  optimistic versus reality, not pessimistic.
- Fills are checked bar-by-bar starting at candle i+1. If a candle's
  high/low range touches both SL and TP, SL is assumed to fill first
  (conservative — protects against overstating performance).
- A flat 0.1% taker fee is applied per side (entry + exit = ~0.2%
  round-trip), matching Binance spot default taker fee. If you have
  BNB fee discount or VIP tier, real costs will be slightly lower.
- Only one position open at a time (matches how bot.py is used).
- "Win rate" counts TP hits as wins, SL hits as losses. A trade that's
  still open at the end of the data window is closed at the last
  available price for accounting purposes (marked separately).
"""

import requests
import pandas as pd
import numpy as np
import time
import sys

sys.path.insert(0, ".")
from bot import (
    compute_indicators, generate_signal, get_htf_trend,
    SL_ATR_MULT, TP_ATR_MULT, RSI_OVERSOLD, RSI_OVERBOUGHT,
    ACCOUNT_SIZE_USDT, RISK_PCT_PER_TRADE, MIN_NOTIONAL_USDT,
)

BINANCE_URL = "https://api.binance.com/api/v3/klines"
TAKER_FEE   = 0.001   # 0.1% per side, Binance spot default

# ─── Backtest config ───────────────────────────────────────────────────
SYMBOL       = "ETHUSDT"
ENTRY_TF     = "15m"
HTF_TF       = "1h"
DAYS_BACK    = 90          # how much history to pull (Binance limit: 1000 candles/request, paginated below)
USE_HTF_FILTER = True      # set False to test without the 1h filter for comparison
# ─────────────────────────────────────────────────────────────────────────


def fetch_klines_paginated(symbol: str, interval: str, days_back: int) -> pd.DataFrame:
    """
    Binance caps each request at 1000 candles, so for longer lookback
    windows we paginate backwards using the `endTime` param.
    """
    interval_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }[interval]

    end_time = int(time.time() * 1000)
    start_time = end_time - days_back * 86_400_000

    all_rows = []
    cursor = end_time
    while cursor > start_time:
        params = {
            "symbol": symbol, "interval": interval,
            "limit": 1000, "endTime": cursor,
        }
        resp = requests.get(BINANCE_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows = rows + all_rows
        cursor = rows[0][0] - interval_ms   # move window back before earliest candle
        time.sleep(0.25)                    # be polite to the public API / avoid rate limits
        if rows[0][0] <= start_time:
            break

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades",
        "taker_base", "taker_quote", "ignore"
    ])
    df = df.drop_duplicates(subset="open_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    # keep only the requested window
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days_back)
    df = df[df["open_time"] >= cutoff].reset_index(drop=True)
    return df


def build_htf_lookup(df_htf: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute the HTF trend for every closed HTF candle, indexed by the
    candle's close_time, so we can look up "trend as of time T" without
    ever peeking at a candle that hadn't closed yet.
    """
    df_htf = compute_indicators(df_htf.copy())
    trends = []
    for i in range(len(df_htf)):
        if i < 21:  # not enough warmup for EMA21 to be meaningful
            trends.append("FLAT")
            continue
        sub = df_htf.iloc[: i + 1]
        trends.append(get_htf_trend(sub))
    df_htf["trend"] = trends
    # close_time approximated as next candle's open_time minus 1ms; use open_time + interval instead
    return df_htf[["open_time", "trend"]]


def htf_trend_as_of(htf_lookup: pd.DataFrame, ts: pd.Timestamp, htf_interval_td: pd.Timedelta) -> str:
    """Return the trend from the most recently CLOSED HTF candle at or before ts."""
    closed = htf_lookup[htf_lookup["open_time"] + htf_interval_td <= ts]
    if closed.empty:
        return None
    return closed.iloc[-1]["trend"]


def simulate_trade(df_entry: pd.DataFrame, entry_idx: int, signal: str, entry_price: float,
                    sl: float, tp: float) -> dict:
    """
    Walk forward candle-by-candle from entry_idx+1 to find whether SL or
    TP was hit first. Returns outcome dict. If neither hit by end of data,
    marks as 'OPEN' and uses last close for unrealized P&L accounting.
    """
    for j in range(entry_idx + 1, len(df_entry)):
        bar = df_entry.iloc[j]
        hit_tp = (bar["high"] >= tp) if signal == "BUY" else (bar["low"] <= tp)
        hit_sl = (bar["low"] <= sl) if signal == "BUY" else (bar["high"] >= sl)

        if hit_sl and hit_tp:
            # Conservative assumption: stop-loss fills first when both are touched in the same bar
            return {"outcome": "SL", "exit_price": sl, "exit_idx": j}
        elif hit_sl:
            return {"outcome": "SL", "exit_price": sl, "exit_idx": j}
        elif hit_tp:
            return {"outcome": "TP", "exit_price": tp, "exit_idx": j}

    # Ran off the end of the dataset without hitting either
    last_close = df_entry.iloc[-1]["close"]
    return {"outcome": "OPEN", "exit_price": last_close, "exit_idx": len(df_entry) - 1}


def run_backtest(df_entry: pd.DataFrame, df_htf: pd.DataFrame, use_htf_filter: bool = True) -> dict:
    df_entry = compute_indicators(df_entry.copy())
    htf_lookup = build_htf_lookup(df_htf) if use_htf_filter else None
    htf_interval_td = pd.Timedelta(hours=1)

    trades = []
    i = 30   # warmup period for indicators (EMA26/MACD/ATR need this much history)
    in_position = False

    while i < len(df_entry) - 1:
        if in_position:
            i += 1
            continue

        sub = df_entry.iloc[: i + 1]
        ts = sub.iloc[-1]["open_time"]

        htf_trend = None
        if use_htf_filter:
            htf_trend = htf_trend_as_of(htf_lookup, ts, htf_interval_td)

        result = generate_signal(sub, htf_trend=htf_trend)

        if result["signal"] in ("BUY", "SELL"):
            outcome = simulate_trade(df_entry, i, result["signal"], result["entry"], result["sl"], result["tp"])

            entry_price = result["entry"]
            exit_price = outcome["exit_price"]

            if result["signal"] == "BUY":
                raw_pct = (exit_price - entry_price) / entry_price
            else:
                raw_pct = (entry_price - exit_price) / entry_price

            net_pct = raw_pct - (2 * TAKER_FEE)  # round-trip fee

            risk_usd = ACCOUNT_SIZE_USDT * RISK_PCT_PER_TRADE
            stop_distance_pct = abs(entry_price - result["sl"]) / entry_price
            position_notional = risk_usd / stop_distance_pct if stop_distance_pct > 0 else 0
            position_notional = min(position_notional, ACCOUNT_SIZE_USDT)
            pnl_usd = position_notional * net_pct

            trades.append({
                "entry_time": ts, "exit_time": df_entry.iloc[outcome["exit_idx"]]["open_time"],
                "signal": result["signal"], "entry_price": entry_price, "exit_price": exit_price,
                "sl": result["sl"], "tp": result["tp"], "outcome": outcome["outcome"],
                "raw_pct": round(raw_pct * 100, 3), "net_pct": round(net_pct * 100, 3),
                "position_notional": round(position_notional, 2), "pnl_usd": round(pnl_usd, 4),
                "score": result["score"], "htf_trend": htf_trend,
            })

            i = outcome["exit_idx"] + 1
        else:
            i += 1

    return summarize(trades)


def summarize(trades: list) -> dict:
    if not trades:
        return {"trades": [], "summary": {"n_trades": 0, "message": "No signals fired in this window."}}

    df = pd.DataFrame(trades)
    closed = df[df["outcome"] != "OPEN"]
    wins = closed[closed["outcome"] == "TP"]
    losses = closed[closed["outcome"] == "SL"]

    n = len(closed)
    win_rate = len(wins) / n * 100 if n > 0 else 0
    avg_win_pct = wins["net_pct"].mean() if len(wins) > 0 else 0
    avg_loss_pct = losses["net_pct"].mean() if len(losses) > 0 else 0
    expectancy_pct = closed["net_pct"].mean() if n > 0 else 0

    total_pnl = closed["pnl_usd"].sum()
    equity_curve = closed["pnl_usd"].cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

    # Simple Sharpe-like ratio on per-trade % returns (not annualized — just a consistency signal)
    if closed["net_pct"].std() > 0:
        sharpe_like = closed["net_pct"].mean() / closed["net_pct"].std()
    else:
        sharpe_like = 0.0

    summary = {
        "n_trades_total": len(df),
        "n_trades_closed": n,
        "n_open_at_end": len(df) - n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win_pct, 3),
        "avg_loss_pct": round(avg_loss_pct, 3),
        "expectancy_pct_per_trade": round(expectancy_pct, 3),
        "total_pnl_usd": round(total_pnl, 2),
        "max_drawdown_usd": round(max_drawdown, 2),
        "consistency_ratio": round(sharpe_like, 2),
        "buy_trades": int((df["signal"] == "BUY").sum()),
        "sell_trades": int((df["signal"] == "SELL").sum()),
    }
    return {"trades": trades, "summary": summary}


def print_summary(label: str, result: dict):
    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if s.get("n_trades", -1) == 0:
        print("  " + s["message"])
        return
    print(f"  Total signals fired   : {s['n_trades_total']}  (BUY: {s['buy_trades']}, SELL: {s['sell_trades']})")
    print(f"  Closed trades         : {s['n_trades_closed']}  (still open at end: {s['n_open_at_end']})")
    print(f"  Win rate              : {s['win_rate_pct']}%  ({s['n_wins']}W / {s['n_losses']}L)")
    print(f"  Avg win / avg loss    : {s['avg_win_pct']}% / {s['avg_loss_pct']}%")
    print(f"  Expectancy per trade  : {s['expectancy_pct_per_trade']}%  (net of {TAKER_FEE*200:.2f}% round-trip fees)")
    print(f"  Total P&L on ${ACCOUNT_SIZE_USDT:.0f} acct  : ${s['total_pnl_usd']}")
    print(f"  Max drawdown          : ${s['max_drawdown_usd']}")
    print(f"  Consistency ratio*    : {s['consistency_ratio']}  (mean return / std dev — higher & positive is steadier)")
    print(f"{'='*60}")


def main():
    print(f"Fetching {DAYS_BACK} days of {SYMBOL} {ENTRY_TF} candles...")
    df_entry = fetch_klines_paginated(SYMBOL, ENTRY_TF, DAYS_BACK)
    print(f"  → {len(df_entry)} candles fetched ({df_entry['open_time'].min()} to {df_entry['open_time'].max()})")

    print(f"Fetching {DAYS_BACK} days of {SYMBOL} {HTF_TF} candles...")
    df_htf = fetch_klines_paginated(SYMBOL, HTF_TF, DAYS_BACK)
    print(f"  → {len(df_htf)} candles fetched")

    print("\nRunning backtest WITH 1h trend filter...")
    result_with_filter = run_backtest(df_entry, df_htf, use_htf_filter=True)
    print_summary(f"WITH 1h HTF Filter  |  {ENTRY_TF} entries  |  {DAYS_BACK}d window", result_with_filter)

    print("\nRunning backtest WITHOUT 1h trend filter (for comparison)...")
    result_no_filter = run_backtest(df_entry, df_htf, use_htf_filter=False)
    print_summary(f"WITHOUT HTF Filter (raw 5-indicator score only)  |  {DAYS_BACK}d window", result_no_filter)

    # Save trade logs to CSV for your own inspection
    if result_with_filter["trades"]:
        pd.DataFrame(result_with_filter["trades"]).to_csv("backtest_trades_with_filter.csv", index=False)
        print("\nSaved: backtest_trades_with_filter.csv")
    if result_no_filter["trades"]:
        pd.DataFrame(result_no_filter["trades"]).to_csv("backtest_trades_no_filter.csv", index=False)
        print("Saved: backtest_trades_no_filter.csv")

    print("""
NOTE ON INTERPRETING THESE RESULTS:
- This window is one slice of market history (one regime: trending,
  ranging, or volatile). A strategy that worked in this window may not
  work in the next one — markets change character. Re-run periodically.
- Small sample sizes (under ~30 closed trades) make win rate and
  expectancy numbers noisy. Treat results as directional, not precise.
- This does not model partial fills, exchange downtime, or slippage
  beyond the flat fee assumption. Real execution will be slightly worse.
""")


if __name__ == "__main__":
    main()

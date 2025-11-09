import os
import time
import json
import re
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from binance.client import Client
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Config
# -------------------------
CONFIG = {
    "SYMBOL": os.getenv("SYMBOL", "BTCUSDT"),

    "INTERVALS": os.getenv("INTERVALS", "1m,5m,15m").split(","),
    "TRIGGER_INTERVAL": os.getenv("TRIGGER_INTERVAL", "5m"),
    "LOOKBACK": int(os.getenv("LOOKBACK", 200)),

    # LLM (Ollama) — single model setup
    "FAST_MODEL": os.getenv("FAST_MODEL", "llama3:8b"),
    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
    "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "LLM_TOP_P": float(os.getenv("LLM_TOP_P", "0.9")),

    # Signal filtering
    "CONF_THRESHOLD": int(os.getenv("CONF_THRESHOLD", 90)),
    "REFRESH_SEC": int(os.getenv("REFRESH_SEC", 30)),
    "MIN_BARS_BETWEEN_FLIPS": int(os.getenv("MIN_BARS_BETWEEN_FLIPS", 3)),

    # Position tracking
    "START_POSITION": os.getenv("START_POSITION", "SHORT").upper(),
    "START_ENTRY": float(os.getenv("START_ENTRY", "102573")),
    "START_LEVERAGE": float(os.getenv("START_LEVERAGE", "10")),

    # Macro assets
    "MACRO_ASSETS": os.getenv("MACRO_ASSETS", "^GSPC,^NDX,^STOXX50E,GC=F,SI=F,^VIX,^TNX").split(","),
    "MACRO_NAMES": {
        "^GSPC": "S&P 500",
        "^NDX": "NASDAQ 100",
        "^STOXX50E": "Euro Stoxx 50",
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "^VIX": "Volatility Index (VIX)",
        "^TNX": "US 10Y Treasury Yield"
    }
}


client = Client()  # public endpoints only

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

# -------------------------
# Market data & indicators
# -------------------------
def fetch_klines(symbol: str, interval: str, limit: int):
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    ema_fast = EMAIndicator(close, window=21).ema_indicator()
    ema_slow = EMAIndicator(close, window=55).ema_indicator()
    rsi = RSIIndicator(close, window=14).rsi()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low  = bb.bollinger_lband()

    # CVD (simple proxy: up-candle adds vol, down-candle subtracts vol)
    cvd, cum = [], 0
    for _, row in df.iterrows():
        if row["close"] > row["open"]:
            cum += row["volume"]
        elif row["close"] < row["open"]:
            cum -= row["volume"]
        cvd.append(cum)

    out = df.copy()
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["rsi"] = rsi
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["bb_high"] = bb_high
    out["bb_low"] = bb_low
    out["cvd"] = cvd
    return out.dropna()

def summarize_indicators(df: pd.DataFrame):
    row = df.iloc[-1]
    if len(df) >= 2:
        row_prev = df.iloc[-2]
        prev_high = round(float(row_prev["high"]), 2)
        prev_low = round(float(row_prev["low"]), 2)
    else:
        prev_high = round(float(row["high"]), 2)
        prev_low = round(float(row["low"]), 2)

    trend = "uptrend" if row["ema_fast"] > row["ema_slow"] else "downtrend"

    return {
        "close": round(float(row["close"]), 2),
        "ema_fast": round(float(row["ema_fast"]), 2),
        "ema_slow": round(float(row["ema_slow"]), 2),
        "trend": trend,
        "rsi": round(float(row["rsi"]), 2),
        "macd_hist": round(float(row["macd_hist"]), 5),
        "bb_low": round(float(row["bb_low"]), 2),
        "bb_high": round(float(row["bb_high"]), 2),
        "prev_high": prev_high,
        "prev_low": prev_low,
        "cvd": round(float(row["cvd"]), 2),
    }

def detect_support_resistance(df: pd.DataFrame, window: int = 20):
    highs = df["high"].rolling(window).max().dropna()
    lows = df["low"].rolling(window).min().dropna()
    return {
        "support": round(lows.iloc[-1], 2) if len(lows) else None,
        "resistance": round(highs.iloc[-1], 2) if len(highs) else None
    }

# -------------------------
# Order book & live price
# -------------------------
def summarize_order_book(symbol="BTCUSDT", limit=20):
    ob = client.get_order_book(symbol=symbol, limit=limit)
    bids = [(float(p), float(q)) for p, q in ob["bids"]]
    asks = [(float(p), float(q)) for p, q in ob["asks"]]
    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
    return {
        "imbalance": round(imbalance, 3),
        "best_bid": bids[0][0],
        "best_ask": asks[0][0],
        "spread": round(asks[0][0] - bids[0][0], 2)
    }

# -------------------------
# Macro data
# -------------------------
def get_macro_data(tickers: list):
    out = {}
    try:
        # explicit auto_adjust to avoid FutureWarning
        data = yf.download(tickers, period="5d", interval="1d", progress=False, auto_adjust=False).ffill()
        for t in tickers:
            closes = data["Close"][t].dropna()
            label = CONFIG["MACRO_NAMES"].get(t, t)
            if len(closes) >= 2:
                last, prev = closes.iloc[-1], closes.iloc[-2]
                pct = (last - prev) / prev * 100 if prev else 0.0
                out[label] = {"last": round(float(last), 2), "pct_change": round(float(pct), 2)}
    except Exception as e:
        out["error"] = str(e)
    return out

# -------------------------
# News (CryptoPanic) with simple cache
# -------------------------
_last_news, _last_fetch = ["[No news cached yet]"], 0
def get_latest_news(limit=5, refresh_interval=900):
    global _last_news, _last_fetch
    API_KEY = os.getenv("CRYPTOPANIC_KEY", "")
    now = time.time()
    if not API_KEY:
        return _last_news
    if now - _last_fetch < refresh_interval:
        return _last_news
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&public=true"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        headlines = [p.get("title") for p in data.get("results", []) if p.get("title")]
        if headlines:
            _last_news = headlines[:limit]
            _last_fetch = now
    except Exception:
        pass
    return _last_news

# -------------------------
# Multi-timeframe data (summary only)
# -------------------------
def get_multi_timeframe_data(symbol, intervals, lookback=120):
    data = {}
    for interval in intervals:
        df = fetch_klines(symbol, interval, limit=lookback)
        df = compute_indicators(df)
        indicators_json = summarize_indicators(df)
        data[interval] = {"indicators": indicators_json}
    return data

# -------------------------
# LLM Prompt (tighter, rationale-focused)
# -------------------------
SYSTEM_PROMPT = """You are a disciplined crypto trading assistant.

Decide ONE of:
ENTER_LONG, ENTER_SHORT, CUT_LONG, CUT_SHORT, HOLD_LONG, HOLD_SHORT, HOLD_FLAT.

Constraints:
- Be proactive but justified. Cut losses early and let gainers gain.
- Give more weight to VOLUME/CVD and ORDER BOOK imbalance.
- Use RSI/MACD for momentum, EMA(21/55) for trend, BBands for extremes, and nearest SUPPORT/RESISTANCE for risk.
- Mention news ONLY if game-changing.
- No generic text.

Output exactly 3 lines:
1) Action (one of the 7)
2) Rationale: mention at least TWO of {RSI, MACD, EMA, CVD/volume, order book imbalance, support/resistance, Bollinger bands, higher timeframe trend}
3) Confidence: 0..100
"""

# -------------------------
# Prompt building
# -------------------------
def build_prompt(trading_tf: str, mtf_data: dict, order_book: dict, macro: dict, news: list,
                 live_price: float, position: dict, sr_levels: dict):
    def fmt_tf(tf: str, d: dict) -> str:
        ind = d["indicators"]
        return (f"{tf}: close={ind['close']}, trend={ind['trend']}, rsi={ind['rsi']}, "
                f"macd_hist={ind['macd_hist']}, cvd={ind['cvd'] if 'cvd' in ind else 'NA'}, "
                f"ema_fast={ind['ema_fast']}, ema_slow={ind['ema_slow']}, "
                f"bb_low={ind['bb_low']}, bb_high={ind['bb_high']}")

    ordered = [trading_tf] + [tf for tf in mtf_data.keys() if tf != trading_tf]
    tf_lines = [fmt_tf(tf, mtf_data[tf]) for tf in ordered]

    if position["status"] == "FLAT":
        pos_str = "FLAT"
    else:
        pos_str = f"{position['status']} @ {position['entry_price']} x{position['leverage']}"

    prompt = f"""{SYSTEM_PROMPT}

SYMBOL: {CONFIG['SYMBOL']}
TRADING_TF: {trading_tf}
CURRENT_POSITION: {pos_str}
LIVE_PRICE: {round(live_price,2)}
SUPPORT: {sr_levels.get('support')}  RESISTANCE: {sr_levels.get('resistance')}

TIMEFRAMES:
{chr(10).join(tf_lines)}

ORDER_BOOK: {json.dumps(order_book, separators=(',',':'))}
MACRO: {json.dumps(macro, separators=(',',':'))}
NEWS: {json.dumps(news, separators=(',',':'))}
"""
    return prompt

def ask_llm_core(prompt: str, model_name: str):
    try:
        r = requests.post(
            CONFIG["OLLAMA_URL"],
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": CONFIG["LLM_TEMPERATURE"],
                    "top_p": CONFIG["LLM_TOP_P"]
                }
            },
            timeout=90
        )
        r.raise_for_status()
        return r.json().get("response", "HOLD_FLAT\nNo response\n50")
    except Exception:
        return "HOLD_FLAT\nLLM error\n50"

def parse_llm_output(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    action, rationale, conf = "HOLD_FLAT", "No rationale", 50

    if lines:
        first = lines[0].upper()
        valid = {"ENTER_LONG","ENTER_SHORT","CUT_LONG","CUT_SHORT","HOLD_LONG","HOLD_SHORT","HOLD_FLAT"}
        if first in valid:
            action = first
    if len(lines) > 1:
        rationale = lines[1]
        # sanitize prefixes like "Rationale:", "Action:", etc.
        for bad in ("rationale:", "action:", "decision:"):
            if rationale.lower().startswith(bad):
                rationale = rationale.split(":", 1)[1].strip()
        rationale = rationale[:250]
    if len(lines) > 2:
        m = re.search(r"\b(\d{1,3})\b", lines[2])
        if m:
            conf = max(0, min(100, int(m.group(1))))

    # Confidence guardrail: downgrade low-confidence flips to holds
    if conf < CONFIG["CONF_THRESHOLD"]:
        if action.startswith("ENTER_"):
            action = "HOLD_FLAT"
        elif action.startswith("CUT_"):
            action = action.replace("CUT", "HOLD")

    return action, rationale, conf

# -------------------------
# Fallback rationale (ensures a meaningful explanation if LLM is terse)
# -------------------------
def compose_fallback_rationale(trading_tf: str, mtf_data: dict, order_book: dict, sr_levels: dict) -> str:
    ind = mtf_data[trading_tf]["indicators"]
    parts = []

    # Trend via EMAs
    parts.append(f"EMA trend {ind['trend']} (21={ind['ema_fast']}, 55={ind['ema_slow']})")

    # RSI
    rsi = ind["rsi"]
    if rsi >= 70:
        parts.append(f"RSI {rsi} (overbought)")
    elif rsi <= 30:
        parts.append(f"RSI {rsi} (oversold)")
    else:
        parts.append(f"RSI {rsi} (neutral)")

    # MACD hist
    macd_hist = ind["macd_hist"]
    parts.append(f"MACD hist {macd_hist:+}")

    # CVD (volume bias)
    parts.append(f"CVD {ind['cvd']} (volume bias)")

    # Order book
    imb = order_book["imbalance"]
    ob_side = "bid" if imb > 0 else "ask"
    parts.append(f"OB imbalance {imb:+} ({ob_side}-side heavy)")

    # Proximity to S/R
    sup = sr_levels.get("support")
    res = sr_levels.get("resistance")
    px = ind["close"]
    if sup and res:
        # relative distance
        ds = abs(px - sup) / max(px, 1e-9) * 100
        dr = abs(res - px) / max(px, 1e-9) * 100
        parts.append(f"near S/R (≈{ds:.2f}% from support, ≈{dr:.2f}% from resistance)")

    return "; ".join(parts)

def ensure_rationale(trading_tf: str, mtf_data: dict, order_book: dict, sr_levels: dict, rationale: str) -> str:
    # If the model gave a too-short or non-informative rationale, compose a fallback one.
    keywords = ("rsi", "macd", "ema", "cvd", "order book", "imbalance", "support", "resistance", "bollinger")
    short = len(rationale) < 40
    lacks_signal_words = not any(k in rationale.lower() for k in keywords)

    if short or lacks_signal_words:
        auto = compose_fallback_rationale(trading_tf, mtf_data, order_book, sr_levels)
        # If the LLM provided something, keep it AND append indicators summary
        if rationale and rationale.lower() != "no rationale":
            return f"{rationale} — {auto}"
        return auto
    return rationale

# -------------------------
# Ask with optional deep confirmation
# -------------------------
def ask_llm_with_confirmation(trading_tf, mtf_data, order_book, macro, news, live_price,
                              position, sr_levels):
    """
    Single-model decision (FAST_MODEL only).
    """
    prompt = build_prompt(trading_tf, mtf_data, order_book, macro, news, live_price, position, sr_levels)

    resp = ask_llm_core(prompt, CONFIG["FAST_MODEL"])
    action, rationale, conf = parse_llm_output(resp)

    # Enrich weak rationale
    rationale = ensure_rationale(trading_tf, mtf_data, order_book, sr_levels, rationale)

    return action, rationale, conf


# -------------------------
# Helpers
# -------------------------
def get_latest_close_time(symbol: str, interval: str):
    df = fetch_klines(symbol, interval, limit=2)
    return df.iloc[-1]["close_time"]

# -------------------------
# Main loop
# -------------------------
def main():
    trading_tf = CONFIG["TRIGGER_INTERVAL"]
    if trading_tf not in CONFIG["INTERVALS"]:
        CONFIG["INTERVALS"].insert(0, trading_tf)

    position = {
        "status": CONFIG["START_POSITION"] if CONFIG["START_POSITION"] in {"FLAT","LONG","SHORT"} else "FLAT",
        "entry_price": (CONFIG["START_ENTRY"] if CONFIG["START_POSITION"] in {"LONG","SHORT"} else None),
        "leverage": CONFIG["START_LEVERAGE"]
    }

    print(f"[{now_iso()}] Bot started — {CONFIG['SYMBOL']} trading={trading_tf} — START_POS={position['status']} @ {position['entry_price']} x{position['leverage']}")

    last_close_time = None
    bars_since_action = CONFIG["MIN_BARS_BETWEEN_FLIPS"]

    while True:
        try:
            # Trading TF S/R
            df_trading = compute_indicators(fetch_klines(CONFIG["SYMBOL"], trading_tf, CONFIG["LOOKBACK"]))
            sr_levels = detect_support_resistance(df_trading, window=20)

            # Multi-timeframe summary
            mtf_data = {}
            for tf in CONFIG["INTERVALS"]:
                df_i = compute_indicators(fetch_klines(CONFIG["SYMBOL"], tf, CONFIG["LOOKBACK"]))
                mtf_data[tf] = {"indicators": summarize_indicators(df_i)}

            # Order book / macro / news
            order_book = summarize_order_book(CONFIG["SYMBOL"], limit=20)
            live_price = (order_book["best_bid"] + order_book["best_ask"]) / 2
            macro = get_macro_data(CONFIG["MACRO_ASSETS"])
            news = get_latest_news()

            # New candle closed? (for cooldown in bars)
            latest_close_time = df_trading.iloc[-1]["close_time"]
            if (last_close_time is None) or (latest_close_time > last_close_time):
                bars_since_action += 1
                last_close_time = latest_close_time

            # Ask LLM(s)
            action, rationale, confidence = ask_llm_with_confirmation(
                trading_tf, mtf_data, order_book, macro, news, live_price, position, sr_levels
            )

            # Position-aware execution with cooldown
            flip = False
            if position["status"] == "FLAT":
                if action in {"ENTER_LONG","ENTER_SHORT"} and confidence >= CONFIG["CONF_THRESHOLD"] and bars_since_action >= CONFIG["MIN_BARS_BETWEEN_FLIPS"]:
                    position["status"] = "LONG" if action == "ENTER_LONG" else "SHORT"
                    position["entry_price"] = live_price
                    flip = True
                    bars_since_action = 0
                else:
                    action = "HOLD_FLAT"
            elif position["status"] == "LONG":
                if action in {"CUT_LONG","ENTER_SHORT"} and confidence >= CONFIG["CONF_THRESHOLD"] and bars_since_action >= CONFIG["MIN_BARS_BETWEEN_FLIPS"]:
                    if action == "ENTER_SHORT":
                        position["status"] = "SHORT"
                        position["entry_price"] = live_price
                    else:
                        position["status"] = "FLAT"
                        position["entry_price"] = None
                    flip = True
                    bars_since_action = 0
                else:
                    action = "HOLD_LONG"
            elif position["status"] == "SHORT":
                if action in {"CUT_SHORT","ENTER_LONG"} and confidence >= CONFIG["CONF_THRESHOLD"] and bars_since_action >= CONFIG["MIN_BARS_BETWEEN_FLIPS"]:
                    if action == "ENTER_LONG":
                        position["status"] = "LONG"
                        position["entry_price"] = live_price
                    else:
                        position["status"] = "FLAT"
                        position["entry_price"] = None
                    flip = True
                    bars_since_action = 0
                else:
                    action = "HOLD_SHORT"

            # PnL
            pnl = None
            if position["status"] in {"LONG","SHORT"} and position["entry_price"]:
                direction = 1 if position["status"] == "LONG" else -1
                pnl = ((live_price - position["entry_price"]) / position["entry_price"]) * 100 * direction * position["leverage"]

            # Print
            tf_close = mtf_data[trading_tf]["indicators"]["close"]
            print("="*70)
            print(f"[{now_iso()}] {CONFIG['SYMBOL']} — trading={trading_tf} — POS={position['status']} @ {position['entry_price']} x{position['leverage']}")
            print(f"Live={round(live_price,2)} | TF Close={tf_close}")
            print(f"Support={sr_levels.get('support')} | Resistance={sr_levels.get('resistance')}")
            if pnl is not None:
                print(f"Unrealized PnL (levered): {round(pnl,2)}%")
            print(f"Decision: {action} (conf {confidence}%) {'<== FLIP' if flip else ''}")
            print(f"Rationale: {rationale}")
            print(f"OrderBook -> Imb: {order_book['imbalance']} | Spread: {order_book['spread']}")
            print("="*70)

            time.sleep(CONFIG["REFRESH_SEC"])

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[{now_iso()}] Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()

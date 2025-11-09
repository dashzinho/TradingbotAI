# Crypto Trading Bot (Binance + Ollama AI)

A Python-based **AI-assisted crypto trading bot** that combines **technical indicators**, **macro data**, **order book analysis**, and **LLM reasoning** (via [Ollama](https://ollama.com)) to generate structured trading decisions in real time.

---

## Overview

This bot continuously monitors the **Binance** market and uses a **local LLM model** to decide between:

```
ENTER_LONG | ENTER_SHORT | CUT_LONG | CUT_SHORT | HOLD_LONG | HOLD_SHORT | HOLD_FLAT
```

It fetches live market data, computes technical indicators, evaluates macro conditions, and queries an LLM for concise, rationale-backed trade actions.

> **Note:** This version is for research and signal generation only — it does **not** execute trades.

---

## Features

- Multi-timeframe technical analysis (EMA, RSI, MACD, Bollinger Bands, CVD)
- LLM-based decision engine using **Ollama** (`llama3:8b` by default)
- Order book imbalance and spread analysis
- Macro data integration (S&P 500, NASDAQ, Gold, VIX, Treasury Yield, etc.)
- Optional crypto news headlines from [CryptoPanic](https://cryptopanic.com/)
- Configurable environment and runtime parameters
- Position tracking with simulated leverage and PnL

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the root directory:
```ini
SYMBOL=BTCUSDT
INTERVALS=1m,5m,15m
TRIGGER_INTERVAL=5m
LOOKBACK=200

# Ollama LLM
OLLAMA_URL=http://localhost:11434/api/generate
FAST_MODEL=llama3:8b
LLM_TEMPERATURE=0.1
LLM_TOP_P=0.9

# Logic thresholds
CONF_THRESHOLD=90
REFRESH_SEC=30
MIN_BARS_BETWEEN_FLIPS=3

# Position tracking
START_POSITION=FLAT
START_ENTRY=102573
START_LEVERAGE=10

# Macro data tickers
MACRO_ASSETS=^GSPC,^NDX,^STOXX50E,GC=F,SI=F,^VIX,^TNX

# Optional: CryptoPanic API key
CRYPTOPANIC_KEY=your_api_key_here
```

---

## Usage
Start the local Ollama server:
```bash
ollama serve
```

Then run the bot:
```bash
python bot.py
```

Example console output:
```
======================================================================
[2025-11-09T20:43:00+00:00] BTCUSDT — trading=5m — POS=FLAT @ None x10
Live=102573.2 | Support=102400 | Resistance=103000
Decision: ENTER_LONG (conf 94%) <== FLIP
Rationale: RSI recovery; MACD bullish; order book +0.22 imbalance
======================================================================
```

---

## Indicators
| Category | Indicator | Description |
|-----------|------------|-------------|
| Trend | EMA(21/55) | Detects short/long-term trend direction |
| Momentum | RSI(14), MACD(12,26,9) | Identifies overbought/oversold zones and crossovers |
| Volatility | Bollinger Bands(20,2) | Measures price extremes |
| Volume | CVD (Cumulative Volume Delta) | Tracks buying vs. selling pressure |
| Order Flow | Bid/Ask imbalance | Detects liquidity bias |
| Macro | S&P 500, Gold, VIX, etc. | Provides risk sentiment context |

---

## Disclaimer
This project is for **educational and research purposes only**.  
It does **not execute real trades**, and no guarantee of profitability is implied.  
Use responsibly and at your own risk.

---

## License
MIT License © 2025 André Santos

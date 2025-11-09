🧠 Crypto Trading Bot (Binance + Ollama AI)

A Python-based crypto trading bot that combines technical indicators, macro data, order book analysis, and LLM reasoning (via Ollama) to generate trading decisions in real time.

🚀 Overview

This bot continuously monitors the Binance market and uses a local LLM model to make data-driven trading decisions such as:

ENTER_LONG, ENTER_SHORT, CUT_LONG, CUT_SHORT, HOLD_LONG, HOLD_SHORT, HOLD_FLAT

It runs autonomously, fetching live data, computing indicators, reading macro context, analyzing order book signals, and prompting an LLM for the next action.

⚠️ Note: This version is for signal generation and research only — it does not place real trades.

🧩 Features

📊 Multi-timeframe technical analysis (EMA, RSI, MACD, Bollinger Bands, CVD)

🧠 LLM-based decision making with Ollama (llama3:8b by default)

💹 Order book volume imbalance detection

🌍 Macro data integration via Yahoo Finance (S&P 500, NASDAQ, Gold, VIX, etc.)

📰 Crypto news feed via CryptoPanic

⚙️ Configurable via .env

💼 Position tracking with PnL simulation and cooldown logic

⚙️ Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Configure environment variables

Create a .env file in the project root:

SYMBOL=BTCUSDT
INTERVALS=1m,5m,15m
TRIGGER_INTERVAL=5m
LOOKBACK=200

# LLM (Ollama)
OLLAMA_URL=http://localhost:11434/api/generate
FAST_MODEL=llama3:8b
LLM_TEMPERATURE=0.1
LLM_TOP_P=0.9

# Thresholds
CONF_THRESHOLD=90
REFRESH_SEC=30
MIN_BARS_BETWEEN_FLIPS=3

# Starting position
START_POSITION=FLAT
START_ENTRY=102573
START_LEVERAGE=10

# Macro tickers
MACRO_ASSETS=^GSPC,^NDX,^STOXX50E,GC=F,SI=F,^VIX,^TNX

# Optional CryptoPanic key
CRYPTOPANIC_KEY=your_api_key_here

▶️ Run the Bot

Ensure your Ollama server is running (ollama serve), then start the bot:

python bot.py


Example output:

======================================================================
[2025-11-09T20:43:00+00:00] BTCUSDT — trading=5m — POS=FLAT @ None x10
Live=102573.2 | Support=102400 | Resistance=103000
Decision: ENTER_LONG (conf 94%) <== FLIP
Rationale: RSI recovery; MACD bullish; order book +0.22 imbalance
======================================================================

📈 Indicators
Indicator	Description
EMA(21/55)	Trend direction
RSI(14)	Overbought / oversold
MACD(12,26,9)	Momentum shift
Bollinger Bands(20,2)	Volatility bands
CVD	Volume flow bias
Order Book	Buy/Sell imbalance
Macro	S&P 500, Gold, VIX, etc.

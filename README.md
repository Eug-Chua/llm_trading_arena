# LLM Trading Arena

This project seeks to replicate [Alpha Arena](https://nof1.ai/) by Nof1.ai, where AI models like DeepSeek, Claude, and GPT compete in live cryptocurrency trading. Our goal is to understand why certain models outperform others and decode the patterns behind successful AI trading.

---

## What is This Project?

**Alpha Arena** is a groundbreaking experiment by Nof1.ai that tests whether LLMs can function as systematic trading agents in real-world, dynamic environments. Unlike static AI benchmarks, Alpha Arena evaluates decision-making capabilities with real capital, live market data, and consequential execution.

**Six leading LLMs** (DeepSeek, Claude, GPT-5, Gemini, Grok 4, Qwen 3) trade cryptocurrency perpetual futures on Hyperliquid with:
- **$10,000 each** in real capital
- **Identical prompts** and system configurations
- **Live market data** (prices, technical indicators, volume)
- **No news or narratives** - pure numerical trading
- **~2-3 minute decision cycles** (mid-to-low frequency trading)

### Our Replication Goals

This project aims to replicate and extend Alpha Arena's research by:
1. **Reproducing the exact trading environment** (prompt format, data pipeline, execution logic)
2. **Testing LLMs head-to-head** in controlled, reproducible conditions
3. **Understanding behavioral differences** across models (risk profiles, biases, patterns)

**Key Research Questions:**
- Can LLMs trade systematically with minimal guidance (zero-shot capability)?
- What decision-loop components can safely run autonomously vs. where do models fail?
- Do models exhibit distinct risk profiles, directional biases, and sizing preferences?

*Note: All agents receive the same user prompt template. System prompts are proprietary to Nof1.ai.*

---

## Why This Matters

### The Big Question
Can LLMs actually trade? Do they reliably follow simple risk rules? Which parts of the decision loop can be trusted to run autonomously? Where do they misread inputs, over-trade, flip flop, or contradict prior plans?

### What We're Learning

1. **Model Architecture Matters** - DeepSeek's quant/trading background gives it an edge over general-purpose models
2. **Less is More** - Models perform better with minimal trading rules vs. explicit instructions
3. **Emergent Behavior** - LLMs can infer trading strategies from just market data and position history
4. **Risk Management** - The best models are conservative, not aggressive

### Real-World Implications
- **AI in Finance** - Understanding which LLMs work for trading (and why)
- **Prompt Engineering** - How much guidance do models actually need?
- **Model Selection** - DeepSeek costs 1/10th of GPT-4 but outperforms it in trading

---

## How It Works

The system operates in a continuous loop:

```
1. Fetch Market Data (BTC, ETH, SOL, BNB, XRP, DOGE)
   ↓
2. Calculate Technical Indicators (EMA, MACD, RSI, ATR)
   ↓
3. Generate Alpha Arena-Style Prompt
   ↓
4. Send to LLM (DeepSeek, GPT, Claude, etc.)
   ↓
5. Parse LLM Response (Chain of Thought + Trade Signals)
   ↓
6. Execute Trades (Buy/Hold/Close positions)
   ↓
7. Track Performance (Returns, Sharpe Ratio, etc.)
   ↓
   Loop back to step 1 (every 3 minutes)
```

### Technical Indicators

The system uses 5 core technical indicators to analyze market conditions. All parameters are configured in `config/indicators.yaml`:

#### Price & Trend Indicators
- **EMA (Exponential Moving Average)**: 20-period (short-term trend) and 50-period (long-term trend)
  - Used for trend identification and crossover signals
  - EMA20 > EMA50 = bullish, EMA20 < EMA50 = bearish

#### Momentum & Oscillators
- **RSI (Relative Strength Index)**: 7-period (fast) and 14-period (standard)
  - Identifies overbought (>70) and oversold (<30) conditions
  - Helps time entries and exits based on momentum extremes

- **MACD (Moving Average Convergence Divergence)**: 12/26/9 configuration
  - Fast: 12-period, Slow: 26-period, Signal: 9-period
  - Detects momentum shifts and trend changes
  - Bullish when MACD line crosses above signal line

#### Volatility & Volume
- **ATR (Average True Range)**: 3-period (fast) and 14-period (standard)
  - Measures market volatility for position sizing and stop-loss placement
  - Higher ATR = wider stops, lower position size

- **Volume**: Raw trading volume for trend confirmation
  - High volume + price move = strong trend
  - Low volume + price move = weak/unreliable trend

#### Perpetuals-Specific Metrics
For crypto perpetual futures, the system also provides:
- **Open Interest**: Total outstanding derivative positions (market positioning)
- **Funding Rate**: Periodic payment between longs and shorts (market sentiment)

All indicators operate on **4-hour candle data** for mid-frequency trading decisions.

### Example LLM Response

```
# CHAIN OF THOUGHT

BTC: Current price $114,250 vs entry $110,000. Up $2,000.
Stop-loss at $108k, invalidation below $105k. Price safely above both.
RSI 73 shows momentum without being overbought. MACD positive.
Decision: HOLD - let this winner run.

ETH: Price $4,118 vs entry $4,100. Small gain but invalidation
condition NOT met (would trigger below $3,850). Holding.

# TRADING DECISIONS

{
  "BTC": {"signal": "hold", "quantity": 0.5, ...},
  "ETH": {"signal": "hold", "quantity": 5.0, ...}
}
```

The system parses this, validates it, and executes the trades.

---

## ⚠️ Disclaimer

**This project is for research and educational purposes only.**

- Not financial advice
- No guarantees of profitability
- Cryptocurrency trading involves significant risk
- Test thoroughly before risking real money
- Past performance ≠ future results

We're studying AI capabilities, not selling a trading system.

---

## Acknowledgments

- **Nof1.ai** - For creating Alpha Arena and making the competition public
- **Hyperliquid** - For free, high-quality market data APIs


*Last Updated: October 2025*

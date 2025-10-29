# LLM Trading Arena

This project seeks to replicate [Alpha Arena](https://nof1.ai/) by Nof1.ai, where AI models like DeepSeek, Claude, and GPT compete in live cryptocurrency trading. Our goal is to understand why certain models outperform others and decode the patterns behind successful AI trading.

---

## 🎯 What is This Project?

**Alpha Arena** is a live competition running from 17 Oct 25 to 3 Nov 25 where different LLMs (Large Language Models) trade crypto perpetuals autonomously. Each model receives market data, analyzes it, and makes trading decisions in real-time. As of 28 Oct 25, **DeepSeek handily outperforms** other models including GPT and Claude.

**This project:**
- Seeks to replicate the exact prompt format and trading environment from Alpha Arena
- Tests multiple LLM models head-to-head in controlled conditions
- Analyzes DeepSeek's outperformance

---

## 🔍 Why This Matters

### The Big Question
Can LLMs actually trade? And if so, which ones are best at it?

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

## 📊 How It Works

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

## 🙏 Acknowledgments

- **Nof1.ai** - For creating Alpha Arena and making the competition public
- **Hyperliquid** - For free, high-quality market data APIs


*Last Updated: October 2025*

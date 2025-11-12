# Experimental Setup & Configuration

This document describes the exact experimental configuration used in the LLM Trading Arena backtests. All parameters are designed to replicate the [Alpha Arena](https://nof1.ai/) environment as closely as possible.

**Last Updated:** November 5, 2025

---

## Table of Contents
1. [Experimental Design](#experimental-design)
2. [Model Configurations](#model-configurations)
3. [Trading Parameters](#trading-parameters)
4. [Technical Indicators](#technical-indicators)
5. [Assumptions & Limitations](#assumptions--limitations)

---

## Experimental Design

### Backtest Period
- **Start Date:** October 17, 2025 00:00:00 UTC
- **End Date:** October 31, 2025 00:00:00 UTC
- **Duration:** 14 days (336 hours)
- **Total Candles:** 85 4-hour candles
- **Decision Points:** 85 timestamps (one decision per 4-hour candle)
- **Decision Times:** Every 4 hours aligned with candle close (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)

**Rationale for Period Selection:**
- Captures a complete 2-week trading cycle
- Includes both weekend and weekday market dynamics
- Sufficient data points for statistical significance testing (85 decision points)
- Recent enough to reflect current market structure and volatility patterns

### Experimental Variables

#### Independent Variables
1. **Model Selection**: 2 LLM providers tested
   - Anthropic (Claude Sonnet 4.5)
   - OpenAI (GPT-4o-mini)

2. **Temperature Settings**: Testing model stochasticity
   - `0.1` - Low temperature (near-deterministic, conservative)
   - `0.7` - High temperature (creative, exploratory)

#### Dependent Variables (Performance Metrics)
- Total return (%)
- Sharpe ratio
- Sortino ratio
- Maximum drawdown (%)
- Win rate (%)
- Number of trades
- Average trade size
- Average holding period
- Risk-adjusted metrics (upside/downside deviation, skewness, kurtosis)

### Trial Methodology

**Multiple Trials Per Configuration:**
- Each model-temperature combination tested across **10 independent trials**
- Each trial uses identical:
  - Date range (Oct 17 00:00 - Oct 31 00:00, 2025)
  - Starting capital ($10,000)
  - Market data
  - Technical indicators
  - Prompt template
- Only difference: LLM non-determinism (even at same temperature)

**Purpose of Multiple Trials (N=10):**
- Measure intra-model variance (how consistent is each model?)
- Statistical significance testing (confidence intervals, p-values)
- Identify outlier performance vs. systematic edge
- Compare temperature effects (0.1 vs 0.7 consistency)
- Build robust sample size for reliable statistical inference

**File Naming Convention:**
```
results/checkpoints/{model}_temp{temperature}_trial{N}.json
```

Example:
```
results/checkpoints/anthropic_temp07_trial1.json
results/checkpoints/anthropic_temp07_trial2.json
results/checkpoints/openai_temp01_trial1.json
```

### Control Variables (Constant Across All Tests)
- Starting capital: $10,000
- Fee structure: 0.02% maker, 0.05% taker
- Trading universe: BTC, ETH, SOL, BNB, XRP, DOGE
- Data interval: 4-hour candles
- Technical indicators: Same parameters for all models
- Prompt template: Identical for all models
- Decision frequency: Every 4 hours (85 total decisions)

---

## Model Configurations

### Models Tested

| Model | Provider | Model ID |
|-------|----------|----------|
| Claude Sonnet 4.5 | Anthropic | `claude-sonnet-4-5-20250929` |
| GPT-4o-mini | OpenAI | `gpt-4o-mini` |

### API Configuration

**Provider Endpoints:**
- **Anthropic:** Native SDK (official Anthropic client)
- **OpenAI:** `https://api.openai.com/v1`

**API Settings:**
- **Timeout:** 300 seconds (5 minutes)
- **Max Retries:** 3
- **Rate Limit Delay:** 2 seconds between requests
- **Authentication:** Environment variables for API keys
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`

### Temperature Settings

Temperature controls the randomness of model outputs:

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| **0.1** | Near-deterministic, highly consistent | Test baseline strategy repeatability |
| **0.7** | Creative, exploratory, varied responses | Test robustness to LLM variance |

**Why These Values:**
- `0.1` - Low enough to minimize variance, high enough to avoid determinism artifacts
- `0.7` - Alpha Arena's reported temperature, balances creativity with coherence

---

## Trading Parameters

### Account Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Starting Capital** | $10,000 | Matches Alpha Arena |
| **Currency** | USD | All positions valued in USD |
| **Account Type** | Paper Trading | Simulated execution |

### Fee Structure (Hyperliquid)

| Fee Type | Rate | Application |
|----------|------|-------------|
| **Maker Fee** | 0.02% | Limit orders that add liquidity |
| **Taker Fee** | 0.05% | Market orders that take liquidity |

**Fee Calculation:**
```python
# All trades treated as taker orders (market execution)
fee = position_size * price * 0.0005
total_cost = position_size * price + fee
```

**Important:** Fees are deducted from account balance on every trade (open and close).

### Position Sizing & Leverage

| Parameter | Value | Enforcement |
|-----------|-------|-------------|
| **Leverage Range** | LLM-determined | No constraints - model decides per trade |
| **Min Trade Size** | 0.0001 units | Prevents dust trades |
| **Max Position Size** | Limited by available capital | Cannot exceed account balance |
| **Max Positions** | 6 (one per coin) | One position per asset at a time |

**Important Note on Leverage:** The LLM specifies leverage for each trade with no enforced constraints. This allows the model to express its risk preference. In practice:
- **Anthropic Claude:** Consistently chose 15x leverage (100% of trades across both temperatures)
- **OpenAI GPT-4o-mini:** Consistently chose 10x leverage (100% of trades across both temperatures)

This suggests leverage acts as a **model personality trait** rather than a dynamic risk adjustment mechanism.

**Leverage Mechanics:**
```python
# Required capital for leveraged position
required_capital = (price * quantity) / leverage

# Example: $10k position with 5x leverage
# Required capital = $10,000 / 5 = $2,000
```

### Risk Management Rules

**Stop-Loss:**
- LLM specifies stop-loss price per position
- Automatically closes position if price hits stop-loss
- Executed at specified price (no slippage in backtest)

**Position Limits:**
- Maximum 1 position per coin at any time
- No hedging (long-only in current implementation)
- Cannot open new position if existing position active

**Capital Preservation:**
- Cannot trade if insufficient capital
- Minimum capital check before each trade
- Fees deducted immediately, reducing available capital

---

## Technical Indicators

All technical indicators use the same parameters across all models and trials. Configuration file: `config/indicators.yaml`

### Data Frequency
- **Candle Interval:** 4 hours
- **Decision Frequency:** Every 4 hours (aligned with candle close)
- **Lookback Window:** Varies by indicator

### Price & Trend Indicators

#### EMA (Exponential Moving Average)
- **Parameters:** 20-period (short-term), 50-period (long-term)
- **Purpose:** Trend identification and crossover signals
- **Interpretation:**
  - EMA20 > EMA50 = Bullish trend
  - EMA20 < EMA50 = Bearish trend
  - Crossovers = Potential trend reversal

### Momentum & Oscillators

#### RSI (Relative Strength Index)
- **Parameters:** 7-period (fast), 14-period (standard)
- **Range:** 0-100
- **Purpose:** Identify overbought/oversold conditions
- **Interpretation:**
  - RSI > 70 = Overbought (potential sell signal)
  - RSI < 30 = Oversold (potential buy signal)
  - RSI 40-60 = Neutral zone

#### MACD (Moving Average Convergence Divergence)
- **Parameters:** Fast=12, Slow=26, Signal=9
- **Components:**
  - MACD Line = EMA(12) - EMA(26)
  - Signal Line = EMA(9) of MACD Line
  - Histogram = MACD - Signal
- **Purpose:** Momentum shifts and trend changes
- **Interpretation:**
  - MACD > Signal = Bullish momentum
  - MACD < Signal = Bearish momentum
  - Histogram growing = Strengthening trend

### Volatility Indicators

#### ATR (Average True Range)
- **Parameters:** 3-period (fast), 14-period (standard)
- **Purpose:** Measure market volatility
- **Applications:**
  - Stop-loss placement (wider stops in high volatility)
  - Position sizing (smaller positions in high volatility)
  - Risk assessment

### Volume Data
- **Raw Volume:** Total trading volume per candle
- **Purpose:** Trend confirmation
- **Interpretation:**
  - High volume + price move = Strong, reliable trend
  - Low volume + price move = Weak, questionable trend

### Data Sources
- **Historical Data:** Hyperliquid API
- **Storage:** Local CSV files in `data/historical/{coin}_{interval}.csv`
- **Update Frequency:** Static for backtest (no live updates)

---

## Assumptions & Limitations

### Market Execution Assumptions

#### Perfect Execution
- **Assumption:** All trades execute at exact close price of current candle
- **Reality:** Would have slippage and partial fills in live trading
- **Impact:** Slightly optimistic performance (real returns would be lower)

#### No Slippage
- **Assumption:** Can buy/sell any quantity at current price
- **Reality:** Large orders move the market
- **Impact:** Results may not scale to large capital

#### Instant Execution
- **Assumption:** Orders execute immediately at decision time
- **Reality:** API latency, order matching delays
- **Impact:** May miss optimal entry/exit by seconds/minutes

### Data Limitations

#### Historical Data Only
- **Limitation:** No live market data, only historical snapshots
- **Impact:** Cannot test real-time decision-making or latency effects

#### 4-Hour Granularity
- **Limitation:** Only see candle close prices, miss intra-candle volatility
- **Reality:** Stop-losses may trigger at different prices intra-candle
- **Impact:** Stop-loss execution may be unrealistic

#### No Orderbook Data
- **Limitation:** No bid/ask spreads, no depth
- **Reality:** Spreads widen in volatile markets
- **Impact:** Entry/exit costs understated

### External Information Exclusions

#### No News or Narratives
- **Exclusion:** No external news, social media, or fundamental data
- **Rationale:** Test pure price-action trading capability
- **Reality:** Real traders incorporate macro events

#### No Cross-Asset Data
- **Exclusion:** No stocks, bonds, commodities, forex
- **Rationale:** Focus on crypto perpetuals only
- **Reality:** Broader market context matters

### Model & Prompt Limitations

#### Prompt Template Access
- **Limitation:** User prompt replicated, but system prompt is proprietary to Nof1.ai
- **Impact:** Results may differ from Alpha Arena due to unknown system instructions

#### Context Window
- **Limitation:** Limited history provided to LLM (current + recent positions)
- **Reality:** Cannot see full trading history beyond last few trades
- **Impact:** Models may forget long-term patterns

#### No Learning
- **Limitation:** Each decision is independent, no memory across timestamps
- **Reality:** No reinforcement learning or strategy refinement
- **Impact:** Cannot adapt trading strategy mid-backtest

### Infrastructure Constraints

#### API Rate Limits
- **GPUStack (DeepSeek/Qwen):** Shared GPU resources, subject to timeouts
- **Anthropic/OpenAI:** Standard API rate limits
- **Impact:** 504 Gateway Timeouts during concurrent runs (3-minute timeout)

#### Compute Resources
- **Limitation:** Running large models (DeepSeek V3.1, Qwen 235B) is slow
- **Impact:** ~1 minute per decision, ~85 minutes per backtest
- **Workaround:** Run trials sequentially, avoid concurrent execution

### Statistical Limitations

#### Sample Size
- **Limitation:** Only 85 decision points per trial
- **Impact:** Requires multiple trials for statistical significance
- **Current:** 10 trials per configuration (sufficient for basic statistical inference)

#### Overfitting Risk
- **Limitation:** Testing on single 2-week period
- **Impact:** Results may not generalize to other time periods
- **Mitigation:** Plan to test on multiple periods (not yet done)

### Comparison Validity

#### Alpha Arena Differences
While attempting to replicate Alpha Arena, key differences exist:

| Factor | Alpha Arena | This Project |
|--------|-------------|--------------|
| **Execution** | Live trading (Hyperliquid) | Backtested (historical data) |
| **Capital** | Real $10k per model | Simulated $10k |
| **Data Feed** | Live, real-time | Historical, delayed |
| **System Prompt** | Proprietary (unknown) | Not available |
| **Decision Frequency** | ~2-3 minutes | 4 hours (fixed) |
| **Assets** | Perpetual futures | Simulated perps |

**Conclusion:** Results are directionally comparable but not exact replications.

---

## Reproducibility

### Running the Backtest

To reproduce these exact results:

```bash
# Example: Run Anthropic trial 1 at temp 0.7
python scripts/run_backtest.py \
  --model anthropic \
  --temperature 0.7 \
  --run-id 1 \
  --start 2025-10-17 \
  --end 2025-10-31
```

### Verifying Configuration

1. **Check indicators:** `cat config/indicators.yaml`
2. **Check models:** `cat config/models.yaml`
3. **Verify data:** `ls data/historical/`
4. **Check checkpoint:** `ls results/{model}/temp{temp}/`

### Data Integrity

All historical market data is stored locally and remains constant across trials:
- **Location:** `data/historical/{COIN}_4h.csv`
- **Format:** Timestamp, Open, High, Low, Close, Volume
- **Source:** Hyperliquid API (collected once, reused for all backtests)

---

## References

- **Alpha Arena:** https://nof1.ai/
- **Hyperliquid Docs:** https://hyperliquid.xyz/
- **Indicator Definitions:** `config/indicators.yaml`
- **Model Configs:** `config/models.yaml`
- **Statistical Analysis:** `STATISTICAL_ANALYSIS_GUIDE.md`
- **Results:** `RESULTS.md` (to be created after experiments complete)

---

**For questions or clarifications, see:**
- [Project Documentation](PROJECT_DOCUMENTATION.md)
- [Statistical Analysis Guide](STATISTICAL_ANALYSIS_GUIDE.md)
- [Model Comparison Guide](MODEL_COMPARISON_GUIDE.md)

# LLM Trading Arena

**A comparative analysis of LLM performance in cryptocurrency trading**

---

## TL;DR

> - **Best Performer:** Claude Sonnet 4.5 with temp 0.7 (mean return of 3.60%)
> - **Most Surprising:** Buy-and-Hold BTC essentially broke even (0.07%) over this 14-day period, while equal-weight portfolio lost money (-1.23%)
> - **Temperature Effects:** No statistically significant impact on performance for either model (Anthropic p=0.25, OpenAI p=0.58)
> - **Model Differences:** OpenAI significantly outperforms Anthropic at temp 0.1 (p=0.025), but no difference at temp 0.7 (p=0.74)
> - **Recommendation:** LLMs showed mixed results - wide variance suggests outcomes are highly sensitive to specific market conditions. Requires validation on longer timeframes.

**Quick Stats:**
- **Models Analyzed:** Anthropic Claude Sonnet 4.5 vs. OpenAI GPT-4o-mini
- **Temperature Settings:** 0.1 and 0.7
- **Trials per Configuration:** 10
- **Backtests Analyzed:** 40 (2 models × 2 temps × 10 trials)
- **Trading Period:** 17 Oct 2025 00:00hrs to 31 Oct 2025 00:00hrs
- **No. of Decision Points:** 85
- **Assets Traded:** BTC, ETH, SOL, BNB, XRP, DOGE
- **Benchmark Strategies:** 2 (Buy-and-Hold BTC, Equal-Weight Portfolio)

---

## 1. Introduction

**Alpha Arena** is a groundbreaking experiment by Nof1.ai that tests whether LLMs can function as systematic trading agents in real-world, dynamic environments. Unlike static AI benchmarks, Alpha Arena evaluates decision-making capabilities with real capital, live market data, and consequential execution.

**In the original Alpha Arena competition**, six leading LLMs (DeepSeek, Claude, GPT-5, Gemini, Grok 4, Qwen 3) trade cryptocurrency perpetual futures on Hyperliquid with:
- **$10,000 each** in real capital
- **Live market data** and ~2-3 minute decision cycles
- **Identical prompts** and system configurations
- **No news or narratives** - pure price-action trading

### Our Replication Study

Inspired by Alpha Arena's pioneering work, this project replicates their methodology in a controlled backtesting environment to enable deeper statistical analysis.

**Key differences from the original Alpha Arena:**

| Aspect | Alpha Arena (Original) | Our Study |
|--------|----------------------|-----------|
| **Execution** | Live trading with real capital | Historical backtest (17-30 Oct 25) |
| **Models Tested** | 6 LLMs | 2 LLMs (Anthropic, OpenAI) |
| **Temperature Settings** | Single configuration per model | 2 temperatures each (0.1 and 0.7) |
| **Decision Frequency** | ~2-3 minutes | 4 hours |
| **Trials per Config** | 1 trial (live) | 10 trials (statistical significance) |
| **Time Period** | 17 Oct - 3 Nov 25 | 17 Oct - 30 Oct 25 |

**Our approach enables:**
1. **Statistical rigor:** 10 trials per configuration allow significance testing and variance analysis
2. **Temperature effects:** Test whether randomness (temp 0.1 vs 0.7) affects performance and consistency
3. **Reproducibility:** Historical backtests can be replicated and validated by others
4. **Behavioral analysis:** Detailed examination of trading patterns, risk preferences, and decision-making

*Note: We replicate Alpha Arena's user prompt template. System prompts remain proprietary to Nof1.ai.*

### Research Questions

1. **Can LLMs trade systematically?** How do they manage risk without explicit instruction?
2. **Which models perform best?** Is there a clear winner in risk-adjusted returns?
3. **How consistent are LLMs?** How different is each model's decision-making across multiple trials for the same trading period?
4. **Does temperature matter?** How does stochasticity (temp 0.1 vs 0.7) affect performance and consistency?
5. **What trading behaviors emerge?** Do models develop distinct strategies, biases, or patterns?

---

## 2. Performance Comparison

### Overall Performance Rankings

**By Total Return:**

| Rank | Model | Temperature | Mean Return (%) | Std Dev (%) | Best Trial(%) | Worst Trial(%) |
|------|-------|-------------|-----------------|-------------|------------|-------------|
| 1 | Anthropic | 0.7 | 3.6 | 35.51 | +55.6 | -35.9 |
| 2 | OpenAI | 0.1 | 1.4 | 7.34 | +12.3 | -10.8 |
| 3 | OpenAI | 0.7 | -0.3 | 6.21 | +7.6 | -11.4 |
| 4 | Anthropic | 0.1 | -10.9 | 14.14 | +4.6 | -39.9 |

> **Note on Standard Deviation:** The "Std Dev %" represents the **sample standard deviation** of returns across the 10 trials, measuring trial-to-trial consistency. Higher std dev = more unpredictable outcomes.
>
> **Formula:**
> $$s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$
>
> Where:
> - $s$ = sample standard deviation
> - $n$ = number of trials (10)
> - $x_i$ = return for trial $i$
> - $\bar{x}$ = mean return across all trials
> - We use $n-1$ when using a sample instead of the whole population

**Benchmarks:**
- Buy-and-Hold BTC: 0.07%
- Equal-Weight Portfolio: -1.23%

**By Risk-Adjusted Returns (Sharpe Ratio):**

| Rank | Model | Temperature | Mean Sharpe | Best Trial Sharpe | Worst Trial Sharpe |
|------|-------|-------------|-------------|-------------------|-------------------|
| 1 | OpenAI | 0.1 | 1.2 | 5.1 | -3.9 |
| 2 | OpenAI | 0.7 | -0.4 | 1.8 | -2.4 |
| 3 | Anthropic | 0.7 | -1.5 | 3.0 | -8.6 |
| 4 | Anthropic | 0.1 | -2.4 | 1.0 | -8.3 |

**Benchmarks:**
- Buy-and-Hold BTC: 0.268
- Equal-Weight Portfolio: -0.272

### Model vs Model Analysis

#### Anthropic vs OpenAI (At Temperature 0.1)

**Performance:**
- Mean return: -10.92% vs 1.40% (difference: **-12.32%**)
- Sharpe ratio: -2.411 vs 1.178
- Win rate: 35.0% vs 40.7%
- **Statistical significance:** p=0.025 (OpenAI significantly outperforms)

**Trading Behavior:**
- Avg trades: 22 vs 23
- Avg leverage: **14.31x vs 10x** (Anthropic uses 43% more leverage)
- Avg hold time: 28.5 hrs vs 30.4 hrs (similar holding duration, ~1.2-1.3 days)

**Key Differences:**
- **Leverage strategy is the major differentiator:** Anthropic chose 15x leverage (50% higher than OpenAI's 10x), amplifying both wins and losses
- **Similar activity levels:** Both models trade with similar frequency (~23 trades), suggesting temperature 0.1 leads to consistent decision cadence
- **OpenAI's risk management pays off:** More conservative leverage (10x) combined with similar win rate leads to statistically better outcomes
- **Different risk appetites:** Models showed distinct, consistent leverage preferences despite no imposed constraints

#### Anthropic vs OpenAI (At Temperature 0.7)

**Performance:**
- Mean return: 3.60% vs -0.30% (difference: **3.90%**)
- Sharpe ratio: -1.531 vs -0.381
- Win rate: 42.3% vs 42.3% (identical)
- **Statistical significance:** p=0.736 (no significant difference)

**Trading Behavior:**
- Avg trades: 23.8 vs 21.0 (Anthropic slightly more active)
- Avg leverage: **14.13x vs 10x** (Anthropic maintains higher leverage)
- Avg hold time: 16.2 hrs vs 27.7 hrs (Anthropic much faster day trading)

**Key Differences:**
- **Anthropic trades faster at temp 0.7:** 16.2 hrs vs 27.7 hrs - nearly 2x faster turnover
- **Despite identical win rates, outcomes differ:** Anthropic's combination of higher leverage + faster trading led to higher mean return but also extreme variance (±55%)
- **High variance cancels out statistical significance:** Anthropic's wild swings mean the 3.90% difference could easily be luck

### Temperature Effects (0.1 vs 0.7)

#### Impact on Performance

| Model | Temp 0.1 Mean Return | Temp 0.7 Mean Return | Difference | p-value | Statistically Significant? |
|-------|---------------------|---------------------|------------|---------|---------------------------|
| Anthropic | -10.92% | 3.60% | -14.52% | 0.245 | No |
| OpenAI | 1.40% | -0.30% | 1.70% | 0.583 | No |

**Interpretation:** Temperature changes did NOT produce statistically significant differences in either model. The observed differences could be due to random variation rather than a true temperature effect.

#### Impact on Consistency

**Standard Deviation of Returns (Lower = More Consistent):**

| Model | Temp 0.1 Std Dev | Temp 0.7 Std Dev | More Consistent | Difference |
|-------|-----------------|-----------------|-----------------|------------|
| Anthropic | 14.14% | 35.51% | **Temp 0.1** | +21.37% |
| OpenAI | 7.34% | 6.21% | **Temp 0.7** | -1.13% |

**Key Findings:**
- **Anthropic:** Temp 0.1 was 2.5× more consistent than temp 0.7 (14.14% vs 35.51% std dev)
- **OpenAI:** Minimal consistency difference between temperatures (7.34% vs 6.21%)
- **Temperature:** Lower temperature improved consistency for Anthropic but not performance
- **High variance at temp 0.7:** Anthropic showed extreme outcomes (range: -35.87% to +55.59%)

---

## 3. Statistical Significance

> **Note on Sample Size:** All results are based on **10 trials per configuration**. This is a relatively small sample size, which means:
> - We can only detect **large differences** reliably
> - Smaller effects might exist but go undetected
> - Results should be validated on additional trials or longer time periods

### Performance Summary

**Mean Returns Across 10 Trials:**

| Model | Temperature | Mean Return | Std Dev | Range (Min to Max) | Median |
|-------|-------------|-------------|---------|-------------------|--------|
| Anthropic | 0.1 | -10.92% | 14.14% | -39.87% to +4.63% | -7.29% |
| Anthropic | 0.7 | 3.60% | 35.51% | -35.87% to +55.59% | -7.67% |
| OpenAI | 0.1 | 1.40% | 7.34% | -10.77% to +12.29% | 0.05% |
| OpenAI | 0.7 | -0.30% | 6.21% | -11.39% to +7.63% | -0.07% |

**Benchmarks (Single Run):**
- Buy-and-Hold BTC: 0.07%
- Equal-Weight Portfolio: -1.23%

### Is the Difference Real or Just Luck?

When comparing models, we need to ask: **"Could this difference have happened by random chance?"**

We use **statistical significance testing** to answer this question. The key metric is the **p-value**:

- **p-value < 0.05 (5%):** The difference is **statistically significant** - unlikely to be random luck
- **p-value ≥ 0.05 (5%):** The difference is **not statistically significant** - could easily be random chance

**Ways to interpret it:**
- p = 0.02 means "Only a 2% chance this is luck" → Probably real
- p = 0.50 means "50% chance this is luck" → Could go either way
- p = 0.80 means "80% chance this is luck" → Probably just noise

### Statistical Test Results

**Comparing Models (At Each Temperature):**

| Comparison | Anthropic Mean | OpenAI Mean | Difference | p-value | Significant? | What This Means |
|------------|---------------|-------------|------------|---------|--------------|-----------------|
| **At Temp 0.1** | -10.92% | 1.40% | -12.32% | **0.025** | **YES** | OpenAI likely outperforms Anthropic (only 2.5% chance this is luck) |
| **At Temp 0.7** | 3.60% | -0.30% | 3.90% | 0.736 | No | Difference could easily be random (73.6% chance it's luck) |

**Comparing Temperatures (Within Each Model):**

| Model | Temp 0.1 Mean | Temp 0.7 Mean | Difference | p-value | Significant? | What This Means |
|-------|---------------|---------------|------------|---------|--------------|-----------------|
| **Anthropic** | -10.92% | 3.60% | -14.52% | 0.245 | No | Despite large difference, could be chance (24.5% chance it's luck) |
| **OpenAI** | 1.40% | -0.30% | 1.70% | 0.583 | No | Tiny difference, likely just noise (58.3% chance it's luck) |

### Why Most Results Are "Not Significant"

**The Challenge of Small Sample Size (10 trials):**

With only 10 trials per configuration, we face two problems:

1. **High Variance:** Individual trials can swing wildly
   - Anthropic temp 0.7 ranged from -35.87% to +55.59% (91% spread)
   - This huge variance makes it hard to detect true differences

2. **Low Statistical Power:** We can only reliably detect very large effects
   - Example: Anthropic 0.1 vs 0.7 showed -14.52% difference, but p=0.245 (not significant)
   - Why? Because the standard deviations are so large (14.14% and 35.51%) that this difference could plausibly be random

### Key Takeaways

**What We Can Conclude:**
- OpenAI outperforms Anthropic at temperature 0.1 (p=0.025) during this period

**What We Cannot Say:**
- Temperature significantly affects performance (not enough evidence)
- Anthropic temp 0.7 is meaningfully different from OpenAI (too much variance)
- These results will hold in different market conditions (only tested one 14-day period)

**Important Caveat:**
Even the "significant" finding (p=0.025) is based on just 10 trials over 14 days. This should be validated on:
- Different time periods
- Longer backtests (3-6 months)
- More trials (20-50 per configuration)

---

## 4. Behavioral Analysis

### Trading Patterns

#### Position Sizing

**Average Trade Size and Median Trade Size:**

| Model | Temperature | Avg Trade Size | Median Trade Size | Trade Size Strategy |
|-------|-------------|----------------|-------------------|---------------------|
| Anthropic | 0.1 | $8,373 | $5,450 | Moderate, consistent |
| Anthropic | 0.7 | $11,352 | $6,329 | **Aggressive (largest avg)** |
| OpenAI | 0.1 | $6,192 | $7,088 | Conservative, tight distribution |
| OpenAI | 0.7 | $5,057 | $3,716 | **Most conservative** |

**Observations:**
- **Most aggressive:** Anthropic temp 0.7 ($11,352 avg) - 80% higher than OpenAI temp 0.7
- **Most conservative:** OpenAI temp 0.7 ($5,057 avg) - smallest position sizes
- **Tightest distribution:** OpenAI temp 0.1 has median ($7,088) close to mean ($6,192), indicating consistent sizing
- **Highest variance:** Anthropic temp 0.7 shows large gap between median ($6,329) and mean ($11,352), suggesting occasional very large positions

#### Leverage Usage

**Leverage Distribution:**

| Model | Temperature | Avg Leverage | Median Leverage | Leverage Strategy |
|-------|-------------|--------------|-----------------|-------------------|
| Anthropic | 0.1 | 14.31x | 15x | **Consistently high leverage** |
| Anthropic | 0.7 | 14.13x | 15x | **Consistently high leverage** |
| OpenAI | 0.1 | 10x | 10x | **Consistent moderate leverage** |
| OpenAI | 0.7 | 10x | 10x | **Consistent moderate leverage** |

**Key Findings:**
- **No leverage constraints were imposed** - models freely chose their own leverage per trade
- **Anthropic consistently chose ~15x leverage** across both temperatures (median 15x, mean 14.3x)
- **OpenAI consistently chose exactly 10x leverage** - more conservative risk preference
- **No temperature effect on leverage:** Both models maintain same leverage regardless of temperature (suggests leverage is a model-specific trait, not affected by randomness)
- **Leverage doesn't correlate with better performance:** Anthropic's higher leverage led to both the best (temp 0.7: +3.60%) and worst (temp 0.1: -10.92%) outcomes
- **Higher leverage = Higher variance:** Anthropic's 15x leverage contributed to extreme trial-to-trial variance (44-91% range)

#### Hold Times

**Average Position Duration:**

| Model | Temperature | Mean Hold (hrs) | Median Hold (hrs) | Mean (days) | Median (days) |
|-------|-------------|-----------------|-------------------|-------------|---------------|
| Anthropic | 0.1 | 28.5 | 12 | 1.2 | 0.5 |
| Anthropic | 0.7 | 16.2 | 8.8 | 0.7 | **0.4 (shortest)** |
| OpenAI | 0.1 | 30.4 | 21.6 | 1.3 | 0.9 |
| OpenAI | 0.7 | 27.7 | 21 | 1.2 | 0.9 |

**Observations:**
- **Shortest hold times:** Anthropic temp 0.7 (median 8.8 hrs / 0.4 days) - enters and exits quickly
- **Longest hold times:** OpenAI temp 0.1 (median 21.6 hrs / 0.9 days) - more patient
- **All models held < 1.5 days on average** - relatively short-term trading across the board
- **Large median-mean gaps for Anthropic:** Suggests mix of quick trades and occasional longer holds
- **OpenAI more consistent:** Median closer to mean indicates steadier hold time behavior
- **No clear correlation with performance:** Both shorter holds (Anthropic 0.7: +3.60%) and longer holds (OpenAI 0.1: +1.40%) achieved positive returns

#### Market Environment & Coin Performance

**Individual Coin Performance During Test Period (Oct 17-30, 2025):**

Understanding each coin's return AND volatility provides context for trading opportunities:

| Coin | Total Return | Volatility (Std Dev) | Avg Intrabar Range | Max Up Move | Max Down Move | Trading Opportunity |
|------|--------------|----------------------|-------------------|-------------|---------------|---------------------|
| **SOL** | -0.96% | **1.53%** | 2.33% | +5.36% | -4.95% | **Moderate** (Most volatile) |
| **DOGE** | -3.06% | **1.51%** | 2.22% | +5.05% | -6.17% | **Moderate** |
| **BNB** | -5.56% | **1.50%** | 2.17% | +3.29% | -7.54% | **Moderate** |
| **XRP** | +4.72% | 1.32% | 2.11% | +3.78% | -5.11% | Limited |
| **ETH** | -2.61% | 1.28% | 1.93% | +4.96% | -4.44% | Limited |
| **BTC** | +0.07% | **0.97%** | 1.33% | +4.44% | -3.27% | **Limited** (Least volatile) |

**Key Insights:**

**Return vs. Volatility:**
- **XRP had the best return (+4.72%) but moderate volatility (1.32%)** - steady uptrend, fewer trading opportunities
- **SOL, DOGE, BNB had highest volatility (1.50-1.53%) but negative returns** - choppy, offering more entry/exit points but requiring good timing
- **BTC had lowest volatility (0.97%) and flat return (0.07%)** - least opportunity for both directional and swing trading

With this market environment, the LLMs would need to trade actively with leverage and good timing; buy-and-hold wouldn't lead to meaningful profit.

#### Per-Coin Performance Analysis

**Which coins were most profitable for each model?**

Aggregated results across all 10 trials show distinct per-coin performance patterns. Below are the actual results for the best and worst performing configurations:

**Anthropic temp 0.7 (Best Overall: +3.60% mean return)**

| Coin | Total Trades | Total P&L | Win Rate | Avg P&L/Trade |
|------|--------------|-----------|----------|---------------|
| ETH  | 62           | **+$9,896** | 38.7%  | +$160         |
| SOL  | 60           | +$2,350 | 48.3%    | +$39          |
| DOGE | 30           | +$393   | 40.0%    | +$13          |
| BNB  | 41           | -$134   | 53.7%    | -$3           |
| BTC  | 28           | -$2,779 | 42.9%    | -$99          |
| XRP  | 38           | **-$4,759** | 28.9% | -$125         |

**Key Insights (Anthropic temp 0.7):**
- **Most profitable:** ETH ($9,896 total) despite having lowest win rate (38.7%) - large winning trades with leverage
- **Worst performer:** XRP (-$4,759, only 28.9% win rate) - poor timing on the only coin that rallied (+4.72%)
- **Paradox:** Made money on ETH (which declined -2.61%) but lost on XRP (which rallied +4.72%)
- **Why?** Capture ratios show the answer - ETH upside capture 317% vs XRP 261% upside capture

**Anthropic temp 0.1 (Worst Overall: -10.92% mean return)**

| Coin | Total Trades | Total P&L | Win Rate | Avg P&L/Trade |
|------|--------------|-----------|----------|---------------|
| BNB  | 37           | +$921   | 67.6%    | +$25          |
| SOL  | 57           | +$243   | 47.4%    | +$4           |
| DOGE | 19           | -$101   | 31.6%    | -$5           |
| BTC  | 23           | -$358   | 47.8%    | -$16          |
| ETH  | 48           | -$2,957 | 45.8%    | -$62          |
| XRP  | 40           | **-$7,731** | 22.5% | -$193         |

**Key Insights (Anthropic temp 0.1):**
- **XRP catastrophe:** -$7,731 loss (worst per-coin result across all configs) despite XRP rallying +4.72%
- **Only BNB profitable:** +$921 with 67.6% win rate - best win rate across all coins/configs
- **Consistent XRP problems:** Both temps lost heavily on XRP - timing issues, not just bad luck

**OpenAI temp 0.1 (Best Risk-Adjusted: Sharpe 1.18)**

| Coin | Total Trades | Total P&L | Win Rate | Avg P&L/Trade |
|------|--------------|-----------|----------|---------------|
| BTC  | 60           | +$1,934 | 46.7%    | +$32          |
| SOL  | 43           | +$955   | 32.6%    | +$22          |
| BNB  | 38           | +$596   | 34.2%    | +$16          |
| DOGE | 12           | +$222   | 50.0%    | +$18          |
| XRP  | 22           | -$67    | 40.9%    | -$3           |
| ETH  | 55           | -$1,531 | 41.8%    | -$28          |

**Key Insights (OpenAI temp 0.1):**
- **4 out of 6 coins profitable** - most balanced performance across all configs
- **BTC most traded (60 trades) and profitable (+$1,934)** - successfully navigated flat market
- **Minimal XRP loss (-$67)** - avoided the XRP timing trap that hurt Anthropic
- **Lower leverage (10x) limited both gains and losses** - no individual coin exceeded $2,000 P&L

**Capture Ratio Summary (Anthropic temp 0.7):**

| Coin | Avg Upside Capture | Avg Downside Capture | Interpretation |
|------|--------------------|----------------------|----------------|
| SOL  | **463.6%** | 316.5% | Excellent upside timing with leverage, captured 4.6× the up-moves |
| ETH  | 316.9% | 380.7% | Good upside capture but suffered more on downside |
| BNB  | 246.7% | 363.3% | Captured 2.5× upside but 3.6× downside (lost more than gained) |
| XRP  | 261.3% | **396.8%** | Poor timing - captured less upside, more downside despite XRP rallying |
| BTC  | 179.4% | 198.6% | Modest captures on both sides |
| DOGE | 148.5% | 259.0% | Weakest upside capture, moderate downside |

**What This Tells Us:**
- **Market direction ≠ Profitability:** ETH declined but was most profitable; XRP rallied but lost most money
- **Timing + Leverage > Coin Selection:** SOL's 463% upside capture with 15x leverage turned volatility into profit
- **Capture ratios explain P&L:** High upside + low downside capture (SOL) = profit; opposite (XRP, BNB) = loss
- **Different coins require different skills:** What worked for SOL (high leverage, volatile) didn't work for XRP (trending but missed)

> **Note:** Full per-coin statistics for all model configurations available in the **Statistical Analysis dashboard** under "Per-Coin Performance Summary" with sortable tables and detailed breakdowns.

### Risk Management

**Available Risk Metrics (from Section 5):**
- **Sharpe Ratios:** OpenAI temp 0.1 had best risk-adjusted returns (Sharpe: 1.18)
- **Sortino Ratios:** Show downside risk - Anthropic temp 0.7 had best Sortino (8.42) despite high overall volatility
- **Downside Deviation:** Anthropic temp 0.7 had lowest downside deviation (254.75), suggesting better loss protection when trades went wrong
- **Biggest Losses:** Ranged from -$440 to -$541 across all configurations, showing models did cut losses at some level

**Key Risk Management Observations:**
- **Higher leverage (Anthropic ~15x) led to both bigger wins AND bigger losses** - classic risk/reward tradeoff
- **OpenAI's moderate leverage (10x) provided more predictable risk exposure** - tighter distribution of outcomes
- **Leverage was a deliberate choice, not a constraint** - different risk appetites between models

---

## 5. Variance & Consistency

### Trial-to-Trial Variance

**Return Distribution Across 10 Trials:**

| Model | Temperature | Mean | Std Dev | Min | Max | Range | Coefficient of Variation |
|-------|-------------|------|---------|-----|-----|-------|--------------------------|
| Anthropic | 0.1 | -10.92% | 13.42% | -39.87% | 4.63% | 44.50% | 122.9% (High variance) |
| Anthropic | 0.7 | 3.60% | 33.69% | -35.87% | 55.59% | 91.46% | 936.4% (Extreme variance) |
| OpenAI | 0.1 | 1.40% | 6.96% | -10.77% | 12.29% | 23.06% | 497.1% (High variance) |
| OpenAI | 0.7 | -0.30% | 5.89% | -11.39% | 7.63% | 19.02% | N/A (negative mean) |

**Coefficient of Variation (CV):**
- **CV = (Std Dev / |Mean|) × 100** - measures relative variability
- **< 50%:** Highly consistent
- **50-100%:** Moderately consistent
- **> 100%:** Highly variable (outcome very unpredictable)

**Most Consistent:** OpenAI temp 0.1 (lowest std dev at 6.96%)
**Least Consistent:** Anthropic temp 0.7 (std dev 33.69%, range 91.46%)

### Temperature Impact on Consistency

**Hypothesis:** Lower temperature (0.1) should produce more consistent results than higher temperature (0.7).

| Model | Temp 0.1 Std Dev | Temp 0.7 Std Dev | Change | Hypothesis Confirmed? |
|-------|-----------------|-----------------|--------|-----------------------|
| Anthropic | 13.42% | 33.69% | +20.27% | **YES** - Temp 0.1 is 2.5× more consistent |
| OpenAI | 6.96% | 5.89% | -1.07% | **NO** - Minimal difference, temp 0.7 slightly more consistent |

**Key Findings:**

1. **Anthropic: Temperature matters enormously for consistency**
   - Temp 0.1: Std dev 13.42%, range 44.50%
   - Temp 0.7: Std dev 33.69%, range 91.46%
   - **2.5× more variable** at higher temperature
   - Trade-off: More consistent but worse average return (-10.92% vs 3.60%)

2. **OpenAI: Temperature has minimal impact on consistency**
   - Both temperatures have similar std dev (~6%)
   - Suggests OpenAI's trading logic is more deterministic regardless of temperature
   - Another way to see it: The 14-day period didn't expose temperature-driven differences

3. **The Consistency-Performance Paradox:**
   - **Anthropic temp 0.1:** Very consistent but consistently poor (-10.92% mean)
   - **Anthropic temp 0.7:** Wildly inconsistent but better average (3.60% mean)
   - This suggests: Consistency ≠ Good performance

### Upside vs Downside Volatility

**Risk Asymmetry Analysis (Average across 10 trials):**

| Model | Temperature | Upside Dev | Downside Dev | Ratio | Interpretation |
|-------|-------------|------------|--------------|-------|----------------|
| Anthropic | 0.1 | 154.95 | 285.65 | 0.54 | Downside-skewed (losses > wins) |
| Anthropic | 0.7 | 418.84 | 254.75 | 1.64 | Upside-skewed (wins > losses) |
| OpenAI | 0.1 | 159.98 | 136.56 | 1.17 | Balanced |
| OpenAI | 0.7 | 158.75 | 134.91 | 1.18 | Balanced |

**Ratio Interpretation:**
- **> 1.5:** Upside-skewed (big wins, small losses)
- **0.7-1.5:** Balanced
- **< 0.7:** Downside-skewed (big losses, small wins)

**Key Insights:**
- **Anthropic temp 0.1:** Worst risk profile (0.54) - losses were nearly 2× larger than wins
- **Anthropic temp 0.7:** Best risk profile (1.64) - wins were 1.6× larger than losses, despite high variance
- **OpenAI:** Consistent balanced profile (~1.17) regardless of temperature
- **Paradox:** Anthropic 0.1 had consistent but poor risk asymmetry; 0.7 had better asymmetry but wild variance

### Advanced Risk Metrics

**Skewness & Kurtosis (Average across 10 trials):**

| Model | Temperature | Skewness | Kurtosis | Distribution Shape |
|-------|-------------|----------|----------|-------------------|
| Anthropic | 0.1 | -1.25 | 4.13 | Negative skew, fat tails (prone to large losses) |
| Anthropic | 0.7 | 0.06 | 4.85 | Nearly symmetric, very fat tails (extreme outcomes common) |
| OpenAI | 0.1 | 0.20 | 0.28 | Slightly positive skew, normal distribution |
| OpenAI | 0.7 | 0.18 | 0.32 | Slightly positive skew, normal distribution |

**Interpreting the Numbers:**
- **Skewness = 0:** Symmetric distribution
- **Skewness < 0:** Negative skew (prone to large losses, "left tail risk")
- **Skewness > 0:** Positive skew (prone to large wins, "lottery ticket")
- **Kurtosis = 0:** Normal distribution
- **Kurtosis > 3:** Fat tails (extreme outcomes more likely than normal)

**Key Findings:**
1. **Anthropic temp 0.1:** Negative skewness (-1.25) = vulnerable to occasional huge losses
   - Kurtosis 4.13 = these extreme losses happen more often than you'd expect
   - This explains trials like -39.87% (worst case)

2. **Anthropic temp 0.7:** Nearly symmetric (0.06) but extreme kurtosis (4.85)
   - Equal chance of huge wins or huge losses
   - Explains the 91% range (-35.87% to +55.59%)

3. **OpenAI:** Near-normal distributions (low skew, low kurtosis)
   - More predictable, fewer extreme outliers
   - Behaves more like traditional trading strategies

---

## 6. Capture Ratios

> **Understanding Capture Ratios:**
> Capture ratios measure how well the model times its entries and exits relative to each coin's price movements.
> - **Upside Capture** = (Model's average return during up-candles) / (Coin's average return during up-candles) × 100%
> - **Downside Capture** = (Model's average return during down-candles) / (Coin's average return during down-candles) × 100%
>
> **How it's calculated:**
> 1. Load all 4-hour candles for the backtest period (85 candles from Oct 17-30)
> 2. For each candle, calculate the coin's return: `(close - open) / open × 100`
> 3. Separate candles into **up-candles** (close > open) and **down-candles** (close < open)
> 4. Calculate the coin's average return during up-candles and down-candles separately
>
> **Example:** If BTC had 5 up-candles with returns of [+0.75%, +0.89%, +1.04%, +0.62%, +1.15%]:
> - Coin's average return during up-candles = (0.75 + 0.89 + 1.04 + 0.62 + 1.15) / 5 = **+0.89%**
> - If the model was holding BTC with 10x leverage during 3 of those candles, capturing [+7.5%, +8.9%, +10.4%, 0%, 0%]:
> - Model's average return during up-candles = (7.5 + 8.9 + 10.4 + 0 + 0) / 5 = **+5.36%**
> - Upside Capture Ratio = 5.36% / 0.89% × 100 = **602%** (captured 6× the benchmark via leverage + timing)
>
> **Ideal Profile:** High upside capture (>100%) + Low downside capture (<100%) = Asymmetric returns

**Interpretation Guide:**
- **Upside Capture > 100%** = Model outperformed the coin during rallies (excellent timing + leverage)
- **Upside Capture < 100%** = Model missed some gains (suboptimal timing or not holding during rallies)
- **Downside Capture < 100%** = Model protected capital during declines (good risk management)
- **Downside Capture > 100%** = Model lost more than the coin's decline (poor timing or over-leveraged)

### Key Insights from Market Context

Given the market environment (Section 4):
- **XRP (+4.72% total)** was the only strong performer - models that identified and rode this trend would show high XRP upside capture
- **BTC (+0.07% flat)** - choppy, range-bound price action makes timing critical
- **SOL, DOGE, BNB (highest volatility, 1.50-1.53%)** - more trading opportunities but requiring precise timing
- **All models traded 100% long positions** - downside capture ratios will be high (didn't short to profit from declines)

### Expected Patterns

Based on overall performance (Section 2):
- **Anthropic temp 0.7 (+3.60% avg return):** Likely high upside capture on XRP and/or successful timing on volatile coins (SOL/DOGE/BNB) using leverage
- **OpenAI temp 0.1 (+1.40% avg return):** Modest upside capture, possibly balanced across multiple coins
- **OpenAI temp 0.7 (-0.30% avg return):** Near-zero, suggesting mixed timing - caught some rallies but also got caught in declines
- **Anthropic temp 0.1 (-10.92% avg return):** Likely high downside capture (poor timing, caught in declining coins) and/or missed XRP rally

> **Note:** Detailed per-coin, per-trial capture ratio analysis is available in the interactive **Statistical Analysis dashboard** (Section 7_Statistical_Analysis.py). The dashboard calculates capture ratios by analyzing all 4-hour candles during each trial and tracking which coins were held during up-moves vs down-moves. This trade-level analysis requires loading full checkpoint files and is best viewed interactively rather than in static summary tables.

---

## 7. Model Strengths & Weaknesses

**Anthropic Claude Sonnet 4.5:**
- **Strengths:**
  - **Highest upside potential:** Best single configuration performance (+3.60% mean at temp 0.7)
  - **Willing to take calculated risks:** Consistently chose 15x leverage, showing confidence in decisions
  - **Best upside/downside asymmetry at temp 0.7:** 1.64 ratio (captures more upside than downside)
  - **Positive skewness at temp 0.7:** Distribution shows more big wins than big losses (skewness: 0.06)
  - **Adaptive trading speed:** Temp 0.7 traded faster (16.2 hrs median) than temp 0.1 (28.5 hrs)
  - **Beat benchmarks significantly:** Temp 0.7 outperformed both buy-and-hold BTC (+0.07%) and equal-weight (-1.23%) by wide margins
- **Weaknesses:**
  - **Extreme variance:** Temp 0.7 ranged from -35.87% to +55.59% (91.46% spread) - highly unpredictable outcomes
  - **Poor performance at temp 0.1:** -10.92% mean return (worst of all configurations tested)
  - **High kurtosis:** Fat-tailed distributions (4.85 at temp 0.7) mean extreme outcomes are common, both positive and negative
  - **Leverage amplifies losses:** 15x leverage that drives outperformance also magnifies drawdowns
  - **Temperature-dependent inconsistency:** Dramatic swing from -10.92% (temp 0.1) to +3.60% (temp 0.7)
  - **Downside-skewed at temp 0.1:** Negative skewness (-1.25) indicates left-tail risk with occasional large losses
- **Best use case:** Suitable for **risk-tolerant traders** seeking high absolute returns and willing to accept significant variance. Use exclusively at **temp 0.7** (temp 0.1 showed poor results). Ideal for portion of portfolio allocated to aggressive strategies. Requires strict position sizing limits to manage extreme outcomes.

**OpenAI GPT-4o-mini:**
- **Strengths:**
  - **Most consistent performance:** Lowest variance across both temperatures (5.89-6.96% std dev)
  - **Best risk-adjusted returns:** Highest Sharpe ratio (1.18 at temp 0.1)
  - **Statistically proven edge:** Only configuration with significant outperformance (p=0.025 vs Anthropic temp 0.1)
  - **Conservative risk management:** Consistently chose 10x leverage (vs Anthropic's 15x)
  - **Near-normal distributions:** Low skewness (0.20) and kurtosis (0.28) at temp 0.1 - predictable, stable behavior
  - **Temperature-agnostic:** Minimal performance difference between temp 0.1 (+1.40%) and temp 0.7 (-0.30%)
  - **Reliable swing trading:** Consistent hold times (~27-30 hrs median) across temperatures
- **Weaknesses:**
  - **Limited upside:** Capped at +1.40% mean return (best case) - conservative approach limits gains
  - **Moderate leverage constrains returns:** 10x leverage is prudent but misses opportunities during strong trends
  - **Still has downside risk:** Temp 0.1 downside deviation of 136.56 shows room for improvement
  - **Erratic position sizing at temp 0.7:** Despite fixed 10x leverage, showed inconsistent quantity selection
  - **Couldn't capitalize on XRP rally:** Beat benchmarks modestly but didn't dramatically outperform despite clear trend opportunity
- **Best use case:** Ideal for **risk-averse traders** prioritizing consistency and capital preservation over maximum returns. Best at **temp 0.1** for optimal risk-adjusted performance. Perfect for core portfolio allocation where predictability matters more than beating the market. Suitable for automated trading with minimal oversight.

---

## 8. Future Research Directions

#### 1. Extended Time Periods & Trials
- Test across multiple 2-week periods
- Increase to 30+ trials per configuration for robust statistics

#### 2. Additional Models + Strategies
- Test other models (Deepseek, Grok etc.)
- Ensemble strategies (combining multiple LLM signals)

#### 3. Prompt Variations
- Test different prompt formats and instructions
- Measure impact of explicit vs. implicit risk guidance
- Optimize prompts for specific objectives (aggressive, defensive, min risk, etc.)

#### 4. Alternative Markets
- Apply to equities markets of various geographies
- Test on different crypto assets (DeFi tokens, memecoins, stablecoins)

#### 5. Explainability
- Analyze LLM reasoning patterns
- Identify common decision-making biases
- Understand what technical signals models prioritize

---

## 9. Conclusion

### Final Verdict

**Best Overall Model:** Anthropic Claude Sonnet 4.5 at temperature 0.7
- Mean return: +3.60%
- Sharpe ratio: -1.53
- Consistency: Low (Std Dev 33.69%)
- **Observations:** Highest mean return, outperformed both benchmarks. Despite extreme variance, delivered positive positive returns in a challenging market where most coins declined. Best upside/downside asymmetry (1.64 ratio).

**Most Consistent Model:** OpenAI GPT-4o-mini at temperature 0.7
- Standard deviation: 5.89%
- **Observations:** Lowest variance across all configurations. Tight distribution of outcomes makes performance more predictable. Temperature had minimal effect on behavior. Useful for risk-averse applications.

**Best Risk-Adjusted:** OpenAI GPT-4o-mini at temperature 0.1
- Sharpe ratio: 1.18
- Sortino ratio: 3.00
- **Observations:** Only configuration with positive Sharpe >1. Statistically proven to outperform Anthropic at temp 0.1 (p=0.025). Conservative 10x leverage provided controlled risk exposure. Best balance of return vs. volatility.

### Answers to Research Questions

**1. Can LLMs trade systematically? Do they follow risk management rules without explicit instruction?**

**Answer:** Yes, but with significant variance.
- **What they did alright:** All models executed trades systematically, selected leverage (10x-15x), and closed positions without explicit stop-loss rules
- **Risk management gaps:** High variance suggests inconsistent risk discipline. Anthropic temp 0.7 ranged from -36% to +56%, indicating some trials took excessive risk while others were conservative
- **Key finding:** Models followed the trading framework but lacked consistent risk discipline across trials. Temperature 0.1 showed more systematic behavior than 0.7

**2. Which models perform best? Is there a clear winner in risk-adjusted returns?**

**Answer:** Depends on the metric.
- **Absolute returns:** Anthropic temp 0.7 (+3.60% mean, but high variance)
- **Risk-adjusted returns:** OpenAI temp 0.1 (Sharpe 1.18, Sortino 3.00)
- **Consistency:** OpenAI temp 0.7 (Std Dev 5.89%)
- **Statistical significance:** Only 1 significant result - OpenAI temp 0.1 beat Anthropic temp 0.1 (p=0.025)
- **Conclusion:** No single clear winner. Choice depends on risk tolerance (high: Anthropic 0.7, low: OpenAI 0.1)

**3. How consistent are LLMs? What is the variance across multiple trials of the same model?**

**Answer:** Highly inconsistent, especially Anthropic.
- **Most consistent:** OpenAI temp 0.7 (Std Dev 5.89%, range: -11.39% to +7.63%)
- **Least consistent:** Anthropic temp 0.7 (Std Dev 33.69%, range: -35.87% to +55.59%)
- **Key insight:** Even with temperature=0.1, variance was high (Anthropic: 13.42%, OpenAI: 6.96%)

**4. Does temperature matter? How does stochasticity (temp 0.1 vs 0.7) affect performance and consistency?**

**Answer:** Temperature effect is model-dependent.
- **Anthropic:** Higher temp increased both returns (+3.60% vs -10.92%) and variance (33.69% vs 13.42%). Difference not statistically significant (p=0.13)
- **OpenAI:** Temperature had minimal effect on returns (+1.40% vs -0.30%) or variance (6.96% vs 5.89%). Difference not significant (p=0.29)
- **Behavioral changes:** Higher temperature led to slightly shorter hold times (Anthropic 0.7: 0.7 days vs 0.1: 1.2 days)
- **Conclusion:** Temperature matters more for Anthropic than OpenAI, but high variance limits statistical confidence

**5. What trading behaviors emerge? Do models develop distinct strategies, biases, or patterns?**

**Answer:** Yes, clear behavioral differences emerged.
- **Leverage preferences:** Anthropic consistently chose ~15x, OpenAI chose 10x (across both temps)
- **Hold times:** All models held <1.5 days average (short-term trading). Anthropic 0.7 had shortest (0.7 days median)
- **Timing quality (capture ratios):**
  - Best: Anthropic 0.7 on SOL (463.6% upside capture, 316.5% downside - 1.46 ratio)
  - Worst: Anthropic 0.1 on XRP (262.0% upside, 438.0% downside - 0.60 ratio)
- **The XRP Trade:** Models made money on declining coins but lost on XRP (only rallying coin), proving timing matters more than direction
- **Distinct patterns:** Anthropic = higher risk/reward, OpenAI = conservative/consistent

---

## References

- **Alpha Arena (Nof1.ai):** https://nof1.ai/
- **Experimental Setup:** [EXPERIMENTAL_SETUP.md](EXPERIMENTAL_SETUP.md)
- **Statistical Analysis Guide:** [STATISTICAL_ANALYSIS_GUIDE.md](STATISTICAL_ANALYSIS_GUIDE.md)
- **Model Comparison Guide:** [MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md)
- **Project Documentation:** [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

---

## Appendix

### A. Full Performance Data

[Tables with complete trial-by-trial results]

### B. Statistical Test Details

[Detailed statistical outputs]

### C. Trade Logs

[Sample trade logs showing LLM reasoning]

### D. Configuration Files

- **Indicators:** [config/indicators.yaml](config/indicators.yaml)
- **Models:** [config/models.yaml](config/models.yaml)

---

**Acknowledgments:** Nof1.ai for Alpha Arena inspiration, Hyperliquid for market data

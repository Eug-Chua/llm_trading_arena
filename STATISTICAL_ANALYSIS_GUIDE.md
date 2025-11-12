# Data Science Experiments Guide

This document explains the experimental framework for measuring LLM trading performance variance and conducting statistical analysis.

---

## Understanding the System Architecture

### 1. **Checkpoint Resumption** (for continuing backtests)

When you resume from a checkpoint, you're restoring:
- **Account state**: Cash balance, open positions, trade history
- **Market timestamp**: Where you left off in the historical data
- **NOT LLM memory**: The LLM doesn't "remember" previous decisions

The LLM makes **fresh decisions** at each iteration based on:
1. Current market state (prices, technical indicators)
2. Current account state (positions, P&L)
3. No context from previous iterations

#### Example: Resuming a Backtest

```
Original backtest: Oct 18-25
Resume: Oct 26-31

When resuming on Oct 26:
✓ Account has $11,500 (restored from checkpoint)
✓ BTC position exists (restored from checkpoint)
✓ LLM sees current market conditions on Oct 26
✓ LLM makes NEW decision (not cached/replayed)
```

### 2. **LLM Response Tracking** (for post-analysis)

The system tracks all LLM responses during backtests:
- **Purpose**: Post-analysis and debugging
- **Storage**: Appended to `self.llm_responses` list
- **Export**: Saved to `{checkpoint}_reasoning.json`
- **NOT used for**: Caching, replay, or resumption

#### What Gets Tracked

```python
{
    'response': LLMResponse object,
    'timestamp': '2025-10-18T12:00:00',
    'model': 'anthropic',
    'iteration': 42
}
```

### 3. **No More Caching** (removed for statistical analysis)

**Old Behavior (with cache):**
- MD5 hash of prompt used as cache key
- Identical prompts returned cached responses
- Multiple runs produced **identical results** (deterministic)
- Good for: Reproducibility, saving API costs
- Bad for: Measuring model variance

**New Behavior (cache removed):**
- Every iteration makes **fresh LLM API call**
- Same backtest run twice = **different results**
- Good for: Statistical analysis, measuring variance
- Trade-off: Higher API costs (but necessary for science!)

---

## Running Statistical Experiments

### Objective

Measure the **variance** and **standard deviation** of LLM trading performance across multiple independent trials.

### Experimental Design

**4 Configurations × 10 Trials = 40 Total Backtests**

| Configuration | Provider   | Temperature | Purpose                    |
|---------------|------------|-------------|----------------------------|
| Config A      | Anthropic  | 0.7         | Baseline (creative)        |
| Config B      | Anthropic  | 0.1         | Deterministic              |
| Config C      | OpenAI     | 0.7         | Baseline (creative)        |
| Config D      | OpenAI     | 0.1         | Deterministic              |

### Running Experiments

#### Date Range
- **Start**: 2025-10-17 00:00:00 UTC
- **End**: 2025-10-31 00:00:00 UTC
- **Duration**: 14 days (336 hours)
- **Interval**: 4h (one decision every 4 hours)
- **Total Candles**: 85 4-hour candles
- **Decision Points**: 85 timestamps (one decision per candle)

#### Command Template

```bash
python scripts/run_backtest.py \
    --start 2025-10-17 \
    --end 2025-10-31 \
    --model {anthropic|openai} \
    --temperature {0.7|0.1} \
    --run-id {1-10} \
    --checkpoint results/checkpoints/{model}_temp{temp}_trial{N}.pkl
```

**Note:** The `--interval 4h` flag is now the default and can be omitted. Using 4h intervals:
- ✅ Faster backtests (~78 iterations vs 6000+ for 3m)
- ✅ Works with your available historical data
- ✅ Reduces API costs (fewer LLM calls)
- ✅ More realistic for manual trading strategies

#### Example: Run All Anthropic (temp=0.7) Trials

```bash
# Trial 1
python scripts/run_backtest.py \
    --start 2025-10-18 \
    --end 2025-10-31 \
    --model anthropic \
    --temperature 0.7 \
    --run-id 1 \
    --checkpoint results/checkpoints/anthropic_temp0.7_trial1.pkl

# Trial 2
python scripts/run_backtest.py \
    --start 2025-10-18 \
    --end 2025-10-31 \
    --model anthropic \
    --temperature 0.7 \
    --run-id 2 \
    --checkpoint results/checkpoints/anthropic_temp0.7_trial2.pkl

# ... repeat for trials 3-10
```

#### Example: Run All OpenAI (temp=0.1) Trials

```bash
# Trial 1
python scripts/run_backtest.py \
    --start 2025-10-18 \
    --end 2025-10-31 \
    --model openai \
    --temperature 0.1 \
    --run-id 1 \
    --checkpoint results/checkpoints/openai_temp0.1_trial1.pkl

# ... repeat for trials 2-10
```

---

## Key Metrics to Analyze

### Risk-Adjusted Performance

#### Sharpe Ratio
**Formula**: `(Portfolio Return - Risk-Free Rate) / Standard Deviation of Returns`

**What it measures**: Return per unit of **total volatility** (both upside and downside).

**Your Implementation**:
- Risk-free rate = **0%** (appropriate for crypto markets with 24/7 trading)
- Calculated per-trade, then annualized based on average trade frequency
- Uses sample standard deviation (ddof=1)

**Interpretation**:
- **Sharpe > 2.0**: Excellent risk-adjusted returns
- **Sharpe 1.0 - 2.0**: Good performance
- **Sharpe 0.5 - 1.0**: Acceptable
- **Sharpe < 0.5**: Poor risk-adjusted returns
- **Sharpe < 0**: Losing money (return below risk-free rate)

**Limitation**: Penalizes BOTH upside and downside volatility equally. If your strategy has big winning days, the Sharpe ratio treats that volatility as "risk" even though it's desirable.

#### Sortino Ratio
**Formula**: `(Portfolio Return - Risk-Free Rate) / Downside Deviation`

**What it measures**: Return per unit of **downside risk only** (ignoring upside volatility).

**Key Difference from Sharpe**:
- **Sharpe**: Penalizes ALL volatility (up and down moves)
- **Sortino**: Only penalizes DOWNSIDE volatility (negative returns)

**Your Implementation**:
- Risk-free rate = **0%** (implicit - using total PnL directly)
- Formula: `total_pnl / downside_deviation`
- Downside deviation = standard deviation of **negative trade P&Ls only**

**Why It Matters**:

In leveraged crypto trading, upside volatility is **desirable** - big winning days are profitable! The Sortino ratio recognizes this by only measuring "bad volatility" (losses) as risk.

**Example Comparison**:

```
Strategy A: Returns = +10%, -2%, +8%, -1%, +12%
- Average: +5.4%
- Total Std Dev: 5.9% (Sharpe penalizes the +12% spike!)
- Downside Std Dev: 0.7% (Sortino only penalizes -2% and -1%)
- Sharpe: 0.92
- Sortino: 7.71 (much higher!)

Strategy B: Returns = +3%, +3%, +3%, +3%, +3%
- Average: +3%
- Total Std Dev: 0% (perfectly consistent)
- Downside Std Dev: 0% (no losses)
- Sharpe: Undefined (infinite if std dev = 0, but codebase returns 0)
- Sortino: Undefined (but Strategy B has lower return)
```

Strategy A has **asymmetric upside** (lottery ticket profile) - Sortino captures this better than Sharpe.

**Interpretation**:
- **Sortino > 2.0**: Excellent (strong returns with limited downside)
- **Sortino 1.0 - 2.0**: Good (positive risk-adjusted returns)
- **Sortino < 1.0**: Poor (returns don't compensate for downside risk)
- **Higher is better**: More return per unit of downside risk

**When to Use**:
- **Sortino**: Better for strategies with asymmetric profiles (trend following, momentum, leveraged long)
- **Sharpe**: Better for strategies where you want consistency (market making, arbitrage, delta-neutral)

**For Your LLM Trading System**:

The Sortino ratio is particularly valuable because:
1. **Leveraged positions (10x)**: Upside spikes are profitable, not risky
2. **Asymmetric target**: You WANT positive skew (big wins, small losses)
3. **"Let winners run" strategy**: Sortino rewards holding winners without penalizing the volatility
4. **Crypto volatility**: Bitcoin can move ±5% in a day - that upside volatility is opportunity, not risk

**Example from Your Data**:

If a trial has:
- Total PnL: $2,500
- Downside deviation of negative trades: $150
- Sortino Ratio: 2,500 / 150 = **16.67** (excellent!)

This means the strategy generates $16.67 of profit for every $1 of downside volatility.

**Ideal Profile**: High Sortino + High Sharpe = Consistent profits with limited downside

#### Max Drawdown
**What it measures**: Largest peak-to-trough decline in account value (lower is better).

**Interpretation**: Maximum % you would have lost from a previous high point.

#### Win Rate
**What it measures**: Percentage of profitable trades (# winning trades / total trades × 100).

#### Profit Factor
**What it measures**: Gross profit / Gross loss (must be >1.0 to be profitable overall).

### Capture Ratios (Market Timing Analysis)
Capture ratios measure how well the model times its trades relative to each coin's price movements during the backtest period.

**Key Characteristics:**
- **Coin-Specific Benchmark**: Each coin (BTC, ETH, SOL, etc.) is compared to its own price movements, not the portfolio
- **All-Candle Analysis**: Evaluates performance across ALL 4h candles during the backtest, not just when positions were held
- **Leverage-Adjusted**: Returns are multiplied by leverage (10x leverage on +1% move = +10% strategy return)

**Calculation Method:**
1. Load all 4h candles for the backtest period (e.g., Oct 18-31)
2. For each candle:
   - Calculate coin's return: `(close - open) / open * 100`
   - Check if model held a position during this candle
   - If holding: `strategy_return = coin_return * leverage`
   - If not holding: `strategy_return = 0%` (missed the move)
3. Categorize candles as "up-market" (coin_return > 0) or "down-market" (coin_return < 0)
4. Calculate capture ratios:
   - **Upside Capture** = (avg strategy return in up-markets) / (avg coin return in up-markets) × 100
   - **Downside Capture** = (avg strategy return in down-markets) / (avg coin return in down-markets) × 100

**Interpretation:**
- **Upside Capture > 100%**: Model captured MORE than the coin's gains (good timing + leverage effect)
- **Upside Capture < 100%**: Model captured LESS than the coin's gains (missed opportunities or poor timing)
- **Downside Capture < 100%**: Model avoided losses (good risk management!) ← This is desirable
- **Downside Capture > 100%**: Model lost MORE than the coin declined (poor timing or over-leveraged)

**Ideal Profile:** High upside capture (>100%) + Low downside capture (<100%) = Asymmetric return profile

**Example:**
```
BTC Analysis (10 trials, 78 candles each):
- BTC up-candles: +2.3% avg (when BTC went up)
- Strategy in BTC up-candles: +4.6% avg (with 10x leverage, held 50% of the time)
- Upside Capture: (4.6 / 2.3) × 100 = 200%

- BTC down-candles: -1.8% avg (when BTC went down)
- Strategy in BTC down-candles: -0.5% avg (avoided most losses)
- Downside Capture: (-0.5 / -1.8) × 100 = 28%
```

This means the model:
- Captured 200% of BTC's upside (2x better than holding, due to timing + leverage)
- Only suffered 28% of BTC's downside (avoided 72% of losses)

### Returns
- **Total Return**: Absolute dollar gain/loss
- **Total Return %**: Percentage gain/loss
- **Net Return**: Return after fees and funding costs

### Trade Statistics
- **Total Trades**: Number of completed trades
- **Winning Trades**: Number of profitable trades
- **Losing Trades**: Number of losing trades
- **Avg Trade P&L**: Mean profit/loss per trade

### Distribution Shape Metrics

#### Skewness: Measuring Return Asymmetry

**What it measures**: How asymmetric the distribution of returns is.

**Range**: -∞ to +∞
- **Skewness = 0**: Symmetric distribution (normal bell curve)
- **Skewness > 0**: Positive skew (right tail longer than left tail)
- **Skewness < 0**: Negative skew (left tail longer than right tail)

**Trading Interpretation:**

**Positive Skew (Good)**: "Lottery Ticket" Profile
```
Positive skew = lots of small losses + a few HUGE wins

Visual representation:
     |
 ███ |
████ |        ▂▂
████ |     ▂██▂
████ |  ▂▂████▂
─────┼──────────────►
 Losses  Wins
```
- Many small losing trades, but occasional home runs
- Asymmetric upside: Limited downside, unlimited upside
- Example: Skewness = +3.95 (Trial 10) → Very desirable! Large profitable outliers
- Good for: Risk-seeking strategies, capturing tail events

**Negative Skew (Bad)**: "Penny Picking" Profile
```
Negative skew = lots of small wins + a few CATASTROPHIC losses

Visual representation:
           |
       ███ |
       ████|
    ▂▂████ |
 ▂▂████████|
───────────┼─────►
   Losses  Wins
```
- Many small winning trades, but occasional blow-ups
- Asymmetric downside: Limited upside, catastrophic downside
- Example: Skewness = -3.13 (Trial 3) → Very undesirable! Large loss outliers
- Bad for: "Picking up pennies in front of a steamroller"

**Magnitude Guidelines:**
- **|Skewness| < 0.5**: Roughly symmetric (like a normal distribution)
- **|Skewness| 0.5 to 1.0**: Moderate asymmetry
- **|Skewness| > 1.0**: High asymmetry
- **|Skewness| > 2.0**: Extreme asymmetry (very fat tail on one side)

**Trading Context:**
- **Ideal**: Positive skew → Big wins offset small losses
- **Dangerous**: Negative skew → Big losses wipe out small wins
- **Temperature Impact**: Higher temperature (0.7) creates more variance in skew across trials

**Example from Real Data (Anthropic temp=0.7, 10 trials):**
```
Trial  1: Skewness = -0.53  (slightly negative - small risk)
Trial  2: Skewness = +0.87  (positive - good!)
Trial  3: Skewness = -3.13  (extreme negative - disaster!)
Trial  4: Skewness = -0.49  (slightly negative - small risk)
Trial  5: Skewness = +0.34  (slightly positive - okay)
Trial  6: Skewness = +1.24  (positive - good!)
Trial  7: Skewness = -0.71  (negative - concerning)
Trial  8: Skewness = +1.73  (very positive - great!)
Trial  9: Skewness = +0.84  (positive - good!)
Trial 10: Skewness = +3.95  (extreme positive - excellent!)

Average: +0.06 (nearly neutral - high variance across trials!)
```

This shows that temperature 0.7 is **unpredictable**: 4 trials had good positive skew, 4 trials had bad negative skew, 2 were neutral. You don't know which profile you'll get until after running the backtest.

#### Kurtosis: Measuring Tail Thickness

**What it measures**: How much of the distribution is in the tails (extreme events) vs the center.

**Range**: -∞ to +∞ (technically, but typically -2 to +10)
- **Kurtosis = 0**: Normal distribution (Gaussian bell curve) - this is the baseline
- **Kurtosis > 0**: "Fat tails" - more extreme events than normal distribution
- **Kurtosis < 0**: "Thin tails" - fewer extreme events than normal distribution

**Trading Interpretation:**

**High Kurtosis (Fat Tails)**: Prone to Extreme Events
```
High kurtosis = Occasional extreme outcomes (both wins and losses)

Visual representation (compared to normal distribution):
        |
        |█
        |█
       ▂|█▂
     ▂▂|█▂▂
   ▂▂██|███▂▂   ← Fat tails (more probability here)
 ▂▂████|████▂▂
───────┼───────►
   Losses Wins
```
- Returns cluster around the mean, BUT...
- Occasional extreme wins and extreme losses
- Example: Kurtosis = 15.63 (Trial 10) → Very fat tails! Expect extreme events
- **Risk**: More "black swan" events (both positive and negative)
- **Volatility**: More spiky P&L chart

**Low Kurtosis (Thin Tails)**: Consistent Outcomes
```
Low kurtosis = Outcomes more evenly distributed, fewer extremes

Visual representation (compared to normal distribution):
        |
      ▂▂|▂▂
    ▂███|███▂   ← Thin tails (less probability in extremes)
  ▂█████|█████▂
▂███████|███████▂
────────┼────────►
   Losses  Wins
```
- Returns more evenly spread out
- Fewer extreme outcomes (wins or losses)
- Example: Kurtosis = 0.12 (Trial 5) → Thin tails, more predictable
- **Predictability**: More consistent trade outcomes
- **Volatility**: Smoother P&L chart

**Magnitude Guidelines:**
- **Kurtosis < 0**: Thin tails (platykurtic) - fewer extremes than normal
- **Kurtosis = 0**: Normal tails (mesokurtic) - like a bell curve
- **Kurtosis 0 to 3**: Moderate fat tails
- **Kurtosis > 3**: Very fat tails (leptokurtic) - extreme events common
- **Kurtosis > 10**: Extremely fat tails - almost guaranteed extreme events

**Trading Context:**
- **High Kurtosis**: More "lottery-like" outcomes - big winners and big losers possible
- **Low Kurtosis**: More "consistent" outcomes - predictable but less upside
- **Relationship with Skewness**:
  - High kurtosis + Positive skew = Occasional huge wins (desirable!)
  - High kurtosis + Negative skew = Occasional huge losses (disaster!)

**Example from Real Data (Anthropic temp=0.7, 10 trials):**
```
Trial  1: Kurtosis = 0.97   (moderate fat tails)
Trial  2: Kurtosis = 1.66   (moderate fat tails)
Trial  3: Kurtosis = 11.95  (extreme fat tails - disaster paired with negative skew!)
Trial  4: Kurtosis = 0.12   (thin tails - consistent)
Trial  5: Kurtosis = 0.86   (moderate fat tails)
Trial  6: Kurtosis = 1.60   (moderate fat tails)
Trial  7: Kurtosis = 2.14   (moderate fat tails)
Trial  8: Kurtosis = 3.63   (very fat tails - but paired with positive skew!)
Trial  9: Kurtosis = 2.39   (moderate fat tails)
Trial 10: Kurtosis = 15.63  (extreme fat tails - paired with extreme positive skew!)

Average: +4.85 (fat tails on average)
```

This shows that temperature 0.7 produces **fat tails on average** - expect extreme events more often than a normal distribution would predict.

#### Combined Interpretation: Skewness + Kurtosis

The combination of skewness and kurtosis tells the full story:

**Best Case**: High Kurtosis + Positive Skew
- Example: Trial 10 (Skew +3.95, Kurtosis 15.63)
- Interpretation: "Occasional massive wins, small losses" - Ideal asymmetric profile
- Strategy works: Cut losses quickly, let winners run

**Worst Case**: High Kurtosis + Negative Skew
- Example: Trial 3 (Skew -3.13, Kurtosis 11.95)
- Interpretation: "Occasional catastrophic losses, small wins" - Disaster profile
- Strategy fails: Letting losses run, cutting winners too early

**Most Consistent**: Low Kurtosis + Near-Zero Skew
- Example: Trial 4 (Skew -0.49, Kurtosis 0.12)
- Interpretation: "Predictable, consistent outcomes" - Boring but reliable
- Strategy is mechanical: Similar risk/reward on each trade

**Why These Metrics Matter for LLM Trading:**

1. **Risk Management**: Negative skew warns you that the model might experience rare but catastrophic losses
2. **Temperature Selection**: Lower temperature (0.1) should reduce variance in skewness across trials
3. **Production Deployment**: You want **consistent positive skew** - not a 40% chance of negative skew
4. **Position Sizing**: High kurtosis means you should size positions smaller (expect extreme moves)
5. **Model Selection**: Compare not just mean return, but the entire distribution shape

### Statistical Analysis (Across 10 Trials)

For each configuration, calculate:
- **Mean**: Average performance across trials
- **Standard Deviation**: Measure of variance
- **Min/Max**: Range of outcomes
- **Coefficient of Variation (CV)**: σ/μ (normalized variance)

#### Coefficient of Variation (CV)

**Formula**: `CV = (Standard Deviation / Mean) × 100`

**What it measures**: How much results vary relative to the average (normalized variance).

**Why it's useful**: You can't compare raw standard deviations across different metrics. CV normalizes variance so you can compare consistency across any metric.

**Example:**
```
Metric                  Mean      Std Dev    CV (%)    Interpretation
Return %                5.2%      3.1%       59.6%     High variance - unpredictable
Sharpe Ratio            2.5       0.8        32.0%     High variance - unpredictable
Win Rate                62.3%     8.1%       13.0%     Low variance - consistent!
Max Drawdown            -8.2%     4.5%       54.9%     High variance - unpredictable
Total Trades            47        12         25.5%     Moderate variance
```

**Interpretation Guidelines:**
- **CV < 15%**: Low variance (consistent) ✅
- **CV 15-30%**: Moderate variance ⚠️
- **CV > 30%**: High variance (unpredictable) ❌

**Real-World Meaning:**
- Win Rate CV = 13% → "You can reliably predict ~62% win rate ±8%"
- Return CV = 59.6% → "Return varies wildly, could be +15% or -5%"
- **Key Insight**: A model can have consistent win rate but unpredictable returns (wins often, but SIZE of wins varies)

#### Statistical Significance Testing

**Purpose**: Answer the question: "Is this performance real, or just luck?"

After running 10 trials, you might see:
- Mean return = +5.2%

But is this statistically significant? Or could random chance produce this result?

**Key Metrics:**

1. **T-Statistic**: How many standard errors the mean is from zero
2. **P-Value**: Probability that results are due to random chance
   - p < 0.05 → Statistically significant (95% confident it's not luck) ✅
   - p ≥ 0.05 → Not significant (could be random chance) ❌
3. **Confidence Interval (CI)**: Range where the "true" mean likely falls (95% confidence)

**Example Calculation:**

```python
from scipy import stats

# 10 trial returns
returns = [5.2, 3.8, 7.1, -2.3, 6.5, 4.9, 8.2, 2.1, 5.8, 6.4]

# One-sample t-test (testing if mean is different from 0)
t_stat, p_value = stats.ttest_1samp(returns, 0)

# Calculate confidence interval
mean = np.mean(returns)              # 4.77%
std_error = stats.sem(returns)       # 0.92%
df = len(returns) - 1                # 9 degrees of freedom
margin = stats.t.ppf(0.975, df) * std_error
ci_lower = mean - margin             # 2.69%
ci_upper = mean + margin             # 6.85%

# Results
# t_stat = 5.18
# p_value = 0.001
# CI = [2.69%, 6.85%]
```

**Interpretation:**
- Mean = +4.77%
- CI = [2.69%, 6.85%] (doesn't include 0)
- p = 0.001 (< 0.05)
- **Conclusion**: Performance is statistically significant! We're 95% confident the true mean return is between 2.69% and 6.85%.

**Summary Table Format:**

```
Metric           Mean     Std Err   95% CI          T-Stat   P-Value   Significant?
Return %         4.8%     0.9%      [2.7%, 6.9%]    5.2      0.001     ✅ Yes
Sharpe Ratio     2.5      0.3       [1.9, 3.1]      8.3      <0.001    ✅ Yes
Win Rate         62.3%    2.6%      [56.8%, 67.8%]  24.0     <0.001    ✅ Yes
Sortino Ratio    3.8      0.4       [2.9, 4.7]      9.5      <0.001    ✅ Yes
```

---

## What If Results Are NOT Statistically Significant?

**IMPORTANT**: Non-significant results (p ≥ 0.05) are NOT failures! They provide critical information about model reliability.

### What "Not Significant" Actually Means

**p > 0.05 does NOT mean "the model is bad"**

It means: **"We don't have enough evidence to say the performance is real vs random chance"**

### Common Scenarios & What They Mean

#### Scenario 1: True Zero Performance (Model Doesn't Work)

```python
# 10 trials
returns = [+2.1%, -1.8%, +0.5%, -2.3%, +1.2%, -0.9%, +1.5%, -1.1%, +0.3%, -0.5%]

mean = +0.1%  # Barely positive
p_value = 0.82  # NOT significant
CI = [-1.2%, +1.4%]  # Includes 0
```

**Interpretation:**
- Mean return is essentially zero
- Could be slightly positive or slightly negative
- **Conclusion**: The model has no edge. It's a coin flip.

**Action**: ❌ Don't deploy this model. Investigate why (bad prompts? wrong indicators? inappropriate market conditions?)

#### Scenario 2: High Variance Masks Real Performance

```python
# 10 trials
returns = [+15.2%, -8.3%, +22.1%, -12.5%, +18.7%, -5.2%, +20.3%, -10.1%, +14.8%, -6.5%]

mean = +4.9%  # Positive!
p_value = 0.18  # NOT significant
CI = [-2.3%, +12.1%]  # Wide range, includes 0
std_dev = 12.8%  # HUGE variance
CV = 261%  # Extremely high
```

**Interpretation:**
- Mean is positive (+4.9%), suggesting potential edge
- Results vary wildly: sometimes +20%, sometimes -12%
- High variance makes it impossible to distinguish signal from noise
- **Conclusion**: The model MIGHT have an edge, but variance is so high we can't be confident

**Actions:**
- 🎲 **Too risky for production** - You could deploy and get the -12% trial
- 📊 **Run more trials** - 20-30 trials instead of 10 to reduce uncertainty
- 🔧 **Lower temperature** - Try temp=0.1 to reduce variance
- 💡 **Understand the swings** - Why does trial 3 get +22% but trial 4 gets -12%? Analyze decision differences

#### Scenario 3: Small Sample Size (Need More Data)

```python
# Only 3 trials (not enough!)
returns = [+5.2%, +6.1%, +4.8%]

mean = +5.4%  # Looks good
p_value = 0.08  # Borderline (close to 0.05)
CI = [-0.8%, +11.6%]  # Wide CI
```

**Interpretation:**
- All 3 trials are positive (promising!)
- But sample size too small to draw conclusions
- **Conclusion**: Looks promising, but need more evidence

**Action**: ➕ Run more trials (get to 10+ for statistical power)

#### Scenario 4: Marginally Significant (Borderline)

```python
# 10 trials
returns = [+3.2%, +1.8%, +5.5%, +0.3%, +4.2%, +2.9%, +6.2%, +1.1%, +4.8%, +3.5%]

mean = +3.4%  # Positive
p_value = 0.06  # Close! (just above 0.05)
CI = [-0.1%, +6.9%]  # Barely includes 0
```

**Interpretation:**
- All trials are positive (encouraging!)
- Borderline significant (p = 0.06 vs threshold 0.05)
- **Conclusion**: Probably real, but not quite confident enough

**Actions:**
- ✅ **Cautiously promising** - Strong signal, needs slightly more data
- 🔍 **Run 1-2 more trials** - Might push p < 0.05
- 💼 **Paper trading** - Test in live market with small size while collecting more data

### Interpretation Matrix: Mean Return + P-Value + CV

| Mean Return | P-Value | CV | What It Means | Action |
|-------------|---------|-----|---------------|--------|
| +5.2% | p < 0.01 | 15% | ✅ **GREAT** - Real edge, consistent | Deploy to production |
| +5.2% | p < 0.01 | 45% | ⚠️ **RISKY** - Real edge, but volatile | Use with caution, small positions |
| +5.2% | p = 0.18 | 15% | 🤔 **SMALL SAMPLE** - Consistent but need more data | Run more trials |
| +5.2% | p = 0.18 | 45% | ❌ **UNPREDICTABLE** - High variance, unclear edge | Don't deploy, investigate |
| +0.2% | p = 0.82 | 10% | ❌ **NO EDGE** - Consistently zero | Model doesn't work |
| -3.1% | p < 0.01 | 15% | 🚫 **CONSISTENTLY BAD** - Real negative edge | Model is broken, fix prompts |

### Real Example: Anthropic temp=0.7

```python
# 10 trials - Sharpe ratios
sharpe_ratios = [6.48, -7.29, -7.18, 9.22, -8.51, -4.78, 0.07, -6.60, -1.14, 4.41]

mean = -1.53
p_value = 0.44  # NOT significant
CI = [-5.82, +2.76]  # Wide range including 0
std_dev = 6.53
CV = 426%  # EXTREME variance
```

**Analysis:**

1. **P-value = 0.44** (not significant)
   - Cannot prove this is different from zero
   - Could be negative, neutral, or positive

2. **CV = 426%** (extreme variance)
   - Results vary WILDLY trial-to-trial
   - Trial 4: +9.22 Sharpe (excellent!)
   - Trial 5: -8.51 Sharpe (disaster!)

3. **Confidence Interval = [-5.82, +2.76]**
   - True Sharpe could be anywhere from -5.8 to +2.8
   - Huge uncertainty!

**Interpretation:**
- ❌ **Not production-ready** - Too unstable
- 🔬 **High temperature = high exploration** - This is the price of creativity
- 💡 **Try temp=0.1** - Lower variance, more consistency
- 📊 **Cherry-picking risk** - If you deploy Trial 4 (+9.22), you got lucky. Next time could be Trial 5 (-8.51)

### Actions When Results Are NOT Significant

1. **Run More Trials** (20-30 instead of 10)
   - Increases statistical power
   - Narrows confidence intervals
   - Reduces uncertainty

2. **Lower Temperature** (try 0.1 or 0.0)
   - Reduces variance
   - More deterministic behavior
   - May reveal consistent edge

3. **Try Different Model/Provider**
   - Some LLMs are more consistent
   - Compare Anthropic vs OpenAI vs DeepSeek

4. **Improve Prompts**
   - Add more structure to reduce decision variance
   - Be explicit about risk management rules
   - Provide clearer examples

5. **Longer Backtests** (30-60 days instead of 13)
   - More decisions per trial = less noise
   - Each trial becomes more representative

6. **Accept Reality**
   - If after all improvements still not significant → **Model doesn't work**
   - This is a **successful experiment** (you learned what doesn't work!)
   - Save API costs and move on

### The Value of "Not Significant" Results

**Non-significant results are NOT failures!** They protect you from:

- ❌ Deploying lucky but unreliable models
- ❌ Cherry-picking the best trial and pretending it's typical
- ❌ Confusing noise with signal
- ❌ Losing money on "backtests that worked" but were just luck

**Better to discover unreliability in backtesting than in production with real money!**

Statistical significance testing tells you: "This isn't ready for production. Go back and improve it."

---

## Expected Outcomes

### Hypothesis 1: Temperature Effect
- **Lower temperature (0.1)** → Lower variance, more consistent results
- **Higher temperature (0.7)** → Higher variance, more exploration

### Hypothesis 2: Provider Comparison
- Compare Anthropic vs OpenAI performance
- Measure consistency vs creativity trade-offs

### Hypothesis 3: Risk-Adjusted Returns
- Does higher Sharpe ratio come with lower variance?
- Is consistency worth sacrificing returns?

---

## Data Collection

### Generated Files (Per Trial)

```
results/checkpoints/
├── anthropic_temp0.7_trial1.pkl          # Binary checkpoint
├── anthropic_temp0.7_trial1.json         # Human-readable metadata
├── anthropic_temp0.7_trial1_reasoning.json  # LLM decisions log
├── ...
```

### Analysis Reports (Auto-generated)

```
results/reports/
├── anthropic_temp0.7_trial1_report.md    # Performance summary
├── anthropic_temp0.7_trial1_chart.png    # Equity curve
├── ...
```

---

## Streamlit Dashboard for Analysis

### Running the Dashboard

```bash
streamlit run frontend/Home.py
```

### Available Pages

#### 1. **Home Page** - Inter-Model Comparison
Compare Anthropic vs OpenAI performance:
- **Model Selectors**: Choose one Anthropic checkpoint and one OpenAI checkpoint
- **Equity Curve Comparison**: Side-by-side performance over time
- **Portfolio Allocation**: How each model allocates capital across coins and cash
- **Quick Stats**: Return %, Sharpe Ratio, Win Rate, Total Trades

**Use Case**: "Which model (Anthropic vs OpenAI) performs better overall?"

#### 2. **Statistical Analysis Page** - Intra-Model Variance
Analyze variance within a single configuration:
- **Model + Temperature Selector**: Choose one config (e.g., Anthropic temp=0.7)
- **Equity Curve Overlays**: See all trials overlaid showing the "fan" of outcomes
- **Per-Trial Stats Tables**:
  - Overall Stats: Account Value, Return %, Total PnL, Fees, Win Rate, Biggest Win/Loss, Sharpe, No of Trades
  - Advanced Stats: Avg/Median Trade Size, Avg/Median Hold Time, % Long, Expected Value, Avg/Median Leverage
  - Last row shows "AVERAGE" across all trials
- **Trade P&L KDE Distribution**: Overlaid probability density curves showing how individual trade P&Ls are distributed within each trial
  - Tight curve = consistent trade outcomes
  - Wide curve = varied results
  - Multiple peaks = distinct win/loss clusters
- **Box Plots**: Distribution of Return %, Sharpe, Win Rate, Max Drawdown across trials
- **Metric Correlations**: Scatter matrix showing relationships like:
  - Return vs Win Rate
  - Total Trades vs Avg Hold Time
  - Avg Trade Size vs Return
- **Variance Analysis**: Coefficient of Variation (CV) for key metrics
  - CV < 15% = Low variance (consistent)
  - CV 15-30% = Moderate variance
  - CV > 30% = High variance (unpredictable)

**Use Case**: "How consistent is Anthropic temp=0.7 across 10 trials?"

#### 3. **Coin Analysis Pages** (BTC, ETH, SOL, etc.)
Individual coin performance charts filtered to backtest date range.

### Understanding the Statistical Analysis Page

The Statistical Analysis page focuses on **intra-model variance** - answering:
> "If I run the same model configuration 10 times, how much do the results vary?"

**Key Insights:**
1. **Equity Curve Fan**:
   - Tight fan = consistent performance
   - Wide fan = high variance
   - IQR band shows 25th-75th percentile range

2. **Trade P&L KDE Overlays** (NEW!):
   - Each trial gets its own colored KDE curve
   - Shows distribution of individual trade returns within that trial
   - Compare across trials to see which had more consistent trade sizing

3. **Per-Trial Metrics**:
   - See exact stats for Trial 1, Trial 2, ... Trial 10
   - Aggregate row shows averages
   - Spot outlier trials easily

4. **Correlation Analysis**:
   - Does more trades = higher returns?
   - Does longer hold time improve win rate?
   - Understand metric relationships

---

## Statistical Analysis Script (Alternative to Dashboard)

If you prefer Python scripts over Streamlit, here's how to aggregate metrics across trials:

```python
import json
import pandas as pd
from pathlib import Path

def aggregate_trial_results(config_name: str, num_trials: int = 10):
    """
    Aggregate results across multiple trials

    Args:
        config_name: e.g., "anthropic_temp0.7"
        num_trials: Number of trials to aggregate
    """
    results = []

    for trial_id in range(1, num_trials + 1):
        checkpoint_file = f"results/checkpoints/{config_name}_trial{trial_id}.json"

        with open(checkpoint_file, 'r') as f:
            data = json.load(f)

        results.append({
            'trial_id': trial_id,
            'sharpe_ratio': data['account']['sharpe_ratio'],
            'total_return_pct': data['account']['total_return_percent'],
            'max_drawdown_pct': data['account']['max_drawdown_pct'],
            'win_rate': data['account']['win_rate'],
            'total_trades': data['account']['trade_count'],
            # Add more metrics as needed
        })

    df = pd.DataFrame(results)

    # Calculate statistics
    stats = {
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'cv': df.std() / df.mean()  # Coefficient of variation
    }

    return df, stats

# Usage
df_anthropic_07, stats_anthropic_07 = aggregate_trial_results("anthropic_temp0.7")
df_anthropic_01, stats_anthropic_01 = aggregate_trial_results("anthropic_temp0.1")
df_openai_07, stats_openai_07 = aggregate_trial_results("openai_temp0.7")
df_openai_01, stats_openai_01 = aggregate_trial_results("openai_temp0.1")

print("Anthropic (temp=0.7) Statistics:")
print(stats_anthropic_07)
```

---

## Visualization Examples

### 1. Distribution of Sharpe Ratios

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# Plot distributions for each configuration
sns.histplot(df_anthropic_07['sharpe_ratio'], ax=ax[0, 0], kde=True)
ax[0, 0].set_title('Anthropic (temp=0.7)')

sns.histplot(df_anthropic_01['sharpe_ratio'], ax=ax[0, 1], kde=True)
ax[0, 1].set_title('Anthropic (temp=0.1)')

sns.histplot(df_openai_07['sharpe_ratio'], ax=ax[1, 0], kde=True)
ax[1, 0].set_title('OpenAI (temp=0.7)')

sns.histplot(df_openai_01['sharpe_ratio'], ax=ax[1, 1], kde=True)
ax[1, 1].set_title('OpenAI (temp=0.1)')

plt.tight_layout()
plt.savefig('results/sharpe_ratio_distributions.png')
```

### 2. Box Plot Comparison

```python
# Combine all results
all_results = pd.concat([
    df_anthropic_07.assign(config='Anthropic 0.7'),
    df_anthropic_01.assign(config='Anthropic 0.1'),
    df_openai_07.assign(config='OpenAI 0.7'),
    df_openai_01.assign(config='OpenAI 0.1')
])

plt.figure(figsize=(12, 6))
sns.boxplot(data=all_results, x='config', y='sharpe_ratio')
plt.title('Sharpe Ratio Distribution by Configuration')
plt.ylabel('Sharpe Ratio')
plt.xlabel('Configuration')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/sharpe_ratio_boxplot.png')
```

### 3. Mean vs Variance Trade-off

```python
configs = ['Anthropic 0.7', 'Anthropic 0.1', 'OpenAI 0.7', 'OpenAI 0.1']
means = [stats_anthropic_07['mean']['sharpe_ratio'],
         stats_anthropic_01['mean']['sharpe_ratio'],
         stats_openai_07['mean']['sharpe_ratio'],
         stats_openai_01['mean']['sharpe_ratio']]
stds = [stats_anthropic_07['std']['sharpe_ratio'],
        stats_anthropic_01['std']['sharpe_ratio'],
        stats_openai_07['std']['sharpe_ratio'],
        stats_openai_01['std']['sharpe_ratio']]

plt.figure(figsize=(10, 6))
plt.scatter(stds, means, s=200, alpha=0.6)

for i, config in enumerate(configs):
    plt.annotate(config, (stds[i], means[i]),
                 xytext=(10, 10), textcoords='offset points')

plt.xlabel('Standard Deviation (Risk)')
plt.ylabel('Mean Sharpe Ratio (Return)')
plt.title('Risk-Return Trade-off by Configuration')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/risk_return_tradeoff.png')
```

---

## Interpretation Guidelines

### Low Variance (Good Consistency)
- **Temperature 0.1** should show lower variance
- More **predictable** performance
- Suitable for **production** deployment

### High Variance (Exploration)
- **Temperature 0.7** may show higher variance
- More **creative** strategies
- Potential for **higher returns** but less predictable

### Provider Comparison
- Compare **mean performance** (which is better on average?)
- Compare **consistency** (which is more reliable?)
- Consider **API costs** and **latency** in production

---

## Next Steps After Data Collection

1. **Aggregate Results**: Run analysis script on all 40 trials
2. **Statistical Tests**:
   - T-test: Compare mean Sharpe ratios between configs
   - F-test: Compare variance between configs
   - ANOVA: Multi-group comparison
3. **Publication-Quality Plots**: Generate visualizations for reports
4. **Model Selection**: Choose best config for production based on:
   - Risk-adjusted returns
   - Consistency
   - Cost efficiency
5. **Further Research**:
   - Try different temperature values (0.3, 0.5)
   - Test other LLM providers (DeepSeek, Gemini)
   - Vary decision intervals (1m, 4h)
   - Test different coin portfolios

---

## Why No Caching?

Caching was **removed** to enable true statistical analysis:

❌ **With Cache (Old)**:
- Run 1: Fresh LLM decisions → Result A
- Run 2: Cached responses → Result A (identical)
- Run 3: Cached responses → Result A (identical)
- **Std Dev = 0** (no variance to measure!)

✅ **Without Cache (New)**:
- Run 1: Fresh LLM decisions → Result A
- Run 2: Fresh LLM decisions → Result B (slightly different)
- Run 3: Fresh LLM decisions → Result C (slightly different)
- **Std Dev > 0** (can measure true model variance!)

This is essential for:
- Understanding model **reliability**
- Quantifying **risk** from LLM non-determinism
- Making **informed decisions** about production deployment

---

## Important Notes

### API Costs
- Each trial makes ~78 LLM API calls (one per 4h candle)
- 40 trials × 78 calls = **3,120 API calls**
- Anthropic Claude: ~$0.015 per call = **~$47**
- OpenAI GPT-4: ~$0.03 per call = **~$94**
- Much more affordable than 3m intervals!

### Compute Time
- Each trial takes ~3-6 minutes (depending on API latency)
- 40 trials = **2 - 4 hours** total
- Can run sequentially or in parallel

### Data Integrity
- Use `--run-id` flag to identify trials
- Don't reuse checkpoint filenames
- Keep raw data (don't delete `.pkl` files until analysis complete)

---

## Questions to Answer

After completing the experiments, you should be able to answer:

1. **Which LLM performs better?** (Anthropic vs OpenAI)
2. **What's the consistency trade-off?** (temp 0.1 vs 0.7)
3. **How much variance is acceptable?** (for production deployment)
4. **Is the performance statistically significant?** (p-value < 0.05?)
5. **What's the cost-benefit trade-off?** (performance vs API costs)
6. **Which configuration should we deploy?** (based on data)

---

**Good luck with your experiments! 🚀📊**

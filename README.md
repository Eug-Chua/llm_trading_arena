# LLM Trading Arena

**Rigorous statistical evaluation of LLM-based cryptocurrency trading strategies with proper experimental controls**

Inspired by [Alpha Arena](https://nof1.ai/) by Nof1.ai, this project systematically evaluates whether large language models can trade cryptocurrency markets autonomously, and more importantly, *why* certain configurations perform better than others.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

**What was tested:**
- 2 LLM models: Anthropic Claude Sonnet 4.5, OpenAI GPT-4o-mini
- 2 temperature settings: 0.1 (deterministic) and 0.7 (creative)
- **40 backtests total** (2 models × 2 temps × 10 trials each)
- 13-day period (Oct 17-30, 2025) with 4-hour decision cycles
- 6 cryptocurrencies: BTC, ETH, SOL, BNB, XRP, DOGE

**What was analyzed:**
- Statistical significance testing (t-tests, p-values)
- Risk-adjusted returns (Sharpe, Sortino ratios)
- Per-coin performance breakdown
- Capture ratios (timing quality measurement)
- Behavioral patterns (leverage choices, hold times)
- Benchmark comparisons (buy-and-hold, equal-weight portfolio)

---

## Key Findings

### 1. Only One Statistically Significant Result
Out of 6 model comparisons, only **1 showed statistical significance** (p < 0.05):
- **OpenAI temp 0.1 outperformed Anthropic temp 0.1** (p=0.025)
- Mean difference: 12.32 percentage points (1.40% vs -10.92%)

**Why so few significant results?**
- High variance (Anthropic temp 0.7 ranged from -36% to +56%)
- Small sample size (10 trials per config)
- Short time period (14 days)

### 2. Timing Matters More Than Direction

**The XRP Paradox:**
- XRP was the only coin that rallied (+4.7% during test period)
- Yet Anthropic temp 0.1 **lost $7,731** trading it (worst per-coin result)
- While making money on coins that declined (ETH: -2.6% market, +$9,896 P&L)

**Insight:** Execution quality (timing + leverage) dominated asset selection

### 3. Temperature Effects Are Model-Specific

**Anthropic:**
- Temp 0.1: -10.92% mean (worst overall)
- Temp 0.7: +3.60% mean (best overall)
- **2.5× more variance** at temp 0.7 (35.5% vs 14.1% std dev)

**OpenAI:**
- Temp 0.1: +1.4% mean (best risk-adjusted)
- Temp 0.7: -0.3% mean
- **Minimal variance difference** (7.3% vs 6.2% std dev)

**Takeaway:** Temperature recommendations are not universal across models

### 4. Leverage Was a Deliberate Choice, Not a Constraint

**No leverage limits were imposed** - models freely chose their own:
- Anthropic consistently selected **~15x leverage** (both temps)
- OpenAI consistently selected **10x leverage** (both temps)
- This difference amplified both wins and losses

### 5. Risk/Reward by Coin (Upside vs Downside Capture)

**Best risk/reward - Anthropic temp 0.7 on SOL:**
- Upside capture: **463.6%** vs Downside capture: 316.5%
- Ratio: **1.46** (captures 46% more upside than downside)

**Worst risk/reward - Anthropic temp 0.1 on XRP:**
- Upside capture: 262.0% vs Downside capture: **438.0%**
- Ratio: **0.60** (captures 66% more downside than upside)

**OpenAI's best - temp 0.7 on BTC:**
- Upside capture: **466.7%** vs Downside capture: 405.9%
- Ratio: **1.15** (favorable timing)

---

## Performance Summary

**Ranked by Mean Return:**

| Model | Temp | Mean Return | Std Dev | Sharpe | Best Trial | Worst Trial |
|-------|------|-------------|---------|--------|------------|-------------|
| Anthropic | 0.7 | **+3.6%** | 35.5% | -1.53 | +55.6% | -35.9% |
| OpenAI | 0.1 | +1.4% | 7.3% | **1.18** | +12.3% | -10.8% |
| OpenAI | 0.7 | -0.3% | 6.2% | -0.38 | +7.6% | -11.4% |
| Anthropic | 0.1 | -10.9% | 14.1% | -2.41 | +4.6% | -39.9% |

**Benchmarks:**
- Buy-and-Hold BTC: +0.07%
- Equal-Weight Portfolio (6 coins): -1.23%

**Winner:** Anthropic temp 0.7 beat both benchmarks but with extreme variance

**Most Consistent:** OpenAI temp 0.1 (best Sharpe ratio, only statistically proven edge)

---

## Technical Highlights

### Statistical Rigor
- Proper hypothesis testing (two-sample t-tests)
- Sample standard deviations with Bessel's correction (n-1)

### Advanced Metrics
- **Capture ratios:** Measured upside/downside timing quality per coin
- **Risk asymmetry:** Upside vs downside deviation analysis
- **Distribution shape:** Skewness and kurtosis across trials
- **Per-coin breakdown:** Total P&L, win rate, best/worst trades

### Interactive Dashboard
Built with Streamlit featuring:
- Multi-trial equity curve overlays
- Return distributions and risk-return scatter plots
- Per-coin performance tables (sortable)
- Capture ratio visualizations
- Statistical significance tables

---

## Tech Stack

**Languages & Libraries:**
- Python 3.10+
- Pandas, NumPy (data processing)
- SciPy (statistical testing)
- Plotly (interactive visualizations)
- Streamlit (frontend)

**APIs:**
- Anthropic Claude API
- OpenAI GPT API
- Hyperliquid API

**Statistical Methods:**
- Two-sample t-tests
- Risk-adjusted performance metrics
- Capture ratio analysis

## System Architecture
<img src="system_architecture.png" width="800px">

---

## Project Structure

```
llm_trading_arena/
├── src/
│   ├── agents/               # LLM agents (Claude, GPT)
│   ├── analysis/             # Performance analysis
│   ├── backtesting/          # Core backtesting engine
│   ├── core/                 # Core components
│   ├── data/                 # Data fetching & processing
│   ├── metrics/              # Performance metrics
│   ├── prompts/              # Prompt templates
│   ├── trading/              # Trading logic
│   └── utils/                # Utilities
├── scripts/
│   ├── run_backtest.py       # Main execution script
│   ├── statistical_analysis.py
│   └── benchmark_strategies.py
├── frontend/
│   ├── Home.py               # Streamlit dashboard
│   └── pages/                # Dashboard pages
├── config/                   # Configuration files
├── data/                     # Historical price data
├── results/
│   ├── statistical_analysis/ # CSV summaries
│   └── [model]/temp[XX]/     # Trial checkpoints
├── FINDINGS.md               # Detailed written analysis
├── EXPERIMENTAL_SETUP.md     # Methodology documentation
└── README.md                 # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- API keys from [Anthropic](https://console.anthropic.com/) and/or [OpenAI](https://platform.openai.com/)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/eug-chua/llm_trading_arena.git
cd llm_trading_arena

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
# Create a .env file in the project root with your API keys:
cat > .env << EOF
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
EOF

# Or manually create .env file and add:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-proj-...
```

### Verify Installation

```bash
# Check that historical data is present
ls data/historical/
# Should see: BTC_4h_17-30_Oct_2025.csv, ETH_4h_17-30_Oct_2025.csv, etc.

# Test that config is loaded
python -c "from src.data.indicators import load_indicator_config; print('Config loaded:', load_indicator_config()['data_interval'])"
# Should print: Config loaded: 4h
```

### Run Your First Backtest

```bash
# Run a single trial (takes ~5-10 minutes)
python scripts/run_backtest.py \
  --model anthropic \
  --temperature 0.7 \
  --start 2025-10-17 \
  --end 2025-10-30 \
  --run-id 1

# Results will be saved to: results/checkpoints/anthropic_temp07_trial1.json
```

### Run Multiple Trials for Statistical Analysis

```bash
# Run 10 trials for one configuration (takes ~1 hour)
for i in {1..10}; do
  python scripts/run_backtest.py \
    --model openai \
    --temperature 0.1 \
    --start 2025-10-17 \
    --end 2025-10-30 \
    --run-id $i
done

# Generate statistical summary
python scripts/statistical_analysis.py

# Results will be in: results/statistical_analysis/
```

### View Results

```bash
# Option 1: Interactive dashboard (recommended)
streamlit run frontend/Home.py
# Open browser to http://localhost:8501

# Option 2: View checkpoint JSON directly
cat results/checkpoints/anthropic_temp07_trial1.json | python -m json.tool | head -50

# Option 3: Read analysis documents
cat FINDINGS.md
```

### Troubleshooting

**Import errors:**
```bash
# Make sure you're in the project root directory
pwd  # Should end in /llm_trading_arena

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**API key errors:**
```bash
# Verify .env file exists and has keys
cat .env
# Should show: ANTHROPIC_API_KEY=sk-ant-...

# Test API connection
python -c "import anthropic; print('Anthropic OK')"
python -c "import openai; print('OpenAI OK')"
```

**Missing historical data:**
```bash
# Historical data is included in the repository
# If missing, check that you cloned the full repo (not shallow clone)
git log --oneline | head -5
```

---

## Documentation

**Detailed Guides:**
- **[FINDINGS.md](FINDINGS.md)** - Complete statistical analysis and insights (recommended starting point)
- **[EXPERIMENTAL_SETUP.md](EXPERIMENTAL_SETUP.md)** - Experimental design, parameters, assumptions
- **[STATISTICAL_ANALYSIS_GUIDE.md](STATISTICAL_ANALYSIS_GUIDE.md)** - Understanding variance and significance
- **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)** - System architecture and component interactions

---

## Limitations & Caveats

**This project has important limitations:**

1. **Small Sample Size**
   - 10 trials per configuration (minimal for statistical inference)
   - 13-day period (cannot separate skill from luck)
   - Single time period (regime-dependent results)

2. **No Out-of-Sample Validation**
   - Tested on same period used for benchmarks
   - No walk-forward analysis
   - Results may not generalize to other periods

3. **Backtest vs Reality Gap**
   - Perfect execution (no slippage)
   - Historical data (look-ahead bias possible)
   - No transaction cost variations

4. **Statistical Power**
   - High variance limits ability to detect small effects
   - Most comparisons not statistically significant
   - Cannot make strong causal claims

For production deployment, you would need:
- 3-6 months of data minimum
- 30-50 trials per configuration
- Multiple market regimes tested

---

## Future Improvements

1. **Extended Validation**
   - Test on 3+ additional time periods
   - Walk-forward optimization framework
   - Robustness checks (fees, leverage, intervals)

2. **Machine Learning Components**
   - Train classifier to predict optimal coin selection
   - Feature importance analysis (what drives success?)
   - Regime classification (trending vs ranging markets)

3. **Production Deployment**
   - REST API for predictions
   - Model monitoring dashboard
   - Docker containerization + cloud deployment

4. **Broader Analysis**
   - Factor analysis (Fama-French style)
   - Portfolio optimization
   - Transaction cost modeling

---

## Acknowledgments

- **Nof1.ai** - For creating Alpha Arena and inspiring this research
- **HyperLiquid** - For their open and accessible API

---

**Note:** This is a research/educational project. NFA. Trade at your own risk.

---

*Last Updated: November 2025*

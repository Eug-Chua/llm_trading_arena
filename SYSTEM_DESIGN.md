# LLM Trading Arena - System Design

**Comprehensive System Architecture & Component Interactions**

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Backtesting vs Live Trading](#backtesting-vs-live-trading)
5. [Technology Stack](#technology-stack)

---

## High-Level Architecture

### Backtesting System Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INITIALIZATION                                                         │
│  ───────────────────────────────────────────────────────────────────   │
│  • Load historical data: data/historical/4h/*.parquet                  │
│  • Initialize LLM Agent (Anthropic/OpenAI, temp 0.1/0.7)               │
│  • Initialize Trading Engine (starting capital: $10,000)               │
│  • Load indicator config: config/indicators.yaml                       │
│  • Create timestamps: Oct 17 00:00 → Oct 31 00:00 (4h intervals)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────┐
            │  FOR EACH 4-HOUR TIMESTAMP                    │
            │  (85 decision points over 14 days)            │
            └───────────────┬───────────────────────────────┘
                            │
            ┌───────────────▼────────────────────────────────────────────┐
            │  SINGLE DECISION CYCLE                                     │
            │  ════════════════════════════════════════════════════════  │
            │                                                            │
            │  ┌──────────────────────────────────────────────────────┐ │
            │  │ DATA LAYER                                           │ │
            │  │ ──────────────────────────────────────────────────   │ │
            │  │ HistoricalDataLoader (src/backtesting/)              │ │
            │  │ • Read CSV candles at current timestamp              │ │
            │  │ • Extract OHLC for all 6 coins                       │ │
            │  │                                                      │ │
            │  │ TechnicalIndicators (src/data/indicators.py)         │ │
            │  │ • Calculate EMA(20), EMA(50)                         │ │
            │  │ • Calculate RSI(7), RSI(14)                          │ │
            │  │ • Calculate MACD, ATR(3), ATR(14)                    │ │
            │  │ • Calculate volume averages                          │ │
            │  │                                                      │ │
            │  │ OUTPUT: Market data + indicators, Current prices     │ │
            │  └──────────────┬───────────────────────────────────────┘ │
            │                 │                                           │
            │                 ├──→ Market data + indicators              │
            │                 │                                           │
            │                 ├──→ Current prices only                   │
            │                 │                                           │
            │                 ▼                                           │
            │  ┌──────────────────────────────────────────────────────┐ │
            │  │ EXECUTION LAYER - Position Management (Rule-Based)   │ │
            │  │ ──────────────────────────────────────────────────   │ │
            │  │ INPUT: Current prices ONLY                           │ │
            │  │                                                      │ │
            │  │ TradingEngine (src/trading/trading_engine.py)        │ │
            │  │ • Update prices for all open  positions               │ │
            │  │ • Update unrealized P&L                              │ │
            │  │ • Check stop-loss conditions (rule-based)            │ │
            │  │ • Check profit-target conditions (rule-based)        │ │
            │  │ • Auto-close positions that hit exit criteria        │ │
            │  │ • Apply funding costs (perpetual futures)            │ │
            │  │                                                      │ │
            │  │ NOTE: NO LLM - just checks if price hit thresholds   │ │
            │  └──────────────────────────────────────────────────────┘ │
            │                          │                                 │
            │                          │ Updated account state           │
            │                          ▼                                 │
            │  ┌──────────────────────────────────────────────────────┐ │
            │  │ INTELLIGENCE LAYER                                   │ │
            │  │ ──────────────────────────────────────────────────   │ │
            │  │ INPUT: Market data + indicators + account state      │ │
            │  │                                                      │ │
            │  │ AlphaArenaPrompt (src/prompts/)                      │ │
            │  │ • Combine market data (6 coins × indicators)         │ │
            │  │ • Add account info (balance, positions, P&L)         │ │
            │  │ • Format into Alpha Arena prompt (~12k chars)        │ │
            │  │                                                      │ │
            │  │ LLMAgent (src/agents/llm_agent.py)                   │ │
            │  │ • Send prompt to Claude/GPT API                      │ │
            │  │ • Parse response (chain of thought + trade signals)  │ │
            │  │ • Validate trade signals                             │ │
            │  │                                                      │ │
            │  │ OUTPUT: Trade signals with exit rules                │ │
            │  └──────────────────────────────────────────────────────┘ │
            │                          │                                 │
            │                          │ Trade signals                   │
            │                          ▼                                 │
            │  ┌──────────────────────────────────────────────────────┐ │
            │  │ EXECUTION LAYER - Trade Execution                    │ │
            │  │ ──────────────────────────────────────────────────   │ │
            │  │ INPUT: Trade signals + current prices                │ │
            │  │                                                      │ │
            │  │ TradingEngine (src/trading/trading_engine.py)        │ │
            │  │ • Validate signals (leverage, stop-loss, balance)    │ │
            │  │ • Execute trades:                                    │ │
            │  │   - "buy": Open new position with LLM's exit rules   │ │
            │  │   - "hold": Keep existing position                   │ │
            │  │   - "close_position": Exit position                  │ │
            │  │ • Apply fees (0.05% taker fee)                       │ │
            │  │ • Update account balance                             │ │
            │  └──────────────────────────────────────────────────────┘ │
            │                          │                                 │
            │                          ▼                                 │
            │  ┌──────────────────────────────────────────────────────┐ │
            │  │ ANALYSIS & MONITORING LAYER                          │ │
            │  │ ──────────────────────────────────────────────────   │ │
            │  │ PerformanceTracker (src/trading/performance.py)      │ │
            │  │ • Calculate total return                             │ │
            │  │ • Calculate Sharpe ratio                             │ │
            │  │ • Track max drawdown                                 │ │
            │  │ • Update win rate                                    │ │
            │  │                                                      │ │
            │  │ CheckpointManager (src/backtesting/)                 │ │
            │  │ • Save trade log (timestamp, coin, action, P&L)      │ │
            │  │ • Save account state (balance, positions)            │ │
            │  │ • Save LLM reasoning (chain of thought)              │ │
            │  │ • Save performance metrics                           │ │
            │  │ └──────────────────────────────────────────────────────┘ │
            │                                                            │
            └────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────────┐
                        │  Advance to Next Timestamp     │
                        │  (4 hours later)               │
                        └────────────┬───────────────────┘
                                     │
                                     │ Loop back if more timestamps
                                     └──────────────────────────────────┐
                                                                        │
                                     ┌──────────────────────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────────┐
                        │  All Timestamps Processed?     │
                        │  (Oct 31 00:00 reached?)       │
                        └────────────┬───────────────────┘
                                     │ YES
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FINAL OUTPUT                                                           │
│  ───────────────────────────────────────────────────────────────────   │
│  ReportGenerator (src/analysis/)                                        │
│  • Generate equity curve chart                                          │
│  • Export trades to CSV                                                 │
│  • Calculate final statistics:                                          │
│    - Total return: -10.92% to +55.59% (varies by trial)                │
│    - Win rate, max drawdown, Sharpe ratio                               │
│    - Per-coin performance                                               │
│                                                                         │
│  Saved to: results/checkpoints/{model}_temp{temp}_trial{n}.json         │
│  Charts:   results/reports/{model}_chart_{timestamp}.png                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### How the Decision Interval Works

The system uses `config/indicators.yaml` to determine when decisions happen:

```yaml
# config/indicators.yaml
data_interval: '4h'  # Decisions every 4 hours
```

**In Backtesting Mode:**
- Loads all historical 4-hour candles from CSV files
- Iterates through timestamps sequentially
- No actual sleeping - just advances to next timestamp in dataset
- Example: 2025-10-17 00:00 → 04:00 → 08:00 → 12:00 → ...

**In Live Trading Mode:**
- Fetches real-time data from Hyperliquid API
- Executes decision immediately
- Sleeps for `data_interval` duration (4h = 14,400 seconds)
- Wakes up at next interval to repeat

**Why 4 hours?**
- Balances decision frequency with market noise
- Allows sufficient time for technical indicators to stabilize
- Provides enough data points for statistical analysis (85 decisions)
- Reduces API costs and LLM token usage
- Standard timeframe for swing trading strategies
```

---

## Component Details

### 1. Data Layer

#### 1.1 Hyperliquid API Client (`src/data/hyperliquid_client.py`)

**Purpose:** Fetch live market data from Hyperliquid exchange

**Key Methods:**
```python
get_candles(coin, interval, lookback_hours)
get_market_data(coin)  # funding rate, open interest
get_current_prices(coins)
```

**Data Provided:**
- Raw OHLC candles (Open, High, Low, Close)
- Volume
- Timestamp
- Funding rate (perpetuals)
- Open interest

**Lookback Period:**
- 4-hour interval: Last 240 hours / 10 days (60 candles for historical context)

**API Endpoint:**
```
https://api.hyperliquid.xyz/info
```

---

#### 1.2 Technical Indicators (`src/data/indicators.py`)

**Purpose:** Calculate all technical indicators from raw OHLC data

**Indicators Calculated:**

| Category | Indicators | Library |
|----------|-----------|---------|
| **Trend** | EMA(20), EMA(50) | pandas-ta |
| **Momentum** | RSI(7), RSI(14), MACD | pandas-ta |
| **Volatility** | ATR(3), ATR(14) | pandas-ta |
| **Volume** | VWMA (Volume Weighted Moving Average) | Custom |

**Why We Calculate:**
- Hyperliquid API only provides raw OHLC data
- Standard practice in trading systems
- Full control over calculation parameters

**Dependencies:**
```python
import pandas as pd
import pandas_ta as ta
```

---

#### 1.3 Market Data Pipeline (`src/data/market_data_pipeline.py`)

**Purpose:** Orchestrate data fetching and processing

**Workflow:**
```
1. Fetch 4-hour candles (last 10 days)
2. Fetch market metadata (funding rate, open interest)
3. Calculate technical indicators
4. Combine into structured format
5. Pass to prompt generator
```

**Output Format:**
```python
{
    "BTC": {
        "current_price": 113278,
        "ema20": 113276.79,
        "rsi7": 20.0,
        "macd": 63.8,
        "funding_rate": 0.0013,
        "open_interest": 29380,
        "candles_4h": [...]   # 60 candles for historical context
    },
    # ... ETH, SOL, BNB, XRP, DOGE
}
```

---

### 2. Intelligence Layer

#### 2.1 Prompt Template (`src/prompts/alpha_arena_template.py`)

**Purpose:** Format market data into Alpha Arena-style prompts for LLMs

**Prompt Structure:**
```
┌────────────────────────────────────────────────────┐
│ 1. Session Metadata                                │
│    • Time elapsed, invocation count                │
│    • Current timestamp                             │
├────────────────────────────────────────────────────┤
│ 2. Market Data (All 6 Coins)                       │
│    • Current prices, indicators                    │
│    • Funding rate, open interest                   │
│    • Historical arrays (oldest → newest)           │
│    • 4-hour timeframe                              │
├────────────────────────────────────────────────────┤
│ 3. Account Information                             │
│    • Total return (%)                              │
│    • Available cash                                │
│    • Account value                                 │
│    • Sharpe ratio                                  │
├────────────────────────────────────────────────────┤
│ 4. Current Positions                               │
│    • Entry price, current P&L                      │
│    • Stop-loss, profit-target                      │
│    • Invalidation conditions                       │
├────────────────────────────────────────────────────┤
│ 5. Output Format Instructions                      │
│    • Chain of thought format                       │
│    • Trade signal JSON schema                      │
└────────────────────────────────────────────────────┘
```

**Prompt Size:**
- 6 coins: ~12,000 characters
- 2 coins: ~6,000 characters

**Example Output:**
```
It has been 240 minutes since you started trading. The current time is
2025-10-29 16:14:33 and you've been invoked 60 times.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST

CURRENT MARKET STATE FOR ALL COINS

ALL BTC DATA
current_price = 113278, current_ema20 = 113276.79, current_macd = 63.8,
current_rsi (7 period) = 20.0

Open Interest: Latest: 29380  Average: 28093.9
Funding Rate: 0.0013%

4-hour candle series (oldest → latest):
Mid prices: [113010, 113012, 113015, ..., 113285, 113288, 113291]
EMA indicators (20-period): [112566.27, 112575.387, ..., 112652.458]
MACD indicators: [26.417, 30.663, ..., 53.291]
RSI indicators (7-Period): [68.88, 63.681, ..., 72.661]
...
```

---

#### 2.2 LLM Agent System (`src/agents/`)

**Purpose:** Universal interface for multiple LLM providers

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│          BaseLLMAgent (Abstract)                │
│  • Common parsing logic                         │
│  • Response validation                          │
│  • Trade signal extraction                      │
└──────────────────┬──────────────────────────────┘
                   │
                   │ Inherits
                   ▼
┌─────────────────────────────────────────────────┐
│          LLMAgent (Universal)                   │
│  • Provider routing                             │
│  • API client initialization                    │
│  • Multi-provider support                       │
└─────────────────────────────────────────────────┘
          │              │
          ▼              ▼
    ┌─────────┐    ┌─────────┐
    │ Anthropic│   │  OpenAI │
    │ Claude   │   │  GPT-4o │
    └─────────┘    └─────────┘
```

**Supported Providers (Used in Study):**

| Provider | Model | Status |
|----------|-------|--------|
| **Anthropic** | claude-sonnet-4-5-20250929 | ✅ Used in study |
| **OpenAI** | gpt-4o-mini | ✅ Used in study |

**Key Classes:**

```python
@dataclass
class TradeSignal:
    coin: str                    # BTC, ETH, SOL, etc.
    signal: str                  # "buy" | "hold" | "close_position"
    quantity: float              # Amount to trade
    leverage: int                # 10-20x
    profit_target: float         # Exit price (profit)
    stop_loss: float             # Exit price (loss)
    invalidation_condition: str  # Force close condition
    confidence: float            # 0.0-1.0
    risk_usd: float              # Dollar amount at risk

@dataclass
class AgentResponse:
    chain_of_thought: str                  # LLM reasoning
    trade_signals: Dict[str, TradeSignal]  # Decisions by coin
    raw_response: str                      # Original output
    success: bool                          # API success
    model_name: str                        # Model used
    error: Optional[str] = None            # Error message
```

**Usage:**
```python
# Initialize agent
agent = LLMAgent(provider="anthropic", temperature=0.7)

# Generate decision
response = agent.generate_decision(prompt)

# Process
if response.success:
    for coin, signal in response.trade_signals.items():
        print(f"{coin}: {signal.signal}")
```

---

### 3. Execution Layer

#### 3.1 Trading Engine (`src/trading/trading_engine.py`)

**Purpose:** Execute trades and manage positions based on LLM decisions

**Core Responsibilities:**
1. **Signal Validation**
   - Validate leverage (10-20x range)
   - Check stop-loss and profit-target validity
   - Verify account balance sufficiency

2. **Trade Execution**
   - Open new positions (buy signals)
   - Maintain existing positions (hold signals)
   - Close positions (close_position signals)
   - Apply trading fees (0.05% taker fee)

3. **Position Monitoring**
   - Check stop-loss conditions
   - Check profit-target conditions
   - Check invalidation conditions
   - Update P&L every decision cycle

4. **Risk Management**
   - Prevent over-leveraging
   - Enforce position limits
   - Track total exposure

**Key Methods:**
```python
execute_trade_signal(signal: TradeSignal, current_price: float)
update_positions(market_data: Dict)
check_exit_conditions(position: Position, current_price: float)
calculate_pnl(position: Position, current_price: float)
close_position(coin: str, exit_price: float, reason: str)
```

**Position Tracking:**
```python
@dataclass
class Position:
    coin: str
    entry_price: float
    quantity: float
    leverage: int
    stop_loss: float
    profit_target: float
    invalidation_condition: str
    entry_time: datetime
    unrealized_pnl: float = 0.0
```

---

#### 3.2 Performance Tracker (`src/trading/performance.py`)

**Purpose:** Track returns, Sharpe ratio, and trade statistics

**Metrics Calculated:**

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Total Return** | `(account_value - initial_capital) / initial_capital * 100` | Overall profit/loss |
| **Sharpe Ratio** | `(mean_return - risk_free_rate) / std_dev_returns * √periods` | Risk-adjusted returns |
| **Win Rate** | `winning_trades / total_trades * 100` | Trading accuracy |
| **Max Drawdown** | `min(account_value - peak_value) / peak_value * 100` | Worst decline |
| **Trade Count** | Count of executed trades | Activity level |

**Usage:**
```python
performance = PerformanceTracker(initial_capital=10000)

# Update after each decision
performance.update(
    account_value=current_value,
    timestamp=current_time
)

# Get stats
stats = performance.get_statistics()
# Returns: {
#   "total_return": 3.6,
#   "sharpe_ratio": 0.42,
#   "win_rate": 55.0,
#   "max_drawdown": -8.2
# }
```

---

### 4. Analysis & Monitoring Layer

#### 4.1 Chart Generator (`src/analysis/chart_generator.py`)

**Purpose:** Generate equity curves and performance visualizations

**Charts Produced:**
- Portfolio equity curve (account value over time)
- Entry/exit markers (green = win, red = loss)
- Profit/loss zones (green above $10k, red below)

**Example:**
```python
from src.analysis import ChartGenerator

generator = ChartGenerator()
generator.generate_chart(
    checkpoint_file="results/checkpoints/anthropic_temp07_trial1.json",
    output_path="results/reports/anthropic_chart.png"
)
```

---

#### 4.2 Trade Exporter (`src/analysis/trade_exporter.py`)

**Purpose:** Export trade logs to CSV for analysis

**Output Format:**
```csv
timestamp,coin,action,price,quantity,leverage,pnl,reason
2025-10-17 00:00:00,BTC,buy,67234.5,0.15,15,0.0,New position
2025-10-17 04:00:00,BTC,close,67891.2,0.15,15,984.3,Profit target
```

---

#### 4.3 Report Generator (`src/analysis/report_generator.py`)

**Purpose:** Generate comprehensive performance reports

**Report Sections:**
1. Overall Performance Summary
2. Trade-by-Trade Breakdown
3. Per-Coin Performance
4. Risk Metrics (Sharpe, Max Drawdown)
5. Behavioral Analysis (avg hold time, win rate)

---

### 5. Orchestration Layer

#### 5.1 Trading Orchestrator (`src/core/trading_orchestrator.py`)

**Purpose:** Coordinate all components in the trading loop

**Trading Cycle:**
```
┌─────────────────────────────────────────────────┐
│  1. Fetch Market Data                           │
│     └─▶ HyperliquidClient.get_candles()         │
├─────────────────────────────────────────────────┤
│  2. Calculate Indicators                        │
│     └─▶ TechnicalIndicators.calculate()         │
├─────────────────────────────────────────────────┤
│  3. Generate Prompt                             │
│     └─▶ AlphaArenaPrompt.generate()             │
├─────────────────────────────────────────────────┤
│  4. Get LLM Decision                            │
│     └─▶ LLMAgent.generate_decision()            │
├─────────────────────────────────────────────────┤
│  5. Execute Trades                              │
│     └─▶ TradingEngine.execute_trade_signal()    │
├─────────────────────────────────────────────────┤
│  6. Update Positions & Performance              │
│     └─▶ PerformanceTracker.update()             │
├─────────────────────────────────────────────────┤
│  7. Save Checkpoint                             │
│     └─▶ CheckpointManager.save()                │
├─────────────────────────────────────────────────┤
│  8. Sleep (4 hours for next decision)           │
│     └─▶ time.sleep(14400)                       │
└─────────────────────────────────────────────────┘
           │
           │ Repeat every 4 hours
           ▼
```

---

## Data Flow

### Complete End-to-End Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA ACQUISITION                                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Hyperliquid API                                                       │
│  https://api.hyperliquid.xyz/info                                      │
│         │                                                              │
│         │ GET /candles?coin=BTC&interval=4h&lookback_hours=340         │
│         ▼                                                              │
│  ┌────────────────────────────────────┐                               │
│  │ Raw OHLC Data (85 candles)         │                               │
│  │ [{t:..., o:..., h:..., l:..., c:...}] │                            │
│  └────────────────────────────────────┘                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: FEATURE ENGINEERING                                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  TechnicalIndicators.calculate()                                       │
│         │                                                              │
│         │ Input: OHLC arrays                                           │
│         │ Library: pandas-ta                                           │
│         ▼                                                              │
│  ┌────────────────────────────────────┐                               │
│  │ Calculated Indicators:             │                               │
│  │ • EMA(20): [112566, ..., 112652]   │                               │
│  │ • RSI(14): [68.88, ..., 72.66]     │                               │
│  │ • MACD: [26.42, ..., 53.29]        │                               │
│  │ • ATR(14): [1234.5, ..., 1456.7]   │                               │
│  └────────────────────────────────────┘                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 3: PROMPT GENERATION                                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  AlphaArenaPrompt.generate_prompt()                                    │
│         │                                                              │
│         │ Inputs:                                                      │
│         │ • Market data (6 coins)                                      │
│         │ • Account info (balance, positions)                          │
│         │ • Performance (return, Sharpe)                               │
│         ▼                                                              │
│  ┌────────────────────────────────────┐                               │
│  │ Formatted Prompt (~12,000 chars)   │                               │
│  │                                    │                               │
│  │ ALL BTC DATA                       │                               │
│  │ current_price = 113278             │                               │
│  │ current_ema20 = 113276.79          │                               │
│  │ ...                                │                               │
│  │ [Repeated for all 6 coins]         │                               │
│  │                                    │                               │
│  │ HERE IS YOUR ACCOUNT INFORMATION   │                               │
│  │ Current Total Return: 3.6%         │                               │
│  │ Available Cash: 8234.50            │                               │
│  │ ...                                │                               │
│  └────────────────────────────────────┘                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: LLM DECISION MAKING                                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  LLMAgent.generate_decision()                                          │
│         │                                                              │
│         │ API Call to:                                                 │
│         │ • Anthropic (Claude Sonnet 4.5)                              │
│         │ • OpenAI (GPT-4o-mini)                                       │
│         ▼                                                              │
│  ┌────────────────────────────────────┐                               │
│  │ LLM Response:                      │                               │
│  │                                    │                               │
│  │ # CHAIN OF THOUGHT                 │                               │
│  │ BTC shows bullish divergence...    │                               │
│  │ ETH funding rate too high...       │                               │
│  │                                    │                               │
│  │ # TRADING DECISIONS                │                               │
│  │ {                                  │                               │
│  │   "BTC": {                         │                               │
│  │     "signal": "buy",               │                               │
│  │     "quantity": 0.15,              │                               │
│  │     "leverage": 15,                │                               │
│  │     "stop_loss": 111900,           │                               │
│  │     "profit_target": 115000        │                               │
│  │   }                                │                               │
│  │ }                                  │                               │
│  └────────────────────────────────────┘                               │
│         │                                                              │
│         │ Parse → AgentResponse                                        │
│         ▼                                                              │
│  chain_of_thought: "BTC shows..."                                      │
│  trade_signals: {"BTC": TradeSignal(...)}                              │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: TRADE EXECUTION                                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  TradingEngine.execute_trade_signal()                                  │
│         │                                                              │
│         │ Validation:                                                  │
│         │ • Check leverage (10-20x)                                    │
│         │ • Verify account balance                                     │
│         │ • Validate stop-loss/profit-target                           │
│         ▼                                                              │
│  ┌────────────────────────────────────┐                               │
│  │ Execute Trade:                     │                               │
│  │                                    │                               │
│  │ IF signal = "buy":                 │                               │
│  │   └─▶ Open new position            │                               │
│  │        • Entry: $113,278           │                               │
│  │        • Quantity: 0.15 BTC        │                               │
│  │        • Leverage: 15x             │                               │
│  │        • Cost: $1,132.78           │                               │
│  │        • Fee: $5.66 (0.05%)        │                               │
│  │                                    │                               │
│  │ IF signal = "hold":                │                               │
│  │   └─▶ Keep existing position       │                               │
│  │        • Update unrealized P&L     │                               │
│  │        • Check exit conditions     │                               │
│  │                                    │                               │
│  │ IF signal = "close_position":      │                               │
│  │   └─▶ Exit position                │                               │
│  │        • Exit: $115,234            │                               │
│  │        • Realized P&L: +$293.40    │                               │
│  │        • Fee: $5.76 (0.05%)        │                               │
│  └────────────────────────────────────┘                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 6: PERFORMANCE TRACKING                                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  PerformanceTracker.update()                                           │
│         │                                                              │
│         │ Calculate:                                                   │
│         │ • Account value                                              │
│         │ • Total return                                               │
│         │ • Sharpe ratio                                               │
│         │ • Max drawdown                                               │
│         ▼                                                              │
│  ┌────────────────────────────────────┐                               │
│  │ Performance Metrics:               │                               │
│  │                                    │                               │
│  │ • Account Value: $10,287.74        │                               │
│  │ • Total Return: +2.88%             │                               │
│  │ • Sharpe Ratio: 0.34               │                               │
│  │ • Max Drawdown: -5.23%             │                               │
│  │ • Win Rate: 58.3%                  │                               │
│  │ • Total Trades: 24                 │                               │
│  └────────────────────────────────────┘                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 7: CHECKPOINT & LOGGING                                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  CheckpointManager.save()                                              │
│         │                                                              │
│         │ Save to JSON:                                                │
│         │ • All trades                                                 │
│         │ • Performance metrics                                        │
│         │ • LLM reasoning                                              │
│         │ • Account state                                              │
│         ▼                                                              │
│  results/checkpoints/anthropic_temp07_trial1.json                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Backtesting vs Live Trading

### Backtesting Mode

**Purpose:** Simulate trading on historical data for statistical analysis

**Data Source:** Local historical CSV files
```
data/historical/
├── BTC_4h_17-30_Oct_2025.csv
├── ETH_4h_17-30_Oct_2025.csv
├── SOL_4h_17-30_Oct_2025.csv
├── BNB_4h_17-30_Oct_2025.csv
├── XRP_4h_17-30_Oct_2025.csv
└── DOGE_4h_17-30_Oct_2025.csv
```

**Process:**
```
┌────────────────────────────────────────────┐
│ BacktestEngine                             │
├────────────────────────────────────────────┤
│ 1. Load historical data from CSV           │
│ 2. Iterate through time windows            │
│ 3. For each 4-hour interval:               │
│    └─▶ Generate market snapshot            │
│    └─▶ Call LLM for decision               │
│    └─▶ Execute trade (simulated)           │
│    └─▶ Update positions & P&L              │
│ 4. Save checkpoint after each decision     │
│ 5. Generate final report                   │
└────────────────────────────────────────────┘
```

**Advantages:**
- **Reproducible:** Same data → same results
- **Fast:** No API rate limits
- **Statistical:** Run 10 trials per config
- **Safe:** No real capital at risk

**Output:**
```
results/
├── checkpoints/
│   ├── anthropic_temp01_trial1.json
│   ├── anthropic_temp01_trial2.json
│   └── ...
├── reports/
│   ├── anthropic_chart_031125_1212.png
│   └── ...
└── statistical_analysis/
    ├── anthropic_temp01_overall_stats.csv
    └── ...
```

---

### Live Trading Mode

**Purpose:** Execute real trades with live market data

**Data Source:** Hyperliquid API (real-time)

**Process:**
```
┌────────────────────────────────────────────┐
│ ContinuousLoop                             │
├────────────────────────────────────────────┤
│ while True:                                │
│   1. Fetch live candles from API           │
│   2. Calculate indicators                  │
│   3. Generate prompt                       │
│   4. Get LLM decision                      │
│   5. Execute trade (REAL)                  │
│   6. Save checkpoint                       │
│   7. Sleep 4 hours                         │
│   8. Repeat                                │
└────────────────────────────────────────────┘
```

**Key Differences:**

| Aspect | Backtesting | Live Trading |
|--------|-------------|--------------|
| **Data** | Historical CSV | Real-time API |
| **Speed** | Fast (no sleep) | 4-hour cycles |
| **Capital** | Simulated ($10k) | Real money |
| **Execution** | Simulated fills | Market orders |
| **Risk** | Zero | High |

---

## Technology Stack

### Core Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Core implementation |
| **API Client** | `requests`, `anthropic`, `openai` | LLM & exchange APIs |
| **Data Processing** | `pandas`, `numpy` | Data manipulation |
| **Technical Analysis** | `pandas-ta` | Indicator calculation |
| **Visualization** | `plotly` | Chart generation |
| **Testing** | `pytest` | Unit & integration tests |
| **Logging** | `logging` | Debugging & monitoring |

### Key Dependencies

```python
# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
pandas-ta>=0.3.14b
plotly>=5.14.0
anthropic>=0.25.0
openai>=1.0.0
requests>=2.31.0
pytest>=7.4.0
python-dotenv>=1.0.0
```

### File Structure

```
llm_trading_arena/
├── src/
│   ├── agents/               # LLM agent system
│   │   ├── base_agent.py
│   │   └── llm_agent.py
│   ├── analysis/             # Performance analysis
│   │   ├── chart_generator.py
│   │   ├── trade_exporter.py
│   │   └── report_generator.py
│   ├── backtesting/          # Backtesting engine
│   │   ├── backtest_engine.py
│   │   ├── historical_loader.py
│   │   └── checkpoint_manager.py
│   ├── core/                 # Orchestration
│   │   └── trading_orchestrator.py
│   ├── data/                 # Data fetching & processing
│   │   ├── hyperliquid_client.py
│   │   ├── indicators.py
│   │   └── market_data_pipeline.py
│   ├── metrics/              # Performance metrics
│   ├── prompts/              # Prompt templates
│   │   └── alpha_arena_template.py
│   ├── trading/              # Trading logic
│   │   ├── trading_engine.py
│   │   ├── position.py
│   │   ├── performance.py
│   │   └── continuous_loop.py
│   └── utils/                # Utilities
│       ├── logger.py
│       └── config.py
├── data/
│   └── historical/           # Historical CSV files
├── results/
│   ├── checkpoints/          # JSON checkpoints
│   ├── reports/              # Charts & analysis
│   └── statistical_analysis/ # CSV stats
├── tests/                    # Unit tests
├── scripts/                  # Analysis scripts
│   ├── statistical_analysis.py
│   └── generate_comparison_report.py
└── config/
    └── prompts/
        └── system_prompt.txt
```

---

## Configuration

### Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...

# Trading configuration
INITIAL_CAPITAL=10000
DECISION_INTERVAL_HOURS=4
TRADING_FEE_PCT=0.05

# Data configuration
HYPERLIQUID_API_URL=https://api.hyperliquid.xyz/info
CANDLE_LOOKBACK_4H=340
```

### Trading Parameters

```python
# Constants
INITIAL_CAPITAL = 10000  # USD
DECISION_INTERVAL = 4    # hours
TRADING_FEE = 0.0005     # 0.05%
MIN_LEVERAGE = 10
MAX_LEVERAGE = 20

# Coins
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]

# Timeframes
INTERVALS = {
    "4h": 340     # hours lookback (14 days, 85 candles)
}
```

---

## Performance Optimization

### API Call Efficiency

**Batch Fetching:**
```python
# Fetch all coins in parallel
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [
        executor.submit(client.get_candles, coin, "4h", 340)
        for coin in COINS
    ]
    results = [f.result() for f in futures]
```

**Caching:**
```python
# Cache indicator calculations
@lru_cache(maxsize=100)
def calculate_ema(prices_tuple, period):
    prices = list(prices_tuple)
    return ta.ema(prices, length=period)
```

### Memory Management

**Checkpoint Compression:**
- Only save last 100 decisions (not all)
- Compress JSON with gzip
- Archive old checkpoints

**Data Windowing:**
- Keep only last 60 candles in memory
- Discard older data after indicator calculation

---

## Security Considerations

### API Key Management

```python
# ✅ Good: Load from environment
api_key = os.getenv("ANTHROPIC_API_KEY")

# ❌ Bad: Hardcode in source
api_key = "sk-ant-123456..."  # NEVER DO THIS
```

### Rate Limiting

```python
# Implement exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm_api(prompt):
    return client.messages.create(...)
```

### Input Validation

```python
# Validate LLM outputs
if not (10 <= signal.leverage <= 20):
    logger.warning(f"Invalid leverage: {signal.leverage}")
    return None

if signal.stop_loss >= signal.profit_target:
    logger.warning("Stop-loss must be below profit target")
    return None
```

---

## Monitoring & Debugging

### Logging Levels

```python
# INFO: Normal operation
logger.info(f"[{model_name}] Trade executed: {signal}")

# WARNING: Potential issues
logger.warning(f"[{model_name}] High leverage detected: {leverage}x")

# ERROR: Failures
logger.error(f"[{model_name}] API call failed: {error}")

# DEBUG: Detailed debugging
logger.debug(f"[{model_name}] Prompt length: {len(prompt)} chars")
```

### Health Checks

```python
def health_check():
    """Verify system is operational"""
    checks = {
        "api_connection": test_hyperliquid_api(),
        "llm_access": test_llm_api_key(),
        "data_availability": check_historical_data(),
        "disk_space": check_disk_space()
    }
    return all(checks.values())
```

---

## Testing Strategy

### Unit Tests

```bash
# Test individual components
pytest tests/test_llm_agent.py
pytest tests/test_indicators.py
pytest tests/test_trading_engine.py
```

### Integration Tests

```bash
# Test end-to-end flow
pytest tests/test_backtest_integration.py
```

### Mock Testing

```python
# Mock LLM responses for deterministic tests
@mock.patch('src.agents.llm_agent.LLMAgent._call_llm_api')
def test_trading_decision(mock_llm):
    mock_llm.return_value = MOCK_RESPONSE
    # Test trading logic...
```

---

## Future Enhancements

### Short-Term
- [ ] Add more LLM providers (Gemini, Grok)
- [ ] Implement retry logic for API failures
- [ ] Add real-time monitoring dashboard
- [ ] Optimize prompt size (reduce token usage)

### Long-Term
- [ ] Multi-agent ensemble (vote on decisions)
- [ ] Reinforcement learning from past trades
- [ ] Automated hyperparameter tuning
- [ ] Risk parity portfolio allocation
- [ ] Market regime detection

---

**Document Version:** 1.0
**Last Updated:** November 10, 2025
**Status:** Production-ready for backtesting, experimental for live trading

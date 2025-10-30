"""
Alpha Arena Prompt Template

Generates prompts in the exact format used by nof1.ai Alpha Arena.
Based on scraped prompt analysis from data/prompts/.

Format matches research/PROMPT_DESIGN_ANALYSIS.md specifications.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MarketData:
    """Market data for a single coin"""
    coin: str
    current_price: float
    current_ema20: float
    current_macd: float
    current_rsi_7: float

    # Open interest and funding (for perpetuals)
    oi_latest: float
    oi_average: float
    funding_rate: float

    # Intraday arrays (3-minute intervals, 10 data points)
    prices: List[float]
    ema_20: List[float]
    macd: List[float]
    rsi_7: List[float]
    rsi_14: List[float]

    # Longer-term context (4-hour timeframe)
    ema_20_4h: float
    ema_50_4h: float
    atr_3_4h: float
    atr_14_4h: float
    volume_current: float
    volume_average: float
    macd_4h: List[float]
    rsi_14_4h: List[float]


@dataclass
class Position:
    """Current trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    liquidation_price: float
    unrealized_pnl: float
    leverage: int
    exit_plan: Dict[str, Any]
    confidence: float
    risk_usd: float
    notional_usd: float
    sl_oid: int = -1
    tp_oid: int = -1
    wait_for_fill: bool = False
    entry_oid: int = -1


@dataclass
class AccountInfo:
    """Account state and performance"""
    total_return_percent: float
    available_cash: float
    account_value: float
    positions: List[Position]
    sharpe_ratio: float


class AlphaArenaPrompt:
    """Generate Alpha Arena style prompts"""

    def __init__(self):
        """Initialize prompt generator"""
        self.start_time = datetime.now()
        self.invocation_count = 0

    def _interpret_funding_rate(self, funding_rate: float) -> str:
        """
        Interpret funding rate and provide trading context

        Args:
            funding_rate: Funding rate as decimal (e.g., 0.0001 = 0.01%)

        Returns:
            Human-readable interpretation string
        """
        rate_pct = funding_rate * 100

        # Determine direction and interpretation
        if funding_rate > 0.02:  # > 0.02% is considered high
            return f"⚠️ HIGH POSITIVE (Longs pay shorts {rate_pct:.4f}% every 8h - expensive to hold longs)"
        elif funding_rate > 0.01:  # > 0.01% is moderate positive
            return f"Positive (Longs pay shorts {rate_pct:.4f}% - moderately bullish market)"
        elif funding_rate > 0:
            return f"Slightly positive (Longs pay shorts {rate_pct:.4f}% - mildly bullish)"
        elif funding_rate < -0.02:  # < -0.02% is high negative
            return f"⚠️ HIGH NEGATIVE (Shorts pay longs {abs(rate_pct):.4f}% every 8h - expensive to hold shorts)"
        elif funding_rate < -0.01:  # < -0.01% is moderate negative
            return f"Negative (Shorts pay longs {abs(rate_pct):.4f}% - moderately bearish market)"
        elif funding_rate < 0:
            return f"Slightly negative (Shorts pay longs {abs(rate_pct):.4f}% - mildly bearish)"
        else:
            return "Neutral (No funding payments - balanced market)"

    def format_coin_data(self, data: MarketData) -> str:
        """
        Format market data for a single coin in Alpha Arena style

        Args:
            data: MarketData object with all indicator values

        Returns:
            Formatted string section for this coin
        """
        # Format arrays as comma-separated values
        prices_str = ', '.join([f"{p:.2f}" if p >= 1 else f"{p:.6f}" for p in data.prices])
        ema_str = ', '.join([f"{e:.2f}" if e >= 1 else f"{e:.6f}" for e in data.ema_20])
        macd_str = ', '.join([f"{m:.3f}" for m in data.macd])
        rsi_7_str = ', '.join([f"{r:.3f}" for r in data.rsi_7])
        rsi_14_str = ', '.join([f"{r:.3f}" for r in data.rsi_14])

        # Format 4-hour arrays
        macd_4h_str = ', '.join([f"{m:.3f}" for m in data.macd_4h])
        rsi_14_4h_str = ', '.join([f"{r:.3f}" for r in data.rsi_14_4h])

        # Format price values (handle small vs large numbers)
        def format_price(p: float) -> str:
            if p >= 1:
                return f"{p:.2f}"
            else:
                return f"{p:.6f}"

        # Interpret funding rate
        funding_interpretation = self._interpret_funding_rate(data.funding_rate)

        # Format open interest change
        oi_change = ((data.oi_latest - data.oi_average) / data.oi_average * 100) if data.oi_average > 0 else 0
        oi_trend = "rising" if oi_change > 5 else ("falling" if oi_change < -5 else "stable")

        prompt = f"""ALL {data.coin} DATA
current_price = {format_price(data.current_price)}, current_ema20 = {format_price(data.current_ema20)}, current_macd = {data.current_macd:.3f}, current_rsi (7 period) = {data.current_rsi_7:.3f}
In addition, here is the latest {data.coin} open interest and funding rate for perps (the instrument you are trading):
Open Interest: Latest: {data.oi_latest:.2f} {data.coin}  |  Average: {data.oi_average:.2f} {data.coin}  |  Trend: {oi_trend} ({oi_change:+.1f}%)
Funding Rate: {data.funding_rate * 100:.4f}% (paid every 8 hours) - {funding_interpretation}
Intraday series (3-minute intervals, oldest → latest):
{"Mid prices" if data.coin in ["SOL", "BNB", "XRP", "DOGE"] else "Mid prices"}: [{prices_str}]
EMA indicators (20-period): [{ema_str}]
MACD indicators: [{macd_str}]
RSI indicators (7-Period): [{rsi_7_str}]
RSI indicators (14-Period): [{rsi_14_str}]
Longer-term context (4-hour timeframe):
20-Period EMA: {format_price(data.ema_20_4h)} vs. 50-Period EMA: {format_price(data.ema_50_4h)}
3-Period ATR: {data.atr_3_4h:.3f} vs. 14-Period ATR: {data.atr_14_4h:.3f}
Current Volume: {data.volume_current:.3f} vs. Average Volume: {data.volume_average:.3f}
MACD indicators: [{macd_4h_str}]
RSI indicators (14-Period): [{rsi_14_4h_str}]
"""
        return prompt

    def format_position(self, pos: Position) -> str:
        """
        Format a single position in Alpha Arena style

        Args:
            pos: Position object

        Returns:
            Formatted position string (Python dict repr)
        """
        # Format like Alpha Arena does (Python dict representation)
        position_dict = {
            'symbol': pos.symbol,
            'quantity': pos.quantity,
            'entry_price': pos.entry_price,
            'current_price': pos.current_price,
            'liquidation_price': pos.liquidation_price,
            'unrealized_pnl': pos.unrealized_pnl,
            'leverage': pos.leverage,
            'exit_plan': pos.exit_plan,
            'confidence': pos.confidence,
            'risk_usd': pos.risk_usd,
            'sl_oid': pos.sl_oid,
            'tp_oid': pos.tp_oid,
            'wait_for_fill': pos.wait_for_fill,
            'entry_oid': pos.entry_oid,
            'notional_usd': pos.notional_usd
        }
        return str(position_dict)

    def format_account_info(self, account: AccountInfo) -> str:
        """
        Format account information section

        Args:
            account: AccountInfo object

        Returns:
            Formatted account section
        """
        prompt = f"""HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE
Current Total Return (percent): {account.total_return_percent:.2f}%
Available Cash: {account.available_cash:.2f}
Current Account Value: {account.account_value:.2f}
Current live positions & performance:
"""
        # Add each position or explicitly state none
        if len(account.positions) == 0:
            prompt += "NONE - No open positions currently\n"
        else:
            for pos in account.positions:
                prompt += self.format_position(pos) + "\n"

        prompt += f"Sharpe Ratio: {account.sharpe_ratio:.3f}"

        return prompt

    def generate_prompt(
        self,
        market_data: Dict[str, MarketData],
        account: AccountInfo,
        objective: Optional[str] = None,
        include_output_format: bool = True
    ) -> str:
        """
        Generate complete Alpha Arena style prompt

        Args:
            market_data: Dictionary mapping coin symbol to MarketData
            account: AccountInfo object with positions and performance
            objective: Optional custom objective (uses default if not provided)
            include_output_format: Whether to include output format instructions

        Returns:
            Complete formatted prompt string
        """
        self.invocation_count += 1

        # Calculate elapsed time
        elapsed = datetime.now() - self.start_time
        minutes_elapsed = int(elapsed.total_seconds() / 60)

        # Current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Default objective
        if objective is None:
            objective = (
                "Below, we are providing you with a variety of state data, price data, "
                "and predictive signals so you can discover alpha. Below that is your "
                "current account information, value, performance, positions, etc."
            )

        # Build prompt
        prompt = f"""It has been {minutes_elapsed} minutes since you started trading. The current time is {current_time} and you've been invoked {self.invocation_count} times. {objective}
ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST
Timeframes note: Unless stated otherwise in a section title, intraday series are provided at 3-minute intervals. If a coin uses a different interval, it is explicitly stated in that coin's section.

CURRENT MARKET STATE FOR ALL COINS
"""

        # Add data for each coin (in Alpha Arena order: BTC, ETH, SOL, BNB, XRP, DOGE)
        coin_order = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
        for coin in coin_order:
            if coin in market_data:
                prompt += self.format_coin_data(market_data[coin]) + "\n"
            else:
                logger.warning(f"Missing market data for {coin}")

        # Add account information
        prompt += self.format_account_info(account)

        # Add output format instructions (optional)
        if include_output_format:
            prompt += "\n\n" + self._get_output_format_instructions()

        logger.info(f"Generated prompt (invocation #{self.invocation_count}, {len(prompt)} chars)")

        return prompt

    def _get_output_format_instructions(self) -> str:
        """
        Get output format instructions for the LLM

        Returns:
            Formatted output instructions string
        """
        # Load output format from config file
        from pathlib import Path
        prompt_file = Path(__file__).parent.parent.parent / "config" / "prompts" / "output_format.txt"

        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Output format file not found: {prompt_file}. Using default.")
            # Fallback to default
            return """
REQUIRED OUTPUT FORMAT:

Provide your response in TWO sections:

1. CHAIN OF THOUGHT:
   - Analyze each current position against its invalidation condition
   - Review technical indicators (RSI, MACD, EMA trends)
   - Check if stop-loss or profit targets are near
   - Explain your reasoning for each decision
   - Consider risk management and portfolio balance

2. TRADING DECISIONS:
   For each coin, output JSON in this exact format:

   {
     "COIN": {
       "trade_signal_args": {
         "coin": "COIN",
         "signal": "hold" | "close_position" | "buy",
         "quantity": <number>,
         "profit_target": <price>,
         "stop_loss": <price>,
         "invalidation_condition": "<description>",
         "leverage": <number>,
         "confidence": <0.0-1.0>,
         "risk_usd": <dollar amount>
       }
     }
   }

   Signal types:
   - "hold": Keep existing position (if invalidation condition NOT met)
   - "close_position": Exit position (if invalidation met or profit/stop-loss hit)
   - "buy": Open new position (only if no existing position for that coin)

   IMPORTANT:
   - If you have NO position in a coin and DON'T want to trade it, simply OMIT that coin from your JSON response
   - Only include coins where you have an existing position to hold/close, or want to open a new position
   - Cannot add to existing positions (no pyramiding)
   - Must set stop-loss, profit target, and invalidation condition for every trade
   - Check EACH position's invalidation condition carefully

Example response:

# CHAIN OF THOUGHT
Reviewing my current positions:

ETH: Current price 4108.75, entry 4189.12. Down $461 but above stop-loss (4065.43)
and invalidation level (4000). RSI at 87.6 shows overbought but MACD positive.
Decision: HOLD - no exit conditions met.

BTC: Current 114110, entry 107343. Up $812. Stop-loss at 102026, invalidation
at 105000. Strong uptrend with RSI 73. Decision: HOLD - riding the momentum.

# TRADING DECISIONS
{
  "ETH": {
    "trade_signal_args": {
      "coin": "ETH",
      "signal": "hold",
      "quantity": 5.74,
      "profit_target": 4568.31,
      "stop_loss": 4065.43,
      "invalidation_condition": "If price closes below 4000 on 3-minute candle",
      "leverage": 10,
      "confidence": 0.65,
      "risk_usd": 722.78
    }
  },
  "BTC": {
    "trade_signal_args": {
      "coin": "BTC",
      "signal": "hold",
      "quantity": 0.12,
      "profit_target": 118136.15,
      "stop_loss": 102026.675,
      "invalidation_condition": "If price closes below 105000 on 3-minute candle",
      "leverage": 10,
      "confidence": 0.75,
      "risk_usd": 619.23
    }
  }
}
"""

    def reset(self):
        """Reset invocation counter and start time"""
        self.start_time = datetime.now()
        self.invocation_count = 0
        logger.info("Prompt generator reset")


def create_sample_market_data(coin: str, base_price: float) -> MarketData:
    """
    Create sample market data for testing

    Args:
        coin: Coin symbol
        base_price: Base price for this coin

    Returns:
        MarketData object with realistic sample values
    """
    import random

    # Generate realistic price movement
    prices = []
    current = base_price
    for i in range(10):
        change = random.uniform(-0.01, 0.01) * current
        current += change
        prices.append(current)

    current_price = prices[-1]

    # Calculate EMAs (simplified)
    ema_20_values = [p * 0.998 for p in prices]  # Slightly below price
    current_ema20 = ema_20_values[-1]

    # MACD values
    macd_values = [random.uniform(-50, 50) for _ in range(10)]
    current_macd = macd_values[-1]

    # RSI values
    rsi_7_values = [random.uniform(30, 70) for _ in range(10)]
    rsi_14_values = [random.uniform(40, 60) for _ in range(10)]
    current_rsi_7 = rsi_7_values[-1]

    # 4-hour data
    macd_4h = [random.uniform(-100, 100) for _ in range(10)]
    rsi_14_4h = [random.uniform(35, 65) for _ in range(10)]

    return MarketData(
        coin=coin,
        current_price=current_price,
        current_ema20=current_ema20,
        current_macd=current_macd,
        current_rsi_7=current_rsi_7,
        oi_latest=1000000.0,
        oi_average=950000.0,
        funding_rate=0.00001,
        prices=prices,
        ema_20=ema_20_values,
        macd=macd_values,
        rsi_7=rsi_7_values,
        rsi_14=rsi_14_values,
        ema_20_4h=current_ema20 * 0.995,
        ema_50_4h=current_ema20 * 0.99,
        atr_3_4h=base_price * 0.005,
        atr_14_4h=base_price * 0.008,
        volume_current=100000.0,
        volume_average=150000.0,
        macd_4h=macd_4h,
        rsi_14_4h=rsi_14_4h
    )


def create_sample_account() -> AccountInfo:
    """
    Create sample account for testing

    Returns:
        AccountInfo object with sample positions
    """
    positions = [
        Position(
            symbol='BTC',
            quantity=0.5,
            entry_price=110000.0,
            current_price=114000.0,
            liquidation_price=100000.0,
            unrealized_pnl=2000.0,
            leverage=10,
            exit_plan={
                'profit_target': 120000.0,
                'stop_loss': 108000.0,
                'invalidation_condition': 'If price closes below 105000 on 3-minute candle'
            },
            confidence=0.75,
            risk_usd=1000.0,
            notional_usd=57000.0,
            sl_oid=12345,
            tp_oid=12346,
            entry_oid=12344
        ),
        Position(
            symbol='ETH',
            quantity=5.0,
            entry_price=4000.0,
            current_price=4100.0,
            liquidation_price=3700.0,
            unrealized_pnl=500.0,
            leverage=10,
            exit_plan={
                'profit_target': 4500.0,
                'stop_loss': 3900.0,
                'invalidation_condition': 'If price closes below 3850 on 3-minute candle'
            },
            confidence=0.65,
            risk_usd=500.0,
            notional_usd=20500.0,
            sl_oid=12347,
            tp_oid=12348,
            entry_oid=12349
        )
    ]

    return AccountInfo(
        total_return_percent=25.50,
        available_cash=5000.0,
        account_value=15000.0,
        positions=positions,
        sharpe_ratio=2.45
    )


# Convenience function
def generate_alpha_arena_prompt(
    market_data: Dict[str, MarketData],
    account: AccountInfo
) -> str:
    """
    Generate Alpha Arena prompt (convenience function)

    Args:
        market_data: Dictionary of MarketData by coin
        account: AccountInfo object

    Returns:
        Formatted prompt string
    """
    generator = AlphaArenaPrompt()
    return generator.generate_prompt(market_data, account)

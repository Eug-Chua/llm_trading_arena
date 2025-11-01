"""
Continuous Evaluation Loop

Live trading simulation using real-time market data.
Inherits common orchestration logic from TradingOrchestrator.

Responsibilities:
- Fetch live market data from Hyperliquid API
- Run evaluation loop with configurable intervals
- Log results to JSONL files
- Handle stopping conditions (max iterations, duration)
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..core.trading_orchestrator import TradingOrchestrator
from ..data.market_data_pipeline import MarketDataPipeline
from ..data.indicators import TechnicalIndicators
from ..prompts.alpha_arena_template import AlphaArenaPrompt, MarketData
from ..agents.llm_agent import LLMAgent
from ..trading.trading_engine import TradingEngine
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ContinuousEvaluationLoop(TradingOrchestrator):
    """
    Continuous evaluation loop matching Alpha Arena's operational pattern

    Runs indefinitely, calling LLM every 3 minutes to:
    - Monitor existing positions
    - Check exit conditions
    - Make new trading decisions
    """

    def __init__(
        self,
        llm_agent: LLMAgent,
        starting_capital: float = 10000.0,
        interval_seconds: int = 180,
        coins: Optional[List[str]] = None,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize continuous loop

        Args:
            llm_agent: LLM agent to use for decisions
            starting_capital: Starting capital in USD
            interval_seconds: Seconds between evaluations (default: 180 = 3 min)
            coins: List of coins to trade (default: Alpha Arena coins)
            log_dir: Directory to save logs (default: logs/continuous/)
        """
        # Initialize common components
        trading_engine = TradingEngine(starting_capital=starting_capital)
        prompt_gen = AlphaArenaPrompt()
        indicators = TechnicalIndicators()

        # Call parent constructor
        super().__init__(llm_agent, trading_engine, prompt_gen, indicators)

        # Live-specific components
        self.pipeline = MarketDataPipeline()
        self.interval_seconds = interval_seconds
        self.coins = coins or ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']

        # State tracking
        self.start_time = datetime.now()
        self.iteration_count = 0
        self.last_evaluation_time = None

        # Logging
        self.log_dir = log_dir or Path("logs/continuous")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{session_id}.jsonl"

        logger.info(f"Initialized continuous loop: {llm_agent.model_name}")
        logger.info(f"Starting capital: ${starting_capital:,.2f}")
        logger.info(f"Interval: {interval_seconds}s ({interval_seconds/60:.1f} min)")
        logger.info(f"Coins: {', '.join(self.coins)}")
        logger.info(f"Log file: {self.session_log_file}")

    def run(
        self,
        max_iterations: Optional[int] = None,
        max_duration_hours: Optional[float] = None
    ):
        """
        Run continuous evaluation loop

        Args:
            max_iterations: Maximum iterations (None = infinite)
            max_duration_hours: Maximum runtime in hours (None = infinite)
        """
        logger.info("=" * 80)
        logger.info("STARTING CONTINUOUS EVALUATION LOOP")
        logger.info("=" * 80)

        try:
            while True:
                # Check stopping conditions
                if max_iterations and self.iteration_count >= max_iterations:
                    logger.info(f"Reached max iterations: {max_iterations}")
                    break

                if max_duration_hours:
                    elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
                    if elapsed_hours >= max_duration_hours:
                        logger.info(f"Reached max duration: {max_duration_hours:.1f}h")
                        break

                # Run one iteration
                self._run_iteration()

                # Wait for next interval
                self._wait_for_next_interval()

        except KeyboardInterrupt:
            logger.info("\n⚠️  Loop interrupted by user (Ctrl+C)")

        except Exception as e:
            logger.error(f"❌ Loop crashed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._print_final_summary()

    def _run_iteration(self):
        """Run a single evaluation iteration"""
        self.iteration_count += 1
        iteration_start = datetime.now()

        logger.info("\n" + "=" * 80)
        logger.info(f"ITERATION #{self.iteration_count}")
        logger.info(f"Time: {iteration_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        try:
            # Use parent's template method for common logic
            self.process_iteration(iteration_start)

            # Log performance summary
            summary = self.engine.get_performance_summary()
            logger.info("\n📈 Account Status:")
            logger.info(f"  Value: ${summary['account_value']:,.2f}")
            logger.info(f"  Return: {summary['total_return_percent']:+.2f}%")
            logger.info(f"  Positions: {summary['num_positions']}")
            logger.info(f"  Trades: {summary['total_trades']}")

            self.last_evaluation_time = datetime.now()
            logger.info(f"✅ Iteration #{self.iteration_count} complete")

        except Exception as e:
            logger.error(f"❌ Iteration #{self.iteration_count} failed: {e}")
            import traceback
            traceback.print_exc()

    def _fetch_market_data(self, timestamp: datetime) -> Tuple[Dict, Dict]:
        """Fetch live market data from Hyperliquid API"""
        logger.info("📊 Fetching market data...")

        # Fetch raw data from pipeline
        market_data_raw = self.pipeline.fetch_and_process(
            coins=self.coins,
            lookback_hours_3m=3,
            lookback_hours_4h=240
        )

        # Convert to MarketData objects
        market_data = self._convert_to_market_data(market_data_raw)
        logger.info(f"✓ Fetched data for {len(market_data)} coins")

        # Get current prices
        current_prices = self.pipeline.get_current_prices(coins=self.coins)
        logger.info(f"✓ Current prices: {self._format_prices(current_prices)}")

        # Update funding rates for open positions
        funding_rates = {coin: data.funding_rate for coin, data in market_data.items()}
        self.engine.update_funding_rates(funding_rates)

        return market_data, current_prices

    def _track_results(self, results: Dict, response: Any, market_data: Dict, timestamp: datetime):
        """Track results by logging to JSONL file"""
        # Log trade results to console
        self._log_trade_results(results)

        # Get performance summary
        summary = self.engine.get_performance_summary()

        # Log full iteration data to JSONL
        self._log_iteration({
            'iteration': self.iteration_count,
            'timestamp': timestamp.isoformat(),
            'elapsed_seconds': (datetime.now() - timestamp).total_seconds(),
            'market_data': {coin: data.current_price for coin, data in market_data.items()},
            'account': summary,
            'trades': results,
            'llm_chain_of_thought': response.chain_of_thought,
            'llm_decisions': {
                coin: {
                    'signal': signal.signal,
                    'quantity': signal.quantity,
                    'stop_loss': signal.stop_loss,
                    'profit_target': signal.profit_target,
                    'invalidation_condition': signal.invalidation_condition,
                    'leverage': signal.leverage,
                    'confidence': signal.confidence,
                    'risk_usd': signal.risk_usd,
                    'close_reason': signal.close_reason
                }
                for coin, signal in response.trade_signals.items()
            }
        })

    def _convert_to_market_data(self, raw_data: Dict) -> Dict[str, MarketData]:
        """Convert pipeline output to MarketData objects"""
        market_data = {}

        for coin, data in raw_data.items():
            if data is None:
                logger.warning(f"No data for {coin}, skipping")
                continue

            try:
                market_data[coin] = MarketData(
                    coin=coin,
                    current_price=data['current_price'],
                    current_ema20=data['current_ema20'],
                    current_macd=data['current_macd'],
                    current_rsi_7=data['current_rsi_7'],
                    oi_latest=data.get('open_interest', 0),
                    oi_average=data.get('open_interest', 0) * 0.95,
                    funding_rate=data.get('funding_rate', 0),
                    prices=data['prices_3m'],
                    ema_20=data['ema_20_3m'],
                    macd=data['macd_3m'],
                    rsi_7=data['rsi_7_3m'],
                    rsi_14=data['rsi_14_3m'],
                    ema_20_4h=data['ema_20_4h'],
                    ema_50_4h=data['ema_50_4h'],
                    atr_3_4h=data['atr_3_4h'],
                    atr_14_4h=data['atr_14_4h'],
                    volume_current=data['volume'],
                    volume_average=data['volume_avg'],
                    macd_4h=data['macd_4h'],
                    rsi_14_4h=data['rsi_14_4h']
                )
            except KeyError as e:
                logger.error(f"Missing field for {coin}: {e}")
                continue

        return market_data

    def _log_trade_results(self, results: Dict):
        """Log trade execution results"""
        for coin, result in results.items():
            # Handle both string results and dict results
            if isinstance(result, str):
                logger.info(f"  → {coin}: {result}")
            elif isinstance(result, dict):
                if result.get('success'):
                    action = result.get('action')
                    if action == 'opened':
                        logger.info(f"  ✓ {coin}: OPENED position at ${result['price']:,.2f}")
                    elif action == 'closed':
                        pnl = result.get('pnl', 0)
                        pnl_str = f"{pnl:+,.2f}" if pnl else "N/A"
                        logger.info(f"  ✓ {coin}: CLOSED position, P&L: ${pnl_str}")
                    elif action == 'held':
                        logger.info(f"  → {coin}: HOLD")
                else:
                    logger.warning(f"  ✗ {coin}: {result.get('reason', 'Unknown error')}")

    def _log_iteration(self, data: Dict):
        """Save iteration data to log file"""
        with open(self.session_log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format_prices(self, prices: Dict[str, float]) -> str:
        """Format prices for display"""
        return ', '.join([f"{coin}=${p:,.0f}" for coin, p in prices.items()])

    def _wait_for_next_interval(self):
        """Sleep until next interval, with progress indicator"""
        if self.last_evaluation_time is None:
            time.sleep(self.interval_seconds)
            return

        next_time = self.last_evaluation_time + timedelta(seconds=self.interval_seconds)
        wait_seconds = (next_time - datetime.now()).total_seconds()

        if wait_seconds > 0:
            logger.info(f"\n⏳ Waiting {wait_seconds:.0f}s until next evaluation...")
            time.sleep(wait_seconds)

    def _print_final_summary(self):
        """Print summary when loop ends"""
        elapsed = datetime.now() - self.start_time
        elapsed_hours = elapsed.total_seconds() / 3600

        summary = self.engine.get_performance_summary()

        logger.info("\n" + "=" * 80)
        logger.info("CONTINUOUS LOOP SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {elapsed_hours:.2f} hours ({elapsed})")
        logger.info(f"Total Iterations: {self.iteration_count}")
        logger.info(f"Avg Interval: {elapsed.total_seconds() / max(self.iteration_count, 1):.1f}s")
        logger.info("")
        logger.info("📈 Final Performance:")
        logger.info(f"  Starting Capital: ${summary['starting_capital']:,.2f}")
        logger.info(f"  Final Value: ${summary['account_value']:,.2f}")
        logger.info(f"  Total Return: {summary['total_return_percent']:+.2f}%")
        logger.info(f"  Total Trades: {summary['total_trades']}")
        logger.info(f"  Final Positions: {summary['num_positions']}")
        logger.info("")
        logger.info(f"📝 Session log saved: {self.session_log_file}")
        logger.info("=" * 80)


def run_continuous_loop(
    provider: str = "deepseek",
    model_id: Optional[str] = None,
    starting_capital: float = 10000.0,
    interval_minutes: float = 3.0,
    max_iterations: Optional[int] = None,
    max_duration_hours: Optional[float] = None
):
    """
    Convenience function to run continuous evaluation loop

    Args:
        provider: LLM provider ("deepseek", "openai", "anthropic", etc.)
        model_id: Model identifier (uses default for provider if not specified)
        starting_capital: Starting capital in USD
        interval_minutes: Minutes between evaluations
        max_iterations: Maximum iterations (None = infinite)
        max_duration_hours: Maximum runtime hours (None = infinite)
    """
    from ..agents.llm_agent import LLMAgent

    # Initialize LLM agent
    llm_agent = LLMAgent(provider=provider, model_id=model_id)

    # Create and run loop
    loop = ContinuousEvaluationLoop(
        llm_agent=llm_agent,
        starting_capital=starting_capital,
        interval_seconds=int(interval_minutes * 60)
    )

    loop.run(
        max_iterations=max_iterations,
        max_duration_hours=max_duration_hours
    )

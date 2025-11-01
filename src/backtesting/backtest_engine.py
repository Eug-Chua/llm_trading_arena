"""
Backtest Engine

Event-driven backtesting engine for trading strategies.
Supports checkpoint/resume and non-determinism testing.
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .historical_loader import HistoricalDataLoader
from .checkpoint_manager import CheckpointManager
from ..data.market_data_pipeline import MarketDataPipeline
from ..data.indicators import TechnicalIndicators
from ..prompts.alpha_arena_template import AlphaArenaPrompt, MarketData, AccountInfo, Position as PromptPosition
from ..agents.llm_agent import LLMAgent
from ..trading.trading_engine import TradingEngine
from ..trading.performance import PerformanceTracker
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BacktestEngine:
    """
    Backtest trading strategies on historical data

    Features:
    - Event-driven replay of historical market data
    - Checkpoint save/load for resuming backtests
    - LLM response caching for efficiency
    - Non-determinism testing support
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        coins: List[str],
        model: str = "claude",
        starting_capital: float = 10000.0,
        checkpoint_path: Optional[str] = None,
        use_llm_cache: bool = True,
        run_id: Optional[int] = None,
        data_dir: str = "data/historical",
        checkpoint_dir: str = "results/checkpoints",
        interval: str = "3m"
    ):
        """
        Initialize backtest engine

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            coins: List of coins to trade
            model: LLM model to use ('claude', 'gpt-4o', 'deepseek')
            starting_capital: Starting capital in USD
            checkpoint_path: Optional checkpoint to resume from
            use_llm_cache: Whether to cache LLM responses
            run_id: Optional run identifier for non-determinism testing
            data_dir: Directory with historical data
            checkpoint_dir: Directory for checkpoints
            interval: Decision interval ('1m', '3m', '4h')
        """
        self.start_date = start_date
        self.end_date = end_date
        self.coins = coins
        self.model = model
        self.use_llm_cache = use_llm_cache
        self.run_id = run_id
        self.interval = interval

        # Initialize components
        self.data_loader = HistoricalDataLoader(data_dir)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.prompt_gen = AlphaArenaPrompt()
        self.indicators = TechnicalIndicators()

        # LLM response cache
        self.llm_cache: Dict[str, Any] = {}

        # Resume from checkpoint if provided
        if checkpoint_path:
            self._resume_from_checkpoint(checkpoint_path)
        else:
            # Fresh start
            self.engine = TradingEngine(starting_capital=starting_capital)
            self.llm_agent = LLMAgent(provider=model, validate_api_key=False)
            self.iteration = 0
            self.start_time = datetime.now()

        logger.info(f"Initialized BacktestEngine:")
        logger.info(f"  Date range: {start_date} to {end_date}")
        logger.info(f"  Coins: {', '.join(coins)}")
        logger.info(f"  Model: {model}")
        logger.info(f"  Interval: {interval}")
        logger.info(f"  Starting capital: ${starting_capital:,.2f}")
        logger.info(f"  LLM cache: {'enabled' if use_llm_cache else 'disabled'}")
        if run_id is not None:
            logger.info(f"  Run ID: {run_id}")
        if checkpoint_path:
            logger.info(f"  Resumed from: {checkpoint_path}")

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Load state from checkpoint"""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            use_llm_cache=self.use_llm_cache
        )

        # Restore account state
        account_state = checkpoint["account"]
        self.engine = TradingEngine(
            starting_capital=account_state["starting_capital"]
        )

        # Restore account fields
        self.engine.account.available_cash = account_state["available_cash"]
        self.engine.account.total_return_percent = account_state.get("total_return_percent", 0.0)
        self.engine.account.sharpe_ratio = account_state.get("sharpe_ratio", 0.0)
        self.engine.account.trade_count = account_state.get("trade_count", 0)
        self.engine.account.total_fees_paid = account_state.get("total_fees_paid", 0.0)
        self.engine.account.total_funding_paid = account_state.get("total_funding_paid", 0.0)

        # Restore positions
        from ..trading.position import Position as TradingPosition
        for pos_dict in checkpoint.get("positions", []):
            pos = TradingPosition(
                symbol=pos_dict["symbol"],
                quantity=pos_dict["quantity"],
                entry_price=pos_dict["entry_price"],
                current_price=pos_dict["current_price"],
                leverage=pos_dict["leverage"],
                stop_loss=pos_dict["stop_loss"],
                profit_target=pos_dict["profit_target"],
                invalidation_condition=pos_dict.get("invalidation_condition", ""),
                confidence=pos_dict.get("confidence", 0.0),
                risk_usd=pos_dict.get("risk_usd", 0.0),
                entry_fee=pos_dict.get("entry_fee", 0.0),
                accumulated_funding=pos_dict.get("accumulated_funding", 0.0)
            )
            self.engine.account.positions[pos.symbol] = pos

        # Restore trade history
        self.engine.account.trade_log = checkpoint.get("trade_history", [])

        # Restore LLM cache
        if self.use_llm_cache:
            self.llm_cache = checkpoint.get("llm_cache", {})
        else:
            self.llm_cache = {}

        # Initialize LLM agent (use current model, not checkpoint model)
        self.llm_agent = LLMAgent(provider=self.model, validate_api_key=False)

        # Restore metadata
        metadata = checkpoint.get("metadata", {})
        self.iteration = metadata.get("total_iterations", 0)
        self.start_time = datetime.fromisoformat(checkpoint["checkpoint_date"])

        logger.info(f"Restored from checkpoint:")
        logger.info(f"  Account value: ${self.engine.account.account_value:,.2f}")
        logger.info(f"  Positions: {len(self.engine.account.positions)}")
        logger.info(f"  Trades: {len(self.engine.account.trade_log)}")
        logger.info(f"  Iterations: {self.iteration}")

    def run(
        self,
        checkpoint_path: Optional[str] = None,
        save_every_n_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run backtest

        Args:
            checkpoint_path: Optional path to save final checkpoint
            save_every_n_iterations: Save checkpoint every N iterations

        Returns:
            Dict with backtest results
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)

        # Get timestamps for backtest period
        timestamps = self.data_loader.get_timestamps(
            self.coins,
            self.start_date,
            self.end_date,
            interval=self.interval
        )

        logger.info(f"Processing {len(timestamps)} timestamps...")

        # Event loop
        for idx, timestamp in enumerate(timestamps):
            self.iteration += 1

            if self.iteration % 100 == 0:
                logger.info(
                    f"Iteration {self.iteration}/{len(timestamps)} "
                    f"({timestamp}) - "
                    f"Account: ${self.engine.account.account_value:,.2f}"
                )

            # Process this timestamp
            self._process_timestamp(timestamp, idx, timestamps)

            # Save intermediate checkpoint if requested
            if save_every_n_iterations and self.iteration % save_every_n_iterations == 0:
                temp_checkpoint = f"temp_checkpoint_iter{self.iteration}.pkl"
                self.save_checkpoint(temp_checkpoint, timestamp)

        # Save final checkpoint if requested
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path, timestamps[-1])

            # Auto-generate analysis report after checkpoint is saved
            logger.info("Generating analysis report...")
            self._generate_analysis_report(checkpoint_path)

        # Get final results
        results = self._get_results()

        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final account value: ${results.get('current_value', 0):,.2f}")
        logger.info(f"Total return: {results.get('total_return_pct', 0):+.2f}%")
        logger.info(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Total trades: {results.get('total_trades', 0)}")
        logger.info("=" * 80)

        return results

    def _process_timestamp(self, timestamp: datetime, idx: int, timestamps: list):
        """Process a single timestamp

        Args:
            timestamp: Current timestamp to process
            idx: Index of current timestamp in timestamps list
            timestamps: Full list of timestamps (needed to detect final candle)
        """
        # Log the historical timestamp being processed
        logger.info(f"[{timestamp.strftime('%Y-%m-%d %H:%M')}] Processing iteration {self.iteration}")

        # 1. Load historical candles at this timestamp
        # Adjust lookback based on interval
        if self.interval == '4h':
            # For 4h intervals, use longer lookback
            candles = self.data_loader.get_all_candles_at_time(
                self.coins,
                timestamp,
                lookback_3m=0,  # Don't need 3m data
                lookback_4h=720  # 30 days of 4h candles
            )
        else:
            # For 3m/1m intervals, use default
            candles = self.data_loader.get_all_candles_at_time(
                self.coins,
                timestamp,
                lookback_3m=3,  # 3 hours
                lookback_4h=240  # 10 days
            )

        if not candles:
            logger.warning(f"No data at {timestamp}, skipping")
            return

        # 2. Calculate indicators and build market data
        market_data = {}
        current_prices = {}

        for coin, intervals in candles.items():
            # Get the appropriate timeframe based on interval
            if self.interval == '4h':
                df_primary = intervals['4h']
                df_secondary = intervals['4h']  # Both same for 4h

                if len(df_primary) < 20:
                    logger.warning(f"Insufficient data for {coin} at {timestamp}, skipping")
                    continue
            else:
                df_primary = intervals['3m']  # or '1m' if we add that
                df_secondary = intervals['4h']

                if len(df_primary) < 20 or len(df_secondary) < 20:
                    logger.warning(f"Insufficient data for {coin} at {timestamp}, skipping")
                    continue

            # Calculate indicators (using dataframe directly)
            # EMA
            df_primary = self.indicators.calculate_ema(df_primary, periods=[20, 50])
            df_secondary = self.indicators.calculate_ema(df_secondary, periods=[20, 50])

            # MACD
            df_primary = self.indicators.calculate_macd(df_primary)
            df_secondary = self.indicators.calculate_macd(df_secondary)

            # RSI
            df_primary = self.indicators.calculate_rsi(df_primary, periods=[7, 14])
            df_secondary = self.indicators.calculate_rsi(df_secondary, periods=[7, 14])

            # ATR
            df_secondary = self.indicators.calculate_atr(df_secondary, periods=[3, 14])

            indicators_3m = df_primary
            indicators_4h = df_secondary

            # Get current price (latest close from primary timeframe)
            current_price = float(df_primary.iloc[-1]['close'])
            current_prices[coin] = current_price

            # Build MarketData object
            market_data[coin] = MarketData(
                coin=coin,
                current_price=current_price,
                current_ema20=float(indicators_3m['ema_20'].iloc[-1]),
                current_macd=float(indicators_3m['macd'].iloc[-1]),
                current_rsi_7=float(indicators_3m['rsi_7'].iloc[-1]),
                oi_latest=0.0,  # Not available in historical data
                oi_average=0.0,
                funding_rate=0.0,
                prices=indicators_3m['close'].tail(10).tolist(),
                ema_20=indicators_3m['ema_20'].tail(10).tolist(),
                macd=indicators_3m['macd'].tail(10).tolist(),
                rsi_7=indicators_3m['rsi_7'].tail(10).tolist(),
                rsi_14=indicators_3m['rsi_14'].tail(10).tolist(),
                ema_20_4h=float(indicators_4h['ema_20'].iloc[-1]),
                ema_50_4h=float(indicators_4h['ema_50'].iloc[-1]),
                atr_3_4h=float(indicators_4h['atr_3'].iloc[-1]) if 'atr_3' in indicators_4h else 0.0,
                atr_14_4h=float(indicators_4h['atr_14'].iloc[-1]) if 'atr_14' in indicators_4h else 0.0,
                volume_current=float(df_primary.iloc[-1]['volume']),
                volume_average=float(df_primary['volume'].mean()),
                macd_4h=indicators_4h['macd'].tail(10).tolist(),
                rsi_14_4h=indicators_4h['rsi_14'].tail(10).tolist()
            )

        if not market_data:
            logger.warning(f"No valid market data at {timestamp}")
            return

        # 3. Update prices in trading engine
        self.engine.account.update_prices(current_prices)

        # 4. Check exit conditions (stop-loss/profit-target)
        exits = self.engine.check_exit_conditions(current_prices)
        if exits:
            for exit_info in exits:
                logger.info(
                    f"Auto-exit: {exit_info['symbol']} at ${exit_info['exit_price']:,.2f} "
                    f"({exit_info['reason']})"
                )

        # 5. Build account info for prompt
        account_info = AccountInfo(
            total_return_percent=self.engine.account.total_return_percent,
            available_cash=self.engine.account.available_cash,
            account_value=self.engine.account.account_value,
            positions=[self._position_to_prompt(pos) for pos in self.engine.account.positions.values()],
            sharpe_ratio=self.engine.account.sharpe_ratio
        )

        # 6. Generate prompt
        is_final_candle = (idx == len(timestamps) - 1)  # Check if this is the last timestamp
        prompt = self.prompt_gen.generate_prompt(market_data, account_info, is_final_candle=is_final_candle)

        # 7. Get LLM decision (with caching if enabled)
        response = self._get_llm_decision(prompt, timestamp)

        # 8. Execute trades
        if response and response.trade_signals:
            results = self.engine.execute_signals(response.trade_signals, current_prices, timestamp)
            logger.debug(f"Trade execution results: {results}")

    def _get_llm_decision(self, prompt: str, timestamp: datetime):
        """Get LLM decision with optional caching"""
        # Generate cache key
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        # Check cache if enabled
        if self.use_llm_cache and prompt_hash in self.llm_cache:
            cached = self.llm_cache[prompt_hash]
            logger.debug(f"Using cached LLM response from {cached['timestamp']}")
            return cached['response']

        # Make fresh API call
        try:
            response = self.llm_agent.generate_decision(prompt)

            # Cache response if enabled
            if self.use_llm_cache:
                self.llm_cache[prompt_hash] = {
                    'response': response,
                    'timestamp': timestamp.isoformat(),
                    'model': self.model
                }

            return response

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    def _position_to_prompt(self, pos) -> PromptPosition:
        """Convert trading position to prompt position"""
        return PromptPosition(
            symbol=pos.symbol,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            liquidation_price=pos.liquidation_price,
            unrealized_pnl=pos.unrealized_pnl,
            leverage=pos.leverage,
            exit_plan=pos.exit_plan,
            confidence=pos.confidence,
            risk_usd=pos.risk_usd,
            notional_usd=pos.notional_usd,
            sl_oid=pos.sl_oid,
            tp_oid=pos.tp_oid,
            wait_for_fill=pos.wait_for_fill,
            entry_oid=pos.entry_oid
        )

    def save_checkpoint(self, filepath: str, checkpoint_date: datetime):
        """Save current state to checkpoint"""
        # Build account state dict
        account_state = {
            "starting_capital": self.engine.account.starting_capital,
            "available_cash": self.engine.account.available_cash,
            "account_value": self.engine.account.account_value,
            "total_return_percent": self.engine.account.total_return_percent,
            "sharpe_ratio": self.engine.account.sharpe_ratio,
            "trade_count": self.engine.account.trade_count,
            "total_fees_paid": self.engine.account.total_fees_paid,
            "total_funding_paid": self.engine.account.total_funding_paid
        }

        # Build positions list
        positions = [pos.to_dict() for pos in self.engine.account.positions.values()]

        # Build metadata
        metadata = {
            "model": self.model,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "coins": self.coins,
            "total_iterations": self.iteration,
            "run_id": self.run_id
        }

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            filepath=filepath,
            account_state=account_state,
            trade_history=self.engine.account.trade_log,
            positions=positions,
            checkpoint_date=checkpoint_date,
            metadata=metadata,
            llm_cache=self.llm_cache if self.use_llm_cache else None,
            include_llm_cache=self.use_llm_cache
        )

        # Also save JSON metadata for easy viewing
        json_path = str(filepath).replace('.pkl', '.json')
        checkpoint_data = {
            "checkpoint_id": Path(filepath).stem,
            "created_at": datetime.now().isoformat(),
            "checkpoint_date": checkpoint_date.isoformat(),
            "account": account_state,
            "positions": positions,
            "trade_history": self.engine.account.trade_log,
            "metadata": metadata,
            "llm_cache": self.llm_cache if self.use_llm_cache else {}
        }
        self.checkpoint_manager.save_metadata_json(json_path, checkpoint_data)

        # Export LLM reasoning to readable markdown files
        if self.use_llm_cache and self.llm_cache:
            self._export_llm_reasoning(filepath)

    def _export_llm_reasoning(self, checkpoint_path: str):
        """Export LLM reasoning to a single JSON file for easy review"""

        checkpoint_path = Path(checkpoint_path)

        # Save to results/checkpoints/ folder with matching checkpoint name
        # Generate filename matching checkpoint: checkpoint_name_reasoning.json
        reasoning_file = checkpoint_path.parent / f"{checkpoint_path.stem}_reasoning.json"

        logger.info(f"Exporting LLM reasoning to {reasoning_file}")

        # Sort responses by timestamp
        sorted_responses = sorted(self.llm_cache.items(), key=lambda x: x[1]['timestamp'])

        # Build structured data
        reasoning_data = {
            "backtest_info": {
                "checkpoint_id": checkpoint_path.stem,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "model": self.model,
                "coins": self.coins,
                "interval": self.interval if hasattr(self, 'interval') else None,
                "total_iterations": len(sorted_responses),
                "exported_at": datetime.now().isoformat()
            },
            "iterations": []
        }

        # Export each iteration
        for i, (prompt_hash, response_data) in enumerate(sorted_responses, 1):
            timestamp = response_data['timestamp']
            model = response_data['model']
            agent_response = response_data['response']

            # Get raw response text and parsed signals
            raw_text = agent_response.raw_response if hasattr(agent_response, 'raw_response') else str(agent_response)

            # Extract signals if available
            signals = []
            if hasattr(agent_response, 'trade_signals') and agent_response.trade_signals:
                # trade_signals can be either a dict or a list
                trade_signals_data = agent_response.trade_signals
                if isinstance(trade_signals_data, dict):
                    # If dict, iterate over values
                    for signal in trade_signals_data.values():
                        signals.append({
                            "coin": signal.coin,
                            "signal": signal.signal,
                            "quantity": signal.quantity,
                            "leverage": signal.leverage,
                            "stop_loss": signal.stop_loss,
                            "profit_target": signal.profit_target,
                            "confidence": signal.confidence,
                            "risk_usd": signal.risk_usd,
                            "invalidation_condition": signal.invalidation_condition
                        })
                else:
                    # If list, iterate directly
                    for signal in trade_signals_data:
                        signals.append({
                            "coin": signal.coin,
                            "signal": signal.signal,
                            "quantity": signal.quantity,
                            "leverage": signal.leverage,
                            "stop_loss": signal.stop_loss,
                            "profit_target": signal.profit_target,
                            "confidence": signal.confidence,
                            "risk_usd": signal.risk_usd,
                            "invalidation_condition": signal.invalidation_condition
                        })

            iteration_data = {
                "iteration": i,
                "timestamp": timestamp,
                "model": model,
                "raw_response": raw_text,
                "signals": signals
            }

            reasoning_data["iterations"].append(iteration_data)

        # Save to JSON file
        import json
        with open(reasoning_file, 'w') as f:
            json.dump(reasoning_data, f, indent=2)

        logger.info(f"Exported {len(sorted_responses)} LLM responses to {reasoning_file}")

    def _generate_analysis_report(self, checkpoint_path: str):
        """
        Auto-generate analysis report and chart after backtest completes

        Args:
            checkpoint_path: Path to the checkpoint file that was just saved
        """
        try:
            # Import the analysis functions
            import sys
            from pathlib import Path as PathLib

            # Add scripts directory to path
            scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from analyze_backtest import analyze_checkpoint

            # Generate report (will create both markdown and chart)
            checkpoint_path = Path(checkpoint_path)
            logger.info(f"Analyzing checkpoint: {checkpoint_path}")

            # analyze_checkpoint will auto-generate filenames in results/reports/
            analyze_checkpoint(str(checkpoint_path))

            logger.info("✅ Analysis report generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            logger.warning("You can manually generate the report later with:")
            logger.warning(f"  python scripts/analyze_backtest.py {checkpoint_path}")

    def _get_results(self) -> Dict[str, Any]:
        """Get backtest results"""
        perf_metrics = self.engine.get_detailed_performance()

        return {
            **perf_metrics,
            "model": self.model,
            "run_id": self.run_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "iterations": self.iteration,
            "llm_cache_size": len(self.llm_cache)
        }

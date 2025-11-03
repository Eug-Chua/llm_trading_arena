"""
Trading Orchestrator Base Class

Provides common orchestration logic for trading strategies.
Subclasses implement data fetching and result tracking.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging

from ..prompts.alpha_arena_template import AccountInfo, MarketData, Position as PromptPosition

logger = logging.getLogger(__name__)


class TradingOrchestrator(ABC):
    """
    Base class for trading orchestration

    Implements template method pattern:
    - Common orchestration logic (update prices, check exits, call LLM, execute trades)
    - Abstract methods for data source and result tracking
    - Override hooks for customization (caching, logging, etc.)

    Subclasses:
    - BacktestEngine: Historical data from parquet files
    - ContinuousEvaluationLoop: Live data from Hyperliquid API
    """

    def __init__(
        self,
        llm_agent: Any,
        trading_engine: Any,
        prompt_gen: Any,
        indicators: Any
    ) -> None:
        """
        Initialize orchestrator with dependencies

        Args:
            llm_agent: LLM agent for trading decisions
            trading_engine: Trading engine for execution
            prompt_gen: Prompt generator for LLM
            indicators: Technical indicators calculator
        """
        self.llm_agent = llm_agent
        self.engine = trading_engine
        self.prompt_gen = prompt_gen
        self.indicators = indicators

    def process_iteration(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Process one trading iteration (template method)

        This method defines the algorithm structure. Subclasses customize
        via abstract methods and override hooks.

        Args:
            timestamp: Current timestamp to process

        Returns:
            Dict with iteration results:
            - market_data: Market data used
            - trades: Trade execution results
            - exits: Automatic exits triggered
            - response: LLM response

        Flow:
            1. Fetch market data (subclass implements)
            2. Update prices in trading engine
            3. Check exit conditions (stop-loss/profit-target)
            4. Build account info for prompt
            5. Generate LLM prompt
            6. Get LLM trading decision
            7. Execute trades
            8. Track results (subclass implements)
        """
        # Step 1: Get market data (subclass implements data source)
        market_data, current_prices = self._fetch_market_data(timestamp)

        if not market_data or not current_prices:
            logger.warning(f"No market data at {timestamp}")
            return {}

        # Step 2: Update prices in trading engine
        self.engine.account.update_prices(current_prices)

        # Step 3: Check automatic exit conditions
        exits = self.engine.check_exit_conditions(current_prices)
        self._log_exits(exits)

        # Step 4: Build account info for prompt
        account_info = self._build_account_info()

        # Step 5: Generate prompt
        prompt = self._generate_prompt(market_data, account_info, timestamp)

        # Step 6: Get LLM decision (with error handling)
        response = self._get_llm_decision(prompt, timestamp)
        if not response:
            return {}

        # Step 7: Execute trades (only if signals present)
        results = {}
        if response.trade_signals:
            results = self.engine.execute_signals(
                response.trade_signals,
                current_prices,
                timestamp
            )

        # Step 8: Track results (subclass implements tracking logic)
        if results:
            self._track_results(results, response, market_data, timestamp)

        return {
            'market_data': market_data,
            'trades': results,
            'exits': exits,
            'response': response
        }

    # Abstract methods (subclass MUST implement)

    @abstractmethod
    def _fetch_market_data(self, timestamp: datetime) -> Tuple[Dict, Dict]:
        """
        Fetch market data for timestamp

        Subclass implements data source:
        - BacktestEngine: Load from historical parquet files
        - ContinuousLoop: Fetch from live Hyperliquid API

        Args:
            timestamp: Timestamp to fetch data for

        Returns:
            Tuple of (market_data dict, current_prices dict):
            - market_data: {coin: MarketData object}
            - current_prices: {coin: float price}
        """
        pass

    @abstractmethod
    def _track_results(
        self,
        results: Dict,
        response: Any,
        market_data: Dict,
        timestamp: datetime
    ) -> None:
        """
        Track/log iteration results

        Subclass implements tracking strategy:
        - BacktestEngine: Log to console periodically
        - ContinuousLoop: Save to JSONL file

        Args:
            results: Trade execution results
            response: LLM response
            market_data: Market data used
            timestamp: Current timestamp
        """
        pass

    # Common methods (with optional override hooks)

    def _build_account_info(self) -> AccountInfo:
        """
        Build account info for prompt

        Common logic used by all subclasses.
        Override if you need custom account info.

        Returns:
            AccountInfo object for prompt generation
        """
        return AccountInfo(
            total_return_percent=self.engine.account.total_return_percent,
            available_cash=self.engine.account.available_cash,
            account_value=self.engine.account.account_value,
            positions=[
                self._position_to_prompt(pos)
                for pos in self.engine.account.positions.values()
            ],
            sharpe_ratio=self.engine.account.sharpe_ratio
        )

    def _generate_prompt(
        self,
        market_data: Dict,
        account_info: AccountInfo,
        timestamp: datetime
    ) -> str:
        """
        Generate LLM prompt

        Common logic with hook for final candle detection.
        Override if you need custom prompt generation.

        Args:
            market_data: Market data dict
            account_info: Account info
            timestamp: Current timestamp

        Returns:
            Formatted prompt string
        """
        is_final = self._is_final_timestamp(timestamp)
        prompt: str = self.prompt_gen.generate_prompt(
            market_data,
            account_info,
            is_final_candle=is_final
        )
        return prompt

    def _get_llm_decision(self, prompt: str, timestamp: datetime) -> Optional[Any]:
        """
        Get LLM trading decision

        Common logic with error handling.
        Override to add caching (BacktestEngine does this).

        Args:
            prompt: Formatted prompt
            timestamp: Current timestamp

        Returns:
            LLM response object or None if failed
        """
        try:
            return self.llm_agent.generate_decision(prompt)
        except Exception as e:
            logger.error(f"LLM call failed at {timestamp}: {e}")
            return None

    def _log_exits(self, exits: List[Dict]) -> None:
        """
        Log automatic exits

        Common logging logic.
        Override for custom exit logging.

        Args:
            exits: List of exit info dicts
        """
        if exits:
            for exit_info in exits:
                logger.info(
                    f"Auto-exit: {exit_info['symbol']} at "
                    f"${exit_info['exit_price']:,.2f} ({exit_info['reason']})"
                )

    def _position_to_prompt(self, position: Any) -> PromptPosition:
        """
        Convert Position to PromptPosition

        Common conversion logic.

        Args:
            position: Position object from trading engine

        Returns:
            PromptPosition object for prompt generation
        """
        return PromptPosition(
            symbol=position.symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            current_price=position.current_price,
            liquidation_price=position.liquidation_price,
            unrealized_pnl=position.unrealized_pnl,
            leverage=position.leverage,
            exit_plan=position.exit_plan,
            confidence=position.confidence,
            risk_usd=position.risk_usd,
            notional_usd=position.notional_usd,
            sl_oid=position.sl_oid,
            tp_oid=position.tp_oid,
            wait_for_fill=position.wait_for_fill,
            entry_oid=position.entry_oid
        )

    def _is_final_timestamp(self, timestamp: datetime) -> bool:
        """
        Check if this is the final timestamp

        Default implementation returns False.
        Override if you need to detect final timestamps (BacktestEngine does this).

        Args:
            timestamp: Current timestamp

        Returns:
            True if final timestamp, False otherwise
        """
        return False

"""
Base LLM Trading Agent

Abstract base class for all LLM trading agents. Provides common functionality
for parsing responses and extracting trade signals.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TradeSignal:
    """Structured trading signal from LLM"""
    coin: str
    signal: str  # "hold", "close_position", "buy"
    quantity: float
    profit_target: float
    stop_loss: float
    invalidation_condition: str
    leverage: int
    confidence: float
    risk_usd: float


@dataclass
class AgentResponse:
    """Complete agent response with reasoning and signals"""
    chain_of_thought: str
    trade_signals: Dict[str, TradeSignal]
    raw_response: str
    success: bool
    model_name: str
    error: Optional[str] = None


class BaseLLMAgent(ABC):
    """
    Abstract base class for LLM trading agents

    All LLM agents (DeepSeek, GPT, Claude, etc.) inherit from this class.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        Initialize base agent

        Args:
            model_name: Human-readable model name
            api_key: API key for the LLM provider
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Initialized {model_name} agent (temp={temperature})")

    @abstractmethod
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API and return raw response text

        This method must be implemented by each specific agent.

        Args:
            prompt: Alpha Arena formatted prompt

        Returns:
            Raw response text from LLM

        Raises:
            Exception: If API call fails
        """
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get system prompt for this LLM

        Different models may need slightly different system prompts.

        Returns:
            System prompt string
        """
        pass

    def generate_decision(self, prompt: str) -> AgentResponse:
        """
        Generate trading decision from prompt

        Args:
            prompt: Alpha Arena formatted prompt

        Returns:
            AgentResponse with chain of thought and trade signals
        """
        try:
            logger.info(f"[{self.model_name}] Sending prompt ({len(prompt)} chars)")

            # Call LLM API (implemented by subclass)
            raw_response = self._call_llm_api(prompt)

            logger.info(f"[{self.model_name}] Received response ({len(raw_response)} chars)")

            # Parse response (common logic)
            chain_of_thought, trade_signals = self._parse_response(raw_response)

            return AgentResponse(
                chain_of_thought=chain_of_thought,
                trade_signals=trade_signals,
                raw_response=raw_response,
                success=True,
                model_name=self.model_name
            )

        except Exception as e:
            logger.error(f"[{self.model_name}] Error generating decision: {e}")
            return AgentResponse(
                chain_of_thought="",
                trade_signals={},
                raw_response="",
                success=False,
                model_name=self.model_name,
                error=str(e)
            )

    def _parse_response(self, response: str) -> Tuple[str, Dict[str, TradeSignal]]:
        """
        Parse LLM response into chain of thought and trade signals

        This is common logic used by all agents.

        Args:
            response: Raw LLM response text

        Returns:
            Tuple of (chain_of_thought, trade_signals_dict)
        """
        chain_of_thought = ""
        trade_signals = {}

        try:
            # Extract chain of thought (everything before JSON or "TRADING DECISIONS")
            cot_match = re.search(
                r'#?\s*CHAIN OF THOUGHT[:\s]*(.*?)(?=#?\s*TRADING DECISIONS|$)',
                response,
                re.DOTALL | re.IGNORECASE
            )
            if cot_match:
                chain_of_thought = cot_match.group(1).strip()
            else:
                # Fallback: take everything before first JSON block
                json_start = response.find('{')
                if json_start > 0:
                    chain_of_thought = response[:json_start].strip()
                else:
                    chain_of_thought = response.strip()

            # Extract JSON blocks
            json_blocks = self._extract_json_blocks(response)

            if not json_blocks:
                logger.warning(f"[{self.model_name}] No JSON blocks found in response")
                return chain_of_thought, trade_signals

            # Parse each JSON block
            for json_str in json_blocks:
                try:
                    data = json.loads(json_str)

                    # Handle different JSON formats
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, dict) and 'trade_signal_args' in value:
                                # Format: {"BTC": {"trade_signal_args": {...}}}
                                signal = self._parse_trade_signal(value['trade_signal_args'])
                                if signal:
                                    trade_signals[signal.coin] = signal
                            elif isinstance(value, dict) and 'coin' in value:
                                # Format: {"BTC": {"coin": "BTC", "signal": "hold", ...}}
                                signal = self._parse_trade_signal(value)
                                if signal:
                                    trade_signals[signal.coin] = signal

                except json.JSONDecodeError as e:
                    logger.warning(f"[{self.model_name}] Failed to parse JSON block: {e}")
                    continue

            logger.info(f"[{self.model_name}] Parsed {len(trade_signals)} trade signals")

        except Exception as e:
            logger.error(f"[{self.model_name}] Error parsing response: {e}")

        return chain_of_thought, trade_signals

    def _extract_json_blocks(self, text: str) -> List[str]:
        """
        Extract all JSON blocks from text

        Args:
            text: Text containing JSON blocks

        Returns:
            List of JSON strings
        """
        json_blocks = []
        brace_count = 0
        start_idx = None

        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_blocks.append(text[start_idx:i+1])
                    start_idx = None

        return json_blocks

    def _parse_trade_signal(self, data: dict) -> Optional[TradeSignal]:
        """
        Parse trade signal from dict

        Args:
            data: Dictionary with trade signal fields

        Returns:
            TradeSignal object or None if invalid
        """
        try:
            # Validate required fields
            required = ['coin', 'signal', 'quantity', 'profit_target', 'stop_loss',
                       'invalidation_condition', 'leverage', 'confidence', 'risk_usd']

            for field in required:
                if field not in data:
                    logger.warning(f"[{self.model_name}] Missing required field: {field}")
                    return None

            # Validate signal type
            if data['signal'] not in ['hold', 'close_position', 'buy']:
                logger.warning(f"[{self.model_name}] Invalid signal type: {data['signal']}")
                return None

            # Validate leverage is positive (no range restriction - let LLM decide)
            if data['leverage'] <= 0:
                logger.warning(f"[{self.model_name}] Invalid leverage {data['leverage']} - must be positive")

            # Validate confidence range
            if not (0.0 <= data['confidence'] <= 1.0):
                logger.warning(f"[{self.model_name}] Confidence {data['confidence']} outside 0-1 range")

            return TradeSignal(
                coin=data['coin'],
                signal=data['signal'],
                quantity=float(data['quantity']),
                profit_target=float(data['profit_target']),
                stop_loss=float(data['stop_loss']),
                invalidation_condition=data['invalidation_condition'],
                leverage=int(data['leverage']),
                confidence=float(data['confidence']),
                risk_usd=float(data['risk_usd'])
            )

        except (ValueError, TypeError) as e:
            logger.error(f"[{self.model_name}] Error parsing trade signal: {e}")
            return None

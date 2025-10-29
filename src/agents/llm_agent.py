"""
Universal LLM Trading Agent

Supports multiple LLM providers via a unified interface:
- DeepSeek (OpenAI-compatible)
- OpenAI (GPT models)
- Anthropic (Claude)
- Google (Gemini)
- xAI (Grok)
- Alibaba (Qwen)
"""

import os
from typing import Optional
from openai import OpenAI

from .base_agent import BaseLLMAgent
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMAgent(BaseLLMAgent):
    """
    Universal LLM agent supporting multiple providers

    Automatically routes to the correct API based on provider configuration.
    """

    # Provider configurations
    PROVIDERS = {
        'deepseek': {
            'base_url': 'https://api.deepseek.com',
            'env_var': 'DEEPSEEK_API_KEY',
            'default_model': 'deepseek-chat'
        },
        'openai': {
            'base_url': 'https://api.openai.com/v1',
            'env_var': 'OPENAI_API_KEY',
            'default_model': 'gpt-4-turbo'
        },
        'anthropic': {
            'base_url': None,  # Uses native Anthropic SDK
            'env_var': 'ANTHROPIC_API_KEY',
            'default_model': 'claude-sonnet-4.5-20241022'
        },
        # Add more providers as needed
    }

    def __init__(
        self,
        provider: str = "deepseek",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        Initialize LLM agent

        Args:
            provider: LLM provider ("deepseek", "openai", "anthropic", etc.)
            model_id: Model identifier (uses default for provider if not specified)
            api_key: API key (defaults to environment variable)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: {list(self.PROVIDERS.keys())}")

        self.provider = provider
        provider_config = self.PROVIDERS[provider]

        # Get API key from env if not provided
        if api_key is None:
            api_key = os.getenv(provider_config['env_var'])
            if not api_key:
                logger.warning(f"{provider_config['env_var']} not found. Using placeholder.")
                api_key = "placeholder"

        # Use default model if not specified
        if model_id is None:
            model_id = provider_config['default_model']

        self.model_id = model_id

        # Initialize base class
        super().__init__(
            model_name=f"{provider.capitalize()} ({model_id})",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Initialize API client based on provider
        self._init_client(provider_config)

        logger.info(f"Initialized LLM agent: {self.model_name}")

    def _init_client(self, provider_config: dict):
        """
        Initialize API client based on provider

        Args:
            provider_config: Provider configuration dict
        """
        if self.provider in ['deepseek', 'openai']:
            # OpenAI-compatible API
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=provider_config['base_url']
            )
            self.client_type = 'openai'

        elif self.provider == 'anthropic':
            # Anthropic native SDK
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                self.client_type = 'anthropic'
            except ImportError:
                logger.error("Anthropic SDK not installed. Install with: pip install anthropic")
                raise

        else:
            raise ValueError(f"Client initialization not implemented for: {self.provider}")

    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API and return raw response text

        Args:
            prompt: Alpha Arena formatted prompt

        Returns:
            Raw response text from LLM

        Raises:
            Exception: If API call fails
        """
        if self.client_type == 'openai':
            # OpenAI-compatible API (DeepSeek, OpenAI)
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content

        elif self.client_type == 'anthropic':
            # Anthropic Claude API
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self._get_system_prompt(),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text

        else:
            raise ValueError(f"API call not implemented for client type: {self.client_type}")

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for LLM

        Returns:
            System prompt string
        """
        # Universal system prompt works for all models
        return """You are an expert cryptocurrency trader managing a leveraged portfolio.

Your task is to analyze market data and make trading decisions based on technical indicators,
risk management principles, and existing position exit plans.

Always provide your response in TWO sections:
1. CHAIN OF THOUGHT - Detailed reasoning analyzing each position
2. TRADING DECISIONS - Structured JSON format with trade signals

Be precise with numbers and follow these trading rules:
- Set stop-loss and profit target for every position (MANDATORY)
- Define invalidation condition for each trade (MANDATORY)
- Check EACH position's invalidation condition carefully before deciding to hold or close
- No pyramiding - cannot add to existing positions

Output JSON in this exact format:
{
  "COIN": {
    "trade_signal_args": {
      "coin": "COIN",
      "signal": "hold" | "close_position" | "buy",
      "quantity": <number>,
      "profit_target": <price>,
      "stop_loss": <price>,
      "invalidation_condition": "<description>",
      "leverage": <10-20>,
      "confidence": <0.0-1.0>,
      "risk_usd": <dollar amount>
    }
  }
}"""


def create_agent(
    provider: str = "deepseek",
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> LLMAgent:
    """
    Create LLM agent (convenience function)

    Args:
        provider: LLM provider ("deepseek", "openai", "anthropic", etc.)
        model_id: Model identifier
        api_key: API key
        temperature: Sampling temperature

    Returns:
        LLMAgent instance
    """
    return LLMAgent(
        provider=provider,
        model_id=model_id,
        api_key=api_key,
        temperature=temperature
    )


# Convenience functions for specific providers
def create_deepseek_agent(
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> LLMAgent:
    """Create DeepSeek agent"""
    return LLMAgent(provider="deepseek", api_key=api_key, temperature=temperature)


def create_openai_agent(
    model_id: str = "gpt-4-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> LLMAgent:
    """Create OpenAI agent"""
    return LLMAgent(provider="openai", model_id=model_id, api_key=api_key, temperature=temperature)


def create_claude_agent(
    model_id: str = "claude-sonnet-4.5-20241022",
    api_key: Optional[str] = None,
    temperature: float = 0.7
) -> LLMAgent:
    """Create Claude agent"""
    return LLMAgent(provider="anthropic", model_id=model_id, api_key=api_key, temperature=temperature)

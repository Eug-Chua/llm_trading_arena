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
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from openai import OpenAI

from .base_agent import BaseLLMAgent
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMAgent(BaseLLMAgent):
    """
    Universal LLM agent supporting multiple providers

    Automatically routes to the correct API based on provider configuration.
    Reads model configurations from config/models.yaml
    """

    @staticmethod
    def _load_model_config() -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('models', {})
        except Exception as e:
            logger.warning(f"Error loading config file: {e}. Using defaults.")
            return {}

    # Base provider configurations (API endpoints and env vars)
    BASE_PROVIDERS = {
        'deepseek': {
            'base_url': 'https://api.deepseek.com',
            'env_var': 'DEEPSEEK_API_KEY',
        },
        'openai': {
            'base_url': 'https://api.openai.com/v1',
            'env_var': 'OPENAI_API_KEY',
        },
        'anthropic': {
            'base_url': None,  # Uses native Anthropic SDK
            'env_var': 'ANTHROPIC_API_KEY',
        },
        'google': {
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
            'env_var': 'GOOGLE_API_KEY',
        },
        'xai': {
            'base_url': 'https://api.x.ai/v1',
            'env_var': 'XAI_API_KEY',
        },
        'alibaba': {
            'base_url': 'https://dashscope.aliyuncs.com/api/v1',
            'env_var': 'ALIBABA_API_KEY',
        },
    }

    @classmethod
    def _get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """
        Get provider configuration, merging base config with models.yaml

        Args:
            provider: Provider name (deepseek, openai, anthropic, etc.)

        Returns:
            Provider configuration dict with base_url, env_var, and default_model
        """
        if provider not in cls.BASE_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: {list(cls.BASE_PROVIDERS.keys())}")

        # Start with base config
        config = cls.BASE_PROVIDERS[provider].copy()

        # Load models.yaml to get model_id
        model_configs = cls._load_model_config()

        # Find matching model config by api_provider
        default_model = None
        for _model_key, model_data in model_configs.items():
            if model_data.get('api_provider') == provider:
                default_model = model_data.get('model_id')
                break

        # Fallback defaults if not in models.yaml
        if default_model is None:
            fallback_defaults = {
                'deepseek': 'deepseek-chat',
                'openai': 'gpt-4o',
                'anthropic': 'claude-sonnet-4-5-20250929',
                'google': 'gemini-2.0-flash-exp',
                'xai': 'grok-beta',
                'alibaba': 'qwen-max'
            }
            default_model = fallback_defaults.get(provider, 'unknown')
            logger.warning(
                f"No model_id found in config/models.yaml for provider '{provider}'. "
                f"Using fallback: {default_model}"
            )

        config['default_model'] = default_model
        return config

    def __init__(
        self,
        provider: str = "deepseek",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        validate_api_key: bool = True
    ):
        """
        Initialize LLM agent

        Args:
            provider: LLM provider ("deepseek", "openai", "anthropic", etc.)
            model_id: Model identifier (uses default for provider if not specified)
            api_key: API key (defaults to environment variable)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            validate_api_key: If True, raises error when API key is missing (default: True)
        """
        # Get provider configuration (merges base config + models.yaml)
        provider_config = self._get_provider_config(provider)

        self.provider = provider

        # Get API key from env if not provided
        if api_key is None:
            api_key = os.getenv(provider_config['env_var'])
            if not api_key:
                if validate_api_key:
                    raise ValueError(
                        f"\n❌ Missing API key for {provider}!\n\n"
                        f"Please set {provider_config['env_var']} in your environment:\n"
                        f"  1. Copy .env.example to .env\n"
                        f"  2. Add your API key: {provider_config['env_var']}=sk-...\n"
                        f"  3. Run: export $(cat .env | xargs)  # or source .env\n\n"
                        f"Alternatively, pass api_key parameter directly to LLMAgent().\n"
                    )
                else:
                    logger.warning(f"{provider_config['env_var']} not found. Using placeholder.")
                    api_key = "placeholder"

        # Use default model if not specified (from models.yaml or fallback)
        if model_id is None:
            model_id = provider_config['default_model']
            logger.info(f"Using model from config: {model_id}")

        self.model_id = model_id
        self.validate_api_key = validate_api_key

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
        try:
            if self.client_type == 'openai':
                # OpenAI-compatible API (DeepSeek, OpenAI)
                # GPT-5 models use max_completion_tokens, GPT-4 and earlier use max_tokens
                api_params = {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                }

                # GPT-5 and GPT-4.1 specific parameter handling
                is_new_gpt = self.provider == 'openai' and ('gpt-5' in self.model_id.lower() or 'gpt-4.1' in self.model_id.lower())

                if is_new_gpt:
                    # GPT-5 and GPT-4.1 models: use max_completion_tokens, omit temperature (only supports default 1.0)
                    api_params["max_completion_tokens"] = self.max_tokens
                else:
                    # GPT-4 and earlier: use max_tokens and temperature
                    api_params["max_tokens"] = self.max_tokens
                    api_params["temperature"] = self.temperature

                response = self.client.chat.completions.create(**api_params)

                # Debug logging for empty responses
                content = response.choices[0].message.content
                if not content:
                    logger.warning(f"Empty response from {self.model_id}")
                    logger.warning(f"  Finish reason: {response.choices[0].finish_reason}")
                    logger.warning(f"  Usage: {response.usage}")

                return content

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

        except Exception as e:
            error_msg = str(e).lower()

            # Handle common API errors with helpful messages
            if 'unauthorized' in error_msg or 'authentication' in error_msg or '401' in error_msg:
                raise ValueError(
                    f"\n❌ API Authentication Failed for {self.provider}!\n\n"
                    f"Your API key appears to be invalid or expired.\n"
                    f"Please check {self.PROVIDERS[self.provider]['env_var']} in your .env file.\n"
                ) from e

            elif 'rate limit' in error_msg or '429' in error_msg:
                raise RuntimeError(
                    f"\n⚠️  Rate Limit Exceeded for {self.provider}!\n\n"
                    f"You've hit the API rate limit. Please wait and try again.\n"
                    f"Consider using a different model or provider.\n"
                ) from e

            elif 'quota' in error_msg or 'insufficient' in error_msg:
                raise RuntimeError(
                    f"\n⚠️  API Quota Exceeded for {self.provider}!\n\n"
                    f"You've exceeded your API quota or balance.\n"
                    f"Please check your account and add credits if needed.\n"
                ) from e

            elif 'model' in error_msg and 'not found' in error_msg:
                raise ValueError(
                    f"\n❌ Model Not Found: {self.model_id}\n\n"
                    f"The model '{self.model_id}' is not available for {self.provider}.\n"
                    f"Check the model name or use the default model.\n"
                ) from e

            else:
                # Re-raise with original error for unexpected issues
                logger.error(f"API call failed: {e}")
                raise

    def _get_system_prompt(self) -> str:
        """
        Get system prompt for LLM

        Returns:
            System prompt string
        """
        # Load system prompt from config file
        from pathlib import Path
        prompt_file = Path(__file__).parent.parent.parent / "config" / "prompts" / "system_prompt.txt"

        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.warning(f"System prompt file not found: {prompt_file}. Using default.")
            # Fallback to default
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

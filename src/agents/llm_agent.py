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
        """
        Load model configuration from YAML file

        Raises:
            FileNotFoundError: If config/models.yaml does not exist
            ValueError: If config file is invalid

        Returns:
            Dict containing model and provider configurations
        """
        config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Critical configuration file missing: {config_path}\n"
                f"This file is required for LLM agent initialization.\n"
                f"Please ensure config/models.yaml exists with proper model settings."
            )

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

                if not config:
                    raise ValueError(f"Config file {config_path} is empty")

                if 'models' not in config:
                    raise ValueError(
                        f"Config file {config_path} missing 'models' section.\n"
                        f"Expected structure: models: {{model_name: {{api_provider: ..., model_id: ...}}}}"
                    )

                if 'providers' not in config:
                    raise ValueError(
                        f"Config file {config_path} missing 'providers' section.\n"
                        f"Expected structure: providers: {{provider_name: {{base_url: ..., env_var: ...}}}}"
                    )

                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")

    @classmethod
    def _get_provider_config(cls, provider: str) -> Dict[str, Any]:
        """
        Get provider configuration from models.yaml

        Args:
            provider: Provider name (deepseek, openai, anthropic, etc.)

        Raises:
            ValueError: If provider not found in config or missing required fields

        Returns:
            Provider configuration dict with base_url, env_var, and default_model
        """
        # Load configuration
        full_config = cls._load_model_config()
        providers = full_config.get('providers', {})
        models = full_config.get('models', {})

        # Check if provider exists
        if provider not in providers:
            available = list(providers.keys())
            raise ValueError(
                f"Provider '{provider}' not found in config/models.yaml.\n"
                f"Available providers: {available}\n"
                f"Please add provider configuration to config/models.yaml under 'providers' section."
            )

        # Get provider config
        provider_config = providers[provider].copy()

        # Validate required fields
        required_fields = ['env_var']
        missing = [f for f in required_fields if f not in provider_config]
        if missing:
            raise ValueError(
                f"Provider '{provider}' missing required fields: {missing}\n"
                f"Each provider must have: env_var (and optionally base_url)"
            )

        # Find default model for this provider
        default_model = None
        for model_key, model_data in models.items():
            if model_data.get('api_provider') == provider:
                default_model = model_data.get('model_id')
                if not default_model:
                    raise ValueError(
                        f"Model '{model_key}' for provider '{provider}' missing 'model_id' field"
                    )
                break

        # Fail if no model found
        if default_model is None:
            raise ValueError(
                f"No model found for provider '{provider}' in config/models.yaml.\n"
                f"Please add a model entry with 'api_provider: {provider}' and 'model_id: <model_name>'"
            )

        provider_config['default_model'] = default_model
        return provider_config

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
        if self.provider in ['deepseek', 'openai', 'google', 'xai', 'alibaba']:
            # OpenAI-compatible API
            if 'base_url' not in provider_config:
                raise ValueError(
                    f"Provider '{self.provider}' missing 'base_url' in config/models.yaml.\n"
                    f"OpenAI-compatible providers require a base_url."
                )

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
            raise ValueError(
                f"Provider '{self.provider}' not supported.\n"
                f"Supported providers: deepseek, openai, anthropic, google, xai, alibaba\n"
                f"Please check config/models.yaml for available providers."
            )

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
                # Get env_var from config for error message
                try:
                    provider_config = self._get_provider_config(self.provider)
                    env_var = provider_config.get('env_var', 'API_KEY')
                except Exception:
                    env_var = 'API_KEY'

                raise ValueError(
                    f"\n❌ API Authentication Failed for {self.provider}!\n\n"
                    f"Your API key appears to be invalid or expired.\n"
                    f"Please check {env_var} in your .env file.\n"
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

        Raises:
            FileNotFoundError: If system prompt file is missing

        Returns:
            System prompt string
        """
        # Load system prompt from config file
        from pathlib import Path
        prompt_file = Path(__file__).parent.parent.parent / "config" / "prompts" / "system_prompt.txt"

        if not prompt_file.exists():
            raise FileNotFoundError(
                f"System prompt file missing: {prompt_file}\n"
                f"This file is required for LLM agent to function.\n"
                f"Please create config/prompts/system_prompt.txt with the system prompt."
            )

        with open(prompt_file, 'r') as f:
            prompt = f.read().strip()

        if not prompt:
            raise ValueError(
                f"System prompt file is empty: {prompt_file}\n"
                f"Please add content to the system prompt file."
            )

        return prompt


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

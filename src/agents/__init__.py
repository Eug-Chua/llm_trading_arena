"""
LLM Trading Agents

Provides a unified interface for multiple LLM providers to generate trading decisions.
"""

from .base_agent import BaseLLMAgent, TradeSignal, AgentResponse
from .llm_agent import (
    LLMAgent,
    create_agent,
    create_deepseek_agent,
    create_openai_agent,
    create_claude_agent
)

__all__ = [
    # Base classes
    'BaseLLMAgent',
    'TradeSignal',
    'AgentResponse',

    # Main agent
    'LLMAgent',

    # Convenience functions
    'create_agent',
    'create_deepseek_agent',
    'create_openai_agent',
    'create_claude_agent',
]

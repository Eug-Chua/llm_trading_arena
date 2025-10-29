"""
Tests for LLM trading agents
"""

import pytest
import json
from src.agents import (
    LLMAgent,
    TradeSignal,
    AgentResponse,
    create_agent,
    create_deepseek_agent,
    create_openai_agent
)


def test_agent_initialization_deepseek():
    """Test DeepSeek agent initializes correctly"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    assert agent.provider == "deepseek"
    assert agent.model_id == "deepseek-chat"
    assert agent.temperature == 0.7
    assert agent.max_tokens == 4000
    assert "deepseek" in agent.model_name.lower()


def test_agent_initialization_openai():
    """Test OpenAI agent initializes correctly"""
    agent = LLMAgent(provider="openai", api_key="test_key")

    assert agent.provider == "openai"
    assert agent.model_id == "gpt-4-turbo"
    assert "Openai" in agent.model_name


def test_invalid_provider():
    """Test initialization with invalid provider"""
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMAgent(provider="invalid_provider", api_key="test_key")


def test_custom_model_id():
    """Test initialization with custom model ID"""
    agent = LLMAgent(provider="deepseek", model_id="custom-model", api_key="test_key")

    assert agent.model_id == "custom-model"


def test_convenience_function_deepseek():
    """Test create_deepseek_agent convenience function"""
    agent = create_deepseek_agent(api_key="test_key", temperature=0.5)

    assert isinstance(agent, LLMAgent)
    assert agent.provider == "deepseek"
    assert agent.temperature == 0.5


def test_convenience_function_openai():
    """Test create_openai_agent convenience function"""
    agent = create_openai_agent(model_id="gpt-4", api_key="test_key")

    assert isinstance(agent, LLMAgent)
    assert agent.provider == "openai"
    assert agent.model_id == "gpt-4"


def test_convenience_function_generic():
    """Test generic create_agent function"""
    agent = create_agent(provider="deepseek", api_key="test_key")

    assert isinstance(agent, LLMAgent)
    assert agent.provider == "deepseek"


def test_parse_trade_signal_valid():
    """Test parsing valid trade signal"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    data = {
        'coin': 'BTC',
        'signal': 'hold',
        'quantity': 0.5,
        'profit_target': 120000.0,
        'stop_loss': 108000.0,
        'invalidation_condition': 'If price closes below 105000 on 3-minute candle',
        'leverage': 10,
        'confidence': 0.75,
        'risk_usd': 1000.0
    }

    signal = agent._parse_trade_signal(data)

    assert signal is not None
    assert signal.coin == 'BTC'
    assert signal.signal == 'hold'
    assert signal.quantity == 0.5
    assert signal.leverage == 10
    assert signal.confidence == 0.75


def test_parse_trade_signal_invalid_signal():
    """Test parsing with invalid signal type"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    data = {
        'coin': 'BTC',
        'signal': 'invalid_signal',
        'quantity': 0.5,
        'profit_target': 120000.0,
        'stop_loss': 108000.0,
        'invalidation_condition': 'Some condition',
        'leverage': 10,
        'confidence': 0.75,
        'risk_usd': 1000.0
    }

    signal = agent._parse_trade_signal(data)
    assert signal is None


def test_parse_trade_signal_missing_field():
    """Test parsing with missing required field"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    data = {
        'coin': 'BTC',
        'signal': 'hold',
        # Missing quantity
        'profit_target': 120000.0,
        'stop_loss': 108000.0,
        'invalidation_condition': 'Some condition',
        'leverage': 10,
        'confidence': 0.75,
        'risk_usd': 1000.0
    }

    signal = agent._parse_trade_signal(data)
    assert signal is None


def test_extract_json_blocks():
    """Test extracting JSON blocks from text"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    text = """
    Some reasoning text here.

    {
      "BTC": {
        "signal": "hold"
      }
    }

    More text.

    {
      "ETH": {
        "signal": "buy"
      }
    }
    """

    blocks = agent._extract_json_blocks(text)

    assert len(blocks) == 2
    assert '"BTC"' in blocks[0]
    assert '"ETH"' in blocks[1]


def test_extract_json_blocks_nested():
    """Test extracting nested JSON"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    text = """
    {
      "BTC": {
        "trade_signal_args": {
          "coin": "BTC",
          "signal": "hold"
        }
      }
    }
    """

    blocks = agent._extract_json_blocks(text)

    assert len(blocks) == 1
    parsed = json.loads(blocks[0])
    assert 'BTC' in parsed
    assert 'trade_signal_args' in parsed['BTC']


def test_parse_response_with_valid_format():
    """Test parsing response with chain of thought and JSON"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    response = """
# CHAIN OF THOUGHT

Analyzing current positions:
- BTC: Price above stop-loss, holding
- ETH: Invalidation condition met, closing

# TRADING DECISIONS

{
  "BTC": {
    "trade_signal_args": {
      "coin": "BTC",
      "signal": "hold",
      "quantity": 0.5,
      "profit_target": 120000.0,
      "stop_loss": 108000.0,
      "invalidation_condition": "If price closes below 105000",
      "leverage": 10,
      "confidence": 0.75,
      "risk_usd": 1000.0
    }
  },
  "ETH": {
    "trade_signal_args": {
      "coin": "ETH",
      "signal": "close_position",
      "quantity": 5.0,
      "profit_target": 4500.0,
      "stop_loss": 3900.0,
      "invalidation_condition": "Price below 3850",
      "leverage": 10,
      "confidence": 0.65,
      "risk_usd": 500.0
    }
  }
}
"""

    cot, signals = agent._parse_response(response)

    # Check chain of thought
    assert len(cot) > 0
    assert 'Analyzing current positions' in cot

    # Check trade signals
    assert len(signals) == 2
    assert 'BTC' in signals
    assert 'ETH' in signals
    assert signals['BTC'].signal == 'hold'
    assert signals['ETH'].signal == 'close_position'


def test_parse_response_alternative_format():
    """Test parsing response with alternative JSON format"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    response = """
Some reasoning here.

{
  "BTC": {
    "coin": "BTC",
    "signal": "hold",
    "quantity": 0.5,
    "profit_target": 120000.0,
    "stop_loss": 108000.0,
    "invalidation_condition": "If price closes below 105000",
    "leverage": 10,
    "confidence": 0.75,
    "risk_usd": 1000.0
  }
}
"""

    cot, signals = agent._parse_response(response)

    assert len(signals) == 1
    assert 'BTC' in signals
    assert signals['BTC'].signal == 'hold'


def test_parse_response_no_json():
    """Test parsing response with no JSON"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")

    response = """
Just some text without any JSON blocks.
This should still extract chain of thought.
"""

    cot, signals = agent._parse_response(response)

    assert len(cot) > 0
    assert len(signals) == 0


def test_system_prompt():
    """Test system prompt generation"""
    agent = LLMAgent(provider="deepseek", api_key="test_key")
    system_prompt = agent._get_system_prompt()

    assert "cryptocurrency trader" in system_prompt.lower()
    assert "CHAIN OF THOUGHT" in system_prompt
    assert "TRADING DECISIONS" in system_prompt
    assert "stop-loss" in system_prompt.lower()  # Check for risk management rules


def test_agent_response_dataclass():
    """Test AgentResponse dataclass"""
    response = AgentResponse(
        chain_of_thought="Some reasoning",
        trade_signals={'BTC': TradeSignal(
            coin='BTC',
            signal='hold',
            quantity=0.5,
            profit_target=120000.0,
            stop_loss=108000.0,
            invalidation_condition='Price below 105000',
            leverage=10,
            confidence=0.75,
            risk_usd=1000.0
        )},
        raw_response="Raw text",
        success=True,
        model_name="DeepSeek (deepseek-chat)"
    )

    assert response.success is True
    assert response.error is None
    assert len(response.trade_signals) == 1
    assert 'BTC' in response.trade_signals
    assert response.model_name == "DeepSeek (deepseek-chat)"


def test_trade_signal_dataclass():
    """Test TradeSignal dataclass"""
    signal = TradeSignal(
        coin='ETH',
        signal='buy',
        quantity=5.0,
        profit_target=4500.0,
        stop_loss=3900.0,
        invalidation_condition='Price below 3850',
        leverage=15,
        confidence=0.8,
        risk_usd=500.0
    )

    assert signal.coin == 'ETH'
    assert signal.signal == 'buy'
    assert signal.leverage == 15
    assert signal.confidence == 0.8


def test_provider_configs():
    """Test that provider configurations are defined"""
    assert 'deepseek' in LLMAgent.PROVIDERS
    assert 'openai' in LLMAgent.PROVIDERS
    assert 'anthropic' in LLMAgent.PROVIDERS

    # Check DeepSeek config
    deepseek_config = LLMAgent.PROVIDERS['deepseek']
    assert deepseek_config['base_url'] == 'https://api.deepseek.com'
    assert deepseek_config['env_var'] == 'DEEPSEEK_API_KEY'
    assert deepseek_config['default_model'] == 'deepseek-chat'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

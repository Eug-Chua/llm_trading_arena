"""
Unit tests for TradingOrchestrator base class

Tests the common orchestration logic shared between BacktestEngine and ContinuousLoop.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Tuple

# We'll import this once we create it
# from src.core.trading_orchestrator import TradingOrchestrator


@pytest.fixture
def mock_llm_agent():
    """Mock LLM agent"""
    agent = Mock()
    agent.model_name = "test-model"
    agent.generate_decision = Mock()
    return agent


@pytest.fixture
def mock_trading_engine():
    """Mock trading engine with account"""
    engine = Mock()

    # Mock account
    engine.account = Mock()
    engine.account.total_return_percent = 5.0
    engine.account.available_cash = 9500.0
    engine.account.account_value = 10500.0
    engine.account.sharpe_ratio = 1.2
    engine.account.positions = {}
    engine.account.update_prices = Mock()

    # Mock methods
    engine.check_exit_conditions = Mock(return_value=[])
    engine.execute_signals = Mock(return_value={'executed': 0})

    return engine


@pytest.fixture
def mock_prompt_gen():
    """Mock prompt generator"""
    gen = Mock()
    gen.generate_prompt = Mock(return_value="test prompt")
    return gen


@pytest.fixture
def mock_indicators():
    """Mock technical indicators"""
    indicators = Mock()
    return indicators


@pytest.fixture
def concrete_orchestrator(mock_llm_agent, mock_trading_engine, mock_prompt_gen, mock_indicators):
    """Create a concrete implementation for testing"""
    from src.core.trading_orchestrator import TradingOrchestrator

    class TestOrchestrator(TradingOrchestrator):
        """Concrete implementation for testing"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fetch_called = False
            self.track_called = False

        def _fetch_market_data(self, timestamp: datetime) -> Tuple[Dict, Dict]:
            """Mock implementation"""
            self.fetch_called = True
            market_data = {
                'BTC': Mock(current_price=50000.0),
                'ETH': Mock(current_price=3000.0)
            }
            current_prices = {'BTC': 50000.0, 'ETH': 3000.0}
            return market_data, current_prices

        def _track_results(self, results, response, market_data, timestamp):
            """Mock implementation"""
            self.track_called = True

    return TestOrchestrator(
        mock_llm_agent,
        mock_trading_engine,
        mock_prompt_gen,
        mock_indicators
    )


@pytest.mark.unit
class TestTradingOrchestrator:
    """Test TradingOrchestrator base class"""

    def test_initialization(self, concrete_orchestrator, mock_llm_agent, mock_trading_engine):
        """Test orchestrator initializes with dependencies"""
        assert concrete_orchestrator.llm_agent == mock_llm_agent
        assert concrete_orchestrator.engine == mock_trading_engine
        assert concrete_orchestrator.prompt_gen is not None
        assert concrete_orchestrator.indicators is not None

    def test_process_iteration_full_flow(self, concrete_orchestrator, mock_trading_engine, mock_llm_agent):
        """Test complete iteration flow"""
        # Setup
        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        mock_response = Mock()
        mock_response.trade_signals = {'BTC': Mock()}
        mock_llm_agent.generate_decision.return_value = mock_response

        # Execute
        result = concrete_orchestrator.process_iteration(timestamp)

        # Verify flow
        assert concrete_orchestrator.fetch_called, "Should call _fetch_market_data"
        assert mock_trading_engine.account.update_prices.called, "Should update prices"
        assert mock_trading_engine.check_exit_conditions.called, "Should check exits"
        assert mock_llm_agent.generate_decision.called, "Should call LLM"
        assert mock_trading_engine.execute_signals.called, "Should execute trades"
        assert concrete_orchestrator.track_called, "Should track results"

        # Verify result structure
        assert 'market_data' in result
        assert 'trades' in result
        assert 'exits' in result
        assert 'response' in result

    def test_process_iteration_no_market_data(self, concrete_orchestrator, mock_trading_engine, mock_llm_agent):
        """Test iteration when no market data available"""
        # Override fetch to return empty data
        concrete_orchestrator._fetch_market_data = Mock(return_value=({}, {}))

        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        result = concrete_orchestrator.process_iteration(timestamp)

        # Should return early without calling LLM
        assert result == {}
        assert not mock_llm_agent.generate_decision.called

    def test_process_iteration_llm_failure(self, concrete_orchestrator, mock_trading_engine, mock_llm_agent):
        """Test iteration when LLM call fails"""
        # Make LLM raise exception
        mock_llm_agent.generate_decision.side_effect = Exception("API error")

        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        result = concrete_orchestrator.process_iteration(timestamp)

        # Should handle gracefully and return empty result
        assert result == {}
        assert not mock_trading_engine.execute_signals.called

    def test_process_iteration_no_trade_signals(self, concrete_orchestrator, mock_trading_engine, mock_llm_agent):
        """Test iteration when LLM returns no trade signals"""
        # LLM returns response but no signals
        mock_response = Mock()
        mock_response.trade_signals = {}
        mock_llm_agent.generate_decision.return_value = mock_response

        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        result = concrete_orchestrator.process_iteration(timestamp)

        # Should not execute trades but still return data
        assert not mock_trading_engine.execute_signals.called
        assert 'market_data' in result
        assert 'response' in result
        assert result['trades'] == {}

    def test_build_account_info(self, concrete_orchestrator, mock_trading_engine):
        """Test account info building"""
        # Add mock position (matching actual Position structure)
        mock_position = Mock()
        mock_position.symbol = "BTC"
        mock_position.quantity = 0.5
        mock_position.entry_price = 50000.0
        mock_position.current_price = 51000.0
        mock_position.liquidation_price = 40000.0
        mock_position.unrealized_pnl = 500.0
        mock_position.leverage = 10
        mock_position.exit_plan = {}
        mock_position.confidence = 0.8
        mock_position.risk_usd = 100.0
        mock_position.notional_usd = 5000.0
        mock_position.sl_oid = -1
        mock_position.tp_oid = -1
        mock_position.wait_for_fill = False
        mock_position.entry_oid = 123

        mock_trading_engine.account.positions = {'BTC': mock_position}

        account_info = concrete_orchestrator._build_account_info()

        assert account_info.total_return_percent == 5.0
        assert account_info.available_cash == 9500.0
        assert account_info.account_value == 10500.0
        assert account_info.sharpe_ratio == 1.2
        assert len(account_info.positions) == 1
        assert account_info.positions[0].symbol == "BTC"

    def test_log_exits(self, concrete_orchestrator, caplog):
        """Test exit logging"""
        import logging

        # Ensure logger is configured for this test
        caplog.set_level(logging.INFO, logger='src.core.trading_orchestrator')

        exits = [
            {
                'symbol': 'BTC',
                'exit_price': 51000.0,
                'reason': 'profit_target'
            },
            {
                'symbol': 'ETH',
                'exit_price': 2900.0,
                'reason': 'stop_loss'
            }
        ]

        concrete_orchestrator._log_exits(exits)

        # Check log output (note: prices are formatted with commas)
        assert 'BTC' in caplog.text
        assert '51,000' in caplog.text  # Formatted with comma
        assert 'profit_target' in caplog.text

    def test_generate_prompt_not_final(self, concrete_orchestrator, mock_prompt_gen):
        """Test prompt generation for non-final timestamp"""
        market_data = {'BTC': Mock()}
        account_info = Mock()
        timestamp = datetime(2025, 11, 1, 12, 0, 0)

        # Override _is_final_timestamp to return False
        concrete_orchestrator._is_final_timestamp = Mock(return_value=False)

        prompt = concrete_orchestrator._generate_prompt(market_data, account_info, timestamp)

        mock_prompt_gen.generate_prompt.assert_called_once()
        call_kwargs = mock_prompt_gen.generate_prompt.call_args.kwargs
        assert call_kwargs['is_final_candle'] == False

    def test_generate_prompt_final(self, concrete_orchestrator, mock_prompt_gen):
        """Test prompt generation for final timestamp"""
        market_data = {'BTC': Mock()}
        account_info = Mock()
        timestamp = datetime(2025, 11, 1, 12, 0, 0)

        # Override _is_final_timestamp to return True
        concrete_orchestrator._is_final_timestamp = Mock(return_value=True)

        prompt = concrete_orchestrator._generate_prompt(market_data, account_info, timestamp)

        call_kwargs = mock_prompt_gen.generate_prompt.call_args.kwargs
        assert call_kwargs['is_final_candle'] == True

    def test_position_to_prompt(self, concrete_orchestrator):
        """Test position conversion to prompt format"""
        mock_position = Mock()
        mock_position.symbol = "BTC"
        mock_position.quantity = 0.5
        mock_position.entry_price = 50000.0
        mock_position.current_price = 51000.0
        mock_position.liquidation_price = 40000.0
        mock_position.unrealized_pnl = 500.0
        mock_position.leverage = 10
        mock_position.exit_plan = {}
        mock_position.confidence = 0.8
        mock_position.risk_usd = 100.0
        mock_position.notional_usd = 5000.0
        mock_position.sl_oid = -1
        mock_position.tp_oid = -1
        mock_position.wait_for_fill = False
        mock_position.entry_oid = 123

        prompt_position = concrete_orchestrator._position_to_prompt(mock_position)

        assert prompt_position.symbol == "BTC"
        assert prompt_position.entry_price == 50000.0
        assert prompt_position.quantity == 0.5

    def test_is_final_timestamp_default(self, concrete_orchestrator):
        """Test default implementation returns False"""
        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        assert concrete_orchestrator._is_final_timestamp(timestamp) == False


@pytest.mark.unit
class TestOrchestratorAbstractMethods:
    """Test that abstract methods must be implemented"""

    def test_cannot_instantiate_base_class(self, mock_llm_agent, mock_trading_engine, mock_prompt_gen, mock_indicators):
        """Test that TradingOrchestrator cannot be instantiated directly"""
        from src.core.trading_orchestrator import TradingOrchestrator

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TradingOrchestrator(
                mock_llm_agent,
                mock_trading_engine,
                mock_prompt_gen,
                mock_indicators
            )

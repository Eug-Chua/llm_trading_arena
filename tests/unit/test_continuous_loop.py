"""
Unit tests for ContinuousEvaluationLoop
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from pathlib import Path


@pytest.fixture
def mock_pipeline():
    """Mock market data pipeline"""
    pipeline = Mock()
    pipeline.fetch_and_process = Mock(return_value={
        'BTC': {'current_price': 50000.0, 'funding_rate': 0.0001}
    })
    pipeline.get_current_prices = Mock(return_value={'BTC': 50000.0})
    return pipeline


@pytest.fixture
def mock_llm_agent():
    """Mock LLM agent"""
    agent = Mock()
    agent.model_name = "test-model"
    agent.generate_decision = Mock()
    return agent


@pytest.mark.unit
class TestContinuousLoopInit:
    """Test ContinuousLoop initialization"""

    @patch('src.trading.continuous_loop.MarketDataPipeline')
    @patch('src.trading.continuous_loop.TradingEngine')
    @patch('src.trading.continuous_loop.AlphaArenaPrompt')
    def test_init_fresh_start(self, mock_prompt, mock_engine, mock_pipeline_cls, mock_llm_agent):
        """Test initialization without checkpoint"""
        from src.trading.continuous_loop import ContinuousEvaluationLoop

        loop = ContinuousEvaluationLoop(
            llm_agent=mock_llm_agent,
            starting_capital=10000.0,
            interval_seconds=180,
            coins=['BTC', 'ETH']
        )

        assert loop.llm_agent == mock_llm_agent
        assert loop.interval_seconds == 180
        assert loop.coins == ['BTC', 'ETH']
        assert loop.iteration_count == 0
        assert loop.engine is not None
        assert loop.pipeline is not None
        assert loop.prompt_gen is not None


@pytest.mark.unit
class TestContinuousLoopFetchData:
    """Test _fetch_market_data implementation"""

    @patch('src.trading.continuous_loop.MarketDataPipeline')
    @patch('src.trading.continuous_loop.TradingEngine')
    @patch('src.trading.continuous_loop.AlphaArenaPrompt')
    def test_fetch_market_data_success(self, mock_prompt, mock_engine, mock_pipeline_cls, mock_llm_agent):
        """Test successful market data fetching"""
        from src.trading.continuous_loop import ContinuousEvaluationLoop

        # Setup mock pipeline with correct field names
        mock_pipeline = Mock()
        mock_pipeline.fetch_and_process = Mock(return_value={
            'BTC': {
                'current_price': 50000.0,
                'funding_rate': 0.0001,
                'current_ema20': 49000.0,
                'current_macd': 100.0,
                'current_rsi_7': 70.0,
                'prices_3m': [50000.0] * 10,
                'ema_20_3m': [49000.0] * 10,
                'macd_3m': [100.0] * 10,
                'rsi_7_3m': [70.0] * 10,
                'rsi_14_3m': [65.0] * 10,
                'ema_20_4h': 48000.0,
                'ema_50_4h': 47000.0,
                'atr_3_4h': 500.0,
                'atr_14_4h': 600.0,
                'volume': 1000.0,
                'volume_avg': 900.0,
                'macd_4h': [100.0] * 10,
                'rsi_14_4h': [65.0] * 10
            }
        })
        mock_pipeline.get_current_prices = Mock(return_value={'BTC': 50000.0})
        mock_pipeline_cls.return_value = mock_pipeline

        loop = ContinuousEvaluationLoop(
            llm_agent=mock_llm_agent,
            starting_capital=10000.0,
            coins=['BTC']
        )

        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        market_data, current_prices = loop._fetch_market_data(timestamp)

        assert 'BTC' in market_data
        assert 'BTC' in current_prices
        assert current_prices['BTC'] == 50000.0
        assert market_data['BTC'].coin == 'BTC'
        assert market_data['BTC'].current_price == 50000.0


@pytest.mark.unit
class TestContinuousLoopTracking:
    """Test _track_results implementation"""

    @patch('src.trading.continuous_loop.MarketDataPipeline')
    @patch('src.trading.continuous_loop.TradingEngine')
    @patch('src.trading.continuous_loop.AlphaArenaPrompt')
    def test_track_results_writes_jsonl(self, mock_prompt, mock_engine, mock_pipeline, mock_llm_agent, tmp_path):
        """Test JSONL logging"""
        from src.trading.continuous_loop import ContinuousEvaluationLoop

        loop = ContinuousEvaluationLoop(
            llm_agent=mock_llm_agent,
            starting_capital=10000.0,
            log_dir=tmp_path
        )

        # Prepare test data
        results = {'executed': 1}
        response = Mock()
        response.chain_of_thought = "Test reasoning"
        response.trade_signals = {}
        market_data = {'BTC': Mock(current_price=50000.0)}
        timestamp = datetime(2025, 11, 1, 12, 0, 0)

        # Mock engine performance
        mock_engine_instance = Mock()
        mock_engine_instance.get_performance_summary = Mock(return_value={
            'account_value': 10500.0,
            'total_return_percent': 5.0
        })
        loop.engine = mock_engine_instance

        loop._track_results(results, response, market_data, timestamp)

        # Verify JSONL file was created and written
        assert loop.session_log_file.exists()
        content = loop.session_log_file.read_text()
        assert len(content) > 0

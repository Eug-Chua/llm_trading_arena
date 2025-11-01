"""
Unit tests for BacktestEngine

Tests the backtest-specific implementation of TradingOrchestrator.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import hashlib

# Import after we refactor
# from src.backtesting.backtest_engine import BacktestEngine


@pytest.fixture
def mock_data_loader():
    """Mock historical data loader"""
    loader = Mock()
    loader.get_timestamps = Mock(return_value=[
        datetime(2025, 11, 1, 12, 0, 0),
        datetime(2025, 11, 1, 12, 3, 0),
        datetime(2025, 11, 1, 12, 6, 0),
    ])
    loader.get_all_candles_at_time = Mock(return_value={
        'BTC': {
            '3m': Mock(
                iloc=[-1],  # For accessing last row
                close=[50000.0],
                volume=[1000.0]
            ),
            '4h': Mock(
                iloc=[-1],
                close=[50000.0],
                volume=[10000.0]
            )
        }
    })
    return loader


@pytest.fixture
def mock_checkpoint_manager():
    """Mock checkpoint manager"""
    manager = Mock()
    manager.save_checkpoint = Mock()
    manager.save_metadata_json = Mock()
    manager.load_checkpoint = Mock(return_value={
        'account': {
            'starting_capital': 10000.0,
            'available_cash': 9500.0,
            'total_return_percent': 5.0,
            'sharpe_ratio': 1.2,
            'trade_count': 3,
            'total_fees_paid': 10.0,
            'total_funding_paid': 5.0
        },
        'positions': [],
        'trade_history': [],
        'llm_cache': {},
        'metadata': {
            'total_iterations': 100,
            'model': 'claude'
        },
        'checkpoint_date': '2025-11-01T12:00:00'
    })
    return manager


@pytest.fixture
def mock_indicators():
    """Mock technical indicators"""
    indicators = Mock()

    # Mock dataframe that gets returned
    mock_df = Mock()
    mock_df.iloc = Mock(return_value={'close': 50000.0})
    mock_df.__len__ = Mock(return_value=100)
    mock_df.tail = Mock(return_value=Mock(tolist=Mock(return_value=[1.0] * 10)))
    mock_df.mean = Mock(return_value=50000.0)

    # All indicator methods return the dataframe with new columns
    indicators.calculate_ema = Mock(return_value=mock_df)
    indicators.calculate_macd = Mock(return_value=mock_df)
    indicators.calculate_rsi = Mock(return_value=mock_df)
    indicators.calculate_atr = Mock(return_value=mock_df)

    return indicators


@pytest.fixture
def backtest_engine_setup(mock_data_loader, mock_checkpoint_manager, mock_indicators):
    """Setup for creating BacktestEngine instances"""
    return {
        'data_loader': mock_data_loader,
        'checkpoint_manager': mock_checkpoint_manager,
        'indicators': mock_indicators
    }


@pytest.mark.unit
class TestBacktestEngineInit:
    """Test BacktestEngine initialization"""

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_init_fresh_start(self, mock_indicators, mock_prompt, mock_engine, mock_llm, mock_checkpoint, mock_data_loader):
        """Test initialization without checkpoint"""
        from src.backtesting.backtest_engine import BacktestEngine

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC', 'ETH'],
            model='claude',
            starting_capital=10000.0,
            use_llm_cache=True
        )

        # Verify components initialized
        assert engine.start_date == '2025-01-01'
        assert engine.end_date == '2025-01-31'
        assert engine.coins == ['BTC', 'ETH']
        assert engine.model == 'claude'
        assert engine.use_llm_cache == True
        assert engine.iteration == 0
        assert isinstance(engine.llm_cache, dict)

        # Verify parent class components
        assert engine.llm_agent is not None
        assert engine.engine is not None
        assert engine.prompt_gen is not None
        assert engine.indicators is not None

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_init_from_checkpoint(self, mock_indicators, mock_prompt, mock_engine_cls, mock_llm, mock_checkpoint_cls, mock_data_loader):
        """Test initialization with checkpoint resume"""
        from src.backtesting.backtest_engine import BacktestEngine

        # Setup mock trading engine with account value
        mock_engine_instance = Mock()
        mock_engine_instance.account = Mock()
        mock_engine_instance.account.account_value = 10500.0  # Return float, not Mock
        mock_engine_instance.account.positions = {}
        mock_engine_instance.account.trade_log = []
        mock_engine_cls.return_value = mock_engine_instance

        # Setup checkpoint manager mock
        mock_checkpoint_instance = Mock()
        mock_checkpoint_instance.load_checkpoint = Mock(return_value={
            'account': {
                'starting_capital': 10000.0,
                'available_cash': 9500.0,
                'total_return_percent': 5.0,
                'sharpe_ratio': 1.2,
                'trade_count': 3,
                'total_fees_paid': 10.0,
                'total_funding_paid': 5.0
            },
            'positions': [],
            'trade_history': [],
            'llm_cache': {'test_hash': {'response': 'test'}},
            'metadata': {
                'total_iterations': 100,
                'model': 'claude'
            },
            'checkpoint_date': '2025-11-01T12:00:00'
        })
        mock_checkpoint_cls.return_value = mock_checkpoint_instance

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC', 'ETH'],
            model='claude',
            checkpoint_path='test_checkpoint.pkl',
            use_llm_cache=True
        )

        # Verify checkpoint loaded
        assert engine.iteration == 100
        assert len(engine.llm_cache) == 1
        mock_checkpoint_instance.load_checkpoint.assert_called_once()


@pytest.mark.unit
class TestBacktestEngineFetchData:
    """Test _fetch_market_data implementation"""

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_fetch_market_data_success(self, mock_indicators_cls, mock_prompt, mock_engine, mock_llm, mock_checkpoint, mock_data_loader_cls):
        """Test successful market data fetching"""
        from src.backtesting.backtest_engine import BacktestEngine
        import pandas as pd

        # Use real pandas DataFrame for easier mocking
        mock_df = pd.DataFrame({
            'close': [50000.0] * 100,
            'volume': [1000.0] * 100,
            'ema_20': [50000.0] * 100,
            'ema_50': [50000.0] * 100,
            'macd': [100.0] * 100,
            'rsi_7': [70.0] * 100,
            'rsi_14': [65.0] * 100,
            'atr_3': [500.0] * 100,
            'atr_14': [600.0] * 100,
        })

        mock_data_loader = Mock()
        mock_data_loader.get_all_candles_at_time = Mock(return_value={
            'BTC': {
                '3m': mock_df,
                '4h': mock_df
            }
        })
        mock_data_loader_cls.return_value = mock_data_loader

        mock_indicators = Mock()
        mock_indicators.calculate_ema = Mock(side_effect=lambda df, periods: df)
        mock_indicators.calculate_macd = Mock(side_effect=lambda df: df)
        mock_indicators.calculate_rsi = Mock(side_effect=lambda df, periods: df)
        mock_indicators.calculate_atr = Mock(side_effect=lambda df, periods: df)
        mock_indicators_cls.return_value = mock_indicators

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC'],
            model='claude'
        )

        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        market_data, current_prices = engine._fetch_market_data(timestamp)

        # Verify data returned
        assert 'BTC' in market_data
        assert 'BTC' in current_prices
        assert current_prices['BTC'] == 50000.0
        assert market_data['BTC'].coin == 'BTC'
        assert market_data['BTC'].current_price == 50000.0

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_fetch_market_data_no_candles(self, mock_indicators, mock_prompt, mock_engine, mock_llm, mock_checkpoint, mock_data_loader_cls):
        """Test market data fetch with no candles available"""
        from src.backtesting.backtest_engine import BacktestEngine

        mock_data_loader = Mock()
        mock_data_loader.get_all_candles_at_time = Mock(return_value={})
        mock_data_loader_cls.return_value = mock_data_loader

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC'],
            model='claude'
        )

        timestamp = datetime(2025, 11, 1, 12, 0, 0)
        market_data, current_prices = engine._fetch_market_data(timestamp)

        # Should return empty dicts
        assert market_data == {}
        assert current_prices == {}


@pytest.mark.unit
class TestBacktestEngineLLMCaching:
    """Test LLM caching functionality"""

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_llm_cache_enabled(self, mock_indicators, mock_prompt, mock_engine, mock_llm_cls, mock_checkpoint, mock_data_loader):
        """Test LLM caching when enabled"""
        from src.backtesting.backtest_engine import BacktestEngine

        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.trade_signals = {}
        mock_llm_instance.generate_decision = Mock(return_value=mock_response)
        mock_llm_cls.return_value = mock_llm_instance

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC'],
            model='claude',
            use_llm_cache=True
        )

        prompt = "test prompt"
        timestamp = datetime(2025, 11, 1, 12, 0, 0)

        # First call - should hit API
        response1 = engine._get_llm_decision(prompt, timestamp)
        assert mock_llm_instance.generate_decision.call_count == 1

        # Second call with same prompt - should use cache
        response2 = engine._get_llm_decision(prompt, timestamp)
        assert mock_llm_instance.generate_decision.call_count == 1  # No additional call
        assert response1 is response2  # Same object from cache

        # Verify cache populated
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        assert prompt_hash in engine.llm_cache

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_llm_cache_disabled(self, mock_indicators, mock_prompt, mock_engine, mock_llm_cls, mock_checkpoint, mock_data_loader):
        """Test LLM behavior when caching disabled"""
        from src.backtesting.backtest_engine import BacktestEngine

        # Setup mock LLM
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_llm_instance.generate_decision = Mock(return_value=mock_response)
        mock_llm_cls.return_value = mock_llm_instance

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC'],
            model='claude',
            use_llm_cache=False
        )

        prompt = "test prompt"
        timestamp = datetime(2025, 11, 1, 12, 0, 0)

        # Two calls with same prompt - both should hit API
        response1 = engine._get_llm_decision(prompt, timestamp)
        response2 = engine._get_llm_decision(prompt, timestamp)

        assert mock_llm_instance.generate_decision.call_count == 2  # Both hit API


@pytest.mark.unit
class TestBacktestEngineFinalTimestamp:
    """Test final timestamp detection"""

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_is_final_timestamp(self, mock_indicators, mock_prompt, mock_engine, mock_llm, mock_checkpoint, mock_data_loader_cls):
        """Test final timestamp detection"""
        from src.backtesting.backtest_engine import BacktestEngine

        mock_data_loader = Mock()
        mock_data_loader.get_timestamps = Mock(return_value=[
            datetime(2025, 11, 1, 12, 0, 0),
            datetime(2025, 11, 1, 12, 3, 0),
            datetime(2025, 11, 1, 12, 6, 0),
        ])
        mock_data_loader_cls.return_value = mock_data_loader

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC'],
            model='claude'
        )

        # Set timestamps
        engine.timestamps = [
            datetime(2025, 11, 1, 12, 0, 0),
            datetime(2025, 11, 1, 12, 3, 0),
            datetime(2025, 11, 1, 12, 6, 0),
        ]

        # Test non-final timestamps
        engine.current_idx = 0
        assert engine._is_final_timestamp(engine.timestamps[0]) == False

        engine.current_idx = 1
        assert engine._is_final_timestamp(engine.timestamps[1]) == False

        # Test final timestamp
        engine.current_idx = 2
        assert engine._is_final_timestamp(engine.timestamps[2]) == True


@pytest.mark.unit
class TestBacktestEngineRun:
    """Test backtest run method"""

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_run_backtest(self, mock_indicators, mock_prompt, mock_engine_cls, mock_llm, mock_checkpoint, mock_data_loader_cls):
        """Test running a backtest"""
        from src.backtesting.backtest_engine import BacktestEngine

        # Setup mock data loader
        mock_data_loader = Mock()
        mock_data_loader.get_timestamps = Mock(return_value=[
            datetime(2025, 11, 1, 12, 0, 0),
            datetime(2025, 11, 1, 12, 3, 0),
        ])
        mock_data_loader.get_all_candles_at_time = Mock(return_value={})  # Return empty to skip processing
        mock_data_loader_cls.return_value = mock_data_loader

        # Setup mock engine
        mock_engine_instance = Mock()
        mock_engine_instance.account = Mock()
        mock_engine_instance.account.account_value = 10500.0
        mock_engine_instance.get_detailed_performance = Mock(return_value={
            'current_value': 10500.0,
            'total_return_pct': 5.0,
            'sharpe_ratio': 1.2,
            'total_trades': 3
        })
        mock_engine_cls.return_value = mock_engine_instance

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC'],
            model='claude'
        )

        results = engine.run()

        # Verify results structure
        assert 'current_value' in results
        assert 'total_return_pct' in results
        assert 'model' in results
        assert 'iterations' in results
        assert results['model'] == 'claude'
        assert results['iterations'] == 2  # Two timestamps


@pytest.mark.unit
class TestBacktestEngineCheckpoint:
    """Test checkpoint save/load functionality"""

    @patch('src.backtesting.backtest_engine.HistoricalDataLoader')
    @patch('src.backtesting.backtest_engine.CheckpointManager')
    @patch('src.backtesting.backtest_engine.LLMAgent')
    @patch('src.backtesting.backtest_engine.TradingEngine')
    @patch('src.backtesting.backtest_engine.AlphaArenaPrompt')
    @patch('src.backtesting.backtest_engine.TechnicalIndicators')
    def test_save_checkpoint(self, mock_indicators, mock_prompt, mock_engine_cls, mock_llm, mock_checkpoint_cls, mock_data_loader):
        """Test checkpoint saving"""
        from src.backtesting.backtest_engine import BacktestEngine

        # Setup mock checkpoint manager
        mock_checkpoint = Mock()
        mock_checkpoint.save_checkpoint = Mock()
        mock_checkpoint.save_metadata_json = Mock()
        mock_checkpoint_cls.return_value = mock_checkpoint

        # Setup mock engine with positions
        mock_engine_instance = Mock()
        mock_engine_instance.account = Mock()
        mock_engine_instance.account.starting_capital = 10000.0
        mock_engine_instance.account.available_cash = 9500.0
        mock_engine_instance.account.account_value = 10500.0
        mock_engine_instance.account.total_return_percent = 5.0
        mock_engine_instance.account.sharpe_ratio = 1.2
        mock_engine_instance.account.trade_count = 3
        mock_engine_instance.account.total_fees_paid = 10.0
        mock_engine_instance.account.total_funding_paid = 5.0
        mock_engine_instance.account.positions = {}
        mock_engine_instance.account.trade_log = []
        mock_engine_cls.return_value = mock_engine_instance

        engine = BacktestEngine(
            start_date='2025-01-01',
            end_date='2025-01-31',
            coins=['BTC'],
            model='claude'
        )

        checkpoint_date = datetime(2025, 11, 1, 12, 0, 0)
        engine.save_checkpoint('test_checkpoint.pkl', checkpoint_date)

        # Verify checkpoint manager called
        assert mock_checkpoint.save_checkpoint.called
        assert mock_checkpoint.save_metadata_json.called

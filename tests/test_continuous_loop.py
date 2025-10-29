"""
Tests for Continuous Evaluation Loop
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from src.trading.continuous_loop import ContinuousEvaluationLoop
from src.agents.llm_agent import LLMAgent


class TestContinuousLoop:
    """Test continuous evaluation loop"""

    def test_initialization(self):
        """Test loop initializes correctly"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=180,
                coins=['BTC', 'ETH'],
                log_dir=Path(tmpdir)
            )

            assert loop.iteration_count == 0
            assert loop.start_time is not None
            assert loop.coins == ['BTC', 'ETH']
            assert loop.interval_seconds == 180

    def test_single_iteration(self):
        """Test running a single iteration"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,  # 1 second for fast test
                coins=['BTC'],  # Just BTC for faster test
                log_dir=Path(tmpdir)
            )

            # Run one iteration
            loop._run_iteration()

            assert loop.iteration_count == 1
            assert loop.last_evaluation_time is not None

    def test_multiple_iterations(self):
        """Test running multiple iterations"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC'],
                log_dir=Path(tmpdir)
            )

            # Run 3 iterations
            loop.run(max_iterations=3)

            assert loop.iteration_count == 3

    def test_max_duration_stopping(self):
        """Test loop stops after max duration"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC'],
                log_dir=Path(tmpdir)
            )

            # Run for 0.001 hours (3.6 seconds)
            # Should complete 2-3 iterations
            loop.run(max_duration_hours=0.001)

            assert loop.iteration_count >= 1
            assert loop.iteration_count <= 5

    def test_log_file_creation(self):
        """Test that log files are created"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC'],
                log_dir=log_dir
            )

            # Run one iteration
            loop.run(max_iterations=1)

            # Check log file exists
            assert loop.session_log_file.exists()

            # Check log file has content
            with open(loop.session_log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 1

                # Verify JSON format
                first_entry = json.loads(lines[0])
                assert 'iteration' in first_entry
                assert 'timestamp' in first_entry
                assert 'account' in first_entry

    def test_account_info_conversion(self):
        """Test converting engine state to AccountInfo"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC'],
                log_dir=Path(tmpdir)
            )

            # Get account info
            account_info = loop._get_account_info()

            assert account_info.available_cash == 10000.0
            assert account_info.account_value == 10000.0
            assert account_info.total_return_percent == 0.0
            assert len(account_info.positions) == 0

    def test_market_data_conversion(self):
        """Test converting pipeline data to MarketData objects"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC'],
                log_dir=Path(tmpdir)
            )

            # Create sample pipeline data
            raw_data = {
                'BTC': {
                    'current_price': 100000.0,
                    'current_ema20': 99500.0,
                    'current_macd': 50.0,
                    'current_rsi_7': 65.0,
                    'open_interest': 1000000.0,
                    'funding_rate': 0.0001,
                    'prices_3m': [100000.0] * 10,
                    'ema_20_3m': [99500.0] * 10,
                    'macd_3m': [50.0] * 10,
                    'rsi_7_3m': [65.0] * 10,
                    'rsi_14_3m': [60.0] * 10,
                    'ema_20_4h': 99000.0,
                    'ema_50_4h': 98000.0,
                    'atr_3_4h': 1000.0,
                    'atr_14_4h': 1200.0,
                    'volume': 5000000.0,
                    'volume_avg': 4800000.0,
                    'macd_4h': [50.0] * 10,
                    'rsi_14_4h': [60.0] * 10
                }
            }

            # Convert
            market_data = loop._convert_to_market_data(raw_data)

            assert 'BTC' in market_data
            assert market_data['BTC'].current_price == 100000.0
            assert market_data['BTC'].current_ema20 == 99500.0
            assert market_data['BTC'].oi_latest == 1000000.0
            assert market_data['BTC'].funding_rate == 0.0001

    def test_format_prices(self):
        """Test price formatting"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC'],
                log_dir=Path(tmpdir)
            )

            prices = {'BTC': 100000.5, 'ETH': 4000.25}
            formatted = loop._format_prices(prices)

            assert 'BTC=$100,000' in formatted
            assert 'ETH=$4,000' in formatted

    def test_error_handling_in_iteration(self):
        """Test that errors in iteration don't crash the loop"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['INVALID_COIN'],  # This will cause errors
                log_dir=Path(tmpdir)
            )

            # Should handle error gracefully and not crash
            try:
                loop._run_iteration()
                # Iteration count should still increment even with errors
                assert loop.iteration_count == 1
            except Exception as e:
                pytest.fail(f"Iteration should not crash: {e}")


class TestContinuousLoopIntegration:
    """Integration tests with real data"""

    @pytest.mark.slow
    def test_real_data_iteration(self):
        """Test iteration with real Hyperliquid data"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC'],  # Just BTC to keep it fast
                log_dir=Path(tmpdir)
            )

            # Run one iteration with real data
            loop._run_iteration()

            # Should complete successfully
            assert loop.iteration_count == 1
            assert loop.engine.account.account_value > 0

    @pytest.mark.slow
    def test_full_session(self):
        """Test a full session with multiple iterations"""
        llm_agent = LLMAgent(provider="openai", model_id="gpt-4")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = ContinuousEvaluationLoop(
                llm_agent=llm_agent,
                starting_capital=10000.0,
                interval_seconds=1,
                coins=['BTC', 'ETH'],
                log_dir=Path(tmpdir)
            )

            # Run 5 iterations
            loop.run(max_iterations=5)

            # Verify
            assert loop.iteration_count == 5
            assert loop.session_log_file.exists()

            # Verify log entries
            with open(loop.session_log_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 5  # One entry per iteration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

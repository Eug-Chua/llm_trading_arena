"""
Integration test for BacktestEngine

Verifies the refactored BacktestEngine works end-to-end.
"""

import pytest
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_backtest_runs_successfully():
    """Test that backtest runs without errors"""
    from src.backtesting.backtest_engine import BacktestEngine

    # Create a minimal backtest (just 1 day, 1 coin)
    # Using date from actual data range: 2025-10-18 to 2025-10-30
    # Use 2025-10-20 to ensure sufficient lookback data (3h for 3m interval)
    engine = BacktestEngine(
        start_date='2025-10-20',
        end_date='2025-10-21',  # Next day to include full Oct 20 data
        coins=['BTC'],
        model='anthropic',
        starting_capital=10000.0,
        use_llm_cache=False,  # Don't cache for clean test
        data_dir='data/historical',
        interval='3m',
        temperature=0.0
    )

    # Run backtest (should process just a few timestamps)
    results = engine.run()

    # Verify results structure
    assert 'model' in results
    assert 'iterations' in results
    assert 'current_value' in results
    assert results['model'] == 'anthropic'
    assert results['iterations'] > 0  # At least one iteration

    logger.info(f"Backtest integration test passed")
    logger.info(f"Iterations: {results['iterations']}")
    logger.info(f"Final value: ${results['current_value']:,.2f}")


@pytest.mark.integration
def test_backtest_with_checkpoint():
    """Test checkpoint save functionality"""
    from src.backtesting.backtest_engine import BacktestEngine
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pkl"

        # Using date from actual data range: 2025-10-18 to 2025-10-30
        # Use 2025-10-20 to ensure sufficient lookback data
        engine = BacktestEngine(
            start_date='2025-10-20',
            end_date='2025-10-21',  # Next day to include full Oct 20 data
            coins=['BTC'],
            model='anthropic',
            starting_capital=10000.0,
            use_llm_cache=False,
            checkpoint_dir=tmpdir,
            interval='3m'
        )

        # Run with checkpoint save
        results = engine.run(checkpoint_path=str(checkpoint_path))

        # Verify checkpoint was created
        assert checkpoint_path.exists(), "Checkpoint file should be created"
        assert checkpoint_path.stat().st_size > 0, "Checkpoint should not be empty"

        # Verify JSON metadata was also created
        json_path = checkpoint_path.with_suffix('.json')
        assert json_path.exists(), "Checkpoint JSON should be created"

        logger.info(f"Checkpoint integration test passed")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Size: {checkpoint_path.stat().st_size:,} bytes")

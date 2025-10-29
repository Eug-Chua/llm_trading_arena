"""
Tests for technical indicators module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.indicators import (
    TechnicalIndicators,
    calculate_indicators_for_candles,
    get_alpha_arena_indicators
)


@pytest.fixture
def sample_candles():
    """Generate sample candle data for testing"""
    base_time = int(datetime(2025, 10, 1).timestamp() * 1000)
    candles = []

    # Generate 100 candles (enough for all indicators)
    for i in range(100):
        timestamp = base_time + (i * 3 * 60 * 1000)  # 3-minute intervals
        # Simple uptrend with noise
        base_price = 100000 + (i * 100)
        noise = np.random.randn() * 500

        candles.append({
            't': timestamp,
            'o': base_price + noise,
            'h': base_price + abs(noise) + 100,
            'l': base_price - abs(noise) - 100,
            'c': base_price + noise * 0.5,
            'v': 1000000 + np.random.randn() * 100000
        })

    return candles


def test_prepare_dataframe(sample_candles):
    """Test DataFrame preparation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)

    assert not df.empty
    assert len(df) == len(sample_candles)
    assert 'timestamp' in df.columns
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns

    # Check sorting (oldest to newest)
    assert df['timestamp'].is_monotonic_increasing


def test_calculate_ema(sample_candles):
    """Test EMA calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_ema(df, periods=[20, 50])

    assert 'ema_20' in df.columns
    assert 'ema_50' in df.columns
    assert not df['ema_20'].isna().all()
    assert not df['ema_50'].isna().all()

    # EMA should be close to price
    assert abs(df['ema_20'].iloc[-1] - df['close'].iloc[-1]) < 1000


def test_calculate_macd(sample_candles):
    """Test MACD calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_macd(df)

    assert 'macd' in df.columns
    assert 'macd_signal' in df.columns
    assert 'macd_histogram' in df.columns
    assert not df['macd'].isna().all()

    # Histogram should be macd - signal
    last_histogram = df['macd_histogram'].iloc[-1]
    calculated_histogram = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
    assert abs(last_histogram - calculated_histogram) < 0.01


def test_calculate_rsi(sample_candles):
    """Test RSI calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_rsi(df, periods=[7, 14])

    assert 'rsi_7' in df.columns
    assert 'rsi_14' in df.columns
    assert not df['rsi_7'].isna().all()
    assert not df['rsi_14'].isna().all()

    # RSI should be between 0 and 100
    assert df['rsi_7'].min() >= 0
    assert df['rsi_7'].max() <= 100
    assert df['rsi_14'].min() >= 0
    assert df['rsi_14'].max() <= 100


def test_calculate_atr(sample_candles):
    """Test ATR calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_atr(df, periods=[3, 14])

    assert 'atr_3' in df.columns
    assert 'atr_14' in df.columns
    assert not df['atr_3'].isna().all()
    assert not df['atr_14'].isna().all()

    # ATR should be positive
    assert df['atr_3'].min() >= 0
    assert df['atr_14'].min() >= 0


def test_calculate_bollinger_bands(sample_candles):
    """Test Bollinger Bands calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_bollinger_bands(df)

    assert 'bb_upper' in df.columns
    assert 'bb_middle' in df.columns
    assert 'bb_lower' in df.columns
    assert 'bb_width' in df.columns

    # Upper should be > middle > lower
    last_upper = df['bb_upper'].iloc[-1]
    last_middle = df['bb_middle'].iloc[-1]
    last_lower = df['bb_lower'].iloc[-1]

    assert last_upper > last_middle > last_lower

    # Width should match
    assert abs(df['bb_width'].iloc[-1] - (last_upper - last_lower)) < 0.01


def test_calculate_vwma(sample_candles):
    """Test VWMA calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_vwma(df)

    assert 'vwma_20' in df.columns
    assert not df['vwma_20'].isna().all()


def test_calculate_adx(sample_candles):
    """Test ADX calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_adx(df)

    assert 'adx' in df.columns
    assert 'plus_di' in df.columns
    assert 'minus_di' in df.columns
    assert not df['adx'].isna().all()

    # ADX should be between 0 and 100
    assert df['adx'].min() >= 0
    assert df['adx'].max() <= 100


def test_calculate_supertrend(sample_candles):
    """Test Supertrend calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_supertrend(df)

    assert 'supertrend' in df.columns
    assert 'supertrend_direction' in df.columns
    assert not df['supertrend'].isna().all()

    # Direction should be 1 or -1
    assert df['supertrend_direction'].isin([1, -1]).all()


def test_calculate_cci(sample_candles):
    """Test CCI calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_cci(df)

    assert 'cci_20' in df.columns
    assert not df['cci_20'].isna().all()


def test_calculate_stochastic(sample_candles):
    """Test Stochastic Oscillator calculation"""
    calc = TechnicalIndicators()
    df = calc.prepare_dataframe(sample_candles)
    df = calc.calculate_stochastic(df)

    assert 'stoch_k' in df.columns
    assert 'stoch_d' in df.columns
    assert not df['stoch_k'].isna().all()
    assert not df['stoch_d'].isna().all()

    # Stochastic should be between 0 and 100
    assert df['stoch_k'].min() >= 0
    assert df['stoch_k'].max() <= 100


def test_calculate_all_indicators(sample_candles):
    """Test calculating all indicators at once"""
    calc = TechnicalIndicators()
    df = calc.calculate_all_indicators(sample_candles)

    # Check that all expected indicators are present
    expected_indicators = [
        'ema_20', 'ema_50',
        'macd', 'macd_signal', 'macd_histogram',
        'rsi_7', 'rsi_14',
        'atr_3', 'atr_14',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'vwma_20',
        'adx', 'plus_di', 'minus_di',
        'supertrend', 'supertrend_direction',
        'cci_20',
        'stoch_k', 'stoch_d'
    ]

    for indicator in expected_indicators:
        assert indicator in df.columns, f"Missing indicator: {indicator}"

    assert len(df) == len(sample_candles)


def test_get_latest_indicators(sample_candles):
    """Test getting latest indicator values in Alpha Arena format"""
    calc = TechnicalIndicators()
    df = calc.calculate_all_indicators(sample_candles)
    result = calc.get_latest_indicators(df, lookback=10)

    # Check structure
    assert 'current' in result
    assert 'arrays' in result
    assert 'stats' in result

    # Check current values
    assert 'price' in result['current']
    assert 'ema_20' in result['current']
    assert 'macd' in result['current']
    assert 'rsi_7' in result['current']
    assert 'rsi_14' in result['current']

    # Check arrays
    assert 'prices' in result['arrays']
    assert 'ema_20' in result['arrays']
    assert 'macd' in result['arrays']
    assert 'rsi_7' in result['arrays']

    # Arrays should have 10 data points
    assert len(result['arrays']['prices']) == 10
    assert len(result['arrays']['ema_20']) == 10
    assert len(result['arrays']['macd']) == 10

    # Check stats
    assert 'high' in result['stats']
    assert 'low' in result['stats']
    assert 'mean' in result['stats']
    assert 'volume_current' in result['stats']
    assert 'volume_avg' in result['stats']


def test_convenience_functions(sample_candles):
    """Test convenience functions"""
    # Test calculate_indicators_for_candles
    df = calculate_indicators_for_candles(sample_candles)
    assert not df.empty
    assert 'ema_20' in df.columns
    assert 'macd' in df.columns

    # Test get_alpha_arena_indicators
    result = get_alpha_arena_indicators(sample_candles)
    assert 'current' in result
    assert 'arrays' in result
    assert len(result['arrays']['prices']) == 10


def test_empty_candles():
    """Test handling of empty candles list"""
    calc = TechnicalIndicators()
    df = calc.calculate_all_indicators([])

    assert df.empty


def test_custom_config(sample_candles):
    """Test custom indicator configuration"""
    custom_config = {
        'ema_periods': [10, 30],
        'rsi_periods': [9, 21],
        'atr_periods': [7],
        'macd': {'fast': 8, 'slow': 21, 'signal': 5},
        'bollinger': {'period': 15, 'std_dev': 1.5},
        'vwma_period': 15,
        'adx_period': 10,
        'supertrend': {'period': 7, 'multiplier': 2.5},
        'cci_period': 14,
        'stochastic': {'k_period': 10, 'd_period': 5}
    }

    calc = TechnicalIndicators()
    df = calc.calculate_all_indicators(sample_candles, custom_config)

    # Check custom periods are used
    assert 'ema_10' in df.columns
    assert 'ema_30' in df.columns
    assert 'rsi_9' in df.columns
    assert 'rsi_21' in df.columns
    assert 'atr_7' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

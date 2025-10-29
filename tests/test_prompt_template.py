"""
Tests for Alpha Arena prompt template
"""

import pytest
from src.prompts.alpha_arena_template import (
    AlphaArenaPrompt,
    MarketData,
    Position,
    AccountInfo,
    create_sample_market_data,
    create_sample_account,
    generate_alpha_arena_prompt
)


def test_format_coin_data():
    """Test formatting market data for a single coin"""
    data = create_sample_market_data('BTC', 114000.0)

    generator = AlphaArenaPrompt()
    result = generator.format_coin_data(data)

    # Check key components are present
    assert 'ALL BTC DATA' in result
    assert 'current_price' in result
    assert 'current_ema20' in result
    assert 'current_macd' in result
    assert 'current_rsi (7 period)' in result
    assert 'Open Interest' in result
    assert 'Funding Rate' in result
    assert 'Intraday series' in result
    assert 'Mid prices:' in result
    assert 'EMA indicators (20-period):' in result
    assert 'MACD indicators:' in result
    assert 'RSI indicators (7-Period):' in result
    assert 'RSI indicators (14-Period):' in result
    assert 'Longer-term context (4-hour timeframe):' in result
    assert '20-Period EMA:' in result
    assert '50-Period EMA:' in result
    assert '3-Period ATR:' in result
    assert '14-Period ATR:' in result


def test_format_position():
    """Test formatting a trading position"""
    pos = Position(
        symbol='BTC',
        quantity=0.5,
        entry_price=110000.0,
        current_price=114000.0,
        liquidation_price=100000.0,
        unrealized_pnl=2000.0,
        leverage=10,
        exit_plan={'profit_target': 120000.0, 'stop_loss': 108000.0},
        confidence=0.75,
        risk_usd=1000.0,
        notional_usd=57000.0
    )

    generator = AlphaArenaPrompt()
    result = generator.format_position(pos)

    # Should be formatted as Python dict
    assert 'symbol' in result
    assert 'BTC' in result
    assert 'quantity' in result
    assert 'entry_price' in result
    assert 'unrealized_pnl' in result
    assert 'exit_plan' in result


def test_format_account_info():
    """Test formatting account information"""
    account = create_sample_account()

    generator = AlphaArenaPrompt()
    result = generator.format_account_info(account)

    assert 'HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE' in result
    assert 'Current Total Return (percent):' in result
    assert 'Available Cash:' in result
    assert 'Current Account Value:' in result
    assert 'Current live positions & performance:' in result
    assert 'Sharpe Ratio:' in result
    assert 'BTC' in result  # Should have BTC position
    assert 'ETH' in result  # Should have ETH position


def test_generate_complete_prompt():
    """Test generating a complete Alpha Arena prompt"""
    # Create market data for all coins
    market_data = {
        'BTC': create_sample_market_data('BTC', 114000.0),
        'ETH': create_sample_market_data('ETH', 4100.0),
        'SOL': create_sample_market_data('SOL', 200.0),
        'BNB': create_sample_market_data('BNB', 1130.0),
        'XRP': create_sample_market_data('XRP', 2.64),
        'DOGE': create_sample_market_data('DOGE', 0.20)
    }

    account = create_sample_account()

    generator = AlphaArenaPrompt()
    prompt = generator.generate_prompt(market_data, account)

    # Check header
    assert 'It has been' in prompt
    assert 'minutes since you started trading' in prompt
    assert "you've been invoked" in prompt
    assert 'ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST' in prompt
    assert 'CURRENT MARKET STATE FOR ALL COINS' in prompt

    # Check all coins are present
    assert 'ALL BTC DATA' in prompt
    assert 'ALL ETH DATA' in prompt
    assert 'ALL SOL DATA' in prompt
    assert 'ALL BNB DATA' in prompt
    assert 'ALL XRP DATA' in prompt
    assert 'ALL DOGE DATA' in prompt

    # Check account section
    assert 'HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE' in prompt
    assert 'Sharpe Ratio:' in prompt

    # Should be substantial prompt
    assert len(prompt) > 5000  # Alpha Arena prompts are ~15k-25k chars


def test_invocation_counter():
    """Test that invocation counter increments"""
    generator = AlphaArenaPrompt()
    market_data = {'BTC': create_sample_market_data('BTC', 114000.0)}
    account = create_sample_account()

    prompt1 = generator.generate_prompt(market_data, account)
    assert "you've been invoked 1 times" in prompt1

    prompt2 = generator.generate_prompt(market_data, account)
    assert "you've been invoked 2 times" in prompt2

    prompt3 = generator.generate_prompt(market_data, account)
    assert "you've been invoked 3 times" in prompt3


def test_generator_reset():
    """Test resetting the generator"""
    generator = AlphaArenaPrompt()
    market_data = {'BTC': create_sample_market_data('BTC', 114000.0)}
    account = create_sample_account()

    # Generate a few prompts
    generator.generate_prompt(market_data, account)
    generator.generate_prompt(market_data, account)
    generator.generate_prompt(market_data, account)

    # Reset
    generator.reset()

    # Should start from 1 again
    prompt = generator.generate_prompt(market_data, account)
    assert "you've been invoked 1 times" in prompt


def test_convenience_function():
    """Test the convenience function"""
    market_data = {
        'BTC': create_sample_market_data('BTC', 114000.0),
        'ETH': create_sample_market_data('ETH', 4100.0)
    }
    account = create_sample_account()

    prompt = generate_alpha_arena_prompt(market_data, account)

    assert 'ALL BTC DATA' in prompt
    assert 'ALL ETH DATA' in prompt
    assert 'HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE' in prompt


def test_price_formatting():
    """Test proper price formatting for different magnitudes"""
    # Large price (BTC)
    btc_data = create_sample_market_data('BTC', 114000.0)
    generator = AlphaArenaPrompt()
    btc_prompt = generator.format_coin_data(btc_data)

    # Should format with 2 decimals
    assert '114' in btc_prompt  # Some variant of 114xxx

    # Small price (DOGE)
    doge_data = create_sample_market_data('DOGE', 0.20)
    doge_prompt = generator.format_coin_data(doge_data)

    # Should format with more decimals
    assert '0.' in doge_prompt


def test_arrays_have_correct_length():
    """Test that all arrays have 10 data points"""
    data = create_sample_market_data('BTC', 114000.0)

    assert len(data.prices) == 10
    assert len(data.ema_20) == 10
    assert len(data.macd) == 10
    assert len(data.rsi_7) == 10
    assert len(data.rsi_14) == 10
    assert len(data.macd_4h) == 10
    assert len(data.rsi_14_4h) == 10


def test_missing_coin_warning(caplog):
    """Test that missing coin data triggers warning"""
    generator = AlphaArenaPrompt()
    market_data = {
        'BTC': create_sample_market_data('BTC', 114000.0)
        # Missing other coins
    }
    account = create_sample_account()

    prompt = generator.generate_prompt(market_data, account)

    # Should still generate prompt
    assert 'ALL BTC DATA' in prompt

    # Should have logged warnings (check with caplog if logging configured)
    # This test might need adjustment based on logging setup


def test_custom_objective():
    """Test using custom objective"""
    market_data = {'BTC': create_sample_market_data('BTC', 114000.0)}
    account = create_sample_account()

    custom_obj = "Your objective is to maximize returns while minimizing risk."

    generator = AlphaArenaPrompt()
    prompt = generator.generate_prompt(market_data, account, objective=custom_obj)

    assert custom_obj in prompt


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

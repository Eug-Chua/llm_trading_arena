"""
Unit tests for TradingEngine Dependency Injection

Tests that TradingEngine correctly uses injected factories.
"""

import pytest
from unittest.mock import Mock, call
from datetime import datetime

from src.trading.trading_engine import TradingEngine
from src.trading.position import Position, Account
from src.trading.factories import default_account_factory, default_position_factory
from src.agents.base_agent import TradeSignal


@pytest.mark.unit
class TestTradingEngineDI:
    """Test dependency injection in TradingEngine"""

    def test_uses_default_account_factory_when_none_provided(self):
        """Test that default account factory is used when none is provided"""
        engine = TradingEngine(starting_capital=10000.0)

        assert engine.account is not None
        assert isinstance(engine.account, Account)
        assert engine.account.starting_capital == 10000.0
        assert engine.account.available_cash == 10000.0

    def test_uses_custom_account_factory(self):
        """Test that custom account factory is used when provided"""
        # Create a mock account
        mock_account = Mock(spec=Account)
        mock_account.starting_capital = 15000.0
        mock_account.available_cash = 15000.0
        mock_account.positions = {}
        mock_account.update_prices = Mock()

        # Create account factory that returns mock
        def custom_account_factory(starting_capital: float) -> Account:
            assert starting_capital == 12345.0
            return mock_account

        engine = TradingEngine(
            starting_capital=12345.0,
            account_factory=custom_account_factory
        )

        assert engine.account == mock_account

    def test_uses_default_position_factory_when_none_provided(self):
        """Test that default position factory creates real Position objects"""
        engine = TradingEngine(starting_capital=10000.0)

        signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        current_prices = {"BTC": 100000.0}
        results = engine.execute_signals({"BTC": signal}, current_prices)

        assert "EXECUTED" in results["BTC"]
        assert engine.account.has_position("BTC")

        # Verify it's a real Position object
        position = engine.account.get_position("BTC")
        assert isinstance(position, Position)
        assert position.symbol == "BTC"
        assert position.quantity == 0.5

    def test_uses_custom_position_factory(self):
        """Test that custom position factory is called with correct parameters"""
        # Create a mock position
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC"
        mock_position.quantity = 0.5
        mock_position.leverage = 10
        mock_position.current_price = 100000.0
        mock_position.notional_usd = 50000.0
        mock_position.unrealized_pnl = 0.0

        # Track position factory calls
        position_factory_calls = []

        def custom_position_factory(**kwargs) -> Position:
            position_factory_calls.append(kwargs)
            return mock_position

        engine = TradingEngine(
            starting_capital=10000.0,
            position_factory=custom_position_factory
        )

        signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        current_prices = {"BTC": 100000.0}
        results = engine.execute_signals({"BTC": signal}, current_prices)

        # Verify position factory was called
        assert len(position_factory_calls) == 1

        # Verify parameters passed to factory
        call_args = position_factory_calls[0]
        assert call_args['symbol'] == "BTC"
        assert call_args['quantity'] == 0.5
        assert call_args['entry_price'] == 100000.0
        assert call_args['current_price'] == 100000.0
        assert call_args['leverage'] == 10
        assert call_args['stop_loss'] == 98000.0
        assert call_args['profit_target'] == 105000.0
        assert call_args['invalidation_condition'] == "Below $97,000"
        assert call_args['confidence'] == 0.8
        assert call_args['risk_usd'] == 100.0
        assert 'entry_fee' in call_args

    def test_position_factory_called_for_each_buy(self):
        """Test that position factory is called once per buy signal"""
        call_count = [0]

        def counting_position_factory(**kwargs) -> Position:
            call_count[0] += 1
            return default_position_factory(**kwargs)

        engine = TradingEngine(
            starting_capital=20000.0,
            position_factory=counting_position_factory
        )

        # Execute two buy signals
        btc_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        eth_signal = TradeSignal(
            coin="ETH",
            signal="buy",
            quantity=5.0,
            leverage=10,
            stop_loss=3900.0,
            profit_target=4200.0,
            invalidation_condition="Below $3,850",
            confidence=0.75,
            risk_usd=100.0
        )

        current_prices = {"BTC": 100000.0, "ETH": 4000.0}
        engine.execute_signals({"BTC": btc_signal, "ETH": eth_signal}, current_prices)

        # Factory should be called twice (once for each buy)
        assert call_count[0] == 2

    def test_position_factory_not_called_for_hold(self):
        """Test that position factory is not called for hold signals"""
        call_count = [0]

        def counting_position_factory(**kwargs) -> Position:
            call_count[0] += 1
            return default_position_factory(**kwargs)

        engine = TradingEngine(
            starting_capital=10000.0,
            position_factory=counting_position_factory
        )

        # First buy to create position
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})
        assert call_count[0] == 1

        # Hold signal should not create new position
        hold_signal = TradeSignal(
            coin="BTC",
            signal="hold",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        engine.execute_signals({"BTC": hold_signal}, {"BTC": 102000.0})
        assert call_count[0] == 1  # Should still be 1

    def test_backward_compatibility_no_factories(self):
        """Test that existing code works without providing factories"""
        # This should work exactly as before
        engine = TradingEngine(starting_capital=10000.0)

        signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        current_prices = {"BTC": 100000.0}
        results = engine.execute_signals({"BTC": signal}, current_prices)

        assert "EXECUTED" in results["BTC"]
        assert engine.account.has_position("BTC")
        position = engine.account.get_position("BTC")
        assert position.symbol == "BTC"

    def test_can_inject_both_factories(self):
        """Test that both account and position factories can be injected together"""
        mock_account = Mock(spec=Account)
        mock_account.starting_capital = 10000.0
        mock_account.available_cash = 10000.0
        mock_account.positions = {}
        mock_account.update_prices = Mock()
        mock_account.has_position = Mock(return_value=False)
        mock_account.add_position = Mock()
        mock_account.total_fees_paid = 0.0
        mock_account.trade_log = []
        mock_account.account_value = 10000.0

        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC"
        mock_position.notional_usd = 50000.0
        mock_position.leverage = 10

        def custom_account_factory(starting_capital: float) -> Account:
            return mock_account

        def custom_position_factory(**kwargs) -> Position:
            return mock_position

        engine = TradingEngine(
            starting_capital=10000.0,
            account_factory=custom_account_factory,
            position_factory=custom_position_factory
        )

        # Verify both factories were used
        assert engine.account == mock_account

        signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        current_prices = {"BTC": 100000.0}
        engine.execute_signals({"BTC": signal}, current_prices)

        # Verify position was added using custom factory
        mock_account.add_position.assert_called_once_with(mock_position)

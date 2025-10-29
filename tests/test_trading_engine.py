"""
Tests for Trading Engine

Tests position management, trade execution, and risk management.
"""

import pytest
from datetime import datetime

from src.trading.position import Position, Account
from src.trading.trading_engine import TradingEngine
from src.agents.base_agent import TradeSignal


class TestPosition:
    """Test Position class"""

    def test_position_creation(self):
        """Test creating a position"""
        pos = Position(
            symbol="BTC",
            quantity=0.5,
            entry_price=100000.0,
            current_price=100000.0,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.75,
            risk_usd=100.0
        )

        assert pos.symbol == "BTC"
        assert pos.quantity == 0.5
        assert pos.leverage == 10

    def test_liquidation_price_calculation(self):
        """Test liquidation price formula"""
        pos = Position(
            symbol="BTC",
            quantity=1.0,
            entry_price=100000.0,
            current_price=100000.0,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        # liquidation_price = entry_price * (1 - 1/leverage)
        # = 100000 * (1 - 1/10) = 100000 * 0.9 = 90000
        assert pos.liquidation_price == 90000.0

    def test_unrealized_pnl_profit(self):
        """Test P&L calculation with profit"""
        pos = Position(
            symbol="BTC",
            quantity=0.5,
            entry_price=100000.0,
            current_price=102000.0,  # Up $2000
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        # P&L = (current - entry) * quantity * leverage
        # = (102000 - 100000) * 0.5 * 10 = 2000 * 0.5 * 10 = 10000
        assert pos.unrealized_pnl == 10000.0

    def test_unrealized_pnl_loss(self):
        """Test P&L calculation with loss"""
        pos = Position(
            symbol="BTC",
            quantity=0.5,
            entry_price=100000.0,
            current_price=98000.0,  # Down $2000
            leverage=10,
            stop_loss=95000.0,
            profit_target=105000.0,
            invalidation_condition="Below $94,000",
            confidence=0.8,
            risk_usd=100.0
        )

        # P&L = (98000 - 100000) * 0.5 * 10 = -2000 * 0.5 * 10 = -10000
        assert pos.unrealized_pnl == -10000.0

    def test_is_liquidated(self):
        """Test liquidation detection"""
        pos = Position(
            symbol="BTC",
            quantity=1.0,
            entry_price=100000.0,
            current_price=89000.0,  # Below liquidation price (90000)
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        assert pos.is_liquidated() is True

    def test_is_stop_loss_hit(self):
        """Test stop-loss detection"""
        pos = Position(
            symbol="BTC",
            quantity=1.0,
            entry_price=100000.0,
            current_price=97000.0,  # Below stop-loss (98000)
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $95,000",
            confidence=0.8,
            risk_usd=100.0
        )

        assert pos.is_stop_loss_hit() is True

    def test_is_profit_target_hit(self):
        """Test profit target detection"""
        pos = Position(
            symbol="BTC",
            quantity=1.0,
            entry_price=100000.0,
            current_price=106000.0,  # Above profit target (105000)
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $95,000",
            confidence=0.8,
            risk_usd=100.0
        )

        assert pos.is_profit_target_hit() is True

    def test_update_price(self):
        """Test price update"""
        pos = Position(
            symbol="BTC",
            quantity=1.0,
            entry_price=100000.0,
            current_price=100000.0,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $95,000",
            confidence=0.8,
            risk_usd=100.0
        )

        pos.update_price(102000.0)
        assert pos.current_price == 102000.0


class TestAccount:
    """Test Account class"""

    def test_account_creation(self):
        """Test creating an account"""
        account = Account(
            starting_capital=10000.0,
            available_cash=10000.0
        )

        assert account.starting_capital == 10000.0
        assert account.available_cash == 10000.0
        assert account.account_value == 10000.0
        assert len(account.positions) == 0

    def test_account_value_with_positions(self):
        """Test account value calculation with open positions"""
        account = Account(
            starting_capital=10000.0,
            available_cash=5000.0
        )

        # Add position with profit
        pos = Position(
            symbol="BTC",
            quantity=0.5,
            entry_price=100000.0,
            current_price=102000.0,  # +$2000, 10x leverage = +$10000 P&L
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        account.add_position(pos)

        # account_value = cash + unrealized P&L
        # = 5000 + 10000 = 15000
        assert account.account_value == 15000.0

    def test_total_return(self):
        """Test total return calculation"""
        account = Account(
            starting_capital=10000.0,
            available_cash=5000.0
        )

        # Add profitable position
        pos = Position(
            symbol="BTC",
            quantity=0.5,
            entry_price=100000.0,
            current_price=102000.0,  # +$10000 P&L with 10x leverage
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        account.add_position(pos)

        # total_return = account_value - starting_capital
        # = 15000 - 10000 = 5000
        assert account.total_return == 5000.0

    def test_has_position(self):
        """Test position existence check"""
        account = Account(
            starting_capital=10000.0,
            available_cash=10000.0
        )

        pos = Position(
            symbol="BTC",
            quantity=1.0,
            entry_price=100000.0,
            current_price=100000.0,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        account.add_position(pos)

        assert account.has_position("BTC") is True
        assert account.has_position("ETH") is False

    def test_update_prices(self):
        """Test updating prices for multiple positions"""
        account = Account(
            starting_capital=10000.0,
            available_cash=5000.0
        )

        # Add BTC position
        btc_pos = Position(
            symbol="BTC",
            quantity=0.5,
            entry_price=100000.0,
            current_price=100000.0,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0
        )

        # Add ETH position
        eth_pos = Position(
            symbol="ETH",
            quantity=5.0,
            entry_price=4000.0,
            current_price=4000.0,
            leverage=10,
            stop_loss=3900.0,
            profit_target=4200.0,
            invalidation_condition="Below $3,850",
            confidence=0.75,
            risk_usd=100.0
        )

        account.add_position(btc_pos)
        account.add_position(eth_pos)

        # Update prices
        new_prices = {
            "BTC": 102000.0,
            "ETH": 4100.0
        }

        account.update_prices(new_prices)

        assert account.get_position("BTC").current_price == 102000.0
        assert account.get_position("ETH").current_price == 4100.0


class TestTradingEngine:
    """Test TradingEngine class"""

    def test_engine_initialization(self):
        """Test creating trading engine"""
        engine = TradingEngine(starting_capital=10000.0)

        assert engine.account.starting_capital == 10000.0
        assert engine.account.available_cash == 10000.0

    def test_buy_signal_success(self):
        """Test successful buy execution"""
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
            risk_usd=100.0,
        )

        current_prices = {"BTC": 100000.0}

        results = engine.execute_signals({"BTC": signal}, current_prices)

        assert "EXECUTED" in results["BTC"]
        assert engine.account.has_position("BTC")

        # Cash should be reduced by: (price * quantity) / leverage
        # = (100000 * 0.5) / 10 = 5000
        assert engine.account.available_cash == 5000.0

    def test_buy_signal_insufficient_cash(self):
        """Test buy rejection due to insufficient cash"""
        engine = TradingEngine(starting_capital=1000.0)  # Only $1000

        signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        current_prices = {"BTC": 100000.0}
        # Required: (100000 * 0.5) / 10 = 5000 > 1000 available

        results = engine.execute_signals({"BTC": signal}, current_prices)

        assert "REJECTED" in results["BTC"]
        assert "Insufficient cash" in results["BTC"]
        assert not engine.account.has_position("BTC")

    def test_buy_signal_no_pyramiding(self):
        """Test no pyramiding rule (cannot add to existing position)"""
        engine = TradingEngine(starting_capital=10000.0)

        # First buy
        signal1 = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        current_prices = {"BTC": 100000.0}
        engine.execute_signals({"BTC": signal1}, current_prices)

        # Try to buy again (should be rejected)
        signal2 = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.3,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        results = engine.execute_signals({"BTC": signal2}, current_prices)

        assert "REJECTED" in results["BTC"]
        assert "no pyramiding" in results["BTC"].lower()

    def test_hold_signal(self):
        """Test hold signal execution"""
        engine = TradingEngine(starting_capital=10000.0)

        # First, open a position
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})

        # Now hold
        hold_signal = TradeSignal(
            coin="BTC",
            signal="hold",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        results = engine.execute_signals({"BTC": hold_signal}, {"BTC": 102000.0})

        assert "HELD" in results["BTC"]
        assert engine.account.has_position("BTC")

    def test_hold_auto_close_stop_loss(self):
        """Test automatic closure when stop-loss is hit"""
        engine = TradingEngine(starting_capital=10000.0)

        # Open position
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})

        # Hold with price below stop-loss
        hold_signal = TradeSignal(
            coin="BTC",
            signal="hold",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        results = engine.execute_signals({"BTC": hold_signal}, {"BTC": 97000.0})

        assert "STOP-LOSS" in results["BTC"]
        assert not engine.account.has_position("BTC")

    def test_hold_auto_close_profit_target(self):
        """Test automatic closure when profit target is hit"""
        engine = TradingEngine(starting_capital=10000.0)

        # Open position
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})

        # Hold with price above profit target
        hold_signal = TradeSignal(
            coin="BTC",
            signal="hold",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        results = engine.execute_signals({"BTC": hold_signal}, {"BTC": 106000.0})

        assert "PROFIT-TARGET" in results["BTC"]
        assert not engine.account.has_position("BTC")

    def test_hold_auto_close_liquidation(self):
        """Test automatic closure when liquidation price is hit"""
        engine = TradingEngine(starting_capital=10000.0)

        # Open position with 10x leverage
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})
        # Liquidation price = 100000 * (1 - 1/10) = 90000

        # Hold with price at liquidation
        hold_signal = TradeSignal(
            coin="BTC",
            signal="hold",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        results = engine.execute_signals({"BTC": hold_signal}, {"BTC": 89000.0})

        assert "LIQUIDATION" in results["BTC"]
        assert not engine.account.has_position("BTC")

    def test_close_signal(self):
        """Test close signal execution"""
        engine = TradingEngine(starting_capital=10000.0)

        # Open position
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})

        # Close position
        close_signal = TradeSignal(
            coin="BTC",
            signal="close_position",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        results = engine.execute_signals({"BTC": close_signal}, {"BTC": 102000.0})

        assert "EXECUTED" in results["BTC"]
        assert "Closed position" in results["BTC"]
        assert not engine.account.has_position("BTC")

    def test_pnl_calculation_on_close(self):
        """Test P&L is calculated correctly when closing position"""
        engine = TradingEngine(starting_capital=10000.0)

        # Open position
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})
        # Cash used: (100000 * 0.5) / 10 = 5000
        # Remaining cash: 5000

        # Close at profit
        close_signal = TradeSignal(
            coin="BTC",
            signal="close_position",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": close_signal}, {"BTC": 102000.0})

        # P&L = (102000 - 100000) * 0.5 * 10 = 10000
        # Returned cash = 5000 (initial) + 10000 (profit) = 15000
        # Total cash = 5000 (remaining) + 15000 = 20000
        assert engine.account.available_cash == 20000.0
        assert engine.account.account_value == 20000.0

    def test_performance_summary(self):
        """Test performance summary generation"""
        engine = TradingEngine(starting_capital=10000.0)

        # Open and close a profitable trade
        buy_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": buy_signal}, {"BTC": 100000.0})

        close_signal = TradeSignal(
            coin="BTC",
            signal="close_position",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        engine.execute_signals({"BTC": close_signal}, {"BTC": 102000.0})

        summary = engine.get_performance_summary()

        assert summary['account_value'] == 20000.0
        assert summary['total_return'] == 10000.0
        assert summary['total_return_percent'] == 100.0
        assert summary['total_trades'] == 1
        assert summary['open_positions'] == 0

    def test_multiple_positions(self):
        """Test managing multiple positions simultaneously"""
        engine = TradingEngine(starting_capital=20000.0)

        # Buy BTC
        btc_signal = TradeSignal(
            coin="BTC",
            signal="buy",
            quantity=0.5,
            leverage=10,
            stop_loss=98000.0,
            profit_target=105000.0,
            invalidation_condition="Below $97,000",
            confidence=0.8,
            risk_usd=100.0,
        )

        # Buy ETH
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

        results = engine.execute_signals(
            {"BTC": btc_signal, "ETH": eth_signal},
            current_prices
        )

        assert "EXECUTED" in results["BTC"]
        assert "EXECUTED" in results["ETH"]
        assert engine.account.has_position("BTC")
        assert engine.account.has_position("ETH")

        # Check cash deduction
        # BTC: (100000 * 0.5) / 10 = 5000
        # ETH: (4000 * 5.0) / 10 = 2000
        # Total used: 7000
        # Remaining: 20000 - 7000 = 13000
        assert engine.account.available_cash == 13000.0

"""
Trading Engine

Core system for executing trades and managing positions.
Implements Alpha Arena trading rules.
"""

from typing import Dict, List, Optional
from datetime import datetime

from .position import Position, Account
from ..agents.base_agent import TradeSignal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TradingEngine:
    """
    Main trading engine

    Executes trade signals from LLM agents and manages positions.
    """

    def __init__(self, starting_capital: float = 10000.0):
        """
        Initialize trading engine

        Args:
            starting_capital: Initial capital in USD
        """
        self.account = Account(
            starting_capital=starting_capital,
            available_cash=starting_capital
        )

        logger.info(f"Initialized trading engine with ${starting_capital:,.2f}")

    def execute_signals(
        self,
        signals: Dict[str, TradeSignal],
        current_prices: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Execute trade signals from LLM

        Args:
            signals: Dict mapping symbol to TradeSignal
            current_prices: Dict mapping symbol to current price

        Returns:
            Dict mapping symbol to execution result
        """
        results = {}

        # Update prices first
        self.account.update_prices(current_prices)

        # Process each signal
        for symbol, signal in signals.items():
            result = self._execute_signal(symbol, signal, current_prices.get(symbol))
            results[symbol] = result

        logger.info(f"Executed {len(signals)} signals: {results}")
        return results

    def _execute_signal(
        self,
        symbol: str,
        signal: TradeSignal,
        current_price: Optional[float]
    ) -> str:
        """
        Execute a single trade signal

        Args:
            symbol: Coin symbol
            signal: TradeSignal object
            current_price: Current market price

        Returns:
            Execution result message
        """
        if current_price is None:
            return "SKIPPED: No price data"

        # Validate signal
        if not self._validate_signal(signal):
            return "REJECTED: Invalid signal"

        if signal.signal == "buy":
            return self._execute_buy(signal, current_price)
        elif signal.signal == "hold":
            return self._execute_hold(signal, current_price)
        elif signal.signal == "close_position":
            return self._execute_close(symbol, current_price)
        else:
            return f"ERROR: Unknown signal type '{signal.signal}'"

    def _validate_signal(self, signal: TradeSignal) -> bool:
        """
        Validate trade signal

        Args:
            signal: TradeSignal to validate

        Returns:
            True if valid
        """
        # Check required fields
        if signal.quantity <= 0:
            logger.warning(f"Invalid quantity {signal.quantity}")
            return False

        if signal.leverage <= 0:
            logger.warning(f"Invalid leverage {signal.leverage}")
            return False

        if signal.stop_loss <= 0 or signal.profit_target <= 0:
            logger.warning(f"Invalid stop-loss or profit target")
            return False

        # Profit target should be above stop-loss (for long positions)
        if signal.profit_target <= signal.stop_loss:
            logger.warning(f"Profit target {signal.profit_target} not above stop-loss {signal.stop_loss}")
            return False

        # Confidence should be between 0 and 1
        if not (0.0 <= signal.confidence <= 1.0):
            logger.warning(f"Invalid confidence {signal.confidence}")
            return False

        return True

    def _execute_buy(self, signal: TradeSignal, current_price: float) -> str:
        """
        Execute buy signal (open new position)

        Args:
            signal: TradeSignal
            current_price: Current market price

        Returns:
            Result message
        """
        # Rule: Cannot add to existing position (no pyramiding)
        if self.account.has_position(signal.coin):
            logger.warning(f"Cannot buy {signal.coin} - position already exists")
            return "REJECTED: Position already exists (no pyramiding)"

        # Calculate required cash (considering leverage)
        # With leverage, you only need: (price * quantity) / leverage
        required_cash = (current_price * signal.quantity) / signal.leverage

        # Check if we have enough cash
        if required_cash > self.account.available_cash:
            logger.warning(f"Insufficient cash for {signal.coin}: need ${required_cash:.2f}, have ${self.account.available_cash:.2f}")
            return f"REJECTED: Insufficient cash (need ${required_cash:.2f})"

        # Create position
        position = Position(
            symbol=signal.coin,
            quantity=signal.quantity,
            entry_price=current_price,
            current_price=current_price,
            leverage=signal.leverage,
            stop_loss=signal.stop_loss,
            profit_target=signal.profit_target,
            invalidation_condition=signal.invalidation_condition,
            confidence=signal.confidence,
            risk_usd=signal.risk_usd
        )

        # Deduct cash
        self.account.available_cash -= required_cash

        # Add position
        self.account.add_position(position)

        # Log trade
        self.account.trade_log.append({
            'timestamp': datetime.now(),
            'action': 'BUY',
            'symbol': signal.coin,
            'quantity': signal.quantity,
            'price': current_price,
            'leverage': signal.leverage,
            'cost': required_cash
        })

        logger.info(f"BOUGHT {signal.coin}: {signal.quantity} @ ${current_price:.2f} ({signal.leverage}x leverage)")
        return f"EXECUTED: Opened position (${required_cash:.2f} capital used)"

    def _execute_hold(self, signal: TradeSignal, current_price: float) -> str:
        """
        Execute hold signal (keep existing position)

        Args:
            signal: TradeSignal
            current_price: Current market price

        Returns:
            Result message
        """
        # Get existing position
        position = self.account.get_position(signal.coin)

        if position is None:
            logger.warning(f"Cannot hold {signal.coin} - no position exists")
            return "REJECTED: No position to hold"

        # Update price
        position.update_price(current_price)

        # Check if we should update exit plan (if LLM changed targets)
        if (signal.stop_loss != position.stop_loss or
            signal.profit_target != position.profit_target):

            logger.info(f"Updating exit plan for {signal.coin}")
            position.stop_loss = signal.stop_loss
            position.profit_target = signal.profit_target
            position.invalidation_condition = signal.invalidation_condition

        # Check if position should be auto-closed
        if position.is_liquidated():
            logger.warning(f"{signal.coin} LIQUIDATED at ${current_price:.2f}")
            return self._force_close(signal.coin, current_price, "LIQUIDATION")

        if position.is_stop_loss_hit():
            logger.info(f"{signal.coin} stop-loss hit at ${current_price:.2f}")
            return self._force_close(signal.coin, current_price, "STOP-LOSS")

        if position.is_profit_target_hit():
            logger.info(f"{signal.coin} profit target hit at ${current_price:.2f}")
            return self._force_close(signal.coin, current_price, "PROFIT-TARGET")

        return f"HELD: P&L ${position.unrealized_pnl:.2f}"

    def _execute_close(self, symbol: str, current_price: float) -> str:
        """
        Execute close signal (exit position)

        Args:
            symbol: Coin symbol
            current_price: Current market price

        Returns:
            Result message
        """
        position = self.account.get_position(symbol)

        if position is None:
            logger.warning(f"Cannot close {symbol} - no position exists")
            return "REJECTED: No position to close"

        return self._force_close(symbol, current_price, "USER-SIGNAL")

    def _force_close(self, symbol: str, current_price: float, reason: str) -> str:
        """
        Force close a position

        Args:
            symbol: Coin symbol
            current_price: Exit price
            reason: Reason for closing

        Returns:
            Result message
        """
        position = self.account.remove_position(symbol)

        if position is None:
            return "ERROR: Position not found"

        # Update price to current
        position.update_price(current_price)

        # Calculate P&L
        pnl = position.unrealized_pnl

        # Return cash (initial capital + P&L)
        initial_capital = (position.entry_price * position.quantity) / position.leverage
        returned_cash = initial_capital + pnl
        self.account.available_cash += returned_cash

        # Log trade
        self.account.trade_log.append({
            'timestamp': datetime.now(),
            'action': 'CLOSE',
            'symbol': symbol,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'reason': reason
        })

        # Update account metrics
        self.account.update_return_percent()

        logger.info(f"CLOSED {symbol}: {position.quantity} @ ${current_price:.2f}, P&L: ${pnl:.2f}, Reason: {reason}")
        return f"EXECUTED: Closed position (P&L ${pnl:.2f}, {reason})"

    def get_account_state(self) -> Account:
        """
        Get current account state

        Returns:
            Account object
        """
        return self.account

    def get_positions(self) -> Dict[str, Position]:
        """
        Get all open positions

        Returns:
            Dict of positions
        """
        return self.account.positions

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary

        Returns:
            Performance metrics dict
        """
        return {
            'account_value': self.account.account_value,
            'total_return': self.account.total_return,
            'total_return_percent': self.account.total_return_percent,
            'available_cash': self.account.available_cash,
            'open_positions': len(self.account.positions),
            'total_trades': self.account.trade_count,
            'sharpe_ratio': self.account.sharpe_ratio
        }

    def __repr__(self) -> str:
        """String representation"""
        return f"TradingEngine({self.account})"

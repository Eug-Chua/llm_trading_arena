"""
Trading Engine

Core system for executing trades and managing positions.
Implements Alpha Arena trading rules.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime

from .position import Position, Account
from .factories import AccountFactory, PositionFactory, default_account_factory, default_position_factory
from ..agents.base_agent import TradeSignal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TradingEngine:
    """
    Main trading engine

    Executes trade signals from LLM agents and manages positions.
    """

    def __init__(
        self,
        starting_capital: float = 10000.0,
        maker_fee: float = 0.0002,  # 0.02% maker fee (Hyperliquid)
        taker_fee: float = 0.0005,   # 0.05% taker fee (Hyperliquid)
        account_factory: Optional[AccountFactory] = None,
        position_factory: Optional[PositionFactory] = None
    ):
        """
        Initialize trading engine

        Args:
            starting_capital: Initial capital in USD
            maker_fee: Maker fee rate (default: 0.02%)
            taker_fee: Taker fee rate (default: 0.05%)
            account_factory: Factory for creating Account instances (optional)
            position_factory: Factory for creating Position instances (optional)
        """
        # Use provided factories or defaults
        _account_factory = account_factory or default_account_factory
        self.position_factory = position_factory or default_position_factory

        # Create account using factory
        self.account = _account_factory(starting_capital)

        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # Track funding rates for each coin
        self.funding_rates: Dict[str, float] = {}

        # Current timestamp for trade logging (set by execute_signals)
        self._current_timestamp = datetime.now()

        logger.info(f"Initialized trading engine with ${starting_capital:,.2f}")
        logger.info(f"Fees: Maker {maker_fee*100:.3f}%, Taker {taker_fee*100:.3f}%")

    def execute_signals(
        self,
        signals: Dict[str, TradeSignal],
        current_prices: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, str]:
        """
        Execute trade signals from LLM

        Args:
            signals: Dict mapping symbol to TradeSignal
            current_prices: Dict mapping symbol to current price
            timestamp: Historical timestamp for backtesting (uses current time if None)

        Returns:
            Dict mapping symbol to execution result
        """
        results = {}

        # Store timestamp for trade logging
        self._current_timestamp = timestamp if timestamp else datetime.now()

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
            return self._execute_close(symbol, current_price, signal)
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

        # Calculate entry fee (on notional value)
        notional_value = current_price * signal.quantity
        entry_fee = notional_value * self.taker_fee  # Assume taker for market orders

        # Total cash needed = margin + fee
        total_cash_needed = required_cash + entry_fee

        # Check if we have enough cash
        if total_cash_needed > self.account.available_cash:
            logger.warning(f"Insufficient cash for {signal.coin}: need ${total_cash_needed:.2f}, have ${self.account.available_cash:.2f}")
            return f"REJECTED: Insufficient cash (need ${total_cash_needed:.2f})"

        # Create position using factory
        position = self.position_factory(
            symbol=signal.coin,
            quantity=signal.quantity,
            entry_price=current_price,
            current_price=current_price,
            leverage=signal.leverage,
            stop_loss=signal.stop_loss,
            profit_target=signal.profit_target,
            invalidation_condition=signal.invalidation_condition,
            confidence=signal.confidence,
            risk_usd=signal.risk_usd,
            entry_fee=entry_fee
        )

        # Deduct cash (margin + fee)
        self.account.available_cash -= total_cash_needed

        # Track fees
        self.account.total_fees_paid += entry_fee

        # Add position
        self.account.add_position(position)

        # Log trade (account value is AFTER the trade)
        self.account.trade_log.append({
            'timestamp': self._current_timestamp,
            'action': 'BUY',
            'symbol': signal.coin,
            'quantity': signal.quantity,
            'price': current_price,
            'leverage': signal.leverage,
            'cost': required_cash,
            'account_value': self.account.account_value
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

    def _execute_close(self, symbol: str, current_price: float, signal: Optional['TradeSignal'] = None) -> str:
        """
        Execute close signal (exit position)

        Args:
            symbol: Coin symbol
            current_price: Current market price
            signal: Optional TradeSignal with close_reason

        Returns:
            Result message
        """
        position = self.account.get_position(symbol)

        if position is None:
            logger.warning(f"Cannot close {symbol} - no position exists")
            return "REJECTED: No position to close"

        # Use close_reason from signal if provided, otherwise default to USER-SIGNAL
        reason = "USER-SIGNAL"
        if signal and signal.close_reason:
            reason = f"USER-SIGNAL ({signal.close_reason})"

        return self._force_close(symbol, current_price, reason)

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

        # Calculate exit fee (on notional value at exit)
        notional_value = current_price * position.quantity
        exit_fee = notional_value * self.taker_fee  # Assume taker for market orders

        # Calculate P&L (already includes entry fee and funding in unrealized_pnl)
        pnl = position.unrealized_pnl

        # Subtract exit fee from PnL
        net_pnl = pnl - exit_fee

        # Return cash (initial capital + net P&L)
        initial_capital = (position.entry_price * position.quantity) / position.leverage
        returned_cash = initial_capital + net_pnl
        self.account.available_cash += returned_cash

        # Track fees
        self.account.total_fees_paid += exit_fee
        self.account.total_funding_paid += position.accumulated_funding

        # Log trade (account value is AFTER the trade)
        self.account.trade_log.append({
            'timestamp': self._current_timestamp,
            'action': 'CLOSE',
            'symbol': symbol,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'gross_pnl': pnl,
            'exit_fee': exit_fee,
            'net_pnl': net_pnl,
            'reason': reason,
            'account_value': self.account.account_value
        })

        # Update account metrics (including Sharpe ratio)
        self.account.update_performance_metrics()

        logger.info(f"CLOSED {symbol}: {position.quantity} @ ${current_price:.2f}, Net P&L: ${net_pnl:.2f} (fee: ${exit_fee:.2f}), Reason: {reason}")
        return f"EXECUTED: Closed position (Net P&L ${net_pnl:.2f}, {reason})"

    def update_funding_rates(self, funding_rates: Dict[str, float]) -> None:
        """
        Update funding rates and apply funding costs to open positions

        Args:
            funding_rates: Dict mapping symbol to current funding rate
        """
        self.funding_rates = funding_rates

        # Apply funding costs to all open positions
        for symbol, position in self.account.positions.items():
            if symbol in funding_rates:
                position.apply_funding_cost(funding_rates[symbol])

    def check_exit_conditions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check if any open positions have hit stop-loss or profit-target

        Args:
            current_prices: Dict mapping symbol to current price

        Returns:
            List of exit info dicts for positions that were closed
        """
        exits = []

        # Update prices first
        self.account.update_prices(current_prices)

        # Check each position
        for symbol, position in list(self.account.positions.items()):
            current_price = current_prices.get(symbol)
            if current_price is None:
                continue

            exit_reason = None

            # Check stop-loss (SL)
            if position.stop_loss > 0:
                if current_price <= position.stop_loss:
                    exit_reason = f"Stop-loss hit (${position.stop_loss:.2f})"

            # Check profit-target (PT)
            if position.profit_target > 0:
                if current_price >= position.profit_target:
                    exit_reason = f"Profit-target hit (${position.profit_target:.2f})"

            # Execute exit if condition met
            if exit_reason:
                # Save PnL before closing
                pnl = position.unrealized_pnl
                result = self._force_close(symbol, current_price, exit_reason)
                exits.append({
                    'symbol': symbol,
                    'exit_price': current_price,
                    'reason': exit_reason,
                    'pnl': pnl,
                    'result': result
                })

        return exits

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
        Get performance summary (basic metrics for backwards compatibility)

        Returns:
            Performance metrics dict
        """
        return {
            'account_value': self.account.account_value,
            'total_return': self.account.total_return,
            'total_return_percent': self.account.total_return_percent,
            'available_cash': self.account.available_cash,
            'num_positions': len(self.account.positions),
            'open_positions': len(self.account.positions),  # Alias for backwards compatibility
            'total_trades': self.account.trade_count,
            'sharpe_ratio': self.account.sharpe_ratio,
            'starting_capital': self.account.starting_capital,
            'total_fees_paid': self.account.total_fees_paid,
            'total_funding_paid': self.account.total_funding_paid
        }

    def get_detailed_performance(self) -> Dict:
        """
        Get detailed performance metrics

        Returns:
            Comprehensive performance metrics including Sharpe ratio, win rate,
            max drawdown, profit factor, and more
        """
        return self.account.get_performance_metrics()

    def __repr__(self) -> str:
        """String representation"""
        return f"TradingEngine({self.account})"

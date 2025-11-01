"""
Position Management

Data structures for tracking trading positions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

from .performance import PerformanceTracker


@dataclass
class Position:
    """
    Represents a single trading position

    This matches the position format used in Alpha Arena prompts.
    """
    symbol: str                      # Coin symbol (BTC, ETH, etc.)
    quantity: float                  # Amount of coin
    entry_price: float               # Price at which position was opened
    current_price: float             # Current market price
    leverage: int                    # Leverage multiplier

    # Risk management
    stop_loss: float                 # Stop-loss price
    profit_target: float             # Take-profit price
    invalidation_condition: str      # When to force close

    # Position metadata
    confidence: float                # 0.0-1.0 confidence score
    risk_usd: float                  # Dollar amount at risk
    entry_time: datetime = field(default_factory=datetime.now)

    # Fee and funding tracking
    entry_fee: float = 0.0           # Fee paid on entry
    accumulated_funding: float = 0.0 # Accumulated funding costs
    last_funding_time: Optional[datetime] = None  # Last funding rate update

    # Order IDs (for live trading)
    entry_oid: int = -1
    sl_oid: int = -1              # Stop-loss order ID
    tp_oid: int = -1              # Take-profit order ID
    wait_for_fill: bool = False

    @property
    def liquidation_price(self) -> float:
        """
        Calculate liquidation price based on leverage

        For long positions:
        liquidation_price = entry_price * (1 - 1/leverage)

        Returns:
            Liquidation price
        """
        return self.entry_price * (1 - 1/self.leverage)

    @property
    def unrealized_pnl(self) -> float:
        """
        Calculate unrealized profit/loss (including fees and funding)

        Returns:
            P&L in USD (net of fees and funding costs)
        """
        price_diff = self.current_price - self.entry_price
        gross_pnl = price_diff * self.quantity * self.leverage

        # Subtract entry fee and accumulated funding costs
        net_pnl = gross_pnl - self.entry_fee - self.accumulated_funding
        return net_pnl

    @property
    def notional_usd(self) -> float:
        """
        Calculate notional position value

        Returns:
            Position value in USD
        """
        return self.current_price * self.quantity

    @property
    def exit_plan(self) -> Dict[str, Any]:
        """
        Get exit plan as dictionary (for prompt formatting)

        Returns:
            Exit plan dict
        """
        return {
            'profit_target': self.profit_target,
            'stop_loss': self.stop_loss,
            'invalidation_condition': self.invalidation_condition
        }

    def update_price(self, new_price: float):
        """
        Update current price

        Args:
            new_price: New market price
        """
        self.current_price = new_price

    def apply_funding_cost(self, funding_rate: float, current_time: Optional[datetime] = None):
        """
        Apply funding rate cost to position

        Funding is typically charged every 8 hours. We calculate the cost based on
        how much time has elapsed since last funding charge.

        Args:
            funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)
            current_time: Current timestamp (defaults to now)
        """
        if current_time is None:
            current_time = datetime.now()

        # Initialize last_funding_time on first call
        if self.last_funding_time is None:
            self.last_funding_time = self.entry_time

        # Calculate hours since last funding charge
        hours_elapsed = (current_time - self.last_funding_time).total_seconds() / 3600

        # Funding is charged every 8 hours
        funding_periods = hours_elapsed / 8.0

        if funding_periods >= 1.0:
            # Calculate funding cost: notional_value * funding_rate * periods
            notional = self.current_price * self.quantity
            funding_cost = notional * funding_rate * funding_periods

            # Add to accumulated funding (positive = cost to us, negative = payment to us)
            self.accumulated_funding += funding_cost

            # Update last funding time
            self.last_funding_time = current_time

    def is_liquidated(self) -> bool:
        """
        Check if position is liquidated

        Returns:
            True if current price <= liquidation price
        """
        return self.current_price <= self.liquidation_price

    def is_stop_loss_hit(self) -> bool:
        """
        Check if stop-loss is hit

        Returns:
            True if current price <= stop-loss
        """
        return self.current_price <= self.stop_loss

    def is_profit_target_hit(self) -> bool:
        """
        Check if profit target is hit

        Returns:
            True if current price >= profit target
        """
        return self.current_price >= self.profit_target

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary (for prompt formatting)

        Returns:
            Position as dict matching Alpha Arena format
        """
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'liquidation_price': self.liquidation_price,
            'unrealized_pnl': self.unrealized_pnl,
            'leverage': self.leverage,
            'exit_plan': self.exit_plan,
            'confidence': self.confidence,
            'risk_usd': self.risk_usd,
            'sl_oid': self.sl_oid,
            'tp_oid': self.tp_oid,
            'wait_for_fill': self.wait_for_fill,
            'entry_oid': self.entry_oid,
            'notional_usd': self.notional_usd,
            'entry_fee': self.entry_fee,
            'accumulated_funding': self.accumulated_funding
        }

    def __repr__(self) -> str:
        """String representation"""
        return (f"Position({self.symbol}, qty={self.quantity:.4f}, "
                f"entry=${self.entry_price:.2f}, current=${self.current_price:.2f}, "
                f"pnl=${self.unrealized_pnl:.2f}, {self.leverage}x)")


@dataclass
class Account:
    """
    Represents trading account state

    Tracks cash, positions, and performance metrics.
    """
    starting_capital: float
    available_cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    # Performance tracking
    total_return_percent: float = 0.0
    sharpe_ratio: float = 0.0
    trade_count: int = 0

    # Fee tracking
    total_fees_paid: float = 0.0        # Total trading fees paid
    total_funding_paid: float = 0.0     # Total funding costs paid

    # Trade history
    closed_positions: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)

    # Performance tracker (initialized lazily)
    _performance_tracker: Optional[PerformanceTracker] = field(default=None, init=False, repr=False)

    @property
    def account_value(self) -> float:
        """
        Calculate total account value (cash + collateral + unrealized P&L)

        Returns:
            Total account value in USD
        """
        # Sum collateral locked in positions + their unrealized P&L
        position_value = sum(pos.capital_used + pos.unrealized_pnl for pos in self.positions.values())
        return self.available_cash + position_value

    @property
    def total_return(self) -> float:
        """
        Calculate total return

        Returns:
            Total return in USD
        """
        return self.account_value - self.starting_capital

    def update_return_percent(self):
        """Update total return percentage"""
        self.total_return_percent = (self.total_return / self.starting_capital) * 100

    def update_performance_metrics(self):
        """
        Update all performance metrics including Sharpe ratio

        This should be called after closing a position to recalculate metrics.
        """
        # Initialize tracker if needed
        if self._performance_tracker is None:
            self._performance_tracker = PerformanceTracker()

        # Update Sharpe ratio
        self.sharpe_ratio = self._performance_tracker.calculate_sharpe_ratio(
            trade_log=self.trade_log,
            starting_capital=self.starting_capital,
            current_value=self.account_value
        )

        # Update return percent
        self.update_return_percent()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics

        Returns:
            Dict with all performance metrics
        """
        # Initialize tracker if needed
        if self._performance_tracker is None:
            self._performance_tracker = PerformanceTracker()

        return self._performance_tracker.calculate_all_metrics(
            trade_log=self.trade_log,
            starting_capital=self.starting_capital,
            current_value=self.account_value,
            total_fees_paid=self.total_fees_paid,
            total_funding_paid=self.total_funding_paid
        )

    def has_position(self, symbol: str) -> bool:
        """
        Check if account has an open position for a symbol

        Args:
            symbol: Coin symbol

        Returns:
            True if position exists
        """
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol

        Args:
            symbol: Coin symbol

        Returns:
            Position or None
        """
        return self.positions.get(symbol)

    def add_position(self, position: Position):
        """
        Add a new position

        Args:
            position: Position object
        """
        self.positions[position.symbol] = position
        self.trade_count += 1

    def remove_position(self, symbol: str) -> Optional[Position]:
        """
        Remove and return a position

        Args:
            symbol: Coin symbol

        Returns:
            Removed position or None
        """
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            self.closed_positions.append(pos)
            return pos
        return None

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions

        Args:
            prices: Dict mapping symbol to current price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

        # Update return after price update
        self.update_return_percent()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert account to dictionary (for prompt formatting)

        Returns:
            Account dict matching Alpha Arena format
        """
        return {
            'total_return_percent': self.total_return_percent,
            'available_cash': self.available_cash,
            'account_value': self.account_value,
            'positions': [pos.to_dict() for pos in self.positions.values()],
            'sharpe_ratio': self.sharpe_ratio
        }

    def __repr__(self) -> str:
        """String representation"""
        return (f"Account(value=${self.account_value:.2f}, "
                f"cash=${self.available_cash:.2f}, "
                f"return={self.total_return_percent:.2f}%, "
                f"positions={len(self.positions)})")

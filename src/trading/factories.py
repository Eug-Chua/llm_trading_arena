"""
Factory Functions for Trading Components

Provides dependency injection interfaces for creating trading objects.
Uses Protocol for type checking without inheritance.
"""

from typing import Protocol
from .position import Position, Account


class AccountFactory(Protocol):
    """Protocol for creating Account instances"""

    def __call__(self, starting_capital: float) -> Account:
        """
        Create an Account instance

        Args:
            starting_capital: Initial capital in USD

        Returns:
            Account instance
        """
        ...


class PositionFactory(Protocol):
    """Protocol for creating Position instances"""

    def __call__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        leverage: int,
        stop_loss: float,
        profit_target: float,
        invalidation_condition: str,
        confidence: float,
        risk_usd: float,
        entry_fee: float
    ) -> Position:
        """
        Create a Position instance

        Args:
            symbol: Coin symbol (BTC, ETH, etc.)
            quantity: Amount of coin
            entry_price: Price at which position was opened
            current_price: Current market price
            leverage: Leverage multiplier
            stop_loss: Stop-loss price
            profit_target: Take-profit price
            invalidation_condition: When to force close
            confidence: 0.0-1.0 confidence score
            risk_usd: Dollar amount at risk
            entry_fee: Fee paid on entry

        Returns:
            Position instance
        """
        ...


def default_account_factory(starting_capital: float) -> Account:
    """
    Default factory for creating Account instances

    Args:
        starting_capital: Initial capital in USD

    Returns:
        Account instance with available_cash = starting_capital
    """
    return Account(
        starting_capital=starting_capital,
        available_cash=starting_capital
    )


def default_position_factory(
    symbol: str,
    quantity: float,
    entry_price: float,
    current_price: float,
    leverage: int,
    stop_loss: float,
    profit_target: float,
    invalidation_condition: str,
    confidence: float,
    risk_usd: float,
    entry_fee: float
) -> Position:
    """
    Default factory for creating Position instances

    Args:
        symbol: Coin symbol (BTC, ETH, etc.)
        quantity: Amount of coin
        entry_price: Price at which position was opened
        current_price: Current market price
        leverage: Leverage multiplier
        stop_loss: Stop-loss price
        profit_target: Take-profit price
        invalidation_condition: When to force close
        confidence: 0.0-1.0 confidence score
        risk_usd: Dollar amount at risk
        entry_fee: Fee paid on entry

    Returns:
        Position instance
    """
    return Position(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        current_price=current_price,
        leverage=leverage,
        stop_loss=stop_loss,
        profit_target=profit_target,
        invalidation_condition=invalidation_condition,
        confidence=confidence,
        risk_usd=risk_usd,
        entry_fee=entry_fee
    )

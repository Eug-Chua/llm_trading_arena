"""
Performance Metrics Module

Calculates risk-adjusted performance metrics for trading strategies.
Implements industry-standard metrics including Sharpe ratio, win rate, drawdown, etc.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceTracker:
    """
    Tracks and calculates trading performance metrics

    This class computes:
    - Sharpe Ratio: Risk-adjusted return metric
    - Win Rate: Percentage of profitable trades
    - Max Drawdown: Largest peak-to-trough decline
    - Average Trade P&L: Mean profit/loss per trade
    - Profit Factor: Gross profit / Gross loss
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance tracker

        Args:
            risk_free_rate: Annual risk-free rate (default: 0% for crypto)
        """
        self.risk_free_rate = risk_free_rate

        # Historical tracking
        self.equity_curve: List[float] = []
        self.equity_timestamps: List[datetime] = []
        self.returns: List[float] = []

    def calculate_sharpe_ratio(
        self,
        trade_log: List[Dict[str, Any]],
        starting_capital: float,
        current_value: Optional[float] = None
    ) -> float:
        """
        Calculate Sharpe ratio from trade log

        Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns

        For crypto trading, we use 0% risk-free rate.
        We calculate returns per trade (not daily) due to irregular trade timing.

        Args:
            trade_log: List of trade dictionaries with 'net_pnl' and 'timestamp'
            starting_capital: Initial capital
            current_value: Current account value (optional)

        Returns:
            Sharpe ratio (annualized)
        """
        if not trade_log:
            return 0.0

        # Extract closed trades only (trades with net_pnl)
        closed_trades = [t for t in trade_log if 'net_pnl' in t and t.get('action') == 'CLOSE']

        if len(closed_trades) < 2:
            # Need at least 2 trades to calculate std dev
            return 0.0

        # Calculate per-trade returns (as percentage of capital at that time)
        trade_returns = []

        for trade in closed_trades:
            pnl = trade['net_pnl']
            # Assume capital grows with each trade (cumulative)
            # For simplicity, we'll use starting capital as denominator
            trade_return = pnl / starting_capital
            trade_returns.append(trade_return)

        # Calculate mean and std dev of returns
        mean_return = np.mean(trade_returns)
        std_return = np.std(trade_returns, ddof=1)  # Sample std dev

        if std_return == 0:
            # No volatility = perfect strategy (or only 1 trade)
            return 0.0

        # Sharpe ratio per trade
        sharpe = (mean_return - self.risk_free_rate) / std_return

        # Annualize: Assume ~365 trading days per year for crypto
        # If we have N trades over T days, annualization factor = sqrt(365 / avg_days_per_trade)
        if len(closed_trades) >= 2:
            first_trade_time = closed_trades[0]['timestamp']
            last_trade_time = closed_trades[-1]['timestamp']

            days_elapsed = (last_trade_time - first_trade_time).total_seconds() / 86400

            if days_elapsed > 0:
                avg_days_per_trade = days_elapsed / len(closed_trades)
                annualization_factor = np.sqrt(365 / avg_days_per_trade)
                sharpe *= annualization_factor

        return sharpe

    def calculate_win_rate(self, trade_log: List[Dict[str, Any]]) -> float:
        """
        Calculate win rate (percentage of profitable trades)

        Args:
            trade_log: List of trade dictionaries

        Returns:
            Win rate as decimal (0.0 to 1.0)
        """
        closed_trades = [t for t in trade_log if 'net_pnl' in t and t.get('action') == 'CLOSE']

        if not closed_trades:
            return 0.0

        winning_trades = sum(1 for t in closed_trades if t['net_pnl'] > 0)
        return winning_trades / len(closed_trades)

    def calculate_max_drawdown(
        self,
        trade_log: List[Dict[str, Any]],
        starting_capital: float
    ) -> Dict[str, float]:
        """
        Calculate maximum drawdown

        Max Drawdown = (Trough Value - Peak Value) / Peak Value

        Args:
            trade_log: List of trade dictionaries
            starting_capital: Initial capital

        Returns:
            Dict with 'max_drawdown' (%), 'peak_value', 'trough_value'
        """
        if not trade_log:
            return {
                'max_drawdown': 0.0,
                'peak_value': starting_capital,
                'trough_value': starting_capital
            }

        # Build equity curve from trade log
        equity = starting_capital
        equity_curve = [starting_capital]

        for trade in trade_log:
            if trade.get('action') == 'CLOSE' and 'net_pnl' in trade:
                equity += trade['net_pnl']
                equity_curve.append(equity)

        # Calculate drawdown at each point
        peak = equity_curve[0]
        max_dd = 0.0
        peak_value = starting_capital
        trough_value = starting_capital

        for value in equity_curve:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak

            if drawdown > max_dd:
                max_dd = drawdown
                peak_value = peak
                trough_value = value

        return {
            'max_drawdown': max_dd * 100,  # Return as percentage
            'peak_value': peak_value,
            'trough_value': trough_value
        }

    def calculate_profit_factor(self, trade_log: List[Dict[str, Any]]) -> float:
        """
        Calculate profit factor

        Profit Factor = Gross Profit / Gross Loss

        A ratio > 1.0 means profitable overall.

        Args:
            trade_log: List of trade dictionaries

        Returns:
            Profit factor
        """
        closed_trades = [t for t in trade_log if 'net_pnl' in t and t.get('action') == 'CLOSE']

        if not closed_trades:
            return 0.0

        gross_profit = sum(t['net_pnl'] for t in closed_trades if t['net_pnl'] > 0)
        gross_loss = abs(sum(t['net_pnl'] for t in closed_trades if t['net_pnl'] < 0))

        if gross_loss == 0:
            # No losing trades = infinite profit factor (cap at 999)
            return 999.0 if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_average_trade_pnl(self, trade_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average trade P&L statistics

        Args:
            trade_log: List of trade dictionaries

        Returns:
            Dict with 'avg_pnl', 'avg_win', 'avg_loss'
        """
        closed_trades = [t for t in trade_log if 'net_pnl' in t and t.get('action') == 'CLOSE']

        if not closed_trades:
            return {
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }

        winning_trades = [t['net_pnl'] for t in closed_trades if t['net_pnl'] > 0]
        losing_trades = [t['net_pnl'] for t in closed_trades if t['net_pnl'] < 0]

        avg_pnl = np.mean([t['net_pnl'] for t in closed_trades])
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0

        return {
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def calculate_all_metrics(
        self,
        trade_log: List[Dict[str, Any]],
        starting_capital: float,
        current_value: float,
        total_fees_paid: float = 0.0,
        total_funding_paid: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics at once

        Args:
            trade_log: List of trade dictionaries
            starting_capital: Initial capital
            current_value: Current account value
            total_fees_paid: Total trading fees paid
            total_funding_paid: Total funding costs paid

        Returns:
            Dict with all performance metrics
        """
        sharpe = self.calculate_sharpe_ratio(trade_log, starting_capital, current_value)
        win_rate = self.calculate_win_rate(trade_log)
        max_dd = self.calculate_max_drawdown(trade_log, starting_capital)
        profit_factor = self.calculate_profit_factor(trade_log)
        avg_pnl = self.calculate_average_trade_pnl(trade_log)

        # Count trades
        closed_trades = [t for t in trade_log if 'net_pnl' in t and t.get('action') == 'CLOSE']
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t['net_pnl'] > 0)
        losing_trades = sum(1 for t in closed_trades if t['net_pnl'] < 0)

        # Calculate total return
        total_return = current_value - starting_capital
        total_return_pct = (total_return / starting_capital) * 100

        # Calculate net return after fees
        net_return = total_return - total_fees_paid - total_funding_paid
        net_return_pct = (net_return / starting_capital) * 100

        metrics = {
            # Core metrics
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,

            # Returns
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'net_return': net_return,
            'net_return_pct': net_return_pct,

            # Drawdown
            'max_drawdown_pct': max_dd['max_drawdown'],
            'peak_value': max_dd['peak_value'],
            'trough_value': max_dd['trough_value'],

            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade_pnl': avg_pnl['avg_pnl'],
            'avg_winning_trade': avg_pnl['avg_win'],
            'avg_losing_trade': avg_pnl['avg_loss'],

            # Costs
            'total_fees_paid': total_fees_paid,
            'total_funding_paid': total_funding_paid,
            'total_costs': total_fees_paid + total_funding_paid,

            # Account state
            'starting_capital': starting_capital,
            'current_value': current_value
        }

        # Log summary
        logger.info(
            f"Performance: Sharpe={sharpe:.2f}, Win Rate={win_rate*100:.1f}%, "
            f"Return={total_return_pct:+.2f}%, Max DD={max_dd['max_drawdown']:.2f}%, "
            f"Trades={total_trades}"
        )

        return metrics

    def update_equity_curve(self, current_value: float, timestamp: Optional[datetime] = None):
        """
        Add a data point to the equity curve

        Args:
            current_value: Current account value
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.equity_curve.append(current_value)
        self.equity_timestamps.append(timestamp)

    def get_equity_curve_data(self) -> Dict[str, List]:
        """
        Get equity curve data for plotting

        Returns:
            Dict with 'timestamps' and 'values' lists
        """
        return {
            'timestamps': self.equity_timestamps,
            'values': self.equity_curve
        }

    def log_performance_summary(self, metrics: Dict[str, Any]):
        """
        Log formatted performance summary

        Args:
            metrics: Metrics dict from calculate_all_metrics()
        """
        logger.info("=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)

        logger.info("RISK-ADJUSTED METRICS")
        logger.info(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
        logger.info(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>7.2f}%")
        logger.info(f"  Win Rate:            {metrics['win_rate']*100:>7.2f}%")
        logger.info(f"  Profit Factor:       {metrics['profit_factor']:>8.2f}")

        logger.info("RETURNS")
        logger.info(f"  Starting Capital:    ${metrics['starting_capital']:>11,.2f}")
        logger.info(f"  Current Value:       ${metrics['current_value']:>11,.2f}")
        logger.info(f"  Gross Return:        ${metrics['total_return']:>11,.2f} ({metrics['total_return_pct']:>+6.2f}%)")
        logger.info(f"  Net Return:          ${metrics['net_return']:>11,.2f} ({metrics['net_return_pct']:>+6.2f}%)")

        logger.info("TRADE STATISTICS")
        logger.info(f"  Total Trades:        {metrics['total_trades']:>8}")
        logger.info(f"  Winning Trades:      {metrics['winning_trades']:>8}")
        logger.info(f"  Losing Trades:       {metrics['losing_trades']:>8}")
        logger.info(f"  Avg Trade P&L:       ${metrics['avg_trade_pnl']:>10,.2f}")
        logger.info(f"  Avg Winning Trade:   ${metrics['avg_winning_trade']:>10,.2f}")
        logger.info(f"  Avg Losing Trade:    ${metrics['avg_losing_trade']:>10,.2f}")

        logger.info("COSTS")
        logger.info(f"  Trading Fees:        ${metrics['total_fees_paid']:>11,.2f}")
        logger.info(f"  Funding Costs:       ${metrics['total_funding_paid']:>11,.2f}")
        logger.info(f"  Total Costs:         ${metrics['total_costs']:>11,.2f}")

        logger.info("=" * 80)

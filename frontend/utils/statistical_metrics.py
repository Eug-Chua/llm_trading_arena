"""
Statistical metrics extraction and calculation utilities.
Handles extraction of trading metrics from checkpoints and statistical significance testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats


def extract_metrics(checkpoint: Dict) -> Dict:
    """
    Extract comprehensive trading metrics from a checkpoint.

    Calculates performance metrics, risk metrics, and distribution statistics
    from a checkpoint's trade history and account data.

    Args:
        checkpoint: Checkpoint dictionary containing account, metadata, and trade_history

    Returns:
        Dictionary containing extracted metrics including:
        - Performance: total return, Sharpe ratio, max drawdown, win rate
        - Trade stats: trade sizes, hold times, leverage usage
        - Risk metrics: Sortino ratio, up/downside deviation, skewness, kurtosis
    """
    account = checkpoint['account']
    metadata = checkpoint.get('metadata', {})
    trade_log = checkpoint.get('trade_history', [])

    # Calculate win rate from trade history
    closed_trades = [t for t in trade_log if t['action'] == 'CLOSE']
    winning_trades = [t for t in closed_trades if t.get('net_pnl', 0) > 0]
    losing_trades = [t for t in closed_trades if t.get('net_pnl', 0) < 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0

    # Calculate max drawdown from trade history
    equity_curve = [account.get('starting_capital', 10000)]
    current_value = account.get('starting_capital', 10000)

    for trade in trade_log:
        if trade['action'] == 'CLOSE' and 'net_pnl' in trade:
            current_value += trade['net_pnl']
            equity_curve.append(current_value)

    # Calculate max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Advanced metrics
    biggest_win = max([t.get('net_pnl', 0) for t in winning_trades]) if winning_trades else 0
    biggest_loss = min([t.get('net_pnl', 0) for t in losing_trades]) if losing_trades else 0

    # Use gross PnL (before fees) so reconciliation works: Final Value = Starting Capital + Gross PnL - Fees
    total_pnl = sum([t.get('gross_pnl', 0) for t in closed_trades])

    # Trade sizes (notional value) and P&Ls
    trade_sizes = []
    hold_times = []
    leverages = []
    long_count = 0
    trade_pnls = []  # Collect all trade P&Ls for distribution analysis

    for trade in closed_trades:
        # Collect P&L
        trade_pnls.append(trade.get('net_pnl', 0))

        # Find corresponding BUY trade
        symbol = trade.get('symbol')
        close_time = trade.get('timestamp')

        # Search backwards for the BUY
        for buy_trade in reversed([t for t in trade_log if t['action'] == 'BUY']):
            if buy_trade.get('symbol') == symbol and buy_trade.get('timestamp') < close_time:
                # Calculate trade size (notional)
                quantity = buy_trade.get('quantity', 0)
                price = buy_trade.get('price', 0)
                trade_sizes.append(quantity * price)

                # Hold time
                time_diff = close_time - buy_trade.get('timestamp')
                hold_hours = time_diff.total_seconds() / 3600
                hold_times.append(hold_hours)

                # Leverage
                leverages.append(buy_trade.get('leverage', 1))

                # Count longs (assume all are long for now)
                long_count += 1
                break

    avg_trade_size = np.mean(trade_sizes) if trade_sizes else 0
    median_trade_size = np.median(trade_sizes) if trade_sizes else 0
    avg_hold = np.mean(hold_times) if hold_times else 0
    median_hold = np.median(hold_times) if hold_times else 0
    pct_long = (long_count / len(closed_trades) * 100) if closed_trades else 0
    expected_value = total_pnl / len(closed_trades) if closed_trades else 0
    avg_leverage = np.mean(leverages) if leverages else 1
    median_leverage = np.median(leverages) if leverages else 1

    # Advanced risk metrics
    # Downside deviation (volatility of negative returns only)
    negative_pnls = [pnl for pnl in trade_pnls if pnl < 0]
    downside_deviation = np.std(negative_pnls) if len(negative_pnls) > 1 else 0

    # Upside deviation (volatility of positive returns only)
    positive_pnls = [pnl for pnl in trade_pnls if pnl > 0]
    upside_deviation = np.std(positive_pnls) if len(positive_pnls) > 1 else 0

    # Sortino ratio (return / downside deviation)
    sortino_ratio = (total_pnl / downside_deviation) if downside_deviation > 0 else 0

    # Upside/Downside capture ratio
    upside_capture = sum(positive_pnls) if positive_pnls else 0
    downside_capture = abs(sum(negative_pnls)) if negative_pnls else 0
    upside_downside_ratio = (upside_capture / downside_capture) if downside_capture > 0 else 0

    # Skewness of trade returns
    skewness = stats.skew(trade_pnls) if len(trade_pnls) > 2 else 0

    # Kurtosis of trade returns
    kurtosis = stats.kurtosis(trade_pnls) if len(trade_pnls) > 2 else 0

    return {
        'model': metadata.get('model', 'unknown'),
        'temperature': metadata.get('temperature', 0.7),
        'run_id': metadata.get('run_id', 0),
        # Overall stats
        'total_return_pct': account.get('total_return_percent', 0),
        'sharpe_ratio': account.get('sharpe_ratio', 0),
        'max_drawdown_pct': max_dd,
        'win_rate': win_rate,
        'total_trades': len(closed_trades),
        'total_fees': account.get('total_fees_paid', 0),
        'final_value': account.get('account_value', 0),
        'starting_capital': account.get('starting_capital', 10000),
        'total_pnl': total_pnl,
        'biggest_win': biggest_win,
        'biggest_loss': biggest_loss,
        # Advanced stats
        'avg_trade_size': avg_trade_size,
        'median_trade_size': median_trade_size,
        'avg_hold_hours': avg_hold,
        'median_hold_hours': median_hold,
        'pct_long': pct_long,
        'expected_value': expected_value,
        'avg_leverage': avg_leverage,
        'median_leverage': median_leverage,
        # Risk & distribution metrics
        'downside_deviation': downside_deviation,
        'upside_deviation': upside_deviation,
        'sortino_ratio': sortino_ratio,
        'upside_downside_ratio': upside_downside_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
        # Keep equity curve for visualization
        'equity_curve': equity_curve,
        # Keep trade PnLs for distribution analysis
        'trade_pnls': trade_pnls,
    }


def group_by_model_config(checkpoints_data: Dict) -> Dict:
    """
    Group checkpoints by model configuration (model + temperature).

    Args:
        checkpoints_data: Dictionary mapping checkpoint paths to checkpoint data

    Returns:
        Dictionary mapping config names (e.g., "anthropic_temp0.7") to lists of trial metrics
    """
    grouped = {}

    for checkpoint_path, checkpoint_data in checkpoints_data.items():
        # Extract metrics
        metrics = extract_metrics(checkpoint_data)
        config_name = f"{metrics['model']}_temp{metrics['temperature']}"

        if config_name not in grouped:
            grouped[config_name] = []

        # Add checkpoint path for reference
        metrics['checkpoint_path'] = checkpoint_path
        grouped[config_name].append(metrics)

    return grouped


def calculate_statistical_significance(trials: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate statistical significance tests for key metrics across trials.

    Performs one-sample t-tests to determine if mean metrics are significantly
    different from zero, and calculates 95% confidence intervals.

    Args:
        trials: List of trial metric dictionaries

    Returns:
        Dictionary mapping metric names to statistical test results including:
        - t_statistic: The t-test statistic
        - p_value: Probability of observing this result by chance
        - mean: Mean value across trials
        - std_dev: Standard deviation
        - std_error: Standard error
        - confidence_interval: 95% CI tuple (lower, upper)
        - is_significant: Boolean indicating if p < 0.05
    """
    if len(trials) < 2:
        return {}

    metrics_to_test = {
        'total_return_pct': [t['total_return_pct'] for t in trials],
        'sharpe_ratio': [t['sharpe_ratio'] for t in trials],
        'win_rate': [t['win_rate'] for t in trials],
        'max_drawdown_pct': [t['max_drawdown_pct'] for t in trials],
        'sortino_ratio': [t['sortino_ratio'] for t in trials],
    }

    results = {}

    for metric_name, values in metrics_to_test.items():
        if len(values) < 2:
            continue

        # One-sample t-test (testing if mean is different from 0)
        t_stat, p_value = stats.ttest_1samp(values, 0)

        # Calculate confidence interval
        mean = np.mean(values)
        std_err = stats.sem(values)
        confidence_level = 0.95
        degrees_freedom = len(values) - 1
        confidence_interval = stats.t.interval(
            confidence_level,
            degrees_freedom,
            loc=mean,
            scale=std_err
        )

        results[metric_name] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean': mean,
            'std_dev': np.std(values),
            'std_error': std_err,
            'confidence_interval': confidence_interval,
            'is_significant': p_value < 0.05,
        }

    return results

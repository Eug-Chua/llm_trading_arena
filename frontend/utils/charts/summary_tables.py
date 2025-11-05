"""
Summary Statistics Tables

This module contains functions for creating summary statistics tables:
- Overall stats table (returns, PnL, win rate, Sharpe, etc.)
- Advanced stats table (trade sizing, hold times, leverage, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def create_overall_stats_table(trials: List[Dict]) -> pd.DataFrame:
    """
    Create Overall Stats table for individual trials (with aggregate row).
    Returns numeric values for proper sorting - formatting should be done in display.
    """
    summary_rows = []

    # Individual trial rows
    for trial in trials:
        # Calculate std dev as % of starting capital
        std_dev_pct = (trial['overall_std_dev'] / trial['starting_capital']) * 100

        summary_rows.append({
            'Trial': trial['run_id'],
            'Acc. Value': trial['final_value'],
            'Return %': trial['total_return_pct'],
            'Total PnL': trial['total_pnl'],
            'Fees': trial['total_fees'],
            'Win Rate': trial['win_rate'],
            'Biggest Win': trial['biggest_win'],
            'Biggest Loss': trial['biggest_loss'],
            'Sharpe': trial['sharpe_ratio'],
            'Std Dev %': std_dev_pct,
            'No of Trades': trial['total_trades']
        })

    # Aggregate row - use string for Trial column to distinguish it
    # Calculate average std dev %
    avg_std_dev_pct = np.mean([(t['overall_std_dev'] / t['starting_capital']) * 100 for t in trials])

    summary_rows.append({
        'Trial': 999999,  # Large number to sort at bottom
        'Acc. Value': np.mean([t['final_value'] for t in trials]),
        'Return %': np.mean([t['total_return_pct'] for t in trials]),
        'Total PnL': np.mean([t['total_pnl'] for t in trials]),
        'Fees': np.mean([t['total_fees'] for t in trials]),
        'Win Rate': np.mean([t['win_rate'] for t in trials]),
        'Biggest Win': np.mean([t['biggest_win'] for t in trials]),
        'Biggest Loss': np.mean([t['biggest_loss'] for t in trials]),
        'Sharpe': np.mean([t['sharpe_ratio'] for t in trials]),
        'Std Dev %': avg_std_dev_pct,
        'No of Trades': int(np.mean([t['total_trades'] for t in trials]))
    })

    df = pd.DataFrame(summary_rows)

    return df


def create_advanced_stats_table(trials: List[Dict]) -> pd.DataFrame:
    """
    Create Advanced Stats table for individual trials (with aggregate row).
    Returns numeric values for proper sorting - formatting should be done in display.
    """
    summary_rows = []

    # Individual trial rows
    for trial in trials:
        summary_rows.append({
            'Trial': trial['run_id'],
            'Acc. Value': trial['final_value'],
            'Avg Trade Size': trial['avg_trade_size'],
            'Median Trade Size': trial['median_trade_size'],
            'Avg Hold (hrs)': trial['avg_hold_hours'],
            'Median Hold (hrs)': trial['median_hold_hours'],
            '% Long': trial['pct_long'],
            'Expected Value': trial['expected_value'],
            'Median Leverage': trial['median_leverage'],
            'Avg Leverage': trial['avg_leverage'],
        })

    # Aggregate row
    summary_rows.append({
        'Trial': 999999,  # Large number to sort at bottom
        'Acc. Value': np.mean([t['final_value'] for t in trials]),
        'Avg Trade Size': np.mean([t['avg_trade_size'] for t in trials]),
        'Median Trade Size': np.mean([t['median_trade_size'] for t in trials]),
        'Avg Hold (hrs)': np.mean([t['avg_hold_hours'] for t in trials]),
        'Median Hold (hrs)': np.mean([t['median_hold_hours'] for t in trials]),
        '% Long': np.mean([t['pct_long'] for t in trials]),
        'Expected Value': np.mean([t['expected_value'] for t in trials]),
        'Median Leverage': np.mean([t['median_leverage'] for t in trials]),
        'Avg Leverage': np.mean([t['avg_leverage'] for t in trials]),
    })

    df = pd.DataFrame(summary_rows)

    return df

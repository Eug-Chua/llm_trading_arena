"""
Coin-level analysis utilities

Calculate per-coin performance metrics aggregated across trials
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def calculate_per_coin_stats(trials: List[Dict]) -> pd.DataFrame:
    """
    Calculate aggregated per-coin statistics across all trials

    For each coin, calculates:
    - Total trades across all trials
    - Total P&L ($)
    - Win rate (%)
    - Average P&L per trade
    - Best single trade
    - Worst single trade

    Args:
        trials: List of trial metadata dictionaries (must have 'checkpoint_path' key)

    Returns:
        DataFrame with per-coin statistics
    """
    import streamlit as st

    coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']

    # Aggregate data across all trials
    coin_data = {coin: {
        'trades': [],
        'pnls': [],
    } for coin in coins}

    for trial in trials:
        # Get the actual checkpoint from session state
        checkpoint_path = trial.get('checkpoint_path')
        if not checkpoint_path:
            continue

        checkpoint = st.session_state.get('loaded_checkpoints', {}).get(checkpoint_path)
        if not checkpoint:
            continue

        trade_log = checkpoint.get('trade_history', [])

        # Group trades by coin
        for trade in trade_log:
            if trade['action'] == 'CLOSE':
                coin = trade.get('symbol')
                if coin in coin_data:
                    pnl = trade.get('net_pnl', 0)
                    coin_data[coin]['trades'].append(trade)
                    coin_data[coin]['pnls'].append(pnl)

    # Calculate summary statistics
    results = []

    for coin in coins:
        trades = coin_data[coin]['trades']
        pnls = coin_data[coin]['pnls']

        if not trades:
            results.append({
                'Coin': coin,
                'Trades': 0,
                'Total P&L': 0,
                'Win Rate': 0,
                'Avg P&L/Trade': 0,
                'Best Trade': 0,
                'Worst Trade': 0,
            })
            continue

        num_trades = len(trades)
        total_pnl = sum(pnls)
        winning_trades = [p for p in pnls if p > 0]
        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
        avg_pnl = total_pnl / num_trades if num_trades > 0 else 0
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0

        results.append({
            'Coin': coin,
            'Trades': num_trades,
            'Total P&L': total_pnl,
            'Win Rate': win_rate,
            'Avg P&L/Trade': avg_pnl,
            'Best Trade': best_trade,
            'Worst Trade': worst_trade,
        })

    # Create DataFrame and sort by Total P&L (descending)
    df = pd.DataFrame(results)
    df = df.sort_values('Total P&L', ascending=False).reset_index(drop=True)

    return df


def calculate_per_coin_capture_ratios_summary(trials: List[Dict], coins: List[str]) -> pd.DataFrame:
    """
    Calculate average capture ratios per coin across all trials

    Args:
        trials: List of trial checkpoint data
        coins: List of coin symbols

    Returns:
        DataFrame with average upside/downside capture per coin
    """
    from frontend.utils.capture_ratios import calculate_coin_capture_ratios
    import streamlit as st

    # Collect capture ratios across trials
    coin_capture_data = {coin: {
        'upside_captures': [],
        'downside_captures': []
    } for coin in coins}

    for trial in trials:
        # Get checkpoint from session state
        checkpoint_path = trial.get('checkpoint_path')
        if not checkpoint_path:
            continue

        checkpoint = st.session_state.get('loaded_checkpoints', {}).get(checkpoint_path)
        if not checkpoint:
            continue

        # Calculate capture ratios for this trial
        capture_ratios = calculate_coin_capture_ratios(checkpoint, coins)

        for coin in coins:
            if coin in capture_ratios:
                coin_capture_data[coin]['upside_captures'].append(
                    capture_ratios[coin]['upside_capture']
                )
                coin_capture_data[coin]['downside_captures'].append(
                    capture_ratios[coin]['downside_capture']
                )

    # Calculate averages
    results = []

    for coin in coins:
        upside_list = coin_capture_data[coin]['upside_captures']
        downside_list = coin_capture_data[coin]['downside_captures']

        if not upside_list:
            continue

        avg_upside = np.mean(upside_list)
        avg_downside = np.mean(downside_list)
        std_upside = np.std(upside_list) if len(upside_list) > 1 else 0
        std_downside = np.std(downside_list) if len(downside_list) > 1 else 0

        results.append({
            'Coin': coin,
            'Avg Upside Capture': avg_upside,
            'Avg Downside Capture': avg_downside,
            'Upside Std Dev': std_upside,
            'Downside Std Dev': std_downside,
            'Trials': len(upside_list)
        })

    df = pd.DataFrame(results)
    return df

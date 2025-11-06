"""
Equity Curve Visualizations

This module contains chart creation functions for equity curve analysis:
- Overlaid equity curves showing trial variance ("fan" chart)
- Account value progression over time
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict


def create_equity_curve_overlay(grouped_data: Dict, config_name: str):
    """
    Create overlaid equity curves for all trials of a single configuration

    This shows the "fan" of possible outcomes from LLM non-determinism

    Args:
        grouped_data: Dictionary mapping config names to lists of trial metadata
        config_name: Name of the configuration to visualize
    """
    trials = grouped_data.get(config_name, [])

    if not trials:
        return None

    fig = go.Figure()

    # Use same color palette as Trade P&L KDE (Set2)
    colors = px.colors.qualitative.Set2

    # Load trade history for each trial
    for idx, trial_meta in enumerate(trials):
        # Get checkpoint using the stored checkpoint_path
        checkpoint_path = trial_meta.get('checkpoint_path')
        if not checkpoint_path:
            continue

        # The checkpoint_path is the stem (filename without extension)
        # Find it in loaded_checkpoints
        checkpoint = st.session_state.get('loaded_checkpoints', {}).get(checkpoint_path)
        if not checkpoint:
            continue

        trade_log = checkpoint.get('trade_history', [])

        # Build equity curve
        timestamps = []
        account_values = []
        starting_capital = trial_meta['starting_capital']

        if trade_log:
            # Add starting point
            first_trade_time = trade_log[0].get('timestamp')
            timestamps.append(first_trade_time)
            account_values.append(starting_capital)

            # Calculate running account value
            current_value = starting_capital
            for trade in trade_log:
                if trade['action'] == 'CLOSE' and 'net_pnl' in trade:
                    current_value += trade['net_pnl']
                    timestamps.append(trade['timestamp'])
                    account_values.append(current_value)
                elif trade['action'] == 'BUY':
                    timestamps.append(trade['timestamp'])
                    account_values.append(current_value)

            # Add final point
            if timestamps:
                timestamps.append(trade_log[-1]['timestamp'])
                account_values.append(trial_meta['final_value'])

        # Add trace for this trial (using Set2 color palette)
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=account_values,
            mode='lines+markers',
            name=f'Trial {trial_meta["run_id"]}',
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            hovertemplate=(
                f"<b>Trial {trial_meta['run_id']}</b><br>" +
                "Time: %{x}<br>" +
                "Value: $%{y:,.2f}<br>" +
                "<extra></extra>"
            )
        ))

    # Add baseline
    fig.add_hline(
        y=10000,
        line_dash='dot',
        line_color='gray',
        opacity=0.6,
        annotation_text='Starting Capital ($10,000)',
        annotation_position='right'
    )

    # Layout
    fig.update_layout(
        title=f'{config_name.replace("_", " ").title()} - All Trials Overlay',
        xaxis_title='Date',
        yaxis_title='Account Value (USD)',
        template='plotly_dark',
        height=600,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

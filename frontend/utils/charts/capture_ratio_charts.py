"""
Capture Ratio Visualizations

This module contains chart creation functions for capture ratio analysis:
- Upside/downside capture ratios by coin
- Market timing performance visualization
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List

# Import from parent utils folder
from frontend.utils.capture_ratios import calculate_coin_capture_ratios


def create_capture_ratios_chart(trials: List[Dict], coins: List[str], extract_metrics_fn=None):
    """
    Create visualization showing upside/downside capture ratios per coin per trial

    Each trial shows 6 pairs of bars (one per coin: BTC, ETH, SOL, BNB, XRP, DOGE)
    Green bars = upside capture, Red bars = downside capture

    Args:
        trials: List of trial metadata dictionaries
        coins: List of coin symbols to analyze
        extract_metrics_fn: Function to extract metrics from checkpoint (optional, for compatibility)
    """
    # Calculate capture ratios for each trial
    trial_capture_data = []

    for trial in trials:
        # Find the checkpoint for this trial
        checkpoint_name = None
        for name, checkpoint in st.session_state.get('loaded_checkpoints', {}).items():
            # Use the passed function or import dynamically
            if extract_metrics_fn:
                meta = extract_metrics_fn(checkpoint)
            else:
                from frontend.utils.statistical_metrics import extract_metrics
                meta = extract_metrics(checkpoint)
            if (meta['model'] == trial['model'] and
                meta['temperature'] == trial['temperature'] and
                meta['run_id'] == trial['run_id']):
                checkpoint_name = name
                break

        if not checkpoint_name:
            continue

        checkpoint = st.session_state['loaded_checkpoints'][checkpoint_name]

        # Calculate capture ratios for this trial
        capture_ratios = calculate_coin_capture_ratios(checkpoint, coins)

        trial_capture_data.append({
            'trial_id': trial['run_id'],
            'capture_ratios': capture_ratios
        })

    if not trial_capture_data:
        return None

    # Create subplots for each coin
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'{coin}' for coin in coins],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for idx, coin in enumerate(coins):
        row = (idx // 3) + 1
        col = (idx % 3) + 1

        trial_ids = []
        upside_captures = []
        downside_captures = []

        for trial_data in trial_capture_data:
            if coin in trial_data['capture_ratios']:
                trial_ids.append(trial_data['trial_id'])
                upside_captures.append(trial_data['capture_ratios'][coin]['upside_capture'])
                downside_captures.append(trial_data['capture_ratios'][coin]['downside_capture'])

        if not trial_ids:
            continue

        # Add upside capture bars (green)
        fig.add_trace(
            go.Bar(
                x=trial_ids,
                y=upside_captures,
                name=f'{coin} Upside',
                marker_color='#00ff88',
                showlegend=(idx == 0),
                legendgroup='upside',
                hovertemplate=f'{coin}<br>Trial %{{x}}<br>Upside Capture: %{{y:.1f}}%<extra></extra>'
            ),
            row=row, col=col
        )

        # Add downside capture bars (red, as negative for visual separation)
        fig.add_trace(
            go.Bar(
                x=trial_ids,
                y=[-d for d in downside_captures],
                name=f'{coin} Downside',
                marker_color='#ff4444',
                showlegend=(idx == 0),
                legendgroup='downside',
                customdata=downside_captures,
                hovertemplate=f'{coin}<br>Trial %{{x}}<br>Downside Capture: %{{customdata:.1f}}%<extra></extra>'
            ),
            row=row, col=col
        )

        # Add reference lines at 100% and -100%
        fig.add_hline(
            y=100, line_dash='dot', line_color='white', opacity=0.3,
            row=row, col=col
        )
        fig.add_hline(
            y=-100, line_dash='dot', line_color='white', opacity=0.3,
            row=row, col=col
        )
        fig.add_hline(
            y=0, line_dash='solid', line_color='gray', opacity=0.5,
            row=row, col=col
        )

        # Update axes for this subplot
        fig.update_xaxes(
            title_text='Trial' if row == 2 else '',
            tickmode='linear',
            tick0=1,
            dtick=1,
            row=row, col=col
        )
        fig.update_yaxes(
            title_text='Capture %' if col == 1 else '',
            row=row, col=col
        )

    fig.update_layout(
        title='Capture Ratios by Coin and Trial',
        template='plotly_dark',
        height=700,
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title=dict(text='')
        )
    )

    return fig

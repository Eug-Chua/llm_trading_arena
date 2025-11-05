"""
Performance and KDE Distribution Charts

This module contains chart creation functions for performance analysis:
- Trial performance scatter plots
- Trade P&L KDE overlays
- Return distribution charts with std dev bands
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List


def create_trial_performance_chart(trials: List[Dict], metric: str, title: str, ylabel: str):
    """Create scatter plot showing each trial's performance for a metric"""

    fig = go.Figure()

    # Use Set2 color palette
    colors = px.colors.qualitative.Set2

    trial_ids = [trial['run_id'] for trial in trials]
    values = [trial[metric] for trial in trials]

    # Add scatter points (no line connection - trials are independent)
    fig.add_trace(go.Scatter(
        x=trial_ids,
        y=values,
        mode='markers',
        name='Trial Performance',
        marker=dict(
            size=14,
            color=[colors[i % len(colors)] for i in range(len(trials))],
            line=dict(color='white', width=2)
        ),
        hovertemplate='Trial %{x}<br>' + ylabel + ': %{y:.2f}<extra></extra>'
    ))

    # Add mean line
    mean_val = np.mean(values)
    fig.add_hline(
        y=mean_val,
        line_dash='dash',
        line_color='white',
        opacity=0.7,
        annotation_text=f'Mean: {mean_val:.2f}',
        annotation_position='right'
    )

    fig.update_layout(
        title=title,
        xaxis_title='Trial Number',
        yaxis_title=ylabel,
        template='plotly_dark',
        height=350,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )

    return fig


def create_trade_pnl_kde_overlay(trials: List[Dict], config_name: str, extract_metrics_fn=None):
    """
    Create overlaid KDE curves showing distribution of individual trade P&Ls for each trial

    Each trial gets its own KDE curve showing how its individual trade returns are distributed
    - Tight KDE = consistent trade sizing
    - Wide KDE = varied trade outcomes

    Args:
        trials: List of trial metadata dictionaries
        config_name: Configuration name for the chart title
        extract_metrics_fn: Function to extract metrics from checkpoint (optional, for compatibility)
    """
    from scipy import stats

    if not trials or len(trials) < 1:
        return None

    fig = go.Figure()

    # Color palette for different trials
    colors = px.colors.qualitative.Set2

    all_pnls = []

    for idx, trial_meta in enumerate(trials):
        # Find the checkpoint for this trial
        checkpoint = None
        for name, cp in st.session_state.get('loaded_checkpoints', {}).items():
            # Use the passed function or import dynamically
            if extract_metrics_fn:
                meta = extract_metrics_fn(cp)
            else:
                from frontend.utils.statistical_metrics import extract_metrics
                meta = extract_metrics(cp)
            if (meta['model'] == trial_meta['model'] and
                meta['temperature'] == trial_meta['temperature'] and
                meta['run_id'] == trial_meta['run_id']):
                checkpoint = cp
                break

        if not checkpoint:
            continue

        # Extract individual trade P&Ls
        trade_log = checkpoint.get('trade_history', [])
        trade_pnls = [t.get('net_pnl', 0) for t in trade_log if t['action'] == 'CLOSE' and 'net_pnl' in t]

        if len(trade_pnls) < 2:
            continue

        all_pnls.extend(trade_pnls)

        # Calculate KDE
        kde = stats.gaussian_kde(trade_pnls, bw_method='scott')

        # Create x-range
        data_min = min(trade_pnls)
        data_max = max(trade_pnls)
        data_range = data_max - data_min
        padding = data_range * 0.5 if data_range > 0 else 100
        x_min = data_min - padding
        x_max = data_max + padding

        x_range = np.linspace(x_min, x_max, 300)
        density = kde(x_range)

        # Add explicit endpoints
        x_range = np.concatenate([[x_min - 10], x_range, [x_max + 10]])
        density = np.concatenate([[0], density, [0]])

        # Color for this trial
        color = colors[idx % len(colors)]

        # Add KDE curve
        fig.add_trace(go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            name=f'Trial {trial_meta["run_id"]}',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.15)'),
            hovertemplate=f'Trial {trial_meta["run_id"]}<br>P&L: $%{{x:,.2f}}<br>Density: %{{y:.6f}}<extra></extra>'
        ))

        # Add mean marker
        mean_pnl = np.mean(trade_pnls)
        mean_density = kde(mean_pnl)[0]
        fig.add_trace(go.Scatter(
            x=[mean_pnl],
            y=[mean_density],
            mode='markers',
            name=f'Trial {trial_meta["run_id"]} Mean',
            marker=dict(size=8, color=color, symbol='diamond'),
            showlegend=False,
            hovertemplate=f'Trial {trial_meta["run_id"]} Mean: ${mean_pnl:,.2f}<extra></extra>'
        ))

    if not all_pnls:
        return None

    # Add vertical line at $0
    fig.add_vline(
        x=0,
        line_dash='dot',
        line_color='gray',
        opacity=0.6,
        annotation_text='$0 (Breakeven)',
        annotation_position='top'
    )

    # Center x-axis on 0
    abs_max = max(abs(min(all_pnls)), abs(max(all_pnls)))
    padding_amt = abs_max * 0.3
    x_range_limit = [-abs_max - padding_amt, abs_max + padding_amt]

    fig.update_layout(
        title=f'Distribution of Individual Trade P&Ls - {config_name}',
        xaxis_title='Trade P&L (USD)',
        yaxis_title='Probability Density',
        template='plotly_dark',
        height=500,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(range=x_range_limit, zeroline=True, tickprefix='$', tickformat=',.0f'),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig


def create_return_distribution_chart(trials: List[Dict]):
    """
    Create visualization showing return distribution with standard deviation bands

    Shows individual trial returns, mean, and ±1σ/±2σ bands to illustrate variance
    """
    if not trials:
        return None

    # Extract returns
    returns = [trial['total_return_pct'] for trial in trials]
    trial_ids = [trial['run_id'] for trial in trials]

    # Calculate statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    fig = go.Figure()

    # Add individual trial points
    colors = px.colors.qualitative.Set2
    fig.add_trace(go.Scatter(
        x=trial_ids,
        y=returns,
        mode='markers',
        name='Trial Returns',
        marker=dict(
            size=14,
            color=[colors[i % len(colors)] for i in range(len(trials))]
        ),
        hovertemplate='Trial %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))

    # Add mean line
    fig.add_hline(
        y=mean_return,
        line_dash='solid',
        line_color='white',
        line_width=2,
        opacity=0.7,
        annotation_text=f'Mean: {mean_return:.2f}%',
    )

    # Add ±1 std dev band (68% of trials should fall here)
    fig.add_hrect(
        y0=mean_return - std_return,
        y1=mean_return + std_return,
        fillcolor='rgba(100, 200, 100, 0.2)',
        line_width=0,
        annotation_text="±1σ (68%)",
        annotation_position="top"
    )

    # Add ±2 std dev band (95% of trials should fall here)
    fig.add_hrect(
        y0=mean_return - 2*std_return,
        y1=mean_return + 2*std_return,
        fillcolor='rgba(100, 150, 200, 0.1)',
        line_width=0,
        annotation_text="±2σ (95%)",
        annotation_position="top"
    )

    # Add reference lines for ±1σ and ±2σ
    fig.add_hline(y=mean_return + std_return, line_dash='dot', line_color='green', opacity=0.5)
    fig.add_hline(y=mean_return - std_return, line_dash='dot', line_color='green', opacity=0.5)
    fig.add_hline(y=mean_return + 2*std_return, line_dash='dot', line_color='blue', opacity=0.3)
    fig.add_hline(y=mean_return - 2*std_return, line_dash='dot', line_color='blue', opacity=0.3)

    # Add zero reference line
    fig.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.5,
                  annotation_text='Break-even', annotation_position='bottom')

    fig.update_layout(
        xaxis_title='Trial Number',
        yaxis_title='Total Return (%)',
        template='plotly_dark',
        height=450,
        hovermode='closest',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        showlegend=False,
        margin=dict(t=20, b=60, l=60, r=20)
    )

    return fig


def create_risk_return_scatter(trials: List[Dict]):
    """
    Create scatter plot showing risk (std dev) vs return for each trial.
    Shows the risk-return tradeoff with a linear trendline.

    Each point represents one trial, positioned by its risk (x-axis) and return (y-axis).
    """
    if not trials:
        return None

    # Extract data
    returns = [trial['total_return_pct'] for trial in trials]
    std_devs_pct = [(trial['overall_std_dev'] / trial['starting_capital']) * 100 for trial in trials]
    trial_ids = [trial['run_id'] for trial in trials]
    sharpe_ratios = [trial['sharpe_ratio'] for trial in trials]

    # Use Set2 color palette (same as Return Distribution chart)
    colors = px.colors.qualitative.Set2

    fig = go.Figure()

    # Add scatter points with Set2 colors
    fig.add_trace(go.Scatter(
        x=std_devs_pct,
        y=returns,
        mode='markers+text',
        marker=dict(
            size=15,
            color=[colors[i % len(colors)] for i in range(len(trials))]
        ),
        text=[f"T{tid}" for tid in trial_ids],
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Risk: %{x:.2f}%<br>' +
            'Return: %{y:.2f}%<br>' +
            'Sharpe Ratio: %{customdata:.2f}<br>' +
            '<extra></extra>'
        ),
        customdata=sharpe_ratios,
        showlegend=False
    ))

    # Add linear trendline (risk-return line)
    if len(returns) > 1:
        # Calculate linear regression (y = mx + b, where y=return, x=risk)
        z = np.polyfit(std_devs_pct, returns, 1)
        p = np.poly1d(z)

        # Generate line points
        x_line = np.linspace(min(std_devs_pct), max(std_devs_pct), 100)
        y_line = p(x_line)

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Risk-Return Trendline',
            line=dict(color='white', width=1.5, dash='dash'),
            hoverinfo='skip',
            showlegend=True
        ))

    fig.update_layout(
        xaxis_title='Std Dev (%)',
        yaxis_title='Return (%)',
        template='plotly_dark',
        height=450,
        hovermode='closest',
        margin=dict(t=20, b=60, l=60, r=20),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

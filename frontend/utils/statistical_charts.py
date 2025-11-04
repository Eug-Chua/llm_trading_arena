"""
Statistical Chart and Table Generation Utilities

This module contains all chart and table creation functions for the Statistical Analysis page.
Includes visualization utilities for:
- Trial performance metrics
- Trade P&L distributions (KDE overlays)
- Risk asymmetry analysis (upside/downside comparisons)
- Distribution shape metrics (skewness, kurtosis)
- Confidence intervals and statistical significance
- Capture ratio analysis per coin
- Equity curve overlays
- Summary statistics tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from typing import Dict, List
from scipy import stats as scipy_stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import capture ratio calculation
from frontend.utils.capture_ratios import calculate_coin_capture_ratios

# Import logger
from src.utils.logger import setup_logger
logger = setup_logger(__name__)


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


def create_upside_downside_comparison(trials: List[Dict]):
    """
    Create side-by-side comparison of upside vs downside deviation across trials
    """
    if not trials:
        return None

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    trial_ids = [trial['run_id'] for trial in trials]
    upside_devs = [trial['upside_deviation'] for trial in trials]
    downside_devs = [trial['downside_deviation'] for trial in trials]

    # Upside deviation bars
    fig.add_trace(go.Bar(
        x=trial_ids,
        y=upside_devs,
        name='Upside Deviation',
        marker_color='#00ff88',  # Bright green
        hovertemplate='Trial %{x}<br>Upside Dev: $%{y:,.2f}<extra></extra>'
    ))

    # Downside deviation bars (negative for visual separation)
    fig.add_trace(go.Bar(
        x=trial_ids,
        y=[-d for d in downside_devs],
        name='Downside Deviation',
        marker_color='#ff4444',  # Bright red
        hovertemplate='Trial %{x}<br>Downside Dev: $%{customdata:,.2f}<extra></extra>',
        customdata=downside_devs
    ))

    fig.update_layout(
        title='Upside vs Downside Volatility by Trial',
        xaxis_title='Trial Number',
        yaxis_title='Deviation ($)',
        template='plotly_dark',
        height=400,
        barmode='relative',
        hovermode='x unified',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=2),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=0.99,
            xanchor='center',
            x=0.5
        )
    )

    return fig


def create_volatility_distribution_boxplot(trials: List[Dict]):
    """
    Create box plots showing distribution of upside and downside deviation across trials
    Helps visualize the spread and consistency of volatility metrics
    """
    if not trials:
        return None

    fig = go.Figure()

    upside_devs = [trial['upside_deviation'] for trial in trials]
    downside_devs = [trial['downside_deviation'] for trial in trials]

    # Add upside deviation box plot
    fig.add_trace(go.Box(
        y=upside_devs,
        name='Upside Deviation',
        marker_color='#00ff88',
        boxmean='sd',  # Show mean and standard deviation
        hovertemplate='Upside Dev: $%{y:,.2f}<extra></extra>'
    ))

    # Add downside deviation box plot
    fig.add_trace(go.Box(
        y=downside_devs,
        name='Downside Deviation',
        marker_color='#ff4444',
        boxmean='sd',
        hovertemplate='Downside Dev: $%{y:,.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Volatility Distribution Across Trials',
        yaxis_title='Deviation ($)',
        template='plotly_dark',
        height=400,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def create_volatility_scatter(trials: List[Dict]):
    """
    Create scatter plot showing upside vs downside deviation for each trial.
    Shows the relationship and asymmetry between upside and downside volatility.
    Expressed as percentage of starting capital for easier interpretation.
    """
    if not trials:
        return None

    fig = go.Figure()

    trial_ids = [trial['run_id'] for trial in trials]
    starting_capital = trials[0].get('starting_capital', 10000)

    # Convert deviations to percentages of starting capital
    upside_devs_pct = [(trial['upside_deviation'] / starting_capital) * 100 for trial in trials]
    downside_devs_pct = [(trial['downside_deviation'] / starting_capital) * 100 for trial in trials]

    # Calculate deviation ratios (upside_dev / downside_dev) for coloring
    ratios = [
        (trial['upside_deviation'] / trial['downside_deviation']) if trial['downside_deviation'] > 0 else 0
        for trial in trials
    ]

    # Scatter points
    fig.add_trace(go.Scatter(
        x=downside_devs_pct,
        y=upside_devs_pct,
        mode='markers+text',
        marker=dict(
            size=15,
            color=ratios,
            colorscale='RdYlGn',  # Red (low ratio) to Green (high ratio)
            showscale=True,
            colorbar=dict(
                title="Deviation<br>Ratio",
                thickness=15,
                len=0.7
            ),
            line=dict(color='white', width=1)
        ),
        text=[f"T{tid}" for tid in trial_ids],
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hovertemplate=(
            '<b>Trial %{text}</b><br>' +
            'Downside Dev: %{x:.2f}%<br>' +
            'Upside Dev: %{y:.2f}%<br>' +
            'Dev Ratio: %{marker.color:.2f}x<br>' +
            '<extra></extra>'
        ),
        showlegend=False
    ))

    # Add diagonal reference line (y = x, where upside = downside)
    max_val = max(max(upside_devs_pct, default=0), max(downside_devs_pct, default=0))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='Equal Volatility',
        hovertemplate='Equal upside/downside<extra></extra>',
        showlegend=True
    ))

    # Add shaded regions
    # Region above diagonal = more upside volatility (desirable)
    # Region below diagonal = more downside volatility (undesirable)

    fig.update_layout(
        title='Upside vs Downside Volatility - Risk Asymmetry Profile',
        xaxis_title='Downside Deviation (% of Starting Capital)',
        yaxis_title='Upside Deviation (% of Starting Capital)',
        template='plotly_dark',
        height=400,
        hovermode='closest',
    )

    # Equal aspect ratio for fair comparison
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    return fig


def create_risk_metrics_heatmap(trials: List[Dict]):
    """
    Create heatmap showing multiple risk metrics across trials.
    Visualizes upside deviation, downside deviation, ratio, skewness, and kurtosis.
    """
    if not trials:
        return None

    trial_ids = [f"Trial {trial['run_id']}" for trial in trials]

    # Collect metrics
    metrics_data = {
        'Upside Dev': [trial['upside_deviation'] for trial in trials],
        'Downside Dev': [trial['downside_deviation'] for trial in trials],
        'Up/Down Ratio': [trial['upside_downside_ratio'] for trial in trials],
        'Skewness': [trial['skewness'] for trial in trials],
        'Kurtosis': [trial['kurtosis'] for trial in trials],
        'Sortino Ratio': [trial['sortino_ratio'] for trial in trials],
    }

    # Normalize each metric to 0-1 scale for consistent color mapping
    normalized_data = []
    metric_names = []

    for metric_name, values in metrics_data.items():
        metric_names.append(metric_name)
        # Min-max normalization
        min_val = min(values)
        max_val = max(values)
        if max_val - min_val > 0:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5] * len(values)  # If all same, use middle value
        normalized_data.append(normalized)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data,
        x=trial_ids,
        y=metric_names,
        colorscale='RdYlGn',  # Red (low) to Green (high)
        hovertemplate=(
            '<b>%{y}</b><br>' +
            '%{x}<br>' +
            'Normalized: %{z:.2f}<br>' +
            '<extra></extra>'
        ),
        colorbar=dict(
            title="Normalized<br>Score",
            thickness=15,
            len=0.7
        )
    ))

    # Add actual values as text annotations
    for i, metric_name in enumerate(metric_names):
        for j in range(len(trial_ids)):
            actual_value = metrics_data[metric_name][j]
            # Format based on metric type
            if 'Dev' in metric_name:
                text = f"${actual_value:,.0f}"
            elif 'Ratio' in metric_name:
                text = f"{actual_value:.2f}x"
            else:
                text = f"{actual_value:.2f}"

            fig.add_annotation(
                x=j,
                y=i,
                text=text,
                showarrow=False,
                font=dict(size=10, color='white'),
                xref='x',
                yref='y'
            )

    fig.update_layout(
        title='Risk Metrics Heatmap - Normalized Comparison Across Trials',
        xaxis_title='',
        yaxis_title='',
        template='plotly_dark',
        height=450,
        xaxis=dict(side='bottom'),
    )

    return fig


def create_distribution_shape_chart(trials: List[Dict]):
    """
    Create chart showing skewness and kurtosis across trials
    """
    if not trials:
        return None

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    trial_ids = [trial['run_id'] for trial in trials]
    skewness = [trial['skewness'] for trial in trials]
    kurtosis = [trial['kurtosis'] for trial in trials]

    # Skewness
    fig.add_trace(go.Scatter(
        x=trial_ids,
        y=skewness,
        mode='lines+markers',
        name='Skewness',
        line=dict(color=colors[4], width=2),
        marker=dict(size=10, line=dict(color='white', width=1)),
        hovertemplate='Trial %{x}<br>Skewness: %{y:.3f}<extra></extra>'
    ))

    # Kurtosis
    fig.add_trace(go.Scatter(
        x=trial_ids,
        y=kurtosis,
        mode='lines+markers',
        name='Kurtosis',
        line=dict(color=colors[5], width=2),
        marker=dict(size=10, line=dict(color='white', width=1)),
        hovertemplate='Trial %{x}<br>Kurtosis: %{y:.3f}<extra></extra>',
        yaxis='y2'
    ))

    # Add reference lines
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5,
                  annotation_text='Symmetric', annotation_position='left')

    fig.update_layout(
        title='Distribution Shape: Skewness & Kurtosis',
        xaxis_title='Trial Number',
        yaxis_title='Skewness',
        yaxis2=dict(title='Kurtosis', overlaying='y', side='right'),
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def create_risk_metrics_summary(trials: List[Dict]):
    """
    Create summary table showing risk asymmetry metrics.
    Returns numeric values for proper sorting - formatting should be done in display.
    """
    summary_data = []

    for trial in trials:
        summary_data.append({
            'Trial': trial['run_id'],
            'Sortino Ratio': trial['sortino_ratio'],
            'Upside/Downside': trial['upside_downside_ratio'],
            'Upside Dev': trial['upside_deviation'],
            'Downside Dev': trial['downside_deviation'],
            'Skewness': trial['skewness'],
            'Kurtosis': trial['kurtosis']
        })

    # Add average row
    summary_data.append({
        'Trial': 999999,  # Large number to sort at bottom
        'Sortino Ratio': np.mean([t['sortino_ratio'] for t in trials]),
        'Upside/Downside': np.mean([t['upside_downside_ratio'] for t in trials]),
        'Upside Dev': np.mean([t['upside_deviation'] for t in trials]),
        'Downside Dev': np.mean([t['downside_deviation'] for t in trials]),
        'Skewness': np.mean([t['skewness'] for t in trials]),
        'Kurtosis': np.mean([t['kurtosis'] for t in trials])
    })

    df = pd.DataFrame(summary_data)

    # Keep Trial as numeric for proper sorting - formatting happens in display layer
    return df


def calculate_statistical_significance(trials: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate statistical significance and confidence intervals for key metrics

    Returns:
        Dict with metrics and their statistical properties:
        - mean, std, confidence_interval, t_statistic, p_value, is_significant
    """
    results = {}

    # Key metrics to analyze
    metrics = {
        'total_return_pct': 'Return %',
        'sharpe_ratio': 'Sharpe Ratio',
        'sortino_ratio': 'Sortino Ratio',
        'win_rate': 'Win Rate %'
    }

    for metric_key, metric_name in metrics.items():
        values = [t[metric_key] for t in trials]
        n = len(values)

        if n < 2:
            continue

        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample standard deviation
        se = std_val / np.sqrt(n)  # Standard error

        # 95% confidence interval using t-distribution
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = scipy_stats.t.ppf(1 - alpha/2, df=n-1)
        margin_of_error = t_critical * se
        ci_lower = mean_val - margin_of_error
        ci_upper = mean_val + margin_of_error

        # One-sample t-test against zero (null hypothesis: mean = 0)
        t_statistic, p_value = scipy_stats.ttest_1samp(values, 0)

        # Determine significance (p < 0.05)
        is_significant = p_value < 0.05

        # Probability of positive outcome (for return-like metrics)
        prob_positive = sum(1 for v in values if v > 0) / n * 100

        results[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'prob_positive': prob_positive,
            'n_trials': n
        }

    return results


def create_confidence_interval_table(trials: List[Dict]) -> pd.DataFrame:
    """
    Create table showing confidence intervals and statistical significance.
    Returns numeric values for proper sorting - formatting should be done in display.
    """
    stats = calculate_statistical_significance(trials)

    table_data = []

    for metric_name, data in stats.items():
        # Store numeric values, CI as separate columns for sorting
        table_data.append({
            'Metric': metric_name,
            'Mean': data['mean'],
            'CI Lower': data['ci_lower'],
            'CI Upper': data['ci_upper'],
            'Std Dev': data['std'],
            'P-Value': data['p_value'],
            'Significant?': "✓ Yes" if data['is_significant'] else "✗ No",
            'Prob > 0': data['prob_positive']
        })

    return pd.DataFrame(table_data)


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
        title='Return Distribution Across Trials',
        xaxis_title='Trial Number',
        yaxis_title='Total Return (%)',
        template='plotly_dark',
        height=500,
        hovermode='closest',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        showlegend=False
    )

    return fig


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


def create_overall_stats_table(trials: List[Dict]) -> pd.DataFrame:
    """
    Create Overall Stats table for individual trials (with aggregate row).
    Returns numeric values for proper sorting - formatting should be done in display.
    """
    summary_rows = []

    # Individual trial rows
    for trial in trials:
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
            'No of Trades': trial['total_trades']
        })

    # Aggregate row - use string for Trial column to distinguish it
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


def create_equity_curve_overlay(grouped_data: Dict, config_name: str, extract_metrics_fn=None):
    """
    Create overlaid equity curves for all trials of a single configuration

    This shows the "fan" of possible outcomes from LLM non-determinism

    Args:
        grouped_data: Dictionary mapping config names to lists of trial metadata
        config_name: Name of the configuration to visualize
        extract_metrics_fn: Function to extract metrics from checkpoint (optional, for compatibility)
    """
    trials = grouped_data.get(config_name, [])

    if not trials:
        return None

    fig = go.Figure()

    # Use same color palette as Trade P&L KDE (Set2)
    colors = px.colors.qualitative.Set2

    # Load trade history for each trial
    for idx, trial_meta in enumerate(trials):
        # Find the checkpoint file
        checkpoint_name = None
        for name, checkpoint in st.session_state.get('loaded_checkpoints', {}).items():
            # Use the passed function or import dynamically
            if extract_metrics_fn:
                meta = extract_metrics_fn(checkpoint)
            else:
                from frontend.utils.statistical_metrics import extract_metrics
                meta = extract_metrics(checkpoint)
            if (meta['model'] == trial_meta['model'] and
                meta['temperature'] == trial_meta['temperature'] and
                meta['run_id'] == trial_meta['run_id']):
                checkpoint_name = name
                break

        if not checkpoint_name:
            continue

        checkpoint = st.session_state['loaded_checkpoints'][checkpoint_name]
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

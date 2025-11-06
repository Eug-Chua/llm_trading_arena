"""
Risk and Volatility Visualization Charts

This module contains chart creation functions for risk analysis:
- Upside vs downside deviation comparisons
- Volatility distribution box plots
- Risk asymmetry scatter plots
- Risk metrics heatmaps
- Distribution shape charts (skewness, kurtosis)
- Risk metrics summary tables
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List


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
        title='Trial Comparison',
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
                thickness=15,
                len=0.8
            ),
            line=dict(color='white', width=0.25)
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
        title='Risk Asymmetry Profile',
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
            'Std Dev': trial['overall_std_dev'],
            'Upside Dev': trial['upside_deviation'],
            'Downside Dev': trial['downside_deviation'],
            'Skewness': trial['skewness'],
            'Kurtosis': trial['kurtosis']
        })

    # Add average row
    summary_data.append({
        'Trial': 999999,  # Large number to sort at bottom
        'Sortino Ratio': np.mean([t['sortino_ratio'] for t in trials]),
        'Std Dev': np.mean([t['overall_std_dev'] for t in trials]),
        'Upside Dev': np.mean([t['upside_deviation'] for t in trials]),
        'Downside Dev': np.mean([t['downside_deviation'] for t in trials]),
        'Skewness': np.mean([t['skewness'] for t in trials]),
        'Kurtosis': np.mean([t['kurtosis'] for t in trials])
    })

    df = pd.DataFrame(summary_data)

    # Keep Trial as numeric for proper sorting - formatting happens in display layer
    return df

"""
Candlestick Chart Creation

This module contains the main candlestick chart creation function:
- Interactive OHLCV chart with technical indicators
- Trade markers with distinctive shapes/colors for each model
- Highlight functionality for specific timestamps
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data.indicators import TechnicalIndicators, load_indicator_config
from frontend.utils.visual_config import get_model_color, get_candlestick_colors, get_indicator_colors, get_chart_height, get_trade_marker_settings

def create_candlestick_chart(df, trades_by_model, show_indicators, selected_models, coin_symbol, interval='4h', highlighted_timestamp=None):
    """Create interactive candlestick chart with indicators and trades

    Args:
        highlighted_timestamp: Optional timestamp to highlight on the chart
    """

    # Load indicator configuration
    config = load_indicator_config()
    rsi_periods = config.get('rsi_periods', [7, 14])

    # Initialize indicator calculator
    indicator_calc = TechnicalIndicators()

    # Create subplots: main chart, RSI, MACD
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{coin_symbol}/USD - {interval.upper()} Chart', f'RSI ({rsi_periods[1]})', 'MACD')
    )

    # Load color configurations
    candlestick_colors = get_candlestick_colors()
    indicator_colors = get_indicator_colors()
    trade_marker_settings = get_trade_marker_settings()

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=coin_symbol,
            increasing_line_color=candlestick_colors.get('increasing', '#00ff88'),
            decreasing_line_color=candlestick_colors.get('decreasing', '#ff4444')
        ),
        row=1, col=1
    )

    # Add EMA indicators if enabled
    ema_periods = config.get('ema_periods', [20, 50])

    # Calculate EMAs (returns DataFrame with columns like 'ema_20', 'ema_50')
    ema_df = indicator_calc.calculate_ema(df, periods=ema_periods)

    if show_indicators['ema20']:
        ema_col = f'ema_{ema_periods[0]}'
        if ema_col in ema_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=ema_df[ema_col],
                    name=f'EMA {ema_periods[0]}',
                    line=dict(color=indicator_colors.get('ema_20', '#00d4ff'), width=1.5),
                    opacity=0.7
                ),
                row=1, col=1
            )

    if show_indicators['ema50']:
        ema_col = f'ema_{ema_periods[1]}'
        if ema_col in ema_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=ema_df[ema_col],
                    name=f'EMA {ema_periods[1]}',
                    line=dict(color=indicator_colors.get('ema_50', '#ffd700'), width=1.5),
                    opacity=0.7
                ),
                row=1, col=1
            )

    # Add trade markers with distinctive colors for each model
    for model_name, trades in trades_by_model.items():
        if model_name not in selected_models:
            continue

        # Extract base model name (remove date suffixes like "_Oct_31")
        base_model_name = model_name.split('_')[0]  # "Anthropic_Oct_31" -> "Anthropic"

        # Get color from visual config
        dot_color = get_model_color(base_model_name)
        marker_symbol = 'circle'

        # Get marker size settings from config
        default_size = trade_marker_settings.get('default_size', 14)
        highlighted_size = trade_marker_settings.get('highlighted_size', 30)
        default_opacity = trade_marker_settings.get('opacity', 0.8)
        highlighted_opacity = trade_marker_settings.get('highlighted_opacity', 0.9)

        # Buy/Open markers
        buys = [t for t in trades if t['action'] == 'BUY']
        if buys:
            buy_times = [t['timestamp'] for t in buys]
            buy_prices = [t['price'] for t in buys]
            buy_hover = [
                f"<b>{model_name} - OPEN</b><br>" +
                f"Price: ${t['price']:,.2f}<br>" +
                f"Quantity: {t['quantity']:.4f}<br>" +
                f"Leverage: {t['leverage']}x<br>" +
                f"Cost: ${t['cost']:,.2f}"
                for t in buys
            ]

            # Calculate marker sizes and opacity - highlight selected timestamp
            marker_sizes = []
            marker_opacities = []
            for t in buys:
                # Normalize timestamps for comparison (convert both to ISO format strings)
                trade_ts = pd.to_datetime(t['timestamp']).isoformat()
                highlighted_ts = pd.to_datetime(highlighted_timestamp).isoformat() if highlighted_timestamp else None

                if highlighted_ts and trade_ts == highlighted_ts:
                    marker_sizes.append(highlighted_size)
                    marker_opacities.append(highlighted_opacity)
                else:
                    marker_sizes.append(default_size)
                    marker_opacities.append(default_opacity)

            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode='markers',
                    name=f'{model_name} - Entry',
                    marker=dict(
                        size=marker_sizes,
                        symbol=marker_symbol,
                        color=dot_color,
                        opacity=marker_opacities,
                        line=dict(width=0)  # No border
                    ),
                    hovertext=buy_hover,
                    hoverinfo='text'
                ),
                row=1, col=1
            )

        # Close/Exit markers
        closes = [t for t in trades if t['action'] == 'CLOSE']
        if closes:
            close_times = [t['timestamp'] for t in closes]
            close_prices = [t['exit_price'] for t in closes]
            close_hover = [
                f"<b>{model_name} - CLOSE</b><br>" +
                f"Exit Price: ${t['exit_price']:,.2f}<br>" +
                f"Entry Price: ${t.get('entry_price', 0):,.2f}<br>" +
                f"P&L: ${t['net_pnl']:,.2f}<br>"
                # f"Reason: {t.get('reason', 'N/A')}"
                for t in closes
            ]

            # Calculate marker sizes and opacity - highlight selected timestamp
            close_marker_sizes = []
            close_marker_opacities = []
            for t in closes:
                # Normalize timestamps for comparison (convert both to ISO format strings)
                trade_ts = pd.to_datetime(t['timestamp']).isoformat()
                highlighted_ts = pd.to_datetime(highlighted_timestamp).isoformat() if highlighted_timestamp else None

                if highlighted_ts and trade_ts == highlighted_ts:
                    close_marker_sizes.append(highlighted_size)
                    close_marker_opacities.append(highlighted_opacity)
                else:
                    close_marker_sizes.append(default_size)
                    close_marker_opacities.append(default_opacity)

            fig.add_trace(
                go.Scatter(
                    x=close_times,
                    y=close_prices,
                    mode='markers',
                    name=f'{model_name} - Exit',
                    marker=dict(
                        size=close_marker_sizes,
                        symbol=marker_symbol,
                        color=dot_color,
                        opacity=close_marker_opacities,
                        line=dict(width=0)  # No border
                    ),
                    hovertext=close_hover,
                    hoverinfo='text',
                    showlegend=False
                ),
                row=1, col=1
            )

    # RSI indicator
    rsi_df = indicator_calc.calculate_rsi(df, periods=rsi_periods)
    rsi_col = f'rsi_{rsi_periods[1]}'

    if rsi_col in rsi_df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=rsi_df[rsi_col],
                name=f'RSI {rsi_periods[1]}',
                line=dict(color=indicator_colors.get('rsi', '#00d4ff'), width=2)
            ),
            row=2, col=1
        )

    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # MACD indicator
    macd_config = config.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
    macd_df = indicator_calc.calculate_macd(
        df,
        fast=macd_config['fast'],
        slow=macd_config['slow'],
        signal=macd_config['signal']
    )

    if 'macd_histogram' in macd_df.columns:
        # MACD histogram
        histogram = macd_df['macd_histogram']
        hist_pos_color = indicator_colors.get('macd_histogram_positive', '#00ff88')
        hist_neg_color = indicator_colors.get('macd_histogram_negative', '#ff4444')
        colors = [hist_pos_color if val >= 0 else hist_neg_color for val in histogram]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=histogram,
                name='MACD Histogram',
                marker_color=colors,
                opacity=0.5
            ),
            row=3, col=1
        )

    if 'macd' in macd_df.columns:
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=macd_df['macd'],
                name='MACD',
                line=dict(color=indicator_colors.get('macd', '#00d4ff'), width=2)
            ),
            row=3, col=1
        )

    if 'macd_signal' in macd_df.columns:
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=macd_df['macd_signal'],
                name='Signal',
                line=dict(color=indicator_colors.get('macd_signal', '#ff6b6b'), width=2)
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        height=get_chart_height('candlestick_chart'),
        template='plotly_dark',
        showlegend=False,  # Hide legends as requested
        hovermode='x unified',
        xaxis3_title='Date'
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    # Remove rangeslider
    fig.update_xaxes(rangeslider_visible=False)

    return fig



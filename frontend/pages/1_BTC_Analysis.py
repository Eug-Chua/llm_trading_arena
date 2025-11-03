"""
LLM Trading Arena - BTC Analysis

Page 2: BTC candlestick chart with indicators and trade markers
"""

import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.indicators import TechnicalIndicators, load_indicator_config

# Page config
st.set_page_config(
    page_title="BTC Analysis - LLM Trading Arena",
    page_icon="₿",
    layout="wide"
)


@st.cache_data
def load_checkpoint(checkpoint_path: str):
    """Load checkpoint data"""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_ohlc_data(coin: str, interval: str = '4h'):
    """Load historical OHLC data"""
    data_path = project_root / "data" / "historical" / interval / f"{coin}.parquet"

    if not data_path.exists():
        return None

    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data
def load_reasoning(reasoning_path: str):
    """Load LLM reasoning JSON"""
    try:
        with open(reasoning_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_coin_trades(checkpoint, coin: str):
    """Extract trades for specific coin"""
    trade_log = checkpoint.get('trade_history', [])
    return [t for t in trade_log if t.get('symbol') == coin]


def create_candlestick_chart(df, trades_by_model, show_indicators, selected_models, interval='4h'):
    """Create interactive candlestick chart with indicators and trades"""

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
        subplot_titles=(f'BTC/USD - {interval.upper()} Chart', f'RSI ({rsi_periods[1]})', 'MACD')
    )

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
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
                    line=dict(color='#00d4ff', width=1.5),
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
                    line=dict(color='#ffd700', width=1.5),
                    opacity=0.7
                ),
                row=1, col=1
            )

    # Add trade markers as simple colored dots
    model_config = {
        'Anthropic': {
            'color': '#D97757'
        },
        'OpenAI': {
            'color': '#ECF4E8'
        }
    }

    for model_name, trades in trades_by_model.items():
        if model_name not in selected_models:
            continue

        config = model_config.get(model_name, {'color': '#ffffff'})
        dot_color = config['color']

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

            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode='markers',
                    name=f'{model_name} - Entry',
                    marker=dict(
                        size=12,
                        color=dot_color,
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
                f"P&L: ${t['net_pnl']:,.2f}<br>" +
                f"Reason: {t.get('reason', 'N/A')}"
                for t in closes
            ]

            fig.add_trace(
                go.Scatter(
                    x=close_times,
                    y=close_prices,
                    mode='markers',
                    name=f'{model_name} - Exit',
                    marker=dict(
                        size=12,
                        color=dot_color
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
                line=dict(color='#00d4ff', width=2)
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
        colors = ['#00ff88' if val >= 0 else '#ff4444' for val in histogram]
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
                line=dict(color='#00d4ff', width=2)
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
                line=dict(color='#ff6b6b', width=2)
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        height=900,
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

def render_llm_reasoning_sidebar(reasoning_data, coin, selected_models):
    """Render LLM reasoning in sidebar"""
    st.sidebar.header(f"Model Chat - {coin}")

    if not reasoning_data:
        st.sidebar.info("No reasoning data available")
        return

    # Filter iterations for this coin
    for model_name, model_reasoning in reasoning_data.items():
        # Match model name case-insensitively
        model_display_name = model_name.title()
        if not any(model_display_name.lower() == m.lower() for m in selected_models):
            continue

        if not model_reasoning:
            continue

        iterations = model_reasoning.get('iterations', [])

        # Filter for iterations mentioning this coin
        coin_iterations = []
        for iteration in iterations:
            signals = iteration.get('signals', [])
            if any(s.get('coin') == coin for s in signals):
                coin_iterations.append(iteration)

        # Display count of iterations being shown (not total)
        num_showing = min(len(coin_iterations), 5)
        st.sidebar.markdown(f"### {model_display_name}")
        st.sidebar.markdown(f"*{num_showing} of {len(coin_iterations)} decisions*")

        # Show latest 5 iterations
        for iteration in coin_iterations[-5:]:
            timestamp = iteration.get('timestamp', 'Unknown')
            raw_response = iteration.get('raw_response', '')

            # Show full Chain of Thought (no truncation)
            with st.sidebar.expander(f"📅 {timestamp}", expanded=False):
                st.markdown(raw_response)

                # Show signals
                signals = [s for s in iteration.get('signals', []) if s.get('coin') == coin]
                for signal in signals:
                    st.markdown(f"**Action:** {signal.get('signal', 'N/A')}")
                    if signal.get('close_reason'):
                        st.markdown(f"**Reason:** {signal.get('close_reason')}")


def main():
    st.title("BTC Analysis")
    st.markdown("**Candlestick chart with technical indicators and LLM trading decisions**")

    # Load configuration
    config = load_indicator_config()
    interval = config.get('data_interval', '4h')

    # Load data
    checkpoint_dir = project_root / "results" / "checkpoints"

    # Get all available checkpoint files
    available_checkpoints = sorted([f.name for f in checkpoint_dir.glob("*.pkl")])

    if not available_checkpoints:
        st.error("❌ No checkpoint files found in results/checkpoints/")
        st.info("Run a backtest first: `python scripts/run_backtest.py --start 2025-10-18 --end 2025-10-30 --model anthropic`")
        return

    checkpoints = {}

    # Anthropic checkpoint selector
    anthropic_files = [f for f in available_checkpoints if 'anthropic' in f.lower()]
    if anthropic_files:
        selected_anthropic = st.sidebar.selectbox(
            "Anthropic Checkpoint",
            options=anthropic_files,
            index=len(anthropic_files) - 1,
            key="anthropic_checkpoint_btc"
        )
        checkpoints['Anthropic'] = load_checkpoint(str(checkpoint_dir / selected_anthropic))

    # OpenAI checkpoint selector
    openai_files = [f for f in available_checkpoints if 'openai' in f.lower()]
    if openai_files:
        selected_openai = st.sidebar.selectbox(
            "OpenAI Checkpoint",
            options=openai_files,
            index=len(openai_files) - 1,
            key="openai_checkpoint_btc"
        )
        checkpoints['OpenAI'] = load_checkpoint(str(checkpoint_dir / selected_openai))

    if not checkpoints:
        st.error("❌ No checkpoint files loaded")
        return

    # Load OHLC data
    df = load_ohlc_data('BTC', interval)

    if df is None:
        st.error("❌ BTC historical data not found. Please run data collection first.")
        st.code(f"python scripts/collect_historical_data.py --coins BTC --interval {interval}")
        return

    # Filter data to backtest date range from metadata
    from datetime import datetime
    min_date = None
    max_date = None

    for checkpoint in checkpoints.values():
        metadata = checkpoint.get('metadata', {})
        if metadata.get('start_date'):
            start_date = datetime.fromisoformat(metadata['start_date'])
            if min_date is None or start_date < min_date:
                min_date = start_date
        if metadata.get('end_date'):
            end_date = datetime.fromisoformat(metadata['end_date'])
            if max_date is None or end_date > max_date:
                max_date = end_date

    # Filter dataframe to backtest period
    if min_date and max_date:
        df = df[(df['timestamp'] >= min_date) & (df['timestamp'] <= max_date)]

    # Load reasoning data (match checkpoint filenames)
    reasoning_data = {}
    if anthropic_files and 'Anthropic' in checkpoints:
        reasoning_path = checkpoint_dir / selected_anthropic.replace('.pkl', '_reasoning.json')
        if reasoning_path.exists():
            reasoning_data['anthropic'] = load_reasoning(str(reasoning_path))

    if openai_files and 'OpenAI' in checkpoints:
        reasoning_path = checkpoint_dir / selected_openai.replace('.pkl', '_reasoning.json')
        if reasoning_path.exists():
            reasoning_data['openai'] = load_reasoning(str(reasoning_path))

    # Controls
    st.sidebar.header("Controls")

    # Model selector
    model_options = list(checkpoints.keys())
    selected_models = st.sidebar.multiselect(
        "Select Model(s)",
        options=model_options,
        default=model_options
    )

    # Indicator toggles
    st.sidebar.markdown("### Indicators")
    show_indicators = {
        'ema20': st.sidebar.checkbox("Show EMA 20", value=True),
        'ema50': st.sidebar.checkbox("Show EMA 50", value=True),
        'volume': st.sidebar.checkbox("Show Volume", value=False)
    }

    # Extract trades by model
    trades_by_model = {}
    for model_name, checkpoint in checkpoints.items():
        trades_by_model[model_name] = get_coin_trades(checkpoint, 'BTC')

    # Render sidebar reasoning
    render_llm_reasoning_sidebar(reasoning_data, 'BTC', selected_models)

    # Main chart
    st.subheader("Price Chart & Indicators")

    chart = create_candlestick_chart(df, trades_by_model, show_indicators, selected_models, interval)
    st.plotly_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()

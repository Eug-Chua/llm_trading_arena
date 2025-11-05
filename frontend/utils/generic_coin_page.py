"""
Generic coin analysis page - reusable for all coins
"""

import streamlit as st
import pickle
import json
import base64
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import from refactored coin chart modules
from frontend.utils.coin_charts.data_loaders import (
    load_ohlc_data,
    get_coin_trades,
)
from frontend.utils.coin_charts.candlestick_chart import create_candlestick_chart
from frontend.utils.coin_charts.reasoning_sidebar import render_llm_reasoning_sidebar
from src.data.indicators import load_indicator_config


@st.cache_data
def load_checkpoint(checkpoint_path: str):
    """Load checkpoint data"""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_reasoning(reasoning_path: str):
    """Load LLM reasoning JSON"""
    try:
        with open(reasoning_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_base64_image(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def render_coin_page(coin_symbol: str):
    """
    Render a coin analysis page

    Args:
        coin_symbol: Coin symbol (e.g., "BTC", "ETH")
    """
    # Check if coin logo exists
    coin_lower = coin_symbol.lower()
    logo_path = project_root / "frontend" / "images" / f"{coin_lower}.png"

    # Display logo with title on main page
    if logo_path.exists():
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; gap: 10px;'>
                <img src='data:image/png;base64,{get_base64_image(logo_path)}' width='50' />
                <h1 style='margin: 0;'>{coin_symbol} Analysis</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title(f"{coin_symbol} Analysis")

    st.markdown("**Candlestick chart with technical indicators and LLM trading decisions**")

    # Load configuration
    config = load_indicator_config()
    interval = config.get('data_interval', '4h')

    # Load data (recursively scan results directory)
    results_dir = project_root / "results"

    # Get all available checkpoint files with natural sort (recursively)
    from frontend.utils.checkpoint_utils import natural_sort_key
    checkpoint_paths = sorted([f for f in results_dir.rglob("*.pkl")], key=natural_sort_key)
    available_checkpoints = [p.name for p in checkpoint_paths]
    checkpoint_path_map = {p.name: p for p in checkpoint_paths}  # Map filename to full path

    if not available_checkpoints:
        st.error("❌ No checkpoint files found in results/")
        st.info("Run a backtest first: `python scripts/run_backtest.py --start 2025-10-18 --end 2025-10-30 --model anthropic --run-id 1`")
        return

    checkpoints = {}
    selected_files = {}  # Track selected filenames for reasoning data

    # Dynamically detect all model providers from checkpoint filenames
    model_providers = set()
    for filename in available_checkpoints:
        # Extract model name (before _temp)
        if '_temp' in filename:
            model_name = filename.split('_temp')[0]
            model_providers.add(model_name)

    # Sort model providers alphabetically
    model_providers = sorted(model_providers)

    # Create checkpoint selector for each detected model
    for model_name in model_providers:
        model_files = [f for f in available_checkpoints if f.startswith(model_name + '_temp')]
        if model_files:
            # Capitalize display name (e.g., "anthropic" -> "Anthropic", "deepseek-terminus" -> "DeepSeek-Terminus")
            display_name = model_name.replace('-', ' ').title().replace(' ', '-')

            selected_file = st.sidebar.selectbox(
                f"{display_name} Checkpoint",
                options=model_files,
                index=len(model_files) - 1,
                key=f"{model_name}_checkpoint_{coin_symbol.lower()}"
            )
            # Get full path from map
            full_path = checkpoint_path_map[selected_file]
            checkpoints[display_name] = load_checkpoint(str(full_path))
            selected_files[display_name] = full_path

    if not checkpoints:
        st.error("❌ No checkpoint files loaded")
        return

    # Load OHLC data
    df = load_ohlc_data(coin_symbol, interval)

    if df is None:
        st.error(f"❌ {coin_symbol} historical data not found. Please run data collection first.")
        st.code(f"python scripts/collect_historical_data.py --coins {coin_symbol} --interval {interval}")
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
    for display_name, full_path in selected_files.items():
        # Reasoning file is in same directory as checkpoint
        reasoning_path = full_path.parent / f"{full_path.stem}_reasoning.json"
        if reasoning_path.exists():
            # Use lowercase model name as key for reasoning data (for backward compatibility)
            model_key = display_name.lower().replace('-', '')
            reasoning_data[model_key] = load_reasoning(str(reasoning_path))

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
        trades_by_model[model_name] = get_coin_trades(checkpoint, coin_symbol)

    # Render sidebar reasoning (this may update session state with highlighted timestamp)
    render_llm_reasoning_sidebar(reasoning_data, coin_symbol, selected_models, trades_by_model)

    # Main chart
    st.subheader("Price Chart & Indicators")

    # Get highlighted timestamp from session state
    highlighted_timestamp = st.session_state.get('highlighted_timestamp', None)

    # Show clear highlight button if a timestamp is highlighted
    if highlighted_timestamp:
        if st.button("Clear highlight"):
            st.session_state.highlighted_timestamp = None
            st.rerun()

    chart = create_candlestick_chart(df, trades_by_model, show_indicators, selected_models, coin_symbol, interval, highlighted_timestamp)
    st.plotly_chart(chart, use_container_width=True)

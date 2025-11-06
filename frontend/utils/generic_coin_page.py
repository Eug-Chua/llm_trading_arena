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
    logo_path = project_root / "frontend" / "images" / "coin_logos" /  f"{coin_lower}.png"

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

    if not checkpoint_paths:
        st.error("❌ No checkpoint files found in results/")
        st.info("Run a backtest first: `python scripts/run_backtest.py --start 2025-10-18 --end 2025-10-30 --model anthropic --run-id 1`")
        return

    # Extract unique models and temperatures from checkpoint paths
    # Path format: results/{model}/temp{XX}/{filename}
    models_with_temps = {}  # {model: {temp_folder: [checkpoint_paths]}}

    for checkpoint_path in checkpoint_paths:
        parts = checkpoint_path.parts
        if len(parts) >= 3:
            model = parts[-3]  # e.g., 'anthropic', 'openai'
            temp_folder = parts[-2]  # e.g., 'temp01', 'temp07'

            # Skip if not in valid structure (e.g., archive files, reports folder)
            if not temp_folder.startswith('temp'):
                continue

            if model not in models_with_temps:
                models_with_temps[model] = {}
            if temp_folder not in models_with_temps[model]:
                models_with_temps[model][temp_folder] = []

            models_with_temps[model][temp_folder].append(checkpoint_path)

    if not models_with_temps:
        st.error("❌ No valid checkpoint structure found")
        return

    # Controls section header (will be populated after loading checkpoints)
    st.sidebar.header("Controls")
    controls_placeholder_model = st.sidebar.empty()
    controls_placeholder_indicators = st.sidebar.empty()

    # Configuration section
    st.sidebar.markdown("---")
    st.sidebar.header("Configuration")

    # Temperature selector (primary control)
    st.sidebar.markdown("### Select Temperature")
    temp_choice = st.sidebar.radio(
        "Temperature",
        options=['0.7', '0.1'],
        index=0,  # Default to 0.7
        label_visibility="collapsed",
        key=f"temp_radio_{coin_symbol.lower()}"
    )

    # Map display to folder name
    temp_folder_map = {
        '0.7': 'temp07',
        '0.1': 'temp01'
    }
    selected_temp_folder = temp_folder_map[temp_choice]

    # Load latest trial for each model at selected temperature
    checkpoints = {}
    selected_files = {}

    for model, temp_data in sorted(models_with_temps.items()):
        if selected_temp_folder in temp_data:
            # Get latest checkpoint (last in naturally sorted list)
            model_checkpoints = sorted(temp_data[selected_temp_folder], key=natural_sort_key)
            if model_checkpoints:
                latest_checkpoint = model_checkpoints[-1]
                display_name = model.replace('-', ' ').title()
                checkpoints[display_name] = load_checkpoint(str(latest_checkpoint))
                selected_files[display_name] = latest_checkpoint

    if not checkpoints:
        st.warning(f"⚠️ No checkpoints found for temperature {temp_choice}")
        return

    # Advanced options (collapsible)
    with st.sidebar.expander("Advanced Options", expanded=False):
        st.markdown("**Select specific trial**")

        # Model selector
        available_models = sorted(models_with_temps.keys())
        override_model = st.selectbox(
            "Model",
            options=['None'] + available_models,
            format_func=lambda x: x.replace('-', ' ').title() if x != 'None' else 'Use default',
            key=f"override_model_{coin_symbol.lower()}"
        )

        if override_model != 'None':
            # Show trials for this model at selected temperature
            if selected_temp_folder in models_with_temps[override_model]:
                trial_checkpoints = sorted(models_with_temps[override_model][selected_temp_folder], key=natural_sort_key)
                trial_names = [p.name for p in trial_checkpoints]

                override_trial = st.selectbox(
                    "Trial",
                    options=trial_names,
                    index=len(trial_names) - 1,
                    key=f"override_trial_{coin_symbol.lower()}"
                )

                # Replace the default checkpoint for this model
                override_checkpoint_path = [p for p in trial_checkpoints if p.name == override_trial][0]
                display_name = override_model.replace('-', ' ').title()
                checkpoints[display_name] = load_checkpoint(str(override_checkpoint_path))
                selected_files[display_name] = override_checkpoint_path

                st.success(f"✓ Using {override_trial} for {display_name}")

    # Now populate Controls section with actual widgets
    with controls_placeholder_model.container():
        # Model selector
        model_options = list(checkpoints.keys())
        selected_models = st.multiselect(
            "Select Model(s)",
            options=model_options,
            default=model_options,
            key=f"model_selector_{coin_symbol.lower()}"
        )

    with controls_placeholder_indicators.container():
        # Indicator toggles
        st.markdown("### Indicators")
        show_indicators = {
            'ema20': st.checkbox("Show EMA 20", value=True, key=f"ema20_{coin_symbol.lower()}"),
            'ema50': st.checkbox("Show EMA 50", value=True, key=f"ema50_{coin_symbol.lower()}"),
            'volume': st.checkbox("Show Volume", value=False, key=f"volume_{coin_symbol.lower()}")
        }

    # Show current configuration (in Configuration section)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current View")
    st.sidebar.markdown(f"**Coin:** {coin_symbol}")
    st.sidebar.markdown(f"**Temperature:** {temp_choice}")
    st.sidebar.markdown(f"**Models shown:** {len(checkpoints)}")
    for model_name, checkpoint_path in selected_files.items():
        st.sidebar.markdown(f"- {model_name}: `{checkpoint_path.name}`")

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

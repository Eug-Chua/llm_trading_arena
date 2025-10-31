"""
Generic coin analysis page - reusable for all coins
"""

import streamlit as st
import pickle
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.coin_analysis import (
    load_ohlc_data,
    get_coin_trades,
    create_candlestick_chart,
    render_llm_reasoning_sidebar
)
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


def render_coin_page(coin_symbol: str):
    """
    Render a coin analysis page

    Args:
        coin_symbol: Coin symbol (e.g., "BTC", "ETH")
    """
    st.title(f"{coin_symbol} Analysis")
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
            key=f"anthropic_checkpoint_{coin_symbol.lower()}"
        )
        checkpoints['Anthropic'] = load_checkpoint(str(checkpoint_dir / selected_anthropic))

    # OpenAI checkpoint selector
    openai_files = [f for f in available_checkpoints if 'openai' in f.lower()]
    if openai_files:
        selected_openai = st.sidebar.selectbox(
            "OpenAI Checkpoint",
            options=openai_files,
            index=len(openai_files) - 1,
            key=f"openai_checkpoint_{coin_symbol.lower()}"
        )
        checkpoints['OpenAI'] = load_checkpoint(str(checkpoint_dir / selected_openai))

    if not checkpoints:
        st.error("❌ No checkpoint files loaded")
        return

    # Load OHLC data
    df = load_ohlc_data(coin_symbol, interval)

    if df is None:
        st.error(f"❌ {coin_symbol} historical data not found. Please run data collection first.")
        st.code(f"python scripts/collect_historical_data.py --coins {coin_symbol} --interval {interval}")
        return

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
        trades_by_model[model_name] = get_coin_trades(checkpoint, coin_symbol)

    # Render sidebar reasoning
    render_llm_reasoning_sidebar(reasoning_data, coin_symbol, selected_models)

    # Main chart
    st.subheader("Price Chart & Indicators")

    chart = create_candlestick_chart(df, trades_by_model, show_indicators, selected_models, coin_symbol, interval)
    st.plotly_chart(chart, use_container_width=True)

"""
LLM Reasoning Sidebar Display

This module contains the sidebar rendering function:
- Display LLM chain-of-thought reasoning
- Show trade decisions and timestamps
- Highlight trade functionality
"""
import pandas as pd
import streamlit as st

def render_llm_reasoning_sidebar(reasoning_data, coin, selected_models, trades_by_model=None):
    """Render LLM reasoning in sidebar with click-to-highlight functionality

    Args:
        trades_by_model: Dict mapping model names to their trades (for checking if highlight is possible)
    """
    st.sidebar.header(f"Model Chat - {coin}")

    if not reasoning_data:
        st.sidebar.info("No reasoning data available")
        return

    # Initialize session state for highlighted timestamp
    if 'highlighted_timestamp' not in st.session_state:
        st.session_state.highlighted_timestamp = None

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
        st.sidebar.markdown(f"### {model_display_name}")
        st.sidebar.markdown(f"*{len(coin_iterations)} total decisions*")

        # Get trade timestamps for this model (to check if highlight is possible)
        model_trade_timestamps = set()
        if trades_by_model:
            # Find matching model in trades_by_model (case-insensitive)
            matching_model = None
            for trade_model_name in trades_by_model.keys():
                if trade_model_name.lower() == model_display_name.lower():
                    matching_model = trade_model_name
                    break

            if matching_model:
                for trade in trades_by_model[matching_model]:
                    trade_ts = pd.to_datetime(trade['timestamp']).isoformat()
                    model_trade_timestamps.add(trade_ts)

        # Show all iterations (reversed to show latest first)
        iterations_to_show = list(reversed(coin_iterations))

        # Show iterations
        for idx, iteration in enumerate(iterations_to_show):
            timestamp = iteration.get('timestamp', 'Unknown')
            raw_response = iteration.get('raw_response', '')

            # Check if there's an actual trade at this timestamp
            timestamp_normalized = pd.to_datetime(timestamp).isoformat() if timestamp != 'Unknown' else None
            has_trade = timestamp_normalized in model_trade_timestamps if timestamp_normalized else False

            # Show full Chain of Thought (no truncation)
            with st.sidebar.expander(f"📅 {timestamp}", expanded=False):
                st.markdown(raw_response)

                # Show signals
                signals = [s for s in iteration.get('signals', []) if s.get('coin') == coin]
                for signal in signals:
                    st.markdown(f"**Action:** {signal.get('signal', 'N/A')}")
                    if signal.get('close_reason'):
                        st.markdown(f"**Reason:** {signal.get('close_reason')}")

            # Show highlight button below the expander if there's an actual trade
            if has_trade:
                button_key = f"highlight_{model_name}_{coin}_{idx}_{timestamp}"
                if st.sidebar.button("🔍 Highlight trade", key=button_key, help="Highlight this trade on the chart"):
                    st.session_state.highlighted_timestamp = timestamp
                    st.rerun()

"""
LLM Trading Arena - Dashboard Home

Page 1: Model Comparison (Leaderboard)
Shows comparative performance between different LLM trading models.
"""

import streamlit as st
import pickle
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Page config
st.set_page_config(
    page_title="LLM Trading Arena",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_data
def load_checkpoint(checkpoint_path: str):
    """Load checkpoint data"""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_reasoning(reasoning_path: str):
    """Load LLM reasoning JSON"""
    import json
    try:
        with open(reasoning_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def extract_metrics(checkpoint):
    """Extract key metrics from checkpoint"""
    account = checkpoint['account']
    metadata = checkpoint.get('metadata', {})
    trade_log = checkpoint.get('trade_history', [])

    # Calculate metrics
    starting_capital = account.get('starting_capital', 10000)
    final_value = account.get('account_value', 0)
    total_return = final_value - starting_capital
    total_return_pct = account.get('total_return_percent', 0)

    # Win rate
    closed_trades = [t for t in trade_log if t['action'] == 'CLOSE']
    winning_trades = [t for t in closed_trades if t.get('net_pnl', 0) > 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0

    # Avg win/loss
    avg_win = sum(t.get('net_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    losing_trades = [t for t in closed_trades if t.get('net_pnl', 0) <= 0]
    avg_loss = sum(t.get('net_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0

    return {
        'model': metadata.get('model', 'Unknown').title(),
        'starting_capital': starting_capital,
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'sharpe_ratio': account.get('sharpe_ratio', 0),
        'win_rate': win_rate,
        'total_trades': account.get('trade_count', 0),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': account.get('max_drawdown_pct', 0)
    }


def create_performance_chart(checkpoints_data):
    """Create interactive performance comparison chart"""
    from frontend.utils.visual_config import get_model_color, get_chart_height

    fig = go.Figure()

    # Track date range from metadata
    min_date = None
    max_date = None

    for model_name, checkpoint in checkpoints_data.items():
        trade_log = checkpoint.get('trade_history', [])
        metadata = checkpoint.get('metadata', {})

        # Update date range from metadata
        if metadata.get('start_date'):
            from datetime import datetime
            start_date = datetime.fromisoformat(metadata['start_date'])
            if min_date is None or start_date < min_date:
                min_date = start_date
        if metadata.get('end_date'):
            from datetime import datetime
            end_date = datetime.fromisoformat(metadata['end_date'])
            if max_date is None or end_date > max_date:
                max_date = end_date

        # Build equity curve by calculating cumulative P&L
        timestamps = []
        account_values = []
        starting_capital = checkpoint['account'].get('starting_capital', 10000)

        # Add starting point
        if trade_log:
            first_trade_time = trade_log[0].get('timestamp')
            timestamps.append(first_trade_time)
            account_values.append(starting_capital)

        # Calculate running account value from cumulative P&L
        current_value = starting_capital
        for trade in trade_log:
            # For CLOSE trades, add the P&L
            if trade['action'] == 'CLOSE' and 'net_pnl' in trade:
                current_value += trade['net_pnl']
                timestamps.append(trade['timestamp'])
                account_values.append(current_value)
            # For BUY trades, subtract fees
            elif trade['action'] == 'BUY' and 'cost' in trade:
                # Fees are included in cost, but we can track the equity curve
                timestamps.append(trade['timestamp'])
                account_values.append(current_value)

        # Get model color from config
        line_color = get_model_color(model_name)

        # Create hover text for the line
        hover_template = (
            f"<b>{model_name}</b><br>" +
            "Date: %{x|%Y-%m-%d %H:%M}<br>" +
            "Account Value: $%{y:,.2f}<br>" +
            "<extra></extra>"
        )

        # Add line trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=account_values,
            mode='lines+markers',
            name=model_name,
            line=dict(color=line_color, width=3),
            marker=dict(size=10),
            hovertemplate=hover_template
        ))

        # Add special marker for the endpoint (with metrics)
        if timestamps and account_values:
            metrics = extract_metrics(checkpoint)

            endpoint_hover = (
                f"<b>{model_name} - Final Results</b><br><br>" +
                f"<b>Final Account Value:</b> ${metrics['final_value']:,.2f}<br>" +
                f"<b>Total Return:</b> ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:+.2f}%)<br>" +
                f"<b>Sharpe Ratio:</b> {metrics['sharpe_ratio']:.2f}<br>" +
                f"<b>Win Rate:</b> {metrics['win_rate']:.1f}%<br>" +
                f"<b>Total Trades:</b> {metrics['total_trades']}<br>" +
                f"<b>Avg Win:</b> ${metrics['avg_win']:,.2f}<br>" +
                f"<b>Avg Loss:</b> ${metrics['avg_loss']:,.2f}<br>" +
                f"<b>Max Drawdown:</b> {metrics['max_drawdown']:.2f}%<br>" +
                "<extra></extra>"
            )

            # Add blinking endpoint marker
            fig.add_trace(go.Scatter(
                x=[timestamps[-1]],
                y=[account_values[-1]],
                mode='markers',
                name=f'{model_name} (Final)',
                marker=dict(
                    size=20,
                    color=line_color,
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                hovertemplate=endpoint_hover,
                showlegend=False
            ))

    # Add baseline at starting capital ($10,000)
    fig.add_hline(
        y=10000,
        line_dash="dot",
        line_color="gray",
        opacity=0.6,
        annotation_text="Starting Capital ($10,000)",
        annotation_position="right"
    )

    # Add benchmark lines (Buy-and-Hold BTC and Equal-Weight Portfolio)
    # Load benchmark historical data for the same period
    import pandas as pd
    try:
        # Load BTC historical data
        btc_file = project_root / "data/archive/historical/4h/BTC.parquet"
        if btc_file.exists() and min_date and max_date:
            btc_df = pd.read_parquet(btc_file)
            btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])

            # Filter to same period as backtests
            mask = (btc_df['timestamp'] >= min_date) & (btc_df['timestamp'] <= max_date)
            btc_period = btc_df[mask].sort_values('timestamp')

            if len(btc_period) > 0:
                # Calculate buy-and-hold equity curve
                entry_price = btc_period.iloc[0]['close']
                btc_timestamps = btc_period['timestamp'].tolist()
                btc_values = [(price / entry_price) * 10000 for price in btc_period['close']]

                # Add Buy-and-Hold BTC line
                fig.add_trace(go.Scatter(
                    x=btc_timestamps,
                    y=btc_values,
                    mode='lines',
                    name='Buy-and-Hold BTC',
                    line=dict(color='rgba(255, 165, 0, 0.3)', width=2, dash='dash'),
                    hovertemplate=(
                        "<b>Buy-and-Hold BTC (Benchmark)</b><br>" +
                        "Date: %{x|%Y-%m-%d %H:%M}<br>" +
                        "Value: $%{y:,.2f}<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=True
                ))

            # Calculate equal-weight portfolio
            coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
            capital_per_coin = 10000 / len(coins)
            equal_weight_values = []

            # Load all coin data
            coin_data = {}
            all_coins_available = True
            for coin in coins:
                coin_file = project_root / f"data/archive/historical/4h/{coin}.parquet"
                if coin_file.exists():
                    df = pd.read_parquet(coin_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    mask = (df['timestamp'] >= min_date) & (df['timestamp'] <= max_date)
                    coin_data[coin] = df[mask].sort_values('timestamp')
                else:
                    all_coins_available = False
                    break

            if all_coins_available and btc_timestamps:
                # Calculate portfolio value at each timestamp
                for ts in btc_timestamps:
                    portfolio_value = 0
                    for coin in coins:
                        coin_df = coin_data[coin]
                        # Find price at this timestamp
                        coin_row = coin_df[coin_df['timestamp'] == ts]
                        if not coin_row.empty:
                            current_price = coin_row.iloc[0]['close']
                            entry_price_coin = coin_data[coin].iloc[0]['close']
                            # Value of this coin position
                            coin_value = capital_per_coin * (current_price / entry_price_coin)
                            portfolio_value += coin_value
                    equal_weight_values.append(portfolio_value)

                # Add Equal-Weight Portfolio line
                if len(equal_weight_values) == len(btc_timestamps):
                    fig.add_trace(go.Scatter(
                        x=btc_timestamps,
                        y=equal_weight_values,
                        mode='lines',
                        name='Equal-Weight Portfolio',
                        line=dict(color='rgba(128, 128, 128, 0.3)', width=2, dash='dot'),
                        hovertemplate=(
                            "<b>Equal-Weight Portfolio (Benchmark)</b><br>" +
                            "Date: %{x|%Y-%m-%d %H:%M}<br>" +
                            "Value: $%{y:,.2f}<br>" +
                            "<extra></extra>"
                        ),
                        showlegend=True
                    ))
    except Exception:
        # Silently skip benchmarks if data loading fails
        pass

    # Layout
    fig.update_layout(
        title={
            'text': "Total Account Value",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#e0e0e0'}
        },
        xaxis_title="Date",
        yaxis_title="Account Value (USD)",
        hovermode='closest',
        template='plotly_dark',
        height=get_chart_height('performance_chart'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[min_date, max_date] if min_date and max_date else None
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickprefix='$',
            tickformat=',.0f'
        )
    )

    return fig


def create_coin_performance_bubble(checkpoint, model_name):
    """
    Create bubble chart showing coin performance impact

    X-axis: Number of trades
    Y-axis: Total P&L (can be negative)
    Bubble size: Average position size
    Bubble color: Win rate %
    """
    from frontend.utils.visual_config import get_coin_color

    coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
    trade_log = checkpoint.get('trade_history', [])

    # Collect data for each coin
    coin_data = []

    for coin in coins:
        # Get all trades for this coin (BUY and CLOSE)
        coin_trades = [t for t in trade_log if t.get('symbol') == coin]
        closed_trades = [t for t in coin_trades if t['action'] == 'CLOSE']

        if closed_trades:
            # Calculate metrics
            num_trades = len(closed_trades)
            total_pnl = sum(t.get('net_pnl', 0) for t in closed_trades)
            winning_trades = [t for t in closed_trades if t.get('net_pnl', 0) > 0]
            win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0

            # Calculate average position size (capital deployed)
            buy_trades = [t for t in coin_trades if t['action'] == 'BUY']
            avg_position_size = sum(t.get('cost', 0) for t in buy_trades) / len(buy_trades) if buy_trades else 0

            coin_data.append({
                'coin': coin,
                'num_trades': num_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'avg_position_size': avg_position_size,
                'color': get_coin_color(coin)
            })

    if not coin_data:
        # No trades at all, return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No trade data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#888888')
        )
        fig.update_layout(
            template='plotly_dark',
            height=500,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Create bubble chart
    fig = go.Figure()

    for coin_info in coin_data:
        # Use coin-specific color from allocation chart
        color = coin_info['color']

        # Bubble size based on average position size (normalized)
        max_position = max(c['avg_position_size'] for c in coin_data)
        bubble_size = (coin_info['avg_position_size'] / max_position * 40 + 10) if max_position > 0 else 20

        fig.add_trace(go.Scatter(
            x=[coin_info['num_trades']],
            y=[coin_info['total_pnl']],
            mode='markers+text',
            marker=dict(
                size=bubble_size,
                color=color,
                opacity=0.6,
                line=dict(width=0)
            ),
            text=coin_info['coin'],
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            hovertemplate=(
                f"<b>{coin_info['coin']}</b><br>"
                f"Trades: {coin_info['num_trades']}<br>"
                f"Total P&L: ${coin_info['total_pnl']:,.2f}<br>"
                f"Win Rate: {coin_info['win_rate']:.1f}%<br>"
                f"Avg Position: ${coin_info['avg_position_size']:,.2f}"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title={
            'text': f'{model_name} - Coin Impact',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#e0e0e0'},
            'y': 0.98,
            'yanchor': 'top'
        },
        xaxis=dict(
            title='Number of Trades',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(size=10, color='#e0e0e0'),
            title_font=dict(size=12, color='#e0e0e0'),
            zeroline=False
        ),
        yaxis=dict(
            title='Total P&L ($)',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickfont=dict(size=10, color='#e0e0e0'),
            title_font=dict(size=12, color='#e0e0e0'),
            tickprefix='$',
            tickformat=',.0f',
            zeroline=False
        ),
        template='plotly_dark',
        height=400,
        hovermode='closest',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


def create_portfolio_allocation_chart(checkpoint, model_name):
    """
    Create stacked area chart showing portfolio allocation over time

    Shows how capital is distributed across coins and cash throughout the backtest
    """
    trade_log = checkpoint.get('trade_history', [])
    account = checkpoint['account']
    starting_capital = account.get('starting_capital', 10000)

    if not trade_log:
        return None

    # Track portfolio state at each timestamp
    timestamps = []
    cash_values = []
    coin_allocations = {}  # {coin: [values over time]}

    # Initialize
    coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
    for coin in coins:
        coin_allocations[coin] = []

    # Track current state
    current_cash = starting_capital
    open_positions = {}  # {coin: {'quantity': X, 'entry_price': Y, 'notional': Z}}

    # Add starting point
    timestamps.append(trade_log[0]['timestamp'])
    cash_values.append(current_cash)
    for coin in coins:
        coin_allocations[coin].append(0)

    # Process each trade
    for trade in trade_log:
        ts = trade['timestamp']
        symbol = trade.get('symbol')
        action = trade['action']

        if action == 'BUY':
            # Open position
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            leverage = trade.get('leverage', 1)

            # Notional value (total position value)
            notional = quantity * price

            # Capital used (margin)
            capital_used = notional / leverage

            open_positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'notional': notional,
                'capital_used': capital_used,
                'current_price': price
            }

            # Update cash (subtract capital used + fees)
            current_cash -= capital_used

        elif action == 'CLOSE':
            # Close position
            pnl = trade.get('net_pnl', 0)
            symbol = trade.get('symbol')

            if symbol in open_positions:
                # Return capital + P&L
                current_cash += open_positions[symbol]['capital_used'] + pnl
                del open_positions[symbol]

        # Record snapshot at this timestamp
        timestamps.append(ts)
        cash_values.append(current_cash)

        for coin in coins:
            if coin in open_positions:
                # Use current market value of position
                pos = open_positions[coin]
                coin_allocations[coin].append(pos['capital_used'])
            else:
                coin_allocations[coin].append(0)

    # Create stacked area chart
    fig = go.Figure()

    # Color scheme for coins
    coin_colors = {
        'BTC': '#F7931A',  # Bitcoin orange
        'ETH': '#627EEA',  # Ethereum blue
        'SOL': '#14F195',  # Solana green
        'BNB': '#F3BA2F',  # Binance yellow
        'XRP': '#23292F',  # XRP black
        'DOGE': '#C2A633', # Doge gold
        'CASH': '#888888'  # Cash gray
    }

    # Add cash first (bottom layer)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cash_values,
        name='Cash',
        mode='lines',
        line=dict(width=0, color=coin_colors['CASH']),
        fillcolor='rgba(136, 136, 136, 0.4)',
        fill='tozeroy',
        stackgroup='one',
        hovertemplate='Cash: $%{y:,.2f}<extra></extra>'
    ))

    # Add each coin
    for coin in coins:
        if any(v > 0 for v in coin_allocations[coin]):
            color = coin_colors.get(coin, '#888888')
            # Convert hex to rgba
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=coin_allocations[coin],
                name=coin,
                mode='lines',
                line=dict(width=0, color=color),
                fillcolor=f'rgba({r}, {g}, {b}, 0.6)',
                stackgroup='one',
                hovertemplate=f'{coin}: $%{{y:,.2f}}<extra></extra>'
            ))

    fig.update_layout(
        title={
            'text': f'{model_name} - Portfolio Allocation Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#e0e0e0'},
            'y': 0.98,
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Capital Allocation (USD)',
        template='plotly_dark',
        height=400,
        hovermode='x unified',
        yaxis=dict(tickprefix='$', tickformat=',.0f'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig


def render_trade_timeline(checkpoints_data, reasoning_data):
    """Render LLM reasoning timeline in sidebar"""
    st.sidebar.header("Trade Timeline")
    st.sidebar.markdown("*Latest trades first*")

    # Collect all trades from all models
    all_trades = []

    for model_name, checkpoint in checkpoints_data.items():
        trade_log = checkpoint.get('trade_history', [])

        for trade in trade_log:
            if trade['action'] in ['BUY', 'CLOSE']:
                all_trades.append({
                    'timestamp': trade.get('timestamp'),
                    'model': model_name,
                    'coin': trade.get('symbol'),
                    'action': trade['action'],
                    'price': trade.get('price') if trade['action'] == 'BUY' else trade.get('exit_price'),
                    'pnl': trade.get('net_pnl') if trade['action'] == 'CLOSE' else None,
                    'reason': trade.get('reason', '')
                })

    # Sort by timestamp (latest first)
    all_trades.sort(key=lambda x: x['timestamp'], reverse=True)

    # Display trades
    for trade in all_trades[:20]:  # Show latest 20 trades
        timestamp_str = trade['timestamp'].strftime('%Y-%m-%d %H:%M')

        # Color based on action
        if trade['action'] == 'BUY':
            icon = "🟢"
        else:
            icon = "🔴" if trade['pnl'] and trade['pnl'] < 0 else "🟢"

        # Create expandable section
        expander_label = f"{icon} {trade['model']} - {trade['coin']} {trade['action']}"
        with st.sidebar.expander(expander_label, expanded=False):
            st.markdown(f"**📅 Time:** {timestamp_str}")
            st.markdown(f"**💰 Price:** ${trade['price']:,.2f}")

            if trade['pnl'] is not None:
                pnl_color = "green" if trade['pnl'] >= 0 else "red"
                st.markdown(f"**📊 P&L:** <span style='color:{pnl_color}'>${trade['pnl']:,.2f}</span>", unsafe_allow_html=True)

            if trade['reason']:
                st.markdown(f"**📝 Reason:** {trade['reason']}")

# Main app
def main():
    st.title("LLM Trading Arena")
    st.markdown("**Comparing LLM performance in autonomous crypto trading**")

    # Load checkpoints (recursively scan results directory)
    results_dir = project_root / "results"

    # Get all available checkpoint files with natural sort (recursively)
    from frontend.utils.checkpoint_utils import natural_sort_key
    checkpoint_paths = sorted([f for f in results_dir.rglob("*.pkl")], key=natural_sort_key)

    if not checkpoint_paths:
        st.error("❌ No checkpoint files found in results/")
        st.info("Run a backtest first: `python scripts/run_backtest.py --start 2025-10-18 --end 2025-10-30 --model anthropic --run-id 1`")
        return

    # Sidebar: Configuration
    st.sidebar.header("Configuration")

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

    # Temperature selector (primary control)
    st.sidebar.markdown("### Select Temperature")
    temp_choice = st.sidebar.radio(
        "Temperature",
        options=['0.7', '0.1'],
        index=0,  # Default to 0.7
        label_visibility="collapsed"
    )

    # Map display to folder name
    temp_folder_map = {
        '0.7': 'temp07',
        '0.1': 'temp01'
    }
    selected_temp_folder = temp_folder_map[temp_choice]

    # Load latest trial for each model at selected temperature
    selected_checkpoints = {}
    selected_files = {}

    from frontend.utils.checkpoint_utils import natural_sort_key

    for model, temp_data in sorted(models_with_temps.items()):
        if selected_temp_folder in temp_data:
            # Get latest checkpoint (last in naturally sorted list)
            checkpoints = sorted(temp_data[selected_temp_folder], key=natural_sort_key)
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                display_name = model.replace('-', ' ').title()
                selected_checkpoints[display_name] = load_checkpoint(str(latest_checkpoint))
                selected_files[display_name] = latest_checkpoint

    if not selected_checkpoints:
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
            format_func=lambda x: x.replace('-', ' ').title() if x != 'None' else 'Use default'
        )

        if override_model != 'None':
            # Show trials for this model at selected temperature
            if selected_temp_folder in models_with_temps[override_model]:
                trial_checkpoints = sorted(models_with_temps[override_model][selected_temp_folder], key=natural_sort_key)
                trial_names = [p.name for p in trial_checkpoints]

                override_trial = st.selectbox(
                    "Trial",
                    options=trial_names,
                    index=len(trial_names) - 1
                )

                # Replace the default checkpoint for this model
                override_checkpoint_path = [p for p in trial_checkpoints if p.name == override_trial][0]
                display_name = override_model.replace('-', ' ').title()
                selected_checkpoints[display_name] = load_checkpoint(str(override_checkpoint_path))
                selected_files[display_name] = override_checkpoint_path

                st.success(f"✓ Using {override_trial} for {display_name}")

    # Show current configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current View")
    st.sidebar.markdown(f"**Temperature:** {temp_choice}")
    st.sidebar.markdown(f"**Models shown:** {len(selected_checkpoints)}")
    for model_name, checkpoint_path in selected_files.items():
        st.sidebar.markdown(f"- {model_name}: `{checkpoint_path.name}`")

    existing_checkpoints = selected_checkpoints

    if not existing_checkpoints:
        st.error("❌ No checkpoint files found. Please run backtests first.")
        st.info("Run: `python scripts/run_backtest.py --model anthropic --start 2025-10-18 --end 2025-10-29`")
        return

    # Load reasoning data (match checkpoint filenames)
    reasoning_data = {}
    for display_name, full_path in selected_files.items():
        # Reasoning file is in same directory as checkpoint
        reasoning_path = full_path.parent / f"{full_path.stem}_reasoning.json"
        if reasoning_path.exists():
            # Use lowercase model name as key for reasoning data (for backward compatibility)
            model_key = display_name.lower().replace('-', '')
            reasoning_data[model_key] = load_reasoning(str(reasoning_path))

    # Render sidebar timeline
    render_trade_timeline(existing_checkpoints, reasoning_data)

    chart = create_performance_chart(existing_checkpoints)
    st.plotly_chart(chart, use_container_width=True)

    # Quick stats below chart
    st.markdown("---")
    st.subheader("Performance Comparison")

    cols = st.columns(len(existing_checkpoints))

    for idx, (model_name, checkpoint) in enumerate(existing_checkpoints.items()):
        metrics = extract_metrics(checkpoint)

        with cols[idx]:
            st.metric(
                label=f"**{model_name}**",
                value=f"${metrics['final_value']:,.2f}",
                delta=f"{metrics['total_return_pct']:+.2f}%"
            )
            st.markdown(f"**Sharpe:** {metrics['sharpe_ratio']:.2f}")
            st.markdown(f"**Win Rate:** {metrics['win_rate']:.1f}%")
            st.markdown(f"**Trades:** {metrics['total_trades']}")

    # Portfolio allocation comparison
    st.markdown("---")
    st.subheader("Portfolio Allocation & Coin Performance")
    st.markdown("*Comparing capital allocation behavior and per-coin impact*")

    # Show allocation + bubble chart for each model (side by side)
    for model_name, checkpoint in existing_checkpoints.items():
        # Create 2 columns for this model
        cols = st.columns(2)

        # Column 1: Allocation chart
        with cols[0]:
            allocation_chart = create_portfolio_allocation_chart(checkpoint, model_name)
            if allocation_chart:
                st.plotly_chart(allocation_chart, use_container_width=True)
            else:
                st.warning(f"No trade data available for {model_name}")

        # Column 2: Bubble chart
        with cols[1]:
            bubble_chart = create_coin_performance_bubble(checkpoint, model_name)
            st.plotly_chart(bubble_chart, use_container_width=True)

        # Add spacing between model rows
        if len(existing_checkpoints) > 1:
            st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

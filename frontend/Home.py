"""
LLM Trading Arena - Dashboard Home

Page 1: Model Comparison (Leaderboard)
Shows comparative performance between different LLM trading models.
"""

import streamlit as st
import pickle
import pandas as pd
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

# Custom CSS for blinking effect
st.markdown("""
<style>
@keyframes blink {
    0%, 50%, 100% { opacity: 1; }
    25%, 75% { opacity: 0.5; }
}
.blinking {
    animation: blink 2s infinite;
}
</style>
""", unsafe_allow_html=True)


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
    fig = go.Figure()

    colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#95e1d3']  # Colors for different models

    # Track date range from metadata
    min_date = None
    max_date = None

    for idx, (model_name, checkpoint) in enumerate(checkpoints_data.items()):
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
            line=dict(color=colors[idx % len(colors)], width=3),
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
                    color=colors[idx % len(colors)],
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

    # Layout
    fig.update_layout(
        title={
            'text': "LLM Trading Performance Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#e0e0e0'}
        },
        xaxis_title="Date",
        yaxis_title="Account Value (USD)",
        hovermode='closest',
        template='plotly_dark',
        height=600,
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
            cost = trade.get('cost', 0)
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
        line=dict(width=0.5, color=coin_colors['CASH']),
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
                line=dict(width=0.5, color=color),
                fillcolor=f'rgba({r}, {g}, {b}, 0.6)',
                stackgroup='one',
                hovertemplate=f'{coin}: $%{{y:,.2f}}<extra></extra>'
            ))

    fig.update_layout(
        title=f'{model_name} - Portfolio Allocation Over Time',
        xaxis_title='Date',
        yaxis_title='Capital Allocation (USD)',
        template='plotly_dark',
        height=500,
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
            color = "#00ff88"
        else:
            icon = "🔴" if trade['pnl'] and trade['pnl'] < 0 else "🟢"
            color = "#ff4444" if trade['pnl'] and trade['pnl'] < 0 else "#00ff88"

        # Create expandable section
        with st.sidebar.expander(f"{icon} {trade['model']} - {trade['coin']} {trade['action']}", expanded=False):
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

    # Load checkpoints
    checkpoint_dir = project_root / "results" / "checkpoints"

    # Get all available checkpoint files with natural sort
    from frontend.utils.checkpoint_utils import natural_sort_key
    available_checkpoints = sorted([f.name for f in checkpoint_dir.glob("*.pkl")], key=natural_sort_key)

    if not available_checkpoints:
        st.error("❌ No checkpoint files found in results/checkpoints/")
        st.info("Run a backtest first: `python scripts/run_backtest.py --start 2025-10-18 --end 2025-10-30 --model anthropic`")
        return

    # Sidebar: Checkpoint selection
    st.sidebar.header("Select Checkpoints")

    selected_checkpoints = {}

    # Anthropic checkpoint selector
    anthropic_files = [f for f in available_checkpoints if 'anthropic' in f.lower()]
    if anthropic_files:
        default_anthropic = anthropic_files[-1]  # Most recent
        selected_anthropic = st.sidebar.selectbox(
            "Anthropic Checkpoint",
            options=anthropic_files,
            index=len(anthropic_files) - 1,
            key="anthropic_checkpoint"
        )
        selected_checkpoints['Anthropic'] = load_checkpoint(str(checkpoint_dir / selected_anthropic))

    # OpenAI checkpoint selector
    openai_files = [f for f in available_checkpoints if 'openai' in f.lower()]
    if openai_files:
        default_openai = openai_files[-1]  # Most recent
        selected_openai = st.sidebar.selectbox(
            "OpenAI Checkpoint",
            options=openai_files,
            index=len(openai_files) - 1,
            key="openai_checkpoint"
        )
        selected_checkpoints['OpenAI'] = load_checkpoint(str(checkpoint_dir / selected_openai))

    existing_checkpoints = selected_checkpoints

    if not existing_checkpoints:
        st.error("❌ No checkpoint files found. Please run backtests first.")
        st.info("Run: `python scripts/run_backtest.py --model anthropic --start 2025-10-18 --end 2025-10-29`")
        return

    # Load reasoning data (match checkpoint filenames)
    reasoning_data = {}
    if anthropic_files and 'Anthropic' in existing_checkpoints:
        reasoning_path = checkpoint_dir / selected_anthropic.replace('.pkl', '_reasoning.json')
        if reasoning_path.exists():
            reasoning_data['anthropic'] = load_reasoning(str(reasoning_path))

    if openai_files and 'OpenAI' in existing_checkpoints:
        reasoning_path = checkpoint_dir / selected_openai.replace('.pkl', '_reasoning.json')
        if reasoning_path.exists():
            reasoning_data['openai'] = load_reasoning(str(reasoning_path))

    # Render sidebar timeline
    render_trade_timeline(existing_checkpoints, reasoning_data)

    # Main content: Performance chart
    st.subheader("Performance Over Time")
    st.markdown("*Hover over the diamond markers at the end to see detailed metrics*")

    chart = create_performance_chart(existing_checkpoints)
    st.plotly_chart(chart, use_container_width=True)

    # Quick stats below chart
    st.markdown("---")
    st.subheader("Quick Comparison")

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
    st.subheader("Portfolio Allocation Behavior")
    st.markdown("*Comparing how each model allocates capital across coins and cash over time*")

    # Create side-by-side allocation charts
    if len(existing_checkpoints) == 2:
        cols = st.columns(2)

        for idx, (model_name, checkpoint) in enumerate(existing_checkpoints.items()):
            with cols[idx]:
                allocation_chart = create_portfolio_allocation_chart(checkpoint, model_name)
                if allocation_chart:
                    st.plotly_chart(allocation_chart, use_container_width=True)
                else:
                    st.warning(f"No trade data available for {model_name}")
    else:
        # Single model or > 2 models - show in single column
        for model_name, checkpoint in existing_checkpoints.items():
            allocation_chart = create_portfolio_allocation_chart(checkpoint, model_name)
            if allocation_chart:
                st.plotly_chart(allocation_chart, use_container_width=True)
            else:
                st.warning(f"No trade data available for {model_name}")


if __name__ == "__main__":
    main()

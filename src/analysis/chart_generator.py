"""
Chart Generator

Generates performance charts for backtest analysis.
"""

from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set modern seaborn style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.color'] = '#2a2a2a'
plt.rcParams['text.color'] = '#e0e0e0'
plt.rcParams['axes.labelcolor'] = '#e0e0e0'
plt.rcParams['xtick.color'] = '#a0a0a0'
plt.rcParams['ytick.color'] = '#a0a0a0'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def generate_performance_chart(trade_log: list, starting_capital: float, output_dir: Path, metadata: dict = None):
    """
    Generate performance charts with trade markers

    Args:
        trade_log: List of trade dictionaries
        starting_capital: Starting account balance
        output_dir: Directory to save chart (results/reports/)
        metadata: Optional metadata dict with model name, dates, etc.

    Returns:
        Chart filename (without path)
    """

    # Calculate account value over time
    timestamps = []
    account_values = []
    trade_markers = {'BUY': [], 'CLOSE': []}

    current_cash = starting_capital
    open_positions = {}  # Track open positions by symbol

    # Use backtest date range if available
    if metadata and 'start_date' in metadata and 'end_date' in metadata:
        start_dt = datetime.fromisoformat(metadata['start_date'])
        end_dt = datetime.fromisoformat(metadata['end_date'])

        # Evenly distribute trades across the backtest period
        total_duration = (end_dt - start_dt).total_seconds()
        time_per_trade = total_duration / len(trade_log) if trade_log else 1

        # Replace trade timestamps with evenly spaced historical timestamps
        for i, trade in enumerate(trade_log):
            historical_timestamp = start_dt + timedelta(seconds=i * time_per_trade)
            trade['historical_timestamp'] = historical_timestamp

        first_timestamp = start_dt
    else:
        # Fallback to original timestamps
        first_timestamp = trade_log[0]['timestamp'] if trade_log else datetime.now()
        for trade in trade_log:
            trade['historical_timestamp'] = trade['timestamp']

    timestamps.append(first_timestamp)
    account_values.append(current_cash)

    for trade in trade_log:
        hist_ts = trade['historical_timestamp']

        if trade['action'] == 'BUY':
            # Deduct cash used (margin + entry fee)
            current_cash -= trade['cost']
            # Track the open position
            open_positions[trade['symbol']] = {
                'cost': trade['cost'],
                'entry_price': trade['price'],
                'quantity': trade['quantity']
            }
            # Account value = available cash (open positions not counted in cash)
            account_value = current_cash
            trade_markers['BUY'].append((hist_ts, account_value, trade['symbol']))

        elif trade['action'] == 'CLOSE':
            symbol = trade['symbol']
            # Return the margin that was locked
            if symbol in open_positions:
                current_cash += open_positions[symbol]['cost']
                del open_positions[symbol]
            # Add the net P&L
            current_cash += trade['net_pnl']
            # Account value = available cash
            account_value = current_cash
            trade_markers['CLOSE'].append((hist_ts, account_value, trade['symbol'], trade['net_pnl']))

        timestamps.append(hist_ts)
        account_values.append(account_value)

    # Create modern styled plot
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

    # Plot: Account Value Over Time with gradient effect
    ax1.plot(timestamps, account_values, linewidth=3, color='#00d4ff', label='Portfolio Value',
             alpha=0.9, zorder=3)

    # Fill area under curve for visual effect
    ax1.fill_between(timestamps, account_values, starting_capital,
                     where=[v >= starting_capital for v in account_values],
                     alpha=0.2, color='#00ff88', interpolate=True, label='Profit Zone')
    ax1.fill_between(timestamps, account_values, starting_capital,
                     where=[v < starting_capital for v in account_values],
                     alpha=0.2, color='#ff4444', interpolate=True, label='Loss Zone')

    # Starting capital line
    ax1.axhline(y=starting_capital, color='#888888', linestyle='--', alpha=0.6,
               linewidth=2, label=f'Initial Capital: ${starting_capital:,.0f}', zorder=2)

    # Mark BUY trades with modern styling
    if trade_markers['BUY']:
        buy_times, buy_values, buy_symbols = zip(*trade_markers['BUY'])
        ax1.scatter(buy_times, buy_values, color='#00ff88', s=150, marker='^',
                   alpha=0.9, label='Entry', zorder=5, edgecolors='#00cc66', linewidths=2)

    # Mark CLOSE trades with profit/loss colors - separate legend entries
    if trade_markers['CLOSE']:
        close_times, close_values, close_symbols, close_pnls = zip(*trade_markers['CLOSE'])

        # Separate wins and losses
        win_times = [t for t, pnl in zip(close_times, close_pnls) if pnl > 0]
        win_values = [v for v, pnl in zip(close_values, close_pnls) if pnl > 0]
        loss_times = [t for t, pnl in zip(close_times, close_pnls) if pnl <= 0]
        loss_values = [v for v, pnl in zip(close_values, close_pnls) if pnl <= 0]

        # Plot winning exits
        if win_times:
            ax1.scatter(win_times, win_values, color='#00ff88', s=150, marker='v',
                       alpha=0.9, label='Exit (Win)', zorder=5, edgecolors='#00cc66', linewidths=2)

        # Plot losing exits
        if loss_times:
            ax1.scatter(loss_times, loss_values, color='#ff4444', s=150, marker='v',
                       alpha=0.9, label='Exit (Loss)', zorder=5, edgecolors='#cc0000', linewidths=2)

    # Styling
    ax1.set_xlabel('Date', fontsize=13, fontweight='600', color='#e0e0e0', labelpad=10)
    ax1.set_ylabel('Account Value (USD)', fontsize=13, fontweight='600', color='#e0e0e0', labelpad=10)
    ax1.set_title('Portfolio Performance', fontsize=18, fontweight='700',
                 color='#ffffff', pad=25, loc='left')

    # Legend with modern styling
    legend = ax1.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True,
                       fontsize=10, edgecolor='#444444', facecolor='#1a1a1a')
    legend.get_frame().set_linewidth(1.5)
    for text in legend.get_texts():
        text.set_color('#e0e0e0')

    # Grid styling
    ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#444444')
    ax1.set_axisbelow(True)

    # Format y-axis with currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Format x-axis based on date range
    date_range = (timestamps[-1] - timestamps[0]).days if len(timestamps) > 1 else 1
    if date_range > 7:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    else:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=10)

    # Improve tick styling
    ax1.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5,
                   colors='#a0a0a0', direction='out')

    # Add subtle spine styling
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')

    plt.tight_layout()

    # Generate chart filename based on model and timestamp
    if metadata:
        model_name = metadata.get('model', 'unknown').lower()
        timestamp = datetime.now().strftime('%d%m%y_%H%M')
        chart_filename = f'{model_name}_chart_{timestamp}.png'
    else:
        chart_filename = 'performance_chart.png'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save chart with high quality
    chart_path = output_dir / chart_filename
    plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='#0a0a0a',
               edgecolor='none')
    plt.close()

    print(f"✓ Chart saved to: {chart_path}")
    return chart_filename

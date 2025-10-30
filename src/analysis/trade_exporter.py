"""
Trade Exporter

Exports trade logs to CSV and markdown formats.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


def export_trades_to_csv(trade_log: list, metadata: dict = None):
    """
    Export trade log as CSV and markdown table

    Args:
        trade_log: List of trade dictionaries
        metadata: Optional metadata dict with model name, etc.

    Returns:
        Tuple of (csv_path, markdown_table_string)
    """
    # Prepare trade data for export
    trades_data = []

    for i, trade in enumerate(trade_log, 1):
        timestamp = trade.get('timestamp', datetime.now())

        if trade['action'] == 'BUY':
            trades_data.append({
                'Trade #': i,
                'Date': timestamp.strftime('%Y-%m-%d'),
                'Time': timestamp.strftime('%H:%M:%S'),
                'Action': 'OPEN',
                'Coin': trade['symbol'],
                'Price': f"${trade['price']:.2f}",
                'Quantity': f"{trade['quantity']:.4f}",
                'Notional': f"${trade['price'] * trade['quantity']:.2f}",
                'Leverage': f"{trade['leverage']}x",
                'Cost': f"${trade['cost']:.2f}",
                'Fee': '-',
                'P&L': '-',
                'Account Value': f"${trade.get('account_value', 0):.2f}",
                'Reason': 'NEW_POSITION'
            })
        elif trade['action'] == 'CLOSE':
            trades_data.append({
                'Trade #': i,
                'Date': timestamp.strftime('%Y-%m-%d'),
                'Time': timestamp.strftime('%H:%M:%S'),
                'Action': 'CLOSE',
                'Coin': trade['symbol'],
                'Price': f"${trade['exit_price']:.2f}",
                'Quantity': f"{trade['quantity']:.4f}",
                'Notional': f"${trade['exit_price'] * trade['quantity']:.2f}",
                'Leverage': '-',
                'Cost': '-',
                'Fee': f"${trade.get('exit_fee', 0):.2f}",
                'P&L': f"${trade['net_pnl']:.2f}",
                'Account Value': f"${trade.get('account_value', 0):.2f}",
                'Reason': trade.get('reason', 'UNKNOWN')
            })

    # Create DataFrame
    df = pd.DataFrame(trades_data)

    # Generate CSV filename
    if metadata:
        model_name = metadata.get('model', 'unknown').lower()
        timestamp = datetime.now().strftime('%d%m%y_%H%M')
        csv_filename = f'{model_name}_trades_{timestamp}.csv'
    else:
        csv_filename = 'trades.csv'

    # CSV goes to results/csv/ directory
    csv_dir = Path("results/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = csv_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"✓ Trade log CSV saved to: {csv_path}")

    # Generate markdown table
    md_table = df.to_markdown(index=False)

    return str(csv_path), md_table

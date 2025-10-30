"""
Report Generator

Generates comprehensive markdown reports from backtest data.
"""

from pathlib import Path
from datetime import datetime


def generate_markdown_report(
    checkpoint: dict,
    checkpoint_path: str,
    chart_filename: str,
    csv_path: str,
    trade_table_md: str,
    output_path: Path = None
):
    """
    Generate comprehensive markdown report from backtest data

    Args:
        checkpoint: Loaded checkpoint dict
        checkpoint_path: Path to checkpoint file
        chart_filename: Filename of the generated chart (for markdown link)
        csv_path: Path to the exported CSV file
        trade_table_md: Markdown table string of trades
        output_path: Optional custom output path

    Returns:
        Path to the generated markdown report
    """
    metadata = checkpoint.get('metadata', {})
    account = checkpoint['account']
    trade_log = checkpoint.get('trade_history', [])

    # Generate markdown report
    lines = []
    lines.append("# Backtest Results Analysis")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Checkpoint:** `{checkpoint_path}`")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Checkpoint ID:** {checkpoint.get('checkpoint_id', 'N/A')}")
    lines.append(f"- **Created At:** {checkpoint.get('created_at', 'N/A')}")
    lines.append(f"- **Checkpoint Date:** {checkpoint.get('checkpoint_date', 'N/A')}")
    lines.append(f"- **Starting Capital:** ${account.get('starting_capital', 0):,.2f}")
    lines.append(f"- **Model:** {metadata.get('model', 'N/A')}")
    lines.append(f"- **Interval:** {metadata.get('interval', 'N/A')}")
    lines.append(f"- **Coins:** {', '.join(metadata.get('coins', []))}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    starting_capital = account.get('starting_capital', 0)
    final_value = account.get('account_value', 0)
    total_return = final_value - starting_capital
    total_return_pct = account.get('total_return_percent', 0)
    sharpe = account.get('sharpe_ratio', 0)
    trade_count = account.get('trade_count', 0)

    # Calculate win rate from trade log
    closed_trades = [t for t in trade_log if t['action'] == 'CLOSE']
    winning_trades = [t for t in closed_trades if t.get('net_pnl', 0) > 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0

    checkpoint_date = checkpoint.get('checkpoint_date', 'N/A')
    lines.append(f"The backtest checkpoint was created on **{checkpoint_date}**.")
    lines.append("")

    # Format return string nicely
    if total_return >= 0:
        return_text = f"a gain of ${total_return:,.2f} (+{total_return_pct:.2f}%)"
    else:
        return_text = f"a loss of ${abs(total_return):,.2f} ({total_return_pct:.2f}%)"

    lines.append(f"Starting with ${starting_capital:,.2f}, the strategy ended with ${final_value:,.2f}, representing {return_text}.")
    lines.append("")
    lines.append(f"The strategy executed {trade_count} trades with a {win_rate:.1f}% win rate and a Sharpe ratio of {sharpe:.2f}.")
    lines.append("")

    # Performance Chart (inserted here, before metrics)
    lines.append("## Performance Chart")
    lines.append("")
    lines.append(f"![Performance Chart]({chart_filename})")
    lines.append("")
    lines.append("*Chart shows account value over time with trade markers:*")
    lines.append("- *Green triangles (▲) = BUY trades*")
    lines.append("- *Green/red triangles (▼) = CLOSE trades (green = profit, red = loss)*")
    lines.append("- *Gray dashed line = starting capital*")
    lines.append("")

    # Performance Metrics
    lines.append("## Performance Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Final Account Value | ${account.get('account_value', 0):,.2f} |")
    lines.append(f"| Total Return | ${total_return:,.2f} ({total_return_pct:+.2f}%) |")
    lines.append(f"| Sharpe Ratio | {sharpe:.3f} |")
    lines.append(f"| Win Rate | {win_rate:.1f}% |")
    lines.append(f"| Available Cash | ${account.get('available_cash', 0):,.2f} |")
    lines.append("")

    # Trade Statistics
    lines.append("## Trade Statistics")
    lines.append("")

    losing_trades = [t for t in closed_trades if t.get('net_pnl', 0) <= 0]
    avg_pnl = sum(t.get('net_pnl', 0) for t in closed_trades) / len(closed_trades) if closed_trades else 0
    avg_win = sum(t.get('net_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get('net_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Trades | {trade_count} |")
    lines.append(f"| Winning Trades | {len(winning_trades)} |")
    lines.append(f"| Losing Trades | {len(losing_trades)} |")
    lines.append(f"| Average P&L per Trade | ${avg_pnl:,.2f} |")
    lines.append(f"| Average Winning Trade | ${avg_win:,.2f} |")
    lines.append(f"| Average Losing Trade | ${avg_loss:,.2f} |")
    lines.append("")

    # Cost Analysis
    lines.append("## Cost Analysis")
    lines.append("")
    total_fees = account.get('total_fees_paid', 0)
    total_funding = account.get('total_funding_paid', 0)
    total_costs = total_fees + total_funding

    lines.append("| Cost Type | Amount |")
    lines.append("|-----------|--------|")
    lines.append(f"| Total Fees Paid | ${total_fees:,.2f} |")
    lines.append(f"| Total Funding Paid | ${total_funding:,.2f} |")
    lines.append(f"| **Total Costs** | **${total_costs:,.2f}** |")
    lines.append("")

    # Trades by Symbol
    lines.append("## Trades by Symbol")
    lines.append("")

    # Group trades by symbol
    trades_by_symbol = {}
    for trade in trade_log:
        symbol = trade['symbol']
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(trade)

    lines.append("| Symbol | Opens | Closes | Total P&L |")
    lines.append("|--------|-------|--------|-----------|")
    for symbol, trades in sorted(trades_by_symbol.items()):
        buys = [t for t in trades if t['action'] == 'BUY']
        closes = [t for t in trades if t['action'] == 'CLOSE']
        total_pnl = sum(t.get('net_pnl', 0) for t in closes)
        lines.append(f"| {symbol} | {len(buys)} | {len(closes)} | ${total_pnl:,.2f} |")
    lines.append("")

    # Detailed Trade Log
    lines.append("## Detailed Trade Log")
    lines.append("")
    lines.append("Complete trade history exported to CSV for analysis and plotting.")
    lines.append("")
    lines.append(f"**CSV Export:** `{csv_path}`")
    lines.append("")
    lines.append(trade_table_md)
    lines.append("")

    # LLM Cache Stats
    lines.append("## LLM Cache Statistics")
    lines.append("")
    lines.append(f"- **Total Cached Responses:** {len(checkpoint.get('llm_cache', {}))}")
    lines.append("")

    # Determine output path
    if output_path is None:
        # Format: modelname_performance_ddmmyy_hhmm.md
        model_name = metadata.get('model', 'unknown').lower()
        timestamp = datetime.now().strftime('%d%m%y_%H%M')
        output_path = Path(f"results/reports/{model_name}_performance_{timestamp}.md")

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Analysis report saved to: {output_path}")
    return output_path

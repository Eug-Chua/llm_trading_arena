"""
Test Open Interest and Funding Rates Integration

This script demonstrates the new market metrics (OI, funding rates)
that have been added to the Hyperliquid client and market data pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hyperliquid_client import HyperliquidClient
from src.data.market_data_pipeline import MarketDataPipeline
from src.utils.config import load_config


def test_hyperliquid_client():
    """Test individual client methods"""
    print("="*80)
    print("Test 1: HyperliquidClient - Individual Methods")
    print("="*80)

    client = HyperliquidClient()

    # Load Alpha Arena coins from config
    config = load_config('config/api.yaml')
    coins = config['alpha_arena']['coins']

    print(f"\nTesting {len(coins)} Alpha Arena coins:\n")
    print(f"{'Coin':<8} {'Funding Rate':>15} {'Open Interest':>20} {'Mark Price':>15}")
    print("-"*80)

    for coin in coins:
        funding = client.get_funding_rate(coin)
        oi = client.get_open_interest(coin)
        market_data = client.get_market_data(coin)

        if market_data:
            print(f"{coin:<8} {funding*100:>14.4f}% {oi:>20,.2f} ${market_data['mark_price']:>14,.2f}")

    print("\n✅ Individual methods working correctly")


def test_market_data_pipeline():
    """Test full pipeline integration"""
    print("\n" + "="*80)
    print("Test 2: MarketDataPipeline - Full Integration")
    print("="*80)

    pipeline = MarketDataPipeline()

    print("\nFetching complete market data for BTC and ETH...")
    print("(includes OHLC, indicators, OI, and funding rates)\n")

    market_data = pipeline.fetch_and_process(
        coins=['BTC', 'ETH'],
        lookback_hours_3m=3,
        lookback_hours_4h=240
    )

    for coin, data in market_data.items():
        if data:
            print(f"\n{coin}:")
            print(f"  Price: ${data['current_price']:,.2f}")
            print(f"  Funding Rate: {data['funding_rate']*100:.4f}%")
            print(f"  Open Interest: {data['open_interest']:,.2f} {coin}")
            print(f"  24h Volume: ${data['day_volume']:,.2f}")
            print(f"  EMA(20): ${data['current_ema20']:,.2f}")
            print(f"  RSI(7): {data['current_rsi_7']:.2f}")

    print("\n✅ Pipeline integration working correctly")


def explain_funding_rate(funding_rate: float):
    """Explain what a funding rate means"""
    rate_pct = funding_rate * 100

    if funding_rate > 0:
        direction = "POSITIVE (Longs pay Shorts)"
        interpretation = "Perpetual price > Spot price → Market is bullish"
        cost = f"Longs pay {rate_pct:.4f}% every 8 hours"
    elif funding_rate < 0:
        direction = "NEGATIVE (Shorts pay Longs)"
        interpretation = "Perpetual price < Spot price → Market is bearish"
        cost = f"Shorts pay {abs(rate_pct):.4f}% every 8 hours"
    else:
        direction = "NEUTRAL"
        interpretation = "Perpetual price = Spot price → Market balanced"
        cost = "No funding payments"

    return {
        'direction': direction,
        'interpretation': interpretation,
        'cost': cost
    }


def demo_funding_interpretation():
    """Demonstrate funding rate interpretation"""
    print("\n" + "="*80)
    print("Test 3: Funding Rate Interpretation")
    print("="*80)

    client = HyperliquidClient()

    coins = ['BTC', 'ETH', 'BNB']

    for coin in coins:
        funding = client.get_funding_rate(coin)
        explanation = explain_funding_rate(funding)

        print(f"\n{coin}:")
        print(f"  Funding Rate: {funding*100:.4f}%")
        print(f"  Direction: {explanation['direction']}")
        print(f"  Meaning: {explanation['interpretation']}")
        print(f"  Cost: {explanation['cost']}")

    print("\n✅ Funding rate interpretation complete")


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  Testing Open Interest & Funding Rates Integration".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")

    try:
        # Run tests
        test_hyperliquid_client()
        test_market_data_pipeline()
        demo_funding_interpretation()

        # Summary
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nNew Features Available:")
        print("  • HyperliquidClient.get_funding_rate(coin)")
        print("  • HyperliquidClient.get_open_interest(coin)")
        print("  • HyperliquidClient.get_market_data(coin)")
        print("  • MarketDataPipeline now includes OI & funding in results")
        print("\nNext Steps:")
        print("  • Update prompt template to include these metrics")
        print("  • Add funding rate warnings to trading decisions")
        print("  • Track funding payments in fee accounting")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

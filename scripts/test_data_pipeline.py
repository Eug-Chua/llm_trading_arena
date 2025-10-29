"""
Test complete market data pipeline with real data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_data_pipeline import MarketDataPipeline

def main():
    print("Testing Market Data Pipeline")
    print("="*80)

    pipeline = MarketDataPipeline()

    # Test with just BTC, ETH, SOL (faster than all 6 coins)
    test_coins = ['BTC', 'ETH', 'SOL']

    print(f"\nFetching data for: {', '.join(test_coins)}")
    print("This will take a few seconds...")
    print()

    try:
        # Fetch and process
        # Need at least 50 candles for indicators
        # 3-min: 50 candles = 150 minutes = 2.5 hours (use 3 for safety)
        # 4-hour: 50 candles = 200 hours = 8.3 days (use 10 days)
        market_data = pipeline.fetch_and_process(
            coins=test_coins,
            lookback_hours_3m=3,   # 3 hours of 3-min data (~60 candles)
            lookback_hours_4h=240  # 10 days of 4-hour data (~60 candles)
        )

        # Display results
        for coin, data in market_data.items():
            if data is None:
                print(f"\n✗ {coin}: FAILED")
                continue

            print(f"\n{'='*80}")
            print(f" {coin} Market Data")
            print(f"{'='*80}")

            # Current values
            print(f"\n📊 Current Values:")
            print(f"  Price: ${data['current_price']:,.2f}")
            print(f"  EMA(20): ${data['current_ema20']:,.2f}")
            print(f"  MACD: {data['current_macd']:.2f}")
            print(f"  RSI(7): {data['current_rsi_7']:.2f}")
            print(f"  Volume: {data['volume']:,.2f}")

            # 3-minute intraday data
            print(f"\n📈 Intraday (3-min) - Last 10 candles:")
            prices_3m = data['prices_3m']
            print(f"  Prices: ${prices_3m[0]:,.2f} → ${prices_3m[-1]:,.2f}")
            print(f"  EMA(20): {len(data['ema_20_3m'])} data points")
            print(f"  MACD: {len(data['macd_3m'])} data points")
            print(f"  RSI(7): {len(data['rsi_7_3m'])} data points")
            print(f"  RSI(14): {len(data['rsi_14_3m'])} data points")

            # 4-hour longer-term data
            print(f"\n📉 Longer-term (4-hour):")
            print(f"  EMA(20): ${data['ema_20_4h']:,.2f}")
            print(f"  EMA(50): ${data['ema_50_4h']:,.2f}")
            print(f"  ATR(3): ${data['atr_3_4h']:,.2f}")
            print(f"  ATR(14): ${data['atr_14_4h']:,.2f}")
            print(f"  MACD (4h): {len(data['macd_4h'])} data points")
            print(f"  RSI(14) (4h): {len(data['rsi_14_4h'])} data points")
            print(f"  Avg Volume: {data['volume_avg']:,.2f}")

            # Validate data
            print(f"\n✓ Validation:")
            checks = [
                ("3-min prices", len(data['prices_3m']) == 10),
                ("3-min EMA(20)", len(data['ema_20_3m']) == 10),
                ("3-min MACD", len(data['macd_3m']) == 10),
                ("4-hour MACD", len(data['macd_4h']) == 10),
                ("4-hour RSI", len(data['rsi_14_4h']) == 10),
                ("Current price > 0", data['current_price'] > 0),
                ("Volume > 0", data['volume'] > 0)
            ]

            for check_name, passed in checks:
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}")

        # Test current prices endpoint
        print(f"\n{'='*80}")
        print(" Current Prices (Quick Fetch)")
        print(f"{'='*80}\n")

        current_prices = pipeline.get_current_prices(coins=test_coins)
        for coin, price in current_prices.items():
            print(f"  {coin}: ${price:,.2f}")

        print(f"\n{'='*80}")
        print("✓ Pipeline test complete!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

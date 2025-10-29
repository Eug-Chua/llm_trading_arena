"""
Test Real LLM Trading Decision

This script tests the full pipeline:
1. Fetch real market data from Hyperliquid
2. Generate Alpha Arena prompt
3. Get LLM trading decision
4. Parse trade signals

This validates the entire flow works with real APIs.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.agents.llm_agent import LLMAgent
from src.data.market_data_pipeline import MarketDataPipeline
from src.prompts.alpha_arena_template import AlphaArenaPrompt, MarketData, AccountInfo
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create output directory
OUTPUT_DIR = project_root / "logs" / "llm_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_trading_decision(provider: str = "anthropic"):
    """
    Test full trading decision pipeline with real LLM

    Args:
        provider: LLM provider ("anthropic" or "openai")
    """
    print("\n" + "=" * 80)
    print(f"TESTING FULL TRADING PIPELINE WITH {provider.upper()}")
    print("=" * 80)

    # Step 1: Initialize LLM agent
    print(f"\n1️⃣  Initializing {provider} agent...")
    agent = LLMAgent(provider=provider, validate_api_key=True)
    print(f"✓ Agent ready: {agent.model_name}")

    # Step 2: Fetch real market data
    print("\n2️⃣  Fetching live market data from Hyperliquid...")
    pipeline = MarketDataPipeline()
    # Use all 6 Alpha Arena coins
    coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
    market_data_raw = pipeline.fetch_and_process(
        coins=coins,
        lookback_hours_3m=3,
        lookback_hours_4h=240
    )
    print(f"✓ Fetched data for: {', '.join(market_data_raw.keys())}")

    # Convert to MarketData objects
    market_data = {}
    for coin, data in market_data_raw.items():
        if data is None:
            continue
        market_data[coin] = MarketData(
            coin=coin,
            current_price=data['current_price'],
            current_ema20=data['current_ema20'],
            current_macd=data['current_macd'],
            current_rsi_7=data['current_rsi_7'],
            oi_latest=data.get('open_interest', 0),
            oi_average=data.get('open_interest', 0) * 0.95,
            funding_rate=data.get('funding_rate', 0),
            prices=data['prices_3m'],
            ema_20=data['ema_20_3m'],
            macd=data['macd_3m'],
            rsi_7=data['rsi_7_3m'],
            rsi_14=data['rsi_14_3m'],
            ema_20_4h=data['ema_20_4h'],
            ema_50_4h=data['ema_50_4h'],
            atr_3_4h=data['atr_3_4h'],
            atr_14_4h=data['atr_14_4h'],
            volume_current=data['volume'],
            volume_average=data['volume_avg'],
            macd_4h=data['macd_4h'],
            rsi_14_4h=data['rsi_14_4h']
        )

    # Step 3: Generate Alpha Arena prompt
    print("\n3️⃣  Generating Alpha Arena prompt...")
    prompt_gen = AlphaArenaPrompt()
    account_info = AccountInfo(
        total_return_percent=0.0,
        available_cash=10000.0,
        account_value=10000.0,
        positions=[],
        sharpe_ratio=0.0
    )
    prompt = prompt_gen.generate_prompt(market_data, account_info)
    print(f"✓ Generated prompt: {len(prompt):,} characters")
    print(f"\nPrompt preview (first 500 chars):")
    print("-" * 80)
    print(prompt[:500])
    print("...\n")

    # Step 4: Get LLM trading decision
    print(f"4️⃣  Sending prompt to {provider} LLM...")
    print("   (This may take 5-15 seconds...)")
    response = agent.generate_decision(prompt)

    # Step 5: Display results
    print("\n✅ SUCCESS! LLM responded with trading decision\n")
    print("=" * 80)
    print("CHAIN OF THOUGHT:")
    print("=" * 80)
    print(response.chain_of_thought[:800])
    if len(response.chain_of_thought) > 800:
        print(f"\n... (truncated {len(response.chain_of_thought) - 800} chars)")

    print("\n" + "=" * 80)
    print("TRADE SIGNALS:")
    print("=" * 80)
    for coin, signal in response.trade_signals.items():
        print(f"\n{coin}:")
        print(f"  Signal: {signal.signal}")
        print(f"  Quantity: {signal.quantity}")
        print(f"  Leverage: {signal.leverage}x")
        print(f"  Stop-loss: ${signal.stop_loss:,.2f}")
        print(f"  Profit target: ${signal.profit_target:,.2f}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Risk: ${signal.risk_usd:,.2f}")

    print("\n" + "=" * 80)
    print(f"✅ FULL PIPELINE TEST PASSED for {provider.upper()}")
    print("=" * 80)
    print("\nThe LLM successfully:")
    print("  ✓ Received real market data from Hyperliquid")
    print("  ✓ Analyzed Alpha Arena formatted prompt")
    print("  ✓ Generated chain-of-thought reasoning")
    print("  ✓ Produced valid trade signals")
    print("\n🎉 Ready for live trading experiments!\n")

    # Step 6: Save full output to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{provider}_{timestamp}.json"

    # Prepare data to save
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": agent.model_name,
        "market_data": {
            coin: {
                "current_price": data.current_price,
                "current_rsi_7": data.current_rsi_7,
                "current_macd": data.current_macd,
                "funding_rate": data.funding_rate,
                "open_interest": data.oi_latest
            }
            for coin, data in market_data.items()
        },
        "prompt": prompt,
        "response": {
            "chain_of_thought": response.chain_of_thought,
            "trade_signals": {
                coin: {
                    "signal": signal.signal,
                    "quantity": signal.quantity,
                    "leverage": signal.leverage,
                    "stop_loss": signal.stop_loss,
                    "profit_target": signal.profit_target,
                    "confidence": signal.confidence,
                    "risk_usd": signal.risk_usd,
                    "invalidation_condition": signal.invalidation_condition
                }
                for coin, signal in response.trade_signals.items()
            }
        }
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"💾 Full output saved to: {output_file}")
    print(f"   View with: cat {output_file}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test full trading pipeline with real LLM')
    parser.add_argument(
        '--provider',
        type=str,
        default='anthropic',
        choices=['anthropic', 'openai'],
        help='LLM provider to test'
    )
    args = parser.parse_args()

    try:
        test_trading_decision(args.provider)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

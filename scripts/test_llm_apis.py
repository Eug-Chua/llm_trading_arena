"""
Test LLM API Connections

This script validates API keys and tests real LLM API calls for all providers.

Usage:
    # Test all providers with API keys available
    python scripts/test_llm_apis.py

    # Test specific provider
    python scripts/test_llm_apis.py --provider deepseek

    # Test with custom API key
    python scripts/test_llm_apis.py --provider deepseek --api-key sk-...
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load .env file using python-dotenv
from dotenv import load_dotenv
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment from {env_file}\n")
else:
    print(f"⚠️  No .env file found at {env_file}")
    print("   Using environment variables from shell\n")

from src.agents.llm_agent import LLMAgent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_provider(provider: str, api_key: str = None) -> bool:
    """
    Test a specific LLM provider

    Args:
        provider: Provider name ("deepseek", "openai", "anthropic")
        api_key: Optional API key (uses env var if not provided)

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Testing {provider.upper()}")
    print("=" * 80)

    try:
        # Initialize agent with validation
        print(f"✓ Initializing {provider} agent...")
        agent = LLMAgent(
            provider=provider,
            api_key=api_key,
            validate_api_key=True  # Enforce API key presence
        )
        print(f"✓ Agent initialized: {agent.model_name}")

        # Create a simple test prompt
        test_prompt = """You are a cryptocurrency trading assistant.

Current market data:
- BTC: $95,000 (up 5% today)
- ETH: $3,500 (up 3% today)

Task: In 1-2 sentences, describe the current market sentiment based on these prices.

Response:"""

        print(f"\n📝 Sending test prompt to {provider}...")
        response = agent._call_llm_api(test_prompt)

        print(f"\n✅ SUCCESS! Response from {provider}:")
        print("-" * 80)
        print(response[:500])  # Show first 500 chars
        if len(response) > 500:
            print(f"\n... (truncated {len(response) - 500} chars)")
        print("-" * 80)

        return True

    except ValueError as e:
        # API key or validation errors
        print(f"\n❌ FAILED: {e}")
        return False

    except RuntimeError as e:
        # Rate limit or quota errors
        print(f"\n⚠️  WARNING: {e}")
        return False

    except Exception as e:
        # Other errors
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test LLM API connections')

    parser.add_argument(
        '--provider',
        type=str,
        choices=['deepseek', 'openai', 'anthropic', 'all'],
        default='all',
        help='LLM provider to test (default: all)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (overrides environment variable)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("LLM API CONNECTION TEST")
    print("=" * 80)
    print("\nThis script validates your API keys and tests real LLM API calls.")
    print("Make sure you have set up your .env file with valid API keys.")

    # Determine which providers to test
    if args.provider == 'all':
        providers_to_test = ['deepseek', 'openai', 'anthropic']
    else:
        providers_to_test = [args.provider]

    # Test each provider
    results = {}
    for provider in providers_to_test:
        results[provider] = test_provider(provider, args.api_key)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    for provider, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {provider}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n⚠️  Some tests failed. Check your API keys in .env file.")
        print("   See .env.example for the required format.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! Your API keys are working correctly.")
        sys.exit(0)


if __name__ == "__main__":
    main()

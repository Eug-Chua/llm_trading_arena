"""
Fetch Historical Trading Data from Alpha Arena Wallet Addresses

This script fetches actual trading history from Hyperliquid for each AI model's wallet.
Stores the data for analysis and validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hyperliquid_client import HyperliquidClient
from src.utils.logger import setup_logger
from src.utils.config import load_config
from datetime import datetime
import json
from pathlib import Path

logger = setup_logger(__name__)

class WalletTradesFetcher:
    """Fetches and stores trading history from AI model wallets"""

    def __init__(self, config_path: str = "config/wallets.yaml", output_dir: str = None):
        self.client = HyperliquidClient()

        # Load wallet configuration
        self.config = load_config(config_path)
        self.wallets = {
            name: info['address']
            for name, info in self.config['wallets'].items()
        }

        # Use output_dir from config if not specified
        if output_dir is None:
            output_dir = self.config['data_collection']['output_dir']

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized WalletTradesFetcher (output: {self.output_dir})")
        logger.info(f"Loaded {len(self.wallets)} wallet addresses from {config_path}")

    def fetch_wallet_fills(self, address: str, model_name: str, existing_fills: list = None):
        """
        Fetch fill history (executed trades) for a wallet address

        Args:
            address: Wallet address (0x...)
            model_name: AI model name for this wallet
            existing_fills: Previously fetched fills (for incremental update)

        Returns:
            List of fills (trades), merged with existing if incremental
        """
        logger.info(f"Fetching fills for {model_name} ({address})...")

        try:
            payload = {
                "type": "userFills",
                "user": address
            }

            fills = self.client._make_request(payload)

            if not fills:
                logger.warning(f"{model_name}: No fills found")
                return existing_fills or []

            logger.info(f"{model_name}: API returned {len(fills)} fills")

            # Incremental update: merge with existing
            if existing_fills:
                # Create a set of existing trade IDs for deduplication
                existing_tids = {f['tid'] for f in existing_fills if 'tid' in f}

                # Add only new fills
                new_fills = [f for f in fills if f.get('tid') not in existing_tids]

                logger.info(f"{model_name}: {len(new_fills)} new fills, {len(existing_fills)} existing")

                # Merge and sort by time
                all_fills = existing_fills + new_fills
                all_fills.sort(key=lambda x: x.get('time', 0))

                return all_fills
            else:
                logger.info(f"{model_name}: First fetch - storing all {len(fills)} fills")
                return fills

        except Exception as e:
            logger.error(f"{model_name}: Failed to fetch fills - {e}")
            return existing_fills or []

    def analyze_fills(self, fills: list, model_name: str):
        """Analyze trading statistics from fills"""

        if not fills:
            return {
                "model": model_name,
                "total_trades": 0,
                "error": "No fills found"
            }

        # Extract stats
        coins_traded = set()
        total_volume = 0
        leverages_used = set()
        timestamps = []

        for fill in fills:
            if isinstance(fill, dict):
                coins_traded.add(fill.get('coin', 'Unknown'))

                # Calculate volume
                px = float(fill.get('px', 0))
                sz = float(fill.get('sz', 0))
                total_volume += px * abs(sz)

                # Track timestamp
                ts = fill.get('time', 0)
                if ts:
                    timestamps.append(ts)

                # Note: Hyperliquid fills don't directly expose leverage
                # We'd need to cross-reference with position data

        timestamps.sort()

        stats = {
            "model": model_name,
            "total_trades": len(fills),
            "coins_traded": sorted(list(coins_traded)),
            "total_volume_usd": round(total_volume, 2),
            "first_trade": datetime.fromtimestamp(timestamps[0] / 1000).isoformat() if timestamps else None,
            "last_trade": datetime.fromtimestamp(timestamps[-1] / 1000).isoformat() if timestamps else None,
            "trade_frequency": len(fills) / max(1, (timestamps[-1] - timestamps[0]) / (1000 * 60 * 60 * 24)) if len(timestamps) > 1 else 0
        }

        return stats

    def fetch_all_wallets(self, incremental: bool = True):
        """
        Fetch trading history for all AI model wallets

        Args:
            incremental: If True, load existing data and append new fills only
        """

        logger.info("="*80)
        logger.info("Fetching Trading History for All AI Models")
        logger.info(f"Mode: {'Incremental Update' if incremental else 'Full Refresh'}")
        logger.info("="*80)

        all_results = {}

        for model_name, address in self.wallets.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {model_name}")
            logger.info(f"Address: {address}")
            logger.info(f"{'='*80}")

            # Load existing data if incremental
            existing_fills = None
            output_file = self.output_dir / f"{model_name.replace(' ', '_').lower()}.json"

            if incremental and output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        existing_data = json.load(f)
                    existing_fills = existing_data.get('fills', [])
                    logger.info(f"{model_name}: Loaded {len(existing_fills)} existing fills")
                except Exception as e:
                    logger.warning(f"{model_name}: Could not load existing data - {e}")

            # Fetch fills (incremental or full)
            fills = self.fetch_wallet_fills(address, model_name, existing_fills)

            # Analyze
            stats = self.analyze_fills(fills, model_name)

            # Store
            result = {
                "model": model_name,
                "address": address,
                "fetched_at": datetime.now().isoformat(),
                "stats": stats,
                "fills": fills  # Store ALL fills (now that we know they're reasonable size)
            }

            all_results[model_name] = result

            # Save individual file (overwrite with merged data)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved to: {output_file}")

            # Print summary
            logger.info(f"\nSummary for {model_name}:")
            logger.info(f"  Total fills: {stats.get('total_trades', 0)}")
            if existing_fills and incremental:
                new_count = len(fills) - len(existing_fills)
                logger.info(f"  New fills this update: {new_count}")
            logger.info(f"  Coins traded: {', '.join(stats.get('coins_traded', []))}")
            logger.info(f"  Volume: ${stats.get('total_volume_usd', 0):,.2f}")
            logger.info(f"  Trade frequency: {stats.get('trade_frequency', 0):.2f} trades/day")

        # Save combined file
        combined_file = self.output_dir / "all_models_summary.json"
        summary = {
            "fetched_at": datetime.now().isoformat(),
            "models": {
                name: result["stats"] for name, result in all_results.items()
            }
        }

        with open(combined_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("Trading History Fetch Complete")
        logger.info(f"Combined summary: {combined_file}")
        logger.info("="*80)

        return all_results

    def compare_models(self):
        """Compare trading patterns across models"""

        logger.info("\n" + "="*80)
        logger.info("Model Comparison")
        logger.info("="*80)

        # Load summaries
        summary_file = self.output_dir / "all_models_summary.json"
        if not summary_file.exists():
            logger.error("Run fetch_all_wallets() first to generate summary")
            return

        with open(summary_file, 'r') as f:
            data = json.load(f)

        models = data.get('models', {})

        # Sort by trade frequency
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1].get('trade_frequency', 0),
            reverse=True
        )

        logger.info("\nTrade Frequency Ranking:")
        for rank, (model, stats) in enumerate(sorted_models, 1):
            freq = stats.get('trade_frequency', 0)
            trades = stats.get('total_trades', 0)
            logger.info(f"  {rank}. {model}: {freq:.2f} trades/day ({trades} total)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fetch wallet trading history')
    parser.add_argument('--full-refresh', action='store_true',
                       help='Re-download all data (default: incremental update)')

    args = parser.parse_args()

    fetcher = WalletTradesFetcher()

    # Fetch all wallet trading history (incremental by default)
    results = fetcher.fetch_all_wallets(incremental=not args.full_refresh)

    # Compare models
    fetcher.compare_models()

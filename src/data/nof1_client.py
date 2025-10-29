"""
Nof1.ai API Client

Fetches real-time trading data, positions, and performance metrics
from the Alpha Arena competition.
"""

import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import time

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Nof1Client:
    """Client for accessing nof1.ai Alpha Arena API"""

    BASE_URL = "https://nof1.ai/api"

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize Nof1 API client

        Args:
            cache_dir: Directory to cache API responses (optional)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LLM-Trading-Arena-Research/1.0'
        })

        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make API request with error handling

        Args:
            endpoint: API endpoint (e.g., 'trades', 'account-totals')
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            logger.debug(f"Requesting {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"Successfully fetched {endpoint}")

            # Cache response if cache_dir is set
            if self.cache_dir:
                self._cache_response(endpoint, data, params)

            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {endpoint}: {e}")
            raise

    def _cache_response(self, endpoint: str, data: Dict, params: Optional[Dict] = None):
        """Cache API response to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = f"_{params}" if params else ""
        filename = f"{endpoint}{param_str}_{timestamp}.json"
        filepath = self.cache_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'endpoint': endpoint,
                'params': params,
                'data': data
            }, f, indent=2)

        logger.debug(f"Cached response to {filepath}")

    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Get all completed trades across all models

        Returns:
            List of trade dictionaries with fields:
            - model_id: Model identifier
            - symbol: Trading pair (BTC, ETH, etc.)
            - entry_price, exit_price: Trade prices
            - entry_time, exit_time: Unix timestamps
            - quantity, leverage: Position details
            - realized_gross_pnl, realized_net_pnl: Profit/loss
            - total_commission_dollars: Fees paid
        """
        response = self._make_request('trades')
        return response.get('trades', [])

    def get_account_totals(self, last_hourly_marker: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get current account state for all models

        Args:
            last_hourly_marker: Optional marker for pagination

        Returns:
            List of account state dictionaries with:
            - model_id: Model identifier
            - timestamp: Current timestamp
            - realized_pnl: Total realized PnL
            - positions: Dict of open positions by symbol
        """
        params = {}
        if last_hourly_marker:
            params['lastHourlyMarker'] = last_hourly_marker

        response = self._make_request('account-totals', params)
        return response.get('accountTotals', [])

    def get_crypto_prices(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current cryptocurrency prices

        Returns:
            Dictionary mapping symbol to price data:
            {
                'BTC': {'symbol': 'BTC', 'price': 113908.5, 'timestamp': 1761626103642},
                'ETH': {'symbol': 'ETH', 'price': 4099.65, 'timestamp': 1761626103642},
                ...
            }
        """
        response = self._make_request('crypto-prices')
        return response.get('prices', {})

    def get_since_inception_values(self) -> List[Dict[str, Any]]:
        """
        Get starting values and metadata for all models

        Returns:
            List of model inception data:
            - model_id: Model identifier
            - nav_since_inception: Starting capital (usually $10,000)
            - inception_date: When model started trading
            - num_invocations: Number of times model was called
        """
        response = self._make_request('since-inception-values')
        return response.get('sinceInceptionValues', [])

    def get_model_positions(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current positions for a specific model

        Args:
            model_id: Model identifier (e.g., 'deepseek-chat-v3.1')

        Returns:
            Dictionary with model's positions or None if not found
        """
        accounts = self.get_account_totals()

        for account in accounts:
            if model_id in account.get('id', ''):
                return {
                    'model_id': model_id,
                    'timestamp': account.get('timestamp'),
                    'realized_pnl': account.get('realized_pnl'),
                    'positions': account.get('positions', {})
                }

        logger.warning(f"No positions found for model: {model_id}")
        return None

    def get_model_trades(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get all trades for a specific model

        Args:
            model_id: Model identifier

        Returns:
            List of trades for the specified model
        """
        all_trades = self.get_trades()
        return [trade for trade in all_trades if trade.get('model_id') == model_id]

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Calculate current leaderboard from account totals

        Returns:
            List of models sorted by performance (NAV)
        """
        accounts = self.get_account_totals()
        inception_data = {
            item['model_id']: item
            for item in self.get_since_inception_values()
        }

        leaderboard = []

        for account in accounts:
            account_id = account.get('id', '')
            # Extract model_id from account id (format: "model-id_marker")
            model_id = account_id.rsplit('_', 1)[0] if '_' in account_id else account_id

            realized_pnl = account.get('realized_pnl', 0)

            # Calculate unrealized PnL from positions
            unrealized_pnl = 0
            positions = account.get('positions', {})
            for position in positions.values():
                unrealized_pnl += position.get('unrealized_pnl', 0)

            # Starting capital
            starting_capital = inception_data.get(model_id, {}).get('nav_since_inception', 10000)

            # Current NAV
            nav = starting_capital + realized_pnl + unrealized_pnl

            # Return percentage
            return_pct = ((nav - starting_capital) / starting_capital) * 100

            leaderboard.append({
                'model_id': model_id,
                'nav': nav,
                'starting_capital': starting_capital,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl + unrealized_pnl,
                'return_pct': return_pct,
                'num_positions': len(positions),
                'timestamp': account.get('timestamp')
            })

        # Sort by timestamp (chronological order) for time-series analysis
        leaderboard.sort(key=lambda x: x['timestamp'])

        return leaderboard

    def monitor_updates(self, interval_seconds: int = 60, callback=None):
        """
        Continuously monitor for updates

        Args:
            interval_seconds: How often to check for updates
            callback: Optional function to call with new data
        """
        logger.info(f"Starting monitoring (interval: {interval_seconds}s)")

        try:
            while True:
                try:
                    data = {
                        'trades': self.get_trades(),
                        'positions': self.get_account_totals(),
                        'prices': self.get_crypto_prices(),
                        'leaderboard': self.get_leaderboard(),
                        'timestamp': datetime.now().isoformat()
                    }

                    if callback:
                        callback(data)

                    logger.info(f"Update fetched at {data['timestamp']}")

                except Exception as e:
                    logger.error(f"Error during monitoring: {e}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")


# Convenience function
def get_client(cache_dir: Optional[str] = None) -> Nof1Client:
    """
    Get a Nof1Client instance

    Args:
        cache_dir: Optional directory path for caching responses

    Returns:
        Nof1Client instance
    """
    cache_path = Path(cache_dir) if cache_dir else None
    return Nof1Client(cache_dir=cache_path)

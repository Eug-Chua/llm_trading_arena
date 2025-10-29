"""
Hyperliquid API Client

Fetches wallet trading data and historical price data from Hyperliquid exchange.
"""

import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)


class HyperliquidClient:
    """Client for accessing Hyperliquid API"""

    def __init__(self, cache_dir: Optional[Path] = None, config_path: str = "config/api.yaml"):
        """
        Initialize Hyperliquid API client

        Args:
            cache_dir: Directory to cache API responses (optional)
            config_path: Path to API configuration file
        """
        # Load API configuration
        self.config = load_config(config_path)
        hyperliquid_config = self.config['hyperliquid']

        self.base_url = hyperliquid_config['base_url']
        self.info_endpoint = hyperliquid_config['info_endpoint']
        self.timeout = hyperliquid_config['timeout_seconds']

        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': hyperliquid_config['user_agent']
        })

        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request with error handling

        Args:
            payload: Request payload

        Returns:
            JSON response data

        Raises:
            requests.RequestException: If request fails
        """
        try:
            logger.debug(f"Requesting {self.info_endpoint} with payload: {payload}")
            response = self.session.post(
                self.info_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            logger.info(f"Successfully fetched {payload.get('type')}")

            # Cache response if cache_dir is set
            if self.cache_dir:
                self._cache_response(payload, data)

            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {payload.get('type')}: {e}")
            raise

    def _cache_response(self, payload: Dict, data: Dict):
        """Cache API response to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_type = payload.get('type', 'unknown')
        filename = f"{request_type}_{timestamp}.json"
        filepath = self.cache_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'payload': payload,
                'data': data
            }, f, indent=2)

        logger.debug(f"Cached response to {filepath}")

    def get_candles(
        self,
        coin: str,
        interval: str = "1m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get historical candlestick data for a coin

        NOTE: Hyperliquid API REQUIRES startTime and endTime parameters.
        If not provided, will default to last {lookback_hours} hours.

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH', 'SOL')
            interval: Candle interval ('1m', '3m', '5m', '15m', '1h', '4h', '1d')
            start_time: Start timestamp in milliseconds (if None, calculated from lookback)
            end_time: End timestamp in milliseconds (if None, uses current time)
            lookback_hours: Hours to look back if start_time not provided (default: 24)

        Returns:
            List of candle dictionaries with:
            - t: Timestamp (milliseconds)
            - T: End timestamp
            - s: Symbol
            - i: Interval
            - o: Open price
            - c: Close price
            - h: High price
            - l: Low price
            - v: Volume
            - n: Number of trades
        """
        # Hyperliquid requires both timestamps - calculate if not provided
        if end_time is None:
            end_time = int(datetime.now().timestamp() * 1000)

        if start_time is None:
            start_time = end_time - (lookback_hours * 60 * 60 * 1000)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time
            }
        }

        response = self._make_request(payload)
        return response

    def get_user_fills(self, user_address: str) -> List[Dict[str, Any]]:
        """
        Get trading history (fills) for a wallet address

        Args:
            user_address: Hyperliquid wallet address (0x...)

        Returns:
            List of trade dictionaries with:
            - coin: Trading pair
            - side: 'Buy' or 'Sell'
            - px: Execution price
            - sz: Size/quantity
            - time: Timestamp (milliseconds)
            - fee: Fee paid
            - closedPnl: Realized PnL (if position closed)
        """
        payload = {
            "type": "userFills",
            "user": user_address
        }

        response = self._make_request(payload)
        return response

    def get_user_state(self, user_address: str) -> Dict[str, Any]:
        """
        Get current state for a wallet (positions, balance, etc.)

        Args:
            user_address: Hyperliquid wallet address

        Returns:
            Dictionary with:
            - assetPositions: List of open positions
            - marginSummary: Account balance and margin info
            - crossMarginSummary: Cross margin details
        """
        payload = {
            "type": "clearinghouseState",
            "user": user_address
        }

        response = self._make_request(payload)
        return response

    def get_meta(self) -> Dict[str, Any]:
        """
        Get metadata about available markets

        Returns:
            Dictionary with universe of available coins and their specs
        """
        payload = {
            "type": "meta"
        }

        response = self._make_request(payload)
        return response

    def get_all_mids(self) -> Dict[str, str]:
        """
        Get current mid prices for all coins

        Returns:
            Dictionary mapping coin symbol to current mid price
        """
        payload = {
            "type": "allMids"
        }

        response = self._make_request(payload)
        return response

    def get_meta_and_asset_contexts(self) -> Dict[str, Any]:
        """
        Get metadata and asset contexts (includes funding rates, OI, volume, etc.)

        Returns:
            Tuple of (meta, asset_contexts) where:
            - meta: Market metadata (universe, margin tables)
            - asset_contexts: List of dicts with per-coin data:
                - funding: Funding rate
                - openInterest: Total open interest
                - markPx: Mark price
                - oraclePx: Oracle price
                - dayNtlVlm: 24h volume (notional)
                - premium: Funding premium
                - prevDayPx: Previous day price
        """
        payload = {
            "type": "metaAndAssetCtxs"
        }

        response = self._make_request(payload)

        # Response is [meta, asset_contexts]
        if isinstance(response, list) and len(response) >= 2:
            return {
                'meta': response[0],
                'asset_contexts': response[1]
            }
        else:
            logger.warning("Unexpected response format from metaAndAssetCtxs")
            return {'meta': {}, 'asset_contexts': []}

    def get_funding_rate(self, coin: str) -> Optional[float]:
        """
        Get current funding rate for a specific coin

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')

        Returns:
            Funding rate as decimal (e.g., 0.0000125 = 0.00125%)
            Returns None if coin not found
        """
        data = self.get_meta_and_asset_contexts()
        asset_contexts = data['asset_contexts']
        meta = data['meta']

        # Find the index of the coin in the universe
        universe = meta.get('universe', [])
        coin_index = None

        for idx, coin_meta in enumerate(universe):
            if coin_meta['name'] == coin:
                coin_index = idx
                break

        if coin_index is None:
            logger.warning(f"Coin {coin} not found in universe")
            return None

        if coin_index >= len(asset_contexts):
            logger.warning(f"Asset context not found for {coin}")
            return None

        ctx = asset_contexts[coin_index]
        funding_rate = float(ctx.get('funding', 0))

        logger.debug(f"Funding rate for {coin}: {funding_rate}")
        return funding_rate

    def get_open_interest(self, coin: str) -> Optional[float]:
        """
        Get current open interest for a specific coin

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')

        Returns:
            Open interest in number of contracts
            Returns None if coin not found
        """
        data = self.get_meta_and_asset_contexts()
        asset_contexts = data['asset_contexts']
        meta = data['meta']

        # Find the index of the coin in the universe
        universe = meta.get('universe', [])
        coin_index = None

        for idx, coin_meta in enumerate(universe):
            if coin_meta['name'] == coin:
                coin_index = idx
                break

        if coin_index is None:
            logger.warning(f"Coin {coin} not found in universe")
            return None

        if coin_index >= len(asset_contexts):
            logger.warning(f"Asset context not found for {coin}")
            return None

        ctx = asset_contexts[coin_index]
        open_interest = float(ctx.get('openInterest', 0))

        logger.debug(f"Open interest for {coin}: {open_interest}")
        return open_interest

    def get_market_data(self, coin: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive market data for a coin

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')

        Returns:
            Dictionary with:
                - funding_rate: Current funding rate
                - open_interest: Total open interest
                - mark_price: Mark price
                - oracle_price: Oracle price
                - day_volume: 24h trading volume
                - prev_day_price: Previous day's price
                - premium: Funding premium
        """
        data = self.get_meta_and_asset_contexts()
        asset_contexts = data['asset_contexts']
        meta = data['meta']

        # Find the index of the coin in the universe
        universe = meta.get('universe', [])
        coin_index = None

        for idx, coin_meta in enumerate(universe):
            if coin_meta['name'] == coin:
                coin_index = idx
                break

        if coin_index is None:
            logger.warning(f"Coin {coin} not found in universe")
            return None

        if coin_index >= len(asset_contexts):
            logger.warning(f"Asset context not found for {coin}")
            return None

        ctx = asset_contexts[coin_index]

        return {
            'coin': coin,
            'funding_rate': float(ctx.get('funding', 0)),
            'open_interest': float(ctx.get('openInterest', 0)),
            'mark_price': float(ctx.get('markPx', 0)),
            'oracle_price': float(ctx.get('oraclePx', 0)),
            'mid_price': float(ctx.get('midPx', 0)),
            'day_volume': float(ctx.get('dayNtlVlm', 0)),
            'prev_day_price': float(ctx.get('prevDayPx', 0)),
            'premium': float(ctx.get('premium', 0))
        }

    def get_candles_for_multiple_coins(
        self,
        coins: List[str],
        interval: str = "3m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get historical candles for multiple coins

        Args:
            coins: List of coin symbols
            interval: Candle interval (use '3m' for Alpha Arena replication)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            Dictionary mapping coin symbol to list of candles
        """
        results = {}

        for coin in coins:
            try:
                logger.info(f"Fetching {interval} candles for {coin}")
                candles = self.get_candles(
                    coin=coin,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time
                )
                results[coin] = candles
            except Exception as e:
                logger.error(f"Failed to fetch candles for {coin}: {e}")
                results[coin] = []

        return results

    def get_wallet_trading_summary(
        self,
        wallet_address: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive trading summary for a wallet

        Args:
            wallet_address: Hyperliquid wallet address

        Returns:
            Dictionary with:
            - fills: All trades
            - state: Current positions and balance
            - total_trades: Number of trades
            - realized_pnl: Total realized PnL
            - win_rate: Percentage of winning trades
        """
        logger.info(f"Fetching trading summary for {wallet_address}")

        # Get fills (trades)
        fills = self.get_user_fills(wallet_address)

        # Get current state
        state = self.get_user_state(wallet_address)

        # Calculate summary statistics
        total_trades = len(fills)
        realized_pnl = sum(
            float(fill.get('closedPnl', 0))
            for fill in fills
            if fill.get('closedPnl')
        )

        winning_trades = sum(
            1 for fill in fills
            if fill.get('closedPnl') and float(fill['closedPnl']) > 0
        )

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'wallet_address': wallet_address,
            'fills': fills,
            'state': state,
            'summary': {
                'total_trades': total_trades,
                'realized_pnl': realized_pnl,
                'winning_trades': winning_trades,
                'win_rate': win_rate
            }
        }


# Convenience function
def get_client(cache_dir: Optional[str] = None) -> HyperliquidClient:
    """
    Get a HyperliquidClient instance

    Args:
        cache_dir: Optional directory path for caching responses

    Returns:
        HyperliquidClient instance
    """
    cache_path = Path(cache_dir) if cache_dir else None
    return HyperliquidClient(cache_dir=cache_path)


def get_alpha_arena_data(
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    cache_dir: Optional[str] = None,
    config_path: str = "config/api.yaml"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get historical price data for all Alpha Arena coins

    Args:
        start_time: Start timestamp in milliseconds (optional)
        end_time: End timestamp in milliseconds (optional)
        cache_dir: Cache directory for responses
        config_path: Path to API configuration file

    Returns:
        Dictionary mapping coin to 3-minute candles
    """
    # Load Alpha Arena configuration
    config = load_config(config_path)
    alpha_arena_coins = config['alpha_arena']['coins']
    default_interval = config['alpha_arena']['default_interval']

    client = get_client(cache_dir=cache_dir)

    logger.info("Fetching Alpha Arena price data for all coins")

    return client.get_candles_for_multiple_coins(
        coins=alpha_arena_coins,
        interval=default_interval,
        start_time=start_time,
        end_time=end_time
    )

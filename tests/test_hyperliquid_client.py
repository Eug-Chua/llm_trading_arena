"""
Unit tests for hyperliquid_client.py
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hyperliquid_client import (
    HyperliquidClient,
    get_client,
    get_alpha_arena_data,
    ALPHA_ARENA_COINS
)


class TestHyperliquidClient:
    """Test suite for HyperliquidClient"""

    @pytest.fixture
    def mock_response(self):
        """Create mock API response"""
        mock = Mock()
        mock.status_code = 200
        mock.raise_for_status = Mock()
        return mock

    @pytest.fixture
    def client(self, tmp_path):
        """Create client with temporary cache directory"""
        return HyperliquidClient(cache_dir=tmp_path)

    def test_client_initialization(self, tmp_path):
        """Test client initialization"""
        client = HyperliquidClient(cache_dir=tmp_path)

        assert client.BASE_URL == "https://api.hyperliquid.xyz"
        assert client.INFO_ENDPOINT == "https://api.hyperliquid.xyz/info"
        assert client.cache_dir == tmp_path
        assert tmp_path.exists()

    def test_client_initialization_without_cache(self):
        """Test client initialization without cache"""
        client = HyperliquidClient()

        assert client.cache_dir is None

    @patch('requests.Session.post')
    def test_get_candles_success(self, mock_post, client, mock_response):
        """Test successful candle fetch"""
        # Mock response
        mock_response.json.return_value = [
            {
                't': 1754300000000,
                'o': 100000.0,
                'h': 101000.0,
                'l': 99000.0,
                'c': 100500.0,
                'v': 1500.5
            },
            {
                't': 1754300180000,
                'o': 100500.0,
                'h': 102000.0,
                'l': 100000.0,
                'c': 101500.0,
                'v': 2000.0
            }
        ]
        mock_post.return_value = mock_response

        # Call method
        candles = client.get_candles(coin='BTC', interval='3m')

        # Assertions
        assert len(candles) == 2
        assert candles[0]['c'] == 100500.0
        assert candles[1]['c'] == 101500.0
        mock_post.assert_called_once()

        # Verify payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['type'] == 'candleSnapshot'
        assert payload['req']['coin'] == 'BTC'
        assert payload['req']['interval'] == '3m'

    @patch('requests.Session.post')
    def test_get_candles_with_time_range(self, mock_post, client, mock_response):
        """Test candle fetch with time range"""
        mock_response.json.return_value = []
        mock_post.return_value = mock_response

        start_time = 1754300000000
        end_time = 1754400000000

        client.get_candles(
            coin='ETH',
            interval='1h',
            start_time=start_time,
            end_time=end_time
        )

        # Verify payload includes time range
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['req']['startTime'] == start_time
        assert payload['req']['endTime'] == end_time

    @patch('requests.Session.post')
    def test_get_user_fills_success(self, mock_post, client, mock_response):
        """Test successful user fills fetch"""
        mock_response.json.return_value = [
            {
                'coin': 'BTC',
                'side': 'Buy',
                'px': '100000.0',
                'sz': '0.5',
                'time': 1754300000000,
                'fee': '50.0',
                'closedPnl': '500.0'
            }
        ]
        mock_post.return_value = mock_response

        wallet = '0x7a8fd8bba33e37361ca6b0cb4518a44681bad2f3'
        fills = client.get_user_fills(wallet)

        # Assertions
        assert len(fills) == 1
        assert fills[0]['coin'] == 'BTC'
        assert fills[0]['side'] == 'Buy'
        assert fills[0]['closedPnl'] == '500.0'

        # Verify payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['type'] == 'userFills'
        assert payload['user'] == wallet

    @patch('requests.Session.post')
    def test_get_user_state_success(self, mock_post, client, mock_response):
        """Test successful user state fetch"""
        mock_response.json.return_value = {
            'assetPositions': [
                {
                    'position': {
                        'coin': 'BTC',
                        'szi': '0.5',
                        'entryPx': '100000.0',
                        'unrealizedPnl': '2500.0',
                        'leverage': {'value': 15}
                    }
                }
            ],
            'marginSummary': {
                'accountValue': '12500.0',
                'totalMarginUsed': '3333.33'
            }
        }
        mock_post.return_value = mock_response

        wallet = '0x7a8fd8bba33e37361ca6b0cb4518a44681bad2f3'
        state = client.get_user_state(wallet)

        # Assertions
        assert 'assetPositions' in state
        assert len(state['assetPositions']) == 1
        assert state['assetPositions'][0]['position']['coin'] == 'BTC'

        # Verify payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['type'] == 'clearinghouseState'
        assert payload['user'] == wallet

    @patch('requests.Session.post')
    def test_get_meta_success(self, mock_post, client, mock_response):
        """Test successful meta fetch"""
        mock_response.json.return_value = {
            'universe': [
                {'name': 'BTC', 'szDecimals': 4},
                {'name': 'ETH', 'szDecimals': 3}
            ]
        }
        mock_post.return_value = mock_response

        meta = client.get_meta()

        # Assertions
        assert 'universe' in meta
        assert len(meta['universe']) == 2

        # Verify payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['type'] == 'meta'

    @patch('requests.Session.post')
    def test_get_all_mids_success(self, mock_post, client, mock_response):
        """Test successful all mids fetch"""
        mock_response.json.return_value = {
            'BTC': '100000.5',
            'ETH': '4000.25',
            'SOL': '150.75'
        }
        mock_post.return_value = mock_response

        mids = client.get_all_mids()

        # Assertions
        assert 'BTC' in mids
        assert 'ETH' in mids
        assert mids['BTC'] == '100000.5'

        # Verify payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['type'] == 'allMids'

    @patch('requests.Session.post')
    def test_get_candles_for_multiple_coins(self, mock_post, client, mock_response):
        """Test fetching candles for multiple coins"""
        mock_response.json.return_value = [
            {'t': 1754300000000, 'o': 100000.0, 'c': 100500.0}
        ]
        mock_post.return_value = mock_response

        coins = ['BTC', 'ETH', 'SOL']
        results = client.get_candles_for_multiple_coins(coins, interval='3m')

        # Assertions
        assert len(results) == 3
        assert 'BTC' in results
        assert 'ETH' in results
        assert 'SOL' in results
        assert mock_post.call_count == 3

    @patch('requests.Session.post')
    def test_get_wallet_trading_summary(self, mock_post, client, mock_response):
        """Test wallet trading summary"""
        def mock_post_side_effect(url, **kwargs):
            mock = Mock()
            mock.raise_for_status = Mock()

            payload = kwargs.get('json', {})

            if payload['type'] == 'userFills':
                mock.json.return_value = [
                    {'coin': 'BTC', 'closedPnl': '500.0'},
                    {'coin': 'ETH', 'closedPnl': '200.0'},
                    {'coin': 'BTC', 'closedPnl': '-100.0'}
                ]
            elif payload['type'] == 'clearinghouseState':
                mock.json.return_value = {
                    'assetPositions': [
                        {'position': {'coin': 'BTC', 'szi': '0.5'}}
                    ]
                }
            return mock

        mock_post.side_effect = mock_post_side_effect

        wallet = '0x7a8fd8bba33e37361ca6b0cb4518a44681bad2f3'
        summary = client.get_wallet_trading_summary(wallet)

        # Assertions
        assert summary['wallet_address'] == wallet
        assert summary['summary']['total_trades'] == 3
        assert summary['summary']['realized_pnl'] == 600.0  # 500 + 200 - 100
        assert summary['summary']['winning_trades'] == 2
        assert summary['summary']['win_rate'] == pytest.approx(66.67, rel=0.01)

    @patch('requests.Session.post')
    def test_request_failure(self, mock_post, client):
        """Test API request failure handling"""
        # Mock failed request
        mock_post.side_effect = Exception("Connection error")

        # Call should raise exception
        with pytest.raises(Exception):
            client.get_candles('BTC')

    def test_cache_response(self, client, tmp_path):
        """Test response caching"""
        payload = {'type': 'candleSnapshot', 'req': {'coin': 'BTC'}}
        data = [{'t': 1754300000000, 'c': 100000.0}]

        # Cache the response
        client._cache_response(payload, data)

        # Check that cache file was created
        cache_files = list(tmp_path.glob("candleSnapshot_*.json"))
        assert len(cache_files) == 1

        # Read and verify cached data
        with open(cache_files[0], 'r') as f:
            cached = json.load(f)

        assert cached['payload'] == payload
        assert cached['data'] == data

    def test_get_client_function(self):
        """Test get_client convenience function"""
        client = get_client()
        assert isinstance(client, HyperliquidClient)
        assert client.cache_dir is None

    def test_get_client_with_cache(self, tmp_path):
        """Test get_client with cache directory"""
        client = get_client(cache_dir=str(tmp_path))
        assert isinstance(client, HyperliquidClient)
        assert client.cache_dir == tmp_path

    @patch('src.data.hyperliquid_client.get_client')
    def test_get_alpha_arena_data(self, mock_get_client):
        """Test get_alpha_arena_data helper function"""
        mock_client = Mock()
        mock_client.get_candles_for_multiple_coins.return_value = {
            'BTC': [{'t': 1754300000000, 'c': 100000.0}],
            'ETH': [{'t': 1754300000000, 'c': 4000.0}]
        }
        mock_get_client.return_value = mock_client

        result = get_alpha_arena_data()

        # Assertions
        mock_client.get_candles_for_multiple_coins.assert_called_once_with(
            coins=ALPHA_ARENA_COINS,
            interval='3m',
            start_time=None,
            end_time=None
        )
        assert 'BTC' in result
        assert 'ETH' in result

    def test_alpha_arena_coins_constant(self):
        """Test ALPHA_ARENA_COINS constant"""
        assert ALPHA_ARENA_COINS == ['BTC', 'ETH', 'SOL', 'BNB', 'DOGE', 'XRP']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

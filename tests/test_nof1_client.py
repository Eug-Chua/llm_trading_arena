"""
Unit tests for nof1_client.py
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nof1_client import Nof1Client, get_client


class TestNof1Client:
    """Test suite for Nof1Client"""

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
        return Nof1Client(cache_dir=tmp_path)

    def test_client_initialization(self, tmp_path):
        """Test client initialization"""
        client = Nof1Client(cache_dir=tmp_path)

        assert client.BASE_URL == "https://nof1.ai/api"
        assert client.cache_dir == tmp_path
        assert tmp_path.exists()

    def test_client_initialization_without_cache(self):
        """Test client initialization without cache"""
        client = Nof1Client()

        assert client.cache_dir is None

    @patch('requests.Session.get')
    def test_get_trades_success(self, mock_get, client, mock_response):
        """Test successful trades fetch"""
        # Mock response
        mock_response.json.return_value = {
            'trades': [
                {
                    'id': 'test_trade_1',
                    'model_id': 'deepseek-chat-v3.1',
                    'symbol': 'BTC',
                    'entry_price': 100000,
                    'exit_price': 105000,
                    'realized_net_pnl': 500
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call method
        trades = client.get_trades()

        # Assertions
        assert len(trades) == 1
        assert trades[0]['model_id'] == 'deepseek-chat-v3.1'
        assert trades[0]['symbol'] == 'BTC'
        mock_get.assert_called_once()

    @patch('requests.Session.get')
    def test_get_account_totals_success(self, mock_get, client, mock_response):
        """Test successful account totals fetch"""
        mock_response.json.return_value = {
            'accountTotals': [
                {
                    'id': 'deepseek-chat-v3.1_123',
                    'timestamp': 1761625992.593592,
                    'realized_pnl': 5000,
                    'positions': {
                        'BTC': {
                            'quantity': 0.5,
                            'entry_price': 100000,
                            'current_price': 105000,
                            'unrealized_pnl': 2500
                        }
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call method
        accounts = client.get_account_totals()

        # Assertions
        assert len(accounts) == 1
        assert accounts[0]['realized_pnl'] == 5000
        assert 'BTC' in accounts[0]['positions']

    @patch('requests.Session.get')
    def test_get_crypto_prices_success(self, mock_get, client, mock_response):
        """Test successful crypto prices fetch"""
        mock_response.json.return_value = {
            'prices': {
                'BTC': {'symbol': 'BTC', 'price': 113908.5, 'timestamp': 1761626103642},
                'ETH': {'symbol': 'ETH', 'price': 4099.65, 'timestamp': 1761626103642}
            }
        }
        mock_get.return_value = mock_response

        # Call method
        prices = client.get_crypto_prices()

        # Assertions
        assert 'BTC' in prices
        assert 'ETH' in prices
        assert prices['BTC']['price'] == 113908.5
        assert prices['ETH']['price'] == 4099.65

    @patch('requests.Session.get')
    def test_get_since_inception_values_success(self, mock_get, client, mock_response):
        """Test successful inception values fetch"""
        mock_response.json.return_value = {
            'sinceInceptionValues': [
                {
                    'model_id': 'deepseek-chat-v3.1',
                    'nav_since_inception': 10000,
                    'inception_date': 1760738685.790017,
                    'num_invocations': 3133
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call method
        values = client.get_since_inception_values()

        # Assertions
        assert len(values) == 1
        assert values[0]['model_id'] == 'deepseek-chat-v3.1'
        assert values[0]['nav_since_inception'] == 10000

    @patch('requests.Session.get')
    def test_get_model_positions_success(self, mock_get, client, mock_response):
        """Test get positions for specific model"""
        mock_response.json.return_value = {
            'accountTotals': [
                {
                    'id': 'deepseek-chat-v3.1_123',
                    'timestamp': 1761625992.593592,
                    'realized_pnl': 5000,
                    'positions': {
                        'BTC': {'quantity': 0.5, 'unrealized_pnl': 2500}
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call method
        positions = client.get_model_positions('deepseek-chat-v3.1')

        # Assertions
        assert positions is not None
        assert positions['model_id'] == 'deepseek-chat-v3.1'
        assert positions['realized_pnl'] == 5000
        assert 'BTC' in positions['positions']

    @patch('requests.Session.get')
    def test_get_model_positions_not_found(self, mock_get, client, mock_response):
        """Test get positions for non-existent model"""
        mock_response.json.return_value = {'accountTotals': []}
        mock_get.return_value = mock_response

        # Call method
        positions = client.get_model_positions('non-existent-model')

        # Assertions
        assert positions is None

    @patch('requests.Session.get')
    def test_get_model_trades_success(self, mock_get, client, mock_response):
        """Test get trades for specific model"""
        mock_response.json.return_value = {
            'trades': [
                {'id': '1', 'model_id': 'deepseek-chat-v3.1', 'symbol': 'BTC'},
                {'id': '2', 'model_id': 'gpt-5', 'symbol': 'ETH'},
                {'id': '3', 'model_id': 'deepseek-chat-v3.1', 'symbol': 'SOL'}
            ]
        }
        mock_get.return_value = mock_response

        # Call method
        trades = client.get_model_trades('deepseek-chat-v3.1')

        # Assertions
        assert len(trades) == 2
        assert all(t['model_id'] == 'deepseek-chat-v3.1' for t in trades)

    @patch('requests.Session.get')
    def test_get_leaderboard_calculation(self, mock_get, client, mock_response):
        """Test leaderboard calculation"""
        # Mock account totals
        def mock_get_side_effect(url, **kwargs):
            mock = Mock()
            mock.raise_for_status = Mock()

            if 'account-totals' in url:
                mock.json.return_value = {
                    'accountTotals': [
                        {
                            'id': 'deepseek-chat-v3.1_123',
                            'timestamp': 1761625992,
                            'realized_pnl': 5000,
                            'positions': {
                                'BTC': {'unrealized_pnl': 2500}
                            }
                        },
                        {
                            'id': 'gpt-5_456',
                            'timestamp': 1761625992,
                            'realized_pnl': -1000,
                            'positions': {}
                        }
                    ]
                }
            elif 'since-inception' in url:
                mock.json.return_value = {
                    'sinceInceptionValues': [
                        {'model_id': 'deepseek-chat-v3.1', 'nav_since_inception': 10000},
                        {'model_id': 'gpt-5', 'nav_since_inception': 10000}
                    ]
                }
            return mock

        mock_get.side_effect = mock_get_side_effect

        # Call method
        leaderboard = client.get_leaderboard()

        # Assertions
        assert len(leaderboard) == 2
        # DeepSeek should be first (better performance)
        assert leaderboard[0]['model_id'] == 'deepseek-chat-v3.1'
        assert leaderboard[0]['nav'] == 17500  # 10000 + 5000 + 2500
        assert leaderboard[0]['return_pct'] == 75.0  # (17500 - 10000) / 10000 * 100

        # GPT-5 should be second
        assert leaderboard[1]['model_id'] == 'gpt-5'
        assert leaderboard[1]['nav'] == 9000  # 10000 - 1000
        assert leaderboard[1]['return_pct'] == -10.0

    @patch('requests.Session.get')
    def test_request_failure(self, mock_get, client):
        """Test API request failure handling"""
        # Mock failed request
        mock_get.side_effect = Exception("Connection error")

        # Call should raise exception
        with pytest.raises(Exception):
            client.get_trades()

    def test_cache_response(self, client, tmp_path):
        """Test response caching"""
        endpoint = "trades"
        data = {'trades': [{'id': '1', 'model_id': 'test'}]}
        params = None

        # Cache the response
        client._cache_response(endpoint, data, params)

        # Check that cache file was created
        cache_files = list(tmp_path.glob("trades_*.json"))
        assert len(cache_files) == 1

        # Read and verify cached data
        with open(cache_files[0], 'r') as f:
            cached = json.load(f)

        assert cached['endpoint'] == 'trades'
        assert cached['data'] == data

    def test_get_client_function(self):
        """Test get_client convenience function"""
        client = get_client()
        assert isinstance(client, Nof1Client)
        assert client.cache_dir is None

    def test_get_client_with_cache(self, tmp_path):
        """Test get_client with cache directory"""
        client = get_client(cache_dir=str(tmp_path))
        assert isinstance(client, Nof1Client)
        assert client.cache_dir == tmp_path


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

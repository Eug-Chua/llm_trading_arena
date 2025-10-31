"""
Technical Indicators Module

Implements all technical indicators used in Alpha Arena prompts:
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Bollinger Bands
- VWMA (Volume Weighted Moving Average)
- ADX (Average Directional Index)
- Supertrend
- CCI (Commodity Channel Index)
- Stochastic Oscillator

Based on research findings from PROMPT_DESIGN_ANALYSIS.md
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

try:
    import pandas_ta as ta
except ImportError:
    ta = None

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def load_indicator_config() -> Dict[str, Any]:
    """
    Load indicator configuration from config file

    Raises:
        FileNotFoundError: If config/indicators.yaml does not exist

    Returns:
        Dict containing indicator configuration
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "indicators.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Critical configuration file missing: {config_path}\n"
            f"This file is required for the trading system to work properly.\n"
            f"Please ensure config/indicators.yaml exists with proper indicator settings."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['data_interval', 'ema_periods', 'rsi_periods', 'macd']
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(
            f"Missing required fields in config/indicators.yaml: {', '.join(missing_fields)}\n"
            f"Please ensure all required indicator settings are defined."
        )

    return config


class TechnicalIndicators:
    """Calculate technical indicators for price data"""

    def __init__(self):
        """Initialize indicator calculator"""
        # Suppress warning - we're using manual calculations instead of pandas-ta
        # if ta is None:
        #     logger.warning("pandas-ta not installed. Install with: pip install pandas-ta")

    @staticmethod
    def prepare_dataframe(candles: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert candle data to pandas DataFrame

        Args:
            candles: List of candle dicts with keys: t, o, h, l, c, v

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not candles:
            logger.warning("Empty candles list provided")
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        # Rename columns to standard names
        column_mapping = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }

        df = df.rename(columns=column_mapping)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by timestamp (oldest to newest)
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.debug(f"Prepared DataFrame with {len(df)} candles")
        return df

    def calculate_ema(
        self,
        df: pd.DataFrame,
        periods: List[int] = [20, 50]
    ) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average

        Args:
            df: DataFrame with 'close' column
            periods: List of EMA periods (default: [20, 50] per research)

        Returns:
            DataFrame with added EMA columns: ema_20, ema_50, etc.
        """
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            logger.debug(f"Calculated EMA-{period}")

        return df

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            DataFrame with added columns: macd, macd_signal, macd_histogram
        """
        # Calculate MACD line
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow

        # Calculate signal line
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()

        # Calculate histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        logger.debug(f"Calculated MACD ({fast}, {slow}, {signal})")
        return df

    def calculate_rsi(
        self,
        df: pd.DataFrame,
        periods: List[int] = [7, 14]
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index

        Per research: Alpha Arena uses both 7-period (short-term)
        and 14-period (standard) RSI

        Args:
            df: DataFrame with 'close' column
            periods: List of RSI periods (default: [7, 14])

        Returns:
            DataFrame with added columns: rsi_7, rsi_14, etc.
        """
        for period in periods:
            # Calculate price changes
            delta = df['close'].diff()

            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain and loss
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            logger.debug(f"Calculated RSI-{period}")

        return df

    def calculate_atr(
        self,
        df: pd.DataFrame,
        periods: List[int] = [3, 14]
    ) -> pd.DataFrame:
        """
        Calculate Average True Range

        Per research: Alpha Arena uses both 3-period and 14-period ATR

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            periods: List of ATR periods (default: [3, 14])

        Returns:
            DataFrame with added columns: atr_3, atr_14, etc.
        """
        # Calculate True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR for each period
        for period in periods:
            df[f'atr_{period}'] = df['true_range'].ewm(span=period, adjust=False).mean()
            logger.debug(f"Calculated ATR-{period}")

        # Clean up temporary columns
        df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)

        return df

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Args:
            df: DataFrame with 'close' column
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2.0)

        Returns:
            DataFrame with added columns: bb_upper, bb_middle, bb_lower, bb_width
        """
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()

        df['bb_upper'] = df['bb_middle'] + (std_dev * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * rolling_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        logger.debug(f"Calculated Bollinger Bands ({period}, {std_dev})")
        return df

    def calculate_vwma(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.DataFrame:
        """
        Calculate Volume Weighted Moving Average

        Args:
            df: DataFrame with 'close' and 'volume' columns
            period: VWMA period (default: 20)

        Returns:
            DataFrame with added column: vwma_{period}
        """
        df[f'vwma_{period}'] = (
            (df['close'] * df['volume']).rolling(window=period).sum() /
            df['volume'].rolling(window=period).sum()
        )

        logger.debug(f"Calculated VWMA-{period}")
        return df

    def calculate_adx(
        self,
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate Average Directional Index

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ADX period (default: 14)

        Returns:
            DataFrame with added columns: adx, plus_di, minus_di
        """
        # Calculate +DM and -DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()

        df['plus_dm'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'],
            0
        )
        df['minus_dm'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'],
            0
        )

        # Calculate True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Smooth +DM, -DM, and TR
        df['plus_dm_smooth'] = df['plus_dm'].ewm(span=period, adjust=False).mean()
        df['minus_dm_smooth'] = df['minus_dm'].ewm(span=period, adjust=False).mean()
        df['tr_smooth'] = df['tr'].ewm(span=period, adjust=False).mean()

        # Calculate +DI and -DI
        df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
        df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])

        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].ewm(span=period, adjust=False).mean()

        # Clean up temporary columns
        temp_cols = ['high_diff', 'low_diff', 'plus_dm', 'minus_dm',
                     'tr1', 'tr2', 'tr3', 'tr', 'plus_dm_smooth',
                     'minus_dm_smooth', 'tr_smooth', 'dx']
        df.drop(temp_cols, axis=1, inplace=True)

        logger.debug(f"Calculated ADX-{period}")
        return df

    def calculate_supertrend(
        self,
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Calculate Supertrend indicator

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 3.0)

        Returns:
            DataFrame with added columns: supertrend, supertrend_direction
        """
        # Calculate ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_st'] = df['tr'].ewm(span=period, adjust=False).mean()

        # Calculate basic bands
        df['hl_avg'] = (df['high'] + df['low']) / 2
        df['basic_upper'] = df['hl_avg'] + (multiplier * df['atr_st'])
        df['basic_lower'] = df['hl_avg'] - (multiplier * df['atr_st'])

        # Calculate final bands
        df['final_upper'] = df['basic_upper']
        df['final_lower'] = df['basic_lower']

        for i in range(1, len(df)):
            # Upper band
            if df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or \
               df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]:
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
            else:
                df.loc[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]

            # Lower band
            if df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or \
               df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]:
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]

        # Determine Supertrend
        df['supertrend'] = np.nan
        df['supertrend_direction'] = 1  # 1 = uptrend, -1 = downtrend

        for i in range(1, len(df)):
            if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1

        # Clean up temporary columns
        temp_cols = ['tr1', 'tr2', 'tr3', 'tr', 'atr_st', 'hl_avg',
                     'basic_upper', 'basic_lower', 'final_upper', 'final_lower']
        df.drop(temp_cols, axis=1, inplace=True)

        logger.debug(f"Calculated Supertrend ({period}, {multiplier})")
        return df

    def calculate_cci(
        self,
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: CCI period (default: 20)

        Returns:
            DataFrame with added column: cci_{period}
        """
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_sma'] = df['tp'].rolling(window=period).mean()
        df['tp_mad'] = df['tp'].rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )

        df[f'cci_{period}'] = (df['tp'] - df['tp_sma']) / (0.015 * df['tp_mad'])

        # Clean up temporary columns
        df.drop(['tp', 'tp_sma', 'tp_mad'], axis=1, inplace=True)

        logger.debug(f"Calculated CCI-{period}")
        return df

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator

        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)

        Returns:
            DataFrame with added columns: stoch_k, stoch_d
        """
        df['lowest_low'] = df['low'].rolling(window=k_period).min()
        df['highest_high'] = df['high'].rolling(window=k_period).max()

        df['stoch_k'] = 100 * (
            (df['close'] - df['lowest_low']) /
            (df['highest_high'] - df['lowest_low'])
        )
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

        # Clean up temporary columns
        df.drop(['lowest_low', 'highest_high'], axis=1, inplace=True)

        logger.debug(f"Calculated Stochastic ({k_period}, {d_period})")
        return df

    def calculate_all_indicators(
        self,
        candles: List[Dict[str, Any]],
        indicator_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators for Alpha Arena

        Args:
            candles: List of candle dictionaries
            indicator_config: Optional config to override default periods

        Returns:
            DataFrame with all indicators calculated
        """
        if not candles:
            logger.warning("No candles provided")
            return pd.DataFrame()

        # Use default config based on research if not provided
        if indicator_config is None:
            indicator_config = load_indicator_config()

        logger.info(f"Calculating all indicators for {len(candles)} candles")

        # Prepare DataFrame
        df = self.prepare_dataframe(candles)

        if df.empty:
            return df

        # Calculate all indicators
        df = self.calculate_ema(df, indicator_config['ema_periods'])
        df = self.calculate_macd(df, **indicator_config['macd'])
        df = self.calculate_rsi(df, indicator_config['rsi_periods'])
        df = self.calculate_atr(df, indicator_config['atr_periods'])
        df = self.calculate_bollinger_bands(df, **indicator_config['bollinger'])
        df = self.calculate_vwma(df, indicator_config['vwma_period'])
        df = self.calculate_adx(df, indicator_config['adx_period'])
        df = self.calculate_supertrend(df, **indicator_config['supertrend'])
        df = self.calculate_cci(df, indicator_config['cci_period'])
        df = self.calculate_stochastic(df, **indicator_config['stochastic'])

        logger.info(f"Calculated {len(df.columns)} total columns including indicators")

        return df

    def get_latest_indicators(
        self,
        df: pd.DataFrame,
        lookback: int = 10
    ) -> Dict[str, Any]:
        """
        Get the latest indicator values formatted for Alpha Arena prompts

        Per research: Alpha Arena provides 10 data points (3-minute intervals)
        plus current values

        Args:
            df: DataFrame with calculated indicators
            lookback: Number of historical values to include (default: 10)

        Returns:
            Dictionary with current values and arrays of historical values
        """
        if df.empty:
            return {}

        # Get the last row (current values)
        current = df.iloc[-1]

        # Get historical arrays (last N values, oldest to newest)
        historical = df.tail(lookback)

        result = {
            'current': {
                'price': float(current['close']),
                'ema_20': float(current['ema_20']) if 'ema_20' in df.columns else None,
                'ema_50': float(current['ema_50']) if 'ema_50' in df.columns else None,
                'macd': float(current['macd']) if 'macd' in df.columns else None,
                'macd_signal': float(current['macd_signal']) if 'macd_signal' in df.columns else None,
                'rsi_7': float(current['rsi_7']) if 'rsi_7' in df.columns else None,
                'rsi_14': float(current['rsi_14']) if 'rsi_14' in df.columns else None,
                'atr_3': float(current['atr_3']) if 'atr_3' in df.columns else None,
                'atr_14': float(current['atr_14']) if 'atr_14' in df.columns else None,
            },
            'arrays': {
                'prices': historical['close'].tolist(),
                'ema_20': historical['ema_20'].tolist() if 'ema_20' in df.columns else [],
                'ema_50': historical['ema_50'].tolist() if 'ema_50' in df.columns else [],
                'macd': historical['macd'].tolist() if 'macd' in df.columns else [],
                'rsi_7': historical['rsi_7'].tolist() if 'rsi_7' in df.columns else [],
                'rsi_14': historical['rsi_14'].tolist() if 'rsi_14' in df.columns else [],
            },
            'stats': {
                'high': float(historical['high'].max()),
                'low': float(historical['low'].min()),
                'mean': float(historical['close'].mean()),
                'volume_current': float(current['volume']),
                'volume_avg': float(historical['volume'].mean()),
            }
        }

        return result


# Convenience function
def calculate_indicators_for_candles(
    candles: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Calculate all indicators for candle data

    Args:
        candles: List of candle dictionaries
        config: Optional indicator configuration

    Returns:
        DataFrame with all indicators
    """
    calculator = TechnicalIndicators()
    return calculator.calculate_all_indicators(candles, config)


def get_alpha_arena_indicators(
    candles: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate indicators in Alpha Arena format

    Args:
        candles: List of candle dictionaries

    Returns:
        Dictionary with current values and 10-point historical arrays
    """
    calculator = TechnicalIndicators()
    df = calculator.calculate_all_indicators(candles)
    return calculator.get_latest_indicators(df, lookback=10)

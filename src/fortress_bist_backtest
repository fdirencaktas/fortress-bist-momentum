"""
BIST-100 Momentum and Mean Reversion Strategies Backtesting Framework
===============================================================================

This module implements a comprehensive backtesting framework for evaluating
momentum and mean reversion trading strategies on BIST-100 stocks from 2018-2025.

Author: Fikri Direnç Aktaş
Date: 2026


The framework includes:
- Data fetching from Yahoo Finance with warm-up period
- Five trading strategies (DMA, Donchian, Combo, RSI, Bollinger Bands)
- Dual currency support (TRY for local, USD for international investors)
- Comprehensive performance metrics and trade statistics
- Robustness checks and sensitivity analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================

SECTOR_CONFIG = {
    'Banking': ['GARAN.IS', 'AKBNK.IS', 'VAKBN.IS', 'ISCTR.IS', 'HALKB.IS'],
    'Insurance': ['ANSGR.IS', 'TURSG.IS'],
    'Industrial': ['FROTO.IS', 'EREGL.IS', 'TOASO.IS'],
    'Retail': ['BIMAS.IS', 'MGROS.IS'],
    'Transportation': ['THYAO.IS', 'TAVHL.IS', 'PGSUS.IS'],
    'Consumer Durable': ['ARCLK.IS', 'VESTL.IS', 'SISE.IS'],
    'Consumer Non-Durable': ['TUKAS.IS', 'ULKER.IS', 'AEFES.IS'],
    'Materials': ['SASA.IS', 'PETKM.IS', 'GUBRF.IS'],
    'Telecom': ['TCELL.IS', 'TTKOM.IS'],
    'Construction And Cement': ['ENKAI.IS', 'CIMSA.IS', 'OYAKC.IS', 'KUYAS.IS', 'BTCIM.IS'],
    'Energy': ['TUPRS.IS', 'ZOREN.IS', 'AKSEN.IS'],
    'Holding': ['KCHOL.IS', 'SAHOL.IS', 'AGHOL.IS', 'RALYH.IS', 'DOHOL.IS'],
    'Defense & Aerospace': ['ASELS.IS', 'OTKAR.IS'],
    'Healthcare': ['ECILC.IS'],
}

SUBPERIODS = [
    ('Pre-COVID', '2018-01-01', '2020-02-29'),
    ('COVID Crash & Recovery', '2020-03-01', '2021-12-31'),
    ('High Inflation Era', '2022-01-01', '2023-12-31'),
    ('Policy Shift', '2024-01-01', '2025-12-31')
]

# Risk-free rates
TRY_RISK_FREE_RATE = 0.25  # 25% average Turkish deposit rate
USD_RISK_FREE_RATE = 0.03  # 3% US Treasury yield

# Transaction cost assumptions
SLIPPAGE_BASELINE = 0.001  # 0.1%
SLIPPAGE_LOW = 0.0005      # 0.05%
SLIPPAGE_HIGH = 0.002      # 0.20%

# Portfolio constraints
MAX_POSITIONS = 15
MIN_CASH_BUFFER = 0.05


# ==============================================================================
# CORE CLASSES
# ==============================================================================

class BISTMomentumStrategy:
    """
    Handles data fetching, resampling, and universe construction for BIST-100 stocks.
    
    This class manages the entire data pipeline from Yahoo Finance, including:
    - Fetching stock price data in batches
    - USD/TRY exchange rate data
    - Filtering stocks with sufficient history
    - Resampling to weekly frequency
    - Creating monthly eligible stock universes
    """
    
    def __init__(
        self,
        data_start: str = '2016-01-01',
        analysis_start: str = '2018-01-01',
        end_date: str = '2025-12-31',
        frequency: str = 'weekly'
    ) -> None:
        """
        Initialize the BIST momentum strategy data handler.
        
        Parameters
        ----------
        data_start : str, default='2016-01-01'
            Start date for data fetching (includes warm-up period)
        analysis_start : str, default='2018-01-01'
            Start date for analysis (signals calculated from this date)
        end_date : str, default='2025-12-31'
            End date for analysis
        frequency : str, default='weekly'
            Data frequency ('weekly' or 'daily')
        """
        self.data_start = data_start
        self.analysis_start = analysis_start
        self.end_date = end_date
        self.frequency = frequency
        self.sectors = SECTOR_CONFIG
        
        # Flatten and deduplicate stock list from sector configuration
        self.stocks = []
        for sector_stocks in SECTOR_CONFIG.values():
            self.stocks.extend(sector_stocks)
        self.stocks = list(dict.fromkeys(self.stocks))
        
        # Initialize data containers
        self.usdtry_full: Optional[pd.Series] = None
        self.prices_try_full: Optional[pd.DataFrame] = None
        self.prices_usd_full: Optional[pd.DataFrame] = None
        self.eligible_stocks: Dict[pd.Timestamp, List[str]] = {}
    
    def resample_to_weekly(self, prices: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """
        Resample daily prices to weekly frequency (Friday close).
        
        Parameters
        ----------
        prices : pd.DataFrame or pd.Series
            Daily price data to resample
            
        Returns
        -------
        pd.DataFrame or pd.Series
            Weekly price data (Friday close)
        """
        if isinstance(prices, pd.DataFrame):
            weekly_prices = prices.resample('W-FRI').last()
            weekly_prices = weekly_prices.dropna(how='all')
        else:
            weekly_prices = prices.resample('W-FRI').last()
            weekly_prices = weekly_prices.dropna()
        return weekly_prices
    
    def fetch_bist_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Fetch BIST-100 stock data and USD/TRY rates with warm-up period.
        
        Downloads data in batches to avoid Yahoo Finance limitations, filters
        stocks with sufficient history, and resamples to weekly frequency.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]
            - TRY-denominated prices for analysis period
            - USD-denominated prices for analysis period
            - USD/TRY exchange rates for analysis period
        """
        print("=" * 60)
        print("📊 DATA FETCHING")
        print("=" * 60)
        print(f"Frequency: {self.frequency.upper()}")
        print(f"Stocks: {len(self.stocks)}")
        print(f"Data period: {self.data_start} to {self.end_date} (includes warm-up)")
        print(f"Analysis period: {self.analysis_start} to {self.end_date}")
        
        # Fetch data in batches
        batch_size = 20
        all_data = []
        
        for i in range(0, len(self.stocks), batch_size):
            batch = self.stocks[i:i + batch_size]
            try:
                data = yf.download(
                    batch + ['TRY=X'],
                    start=self.data_start,
                    end=self.end_date,
                    progress=False
                )['Close']
                all_data.append(data)
                print(f"  ✓ Batch {i // batch_size + 1}: {len(batch)} stocks fetched")
            except Exception as e:
                print(f"  ⚠ Batch {i // batch_size + 1} failed: {e}")
                continue
        
        # Combine all data
        data = pd.concat(all_data, axis=1)
        data = data.loc[:, ~data.columns.duplicated()]
        
        # Extract currency and price data
        self.usdtry_full = data['TRY=X']
        self.prices_try_full = data.drop('TRY=X', axis=1)
        
        # Filter stocks with sufficient history by analysis start
        historical_by_analysis = self.prices_try_full.loc[:self.analysis_start]
        min_days = 252
        valid_stocks = historical_by_analysis.columns[
            historical_by_analysis.count() >= min_days
        ]
        self.prices_try_full = self.prices_try_full[valid_stocks]
        print(f"\n✓ {len(valid_stocks)} stocks have sufficient history by {self.analysis_start}")
        
        # Convert to USD
        self.prices_usd_full = self.prices_try_full.div(self.usdtry_full, axis=0)
        
        # Resample to weekly if requested
        if self.frequency == 'weekly':
            print("Resampling to weekly frequency (Friday close)...")
            self.prices_try_full = self.resample_to_weekly(self.prices_try_full)  # type: ignore
            self.prices_usd_full = self.resample_to_weekly(self.prices_usd_full)  # type: ignore
            self.usdtry_full = self.resample_to_weekly(self.usdtry_full)  # type: ignore
        
        # Slice to analysis period
        prices_try_analysis = self.prices_try_full.loc[self.analysis_start:]
        prices_usd_analysis = self.prices_usd_full.loc[self.analysis_start:]
        usdtry_analysis = self.usdtry_full.loc[self.analysis_start:]
        
        print(f"✓ Data ready: {len(prices_try_analysis.columns)} stocks, {len(prices_try_analysis)} periods")
        print("=" * 60)
        
        return prices_try_analysis, prices_usd_analysis, usdtry_analysis
    
    def create_monthly_universe(self) -> Dict[pd.Timestamp, List[str]]:
        """
        Create monthly eligible stocks list from analysis_start.
        
        For each month, determines which stocks have sufficient historical
        data for signal calculation.
        
        Returns
        -------
        Dict[pd.Timestamp, List[str]]
            Mapping from month start to list of eligible stock symbols
        """
        monthly_dates = pd.date_range(
            start=self.analysis_start,
            end=self.end_date,
            freq='MS'
        )
        
        for month_start in monthly_dates:
            if self.frequency == 'weekly':
                historical_data = self.prices_try_full.loc[:month_start]  # type: ignore
                min_periods = 52
            else:
                historical_data = self.prices_try_full.loc[:month_start]  # type: ignore
                min_periods = 252
            
            eligible = historical_data.columns[
                historical_data.count() >= min_periods
            ]
            self.eligible_stocks[month_start] = list(eligible)
        
        return self.eligible_stocks


class MomentumStrategies:
    """
    Implementation of momentum and mean reversion trading strategies.
    
    All strategies are implemented as static methods that return entry and exit
    signals for a given price DataFrame.
    """
    
    @staticmethod
    def dual_moving_average_weekly(
        prices: Union[pd.DataFrame, pd.Series],
        fast_weeks: int = 21,
        slow_weeks: int = 50
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Weekly Dual Moving Average Crossover strategy.
        
        Generates buy signals when fast EMA crosses above slow EMA,
        and sell signals when fast EMA crosses below slow EMA.
        
        Parameters
        ----------
        prices : pd.DataFrame or pd.Series
            Price data (columns are stocks, index is datetime)
        fast_weeks : int, default=21
            Fast EMA period (weeks)
        slow_weeks : int, default=50
            Slow EMA period (weeks)
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Entry signals and exit signals DataFrames
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        
        ema_fast = prices.ewm(span=fast_weeks, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_weeks, adjust=False).mean()
        
        entries = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        exits = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        
        return entries, exits
    
    @staticmethod
    def donchian_breakout_weekly(
        prices: Union[pd.DataFrame, pd.Series],
        window_weeks: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Weekly Donchian Channel Breakout strategy.
        
        Generates buy signals when price breaks above the highest price of
        the lookback window, and sell signals when price breaks below the
        lowest price.
        
        Parameters
        ----------
        prices : pd.DataFrame or pd.Series
            Price data (columns are stocks, index is datetime)
        window_weeks : int, default=20
            Lookback window for channel calculation (weeks)
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Entry signals, exit signals, and channel midpoint
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        
        upper = prices.rolling(window=window_weeks).max()
        lower = prices.rolling(window=window_weeks).min()
        midpoint = (upper + lower) / 2
        
        entries = prices > upper.shift(1)
        exits = prices < lower.shift(1)
        
        return entries, exits, midpoint
    
    @staticmethod
    def combo_filtered_weekly(
        prices: Union[pd.DataFrame, pd.Series],
        fast_weeks: int = 21,
        slow_weeks: int = 50,
        donchian_weeks: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Combo Strategy: DMA + Donchian Filter.
        
        Requires both DMA uptrend and price above Donchian midpoint to
        generate entry signals, reducing false signals in sideways markets.
        
        Parameters
        ----------
        prices : pd.DataFrame or pd.Series
            Price data (columns are stocks, index is datetime)
        fast_weeks : int, default=21
            Fast EMA period for DMA (weeks)
        slow_weeks : int, default=50
            Slow EMA period for DMA (weeks)
        donchian_weeks : int, default=20
            Lookback window for Donchian channel (weeks)
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Entry signals and exit signals DataFrames
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        
        # DMA condition
        ema_fast = prices.ewm(span=fast_weeks, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_weeks, adjust=False).mean()
        condition1 = ema_fast > ema_slow
        
        # Donchian midpoint condition
        upper = prices.rolling(window=donchian_weeks).max()
        lower = prices.rolling(window=donchian_weeks).min()
        midpoint = (upper + lower) / 2
        condition2 = prices > midpoint
        
        # Combined long condition
        long_condition = condition1 & condition2
        
        entries = long_condition & (~long_condition.shift(1).fillna(False))
        exits = (~long_condition) & long_condition.shift(1).fillna(False)
        
        return entries, exits
    
    @staticmethod
    def rsi_mean_reversion_weekly(
        prices: Union[pd.DataFrame, pd.Series],
        rsi_weeks: int = 14,
        oversold: int = 30,
        overbought: int = 70
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        RSI Mean Reversion Strategy.
        
        Generates buy signals when RSI falls below oversold threshold,
        and sell signals when RSI rises above overbought threshold.
        
        Parameters
        ----------
        prices : pd.DataFrame or pd.Series
            Price data (columns are stocks, index is datetime)
        rsi_weeks : int, default=14
            RSI calculation period (weeks)
        oversold : int, default=30
            Oversold threshold (buy signal)
        overbought : int, default=70
            Overbought threshold (sell signal)
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Entry signals and exit signals DataFrames
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_weeks).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_weeks).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        entries = rsi < oversold
        exits = rsi > overbought
        
        return entries, exits
    
    @staticmethod
    def bollinger_band_mean_reversion_weekly(
        prices: Union[pd.DataFrame, pd.Series],
        window_weeks: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Bollinger Band Mean Reversion Strategy.
        
        Generates buy signals when price falls below lower band,
        and sell signals when price rises above upper band.
        
        Parameters
        ----------
        prices : pd.DataFrame or pd.Series
            Price data (columns are stocks, index is datetime)
        window_weeks : int, default=20
            Moving average and standard deviation window (weeks)
        num_std : float, default=2.0
            Number of standard deviations for bands
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Entry signals and exit signals DataFrames
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        
        rolling_mean = prices.rolling(window=window_weeks).mean()
        rolling_std = prices.rolling(window=window_weeks).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        entries = prices < lower_band
        exits = prices > upper_band
        
        return entries, exits
    
    @staticmethod
    def get_strategies(frequency: str = 'weekly') -> Dict[str, Dict[str, Any]]:
        """
        Return strategy functions with descriptions.
        
        Parameters
        ----------
        frequency : str, default='weekly'
            Data frequency ('weekly' or 'daily')
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping strategy keys to {'func': callable, 'description': str}
        """
        return {
            'DMA_21_50': {
                'func': lambda p: MomentumStrategies.dual_moving_average_weekly(p, 21, 50),
                'description': 'Weekly Dual Moving Average (21/50 weeks)'
            },
            'Donchian_20': {
                'func': lambda p: MomentumStrategies.donchian_breakout_weekly(p, 20)[:2],
                'description': 'Weekly Donchian Breakout (20 weeks)'
            },
            'Combo_Filtered': {
                'func': lambda p: MomentumStrategies.combo_filtered_weekly(p, 21, 50, 20),
                'description': 'Weekly DMA + Donchian Filter'
            },
            'RSI_14': {
                'func': lambda p: MomentumStrategies.rsi_mean_reversion_weekly(p, 14, 30, 70),
                'description': 'Weekly RSI Mean Reversion (14-week, 30/70)'
            },
            'BB_20_2': {
                'func': lambda p: MomentumStrategies.bollinger_band_mean_reversion_weekly(p, 20, 2),
                'description': 'Weekly Bollinger Band (20-week, 2 std)'
            }
        }
    
    @staticmethod
    def get_parameter_variants() -> Dict[str, List[Dict[str, Any]]]:
        """
        Return parameter variants for sensitivity analysis.
        
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary mapping strategy families to lists of variants
        """
        return {
            'DMA': [
                {'name': 'DMA_10_30', 'func': lambda p: MomentumStrategies.dual_moving_average_weekly(p, 10, 30),
                 'desc': 'DMA (10/30 weeks) - Fast'},
                {'name': 'DMA_21_50', 'func': lambda p: MomentumStrategies.dual_moving_average_weekly(p, 21, 50),
                 'desc': 'DMA (21/50 weeks) - Baseline'},
                {'name': 'DMA_30_60', 'func': lambda p: MomentumStrategies.dual_moving_average_weekly(p, 30, 60),
                 'desc': 'DMA (30/60 weeks) - Slow'}
            ],
            'Donchian': [
                {'name': 'Donchian_10', 'func': lambda p: MomentumStrategies.donchian_breakout_weekly(p, 10)[:2],
                 'desc': 'Donchian (10 weeks) - Fast'},
                {'name': 'Donchian_20', 'func': lambda p: MomentumStrategies.donchian_breakout_weekly(p, 20)[:2],
                 'desc': 'Donchian (20 weeks) - Baseline'},
                {'name': 'Donchian_30', 'func': lambda p: MomentumStrategies.donchian_breakout_weekly(p, 30)[:2],
                 'desc': 'Donchian (30 weeks) - Slow'}
            ],
            'RSI': [
                {'name': 'RSI_10', 'func': lambda p: MomentumStrategies.rsi_mean_reversion_weekly(p, 10, 30, 70),
                 'desc': 'RSI (10 weeks) - Fast'},
                {'name': 'RSI_14', 'func': lambda p: MomentumStrategies.rsi_mean_reversion_weekly(p, 14, 30, 70),
                 'desc': 'RSI (14 weeks) - Baseline'},
                {'name': 'RSI_20', 'func': lambda p: MomentumStrategies.rsi_mean_reversion_weekly(p, 20, 30, 70),
                 'desc': 'RSI (20 weeks) - Slow'}
            ],
            'BB': [
                {'name': 'BB_15_2', 'func': lambda p: MomentumStrategies.bollinger_band_mean_reversion_weekly(p, 15, 2),
                 'desc': 'BB (15 weeks, 2σ) - Fast'},
                {'name': 'BB_20_2', 'func': lambda p: MomentumStrategies.bollinger_band_mean_reversion_weekly(p, 20, 2),
                 'desc': 'BB (20 weeks, 2σ) - Baseline'},
                {'name': 'BB_20_2.5', 'func': lambda p: MomentumStrategies.bollinger_band_mean_reversion_weekly(p, 20, 2.5),
                 'desc': 'BB (20 weeks, 2.5σ) - Wider bands'}
            ]
        }


class PortfolioBacktester:
    """
    Manages portfolio construction, position tracking, and trade execution.
    
    This class simulates realistic trading with:
    - Position limits and cash buffers
    - Transaction costs (slippage)
    - Interest on cash balances
    - Trade-level statistics tracking
    """
    
    def __init__(
        self,
        prices_try: pd.DataFrame,
        usdtry: pd.Series,
        eligible_stocks: Dict[pd.Timestamp, List[str]],
        slippage: float = SLIPPAGE_BASELINE
    ) -> None:
        """
        Initialize the portfolio backtester.
        
        Parameters
        ----------
        prices_try : pd.DataFrame
            TRY-denominated stock prices
        usdtry : pd.Series
            USD/TRY exchange rates
        eligible_stocks : Dict[pd.Timestamp, List[str]]
            Monthly eligible stock universes
        slippage : float, default=0.001
            Transaction cost as decimal (0.001 = 0.1%)
        """
        self.prices_try = prices_try
        self.usdtry = usdtry
        self.eligible_stocks = eligible_stocks
        self.slippage = slippage
    
    def run_strategy_try(
        self,
        strategy_func: callable,
        strategy_name: str,
        debug: bool = False
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Run strategy in TRY (local investor perspective).
        
        Parameters
        ----------
        strategy_func : callable
            Function that returns (entries, exits) from price data
        strategy_name : str
            Name of the strategy for debugging
        debug : bool, default=False
            If True, print detailed trade statistics
            
        Returns
        -------
        Tuple[pd.Series, Dict[str, float]]
            Portfolio value series and trade statistics dictionary
        """
        portfolio_try, trade_stats = self._run_strategy_base(strategy_func, strategy_name, debug)
        return portfolio_try, trade_stats
    
    def run_strategy_usd(
        self,
        strategy_func: callable,
        strategy_name: str,
        debug: bool = False
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Run strategy for international investor (USD perspective).
        
        Converts initial USD to TRY, trades in TRY, and converts back to USD.
        
        Parameters
        ----------
        strategy_func : callable
            Function that returns (entries, exits) from price data
        strategy_name : str
            Name of the strategy for debugging
        debug : bool, default=False
            If True, print detailed trade statistics
            
        Returns
        -------
        Tuple[pd.Series, Dict[str, float]]
            USD-denominated portfolio value series and trade statistics
        """
        portfolio_try, trade_stats = self.run_strategy_try(strategy_func, strategy_name, debug)
        
        initial_usd = 1.0
        initial_try = initial_usd * self.usdtry.iloc[0]
        scale_factor = initial_try / portfolio_try.iloc[0]
        portfolio_try_scaled = portfolio_try * scale_factor
        portfolio_usd = portfolio_try_scaled / self.usdtry
        
        return portfolio_usd, trade_stats
    
    def _run_strategy_base(
        self,
        strategy_func: callable,
        strategy_name: str,
        debug: bool = False
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Base strategy runner with trade statistics.
        
        This is the core backtesting engine that simulates trading day by day,
        tracking positions, cash, and generating performance statistics.
        
        Parameters
        ----------
        strategy_func : callable
            Function that returns (entries, exits) from price data
        strategy_name : str
            Name of the strategy for debugging
        debug : bool, default=False
            If True, print detailed trade statistics
            
        Returns
        -------
        Tuple[pd.Series, Dict[str, float]]
            TRY-denominated portfolio value series and trade statistics
        """
        # Generate signals
        result = strategy_func(self.prices_try)
        
        if isinstance(result, tuple):
            if len(result) == 2:
                entries, exits = result
            elif len(result) == 3:
                entries, exits, _ = result
            else:
                raise ValueError(f"Strategy {strategy_name} returned unexpected number of values")
        else:
            entries = result
            exits = ~entries.shift(1).fillna(False)
        
        # Ensure DataFrame format
        if isinstance(entries, pd.Series):
            entries = entries.to_frame()
        if isinstance(exits, pd.Series):
            exits = exits.to_frame()
        
        # Align with price data
        entries = entries.reindex(columns=self.prices_try.columns, fill_value=False)
        exits = exits.reindex(columns=self.prices_try.columns, fill_value=False)
        
        # Initialize portfolio tracking
        portfolio_try = pd.Series(1.0, index=self.prices_try.index)
        cash_try = pd.Series(1.0, index=self.prices_try.index)
        current_positions: Dict[str, float] = {}
        
        # Trade statistics tracking
        total_turnover = 0.0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_gain = 0.0
        total_loss = 0.0
        
        # Interest calculation
        periods_per_year = 52
        weekly_interest_rate = (1 + TRY_RISK_FREE_RATE) ** (1 / periods_per_year) - 1
        
        # Main backtesting loop
        for i, date in enumerate(self.prices_try.index):
            if i == 0:
                continue
            
            month_start = pd.Timestamp(date.year, date.month, 1)
            eligible = self.eligible_stocks.get(month_start, list(self.prices_try.columns))
            
            # --- PROCESS EXITS ---
            if date in exits.index:
                for stock in list(current_positions.keys()):
                    if stock in exits.columns and exits.loc[date, stock]:
                        exit_price_try = self.prices_try.loc[date, stock] * (1 - self.slippage)
                        if not pd.isna(exit_price_try) and exit_price_try > 0:
                            exit_value = current_positions[stock] * exit_price_try
                            cash_try.loc[date:] += exit_value
                            total_turnover += exit_value
                            total_trades += 1
                            
                            # Track trade outcome
                            entry_value = current_positions[f"{stock}_entry_value"]
                            trade_return = (exit_value - entry_value) / entry_value
                            if trade_return > 0:
                                winning_trades += 1
                                total_gain += trade_return
                            else:
                                losing_trades += 1
                                total_loss += abs(trade_return)
                            
                            # Clean up position tracking
                            del current_positions[stock]
                            if f"{stock}_entry_value" in current_positions:
                                del current_positions[f"{stock}_entry_value"]
            
            # --- PROCESS ENTRIES ---
            if date in entries.index:
                entry_candidates = []
                for stock in entries.columns:
                    if (entries.loc[date, stock] and
                        stock in eligible and
                        stock not in current_positions):
                        entry_candidates.append(stock)
                
                for stock in entry_candidates:
                    active_positions = len([k for k in current_positions.keys() if not k.endswith('_entry_value')])
                    if active_positions < MAX_POSITIONS:
                        available_cash_try = cash_try.loc[date] * (1 - MIN_CASH_BUFFER)
                        position_value_try = available_cash_try / (MAX_POSITIONS - active_positions)
                        
                        entry_price_try = self.prices_try.loc[date, stock] * (1 + self.slippage)
                        
                        if (position_value_try > 0 and entry_price_try > 0 and
                            not pd.isna(entry_price_try) and not pd.isna(position_value_try)):
                            shares = position_value_try / entry_price_try
                            current_positions[stock] = shares
                            current_positions[f"{stock}_entry_value"] = position_value_try
                            cash_try.loc[date:] -= position_value_try
                            total_turnover += position_value_try
            
            # --- APPLY INTEREST TO CASH ---
            if i > 0:
                days_between = (date - self.prices_try.index[i - 1]).days
                weeks_between = days_between / 7
                interest_multiplier = (1 + weekly_interest_rate) ** weeks_between
                cash_try.loc[date:] = cash_try.loc[date] * interest_multiplier
            
            # --- UPDATE PORTFOLIO VALUE ---
            portfolio_try.loc[date] = cash_try.loc[date]
            active_positions = [k for k in current_positions.keys() if not k.endswith('_entry_value')]
            for stock in active_positions:
                if stock in self.prices_try.columns:
                    current_price_try = self.prices_try.loc[date, stock]
                    if not pd.isna(current_price_try):
                        portfolio_try.loc[date] += current_positions[stock] * current_price_try
            
            if portfolio_try.loc[date] <= 0:
                portfolio_try.loc[date] = cash_try.loc[date]
        
        # --- CALCULATE TRADE STATISTICS ---
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_gain = total_gain / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        win_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else np.inf
        profit_factor = total_gain / total_loss if total_loss > 0 else np.inf
        expectancy = (win_rate * avg_gain) - ((1 - win_rate) * avg_loss) if total_trades > 0 else 0
        
        months = (self.prices_try.index[-1] - self.prices_try.index[0]).days / 30
        avg_monthly_turnover = (total_turnover / months) / portfolio_try.iloc[0] * 100 if months > 0 else 0
        
        trade_stats = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_gain': avg_gain * 100,
            'avg_loss': avg_loss * 100,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy * 100,
            'avg_monthly_turnover': avg_monthly_turnover
        }
        
        if debug:
            self._print_trade_stats(strategy_name, trade_stats)
        
        return portfolio_try, trade_stats
    
    def _print_trade_stats(self, strategy_name: str, stats: Dict[str, float]) -> None:
        """Print formatted trade statistics for debugging."""
        print(f"\n{strategy_name} - Trade Statistics:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate'] * 100:.1f}%")
        print(f"  Avg Win: {stats['avg_gain']:.1f}%")
        print(f"  Avg Loss: {stats['avg_loss']:.1f}%")
        print(f"  Win/Loss Ratio: {stats['win_loss_ratio']:.2f}x")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Expectancy: {stats['expectancy']:.2f}%")


class PerformanceAnalyzer:
    """
    Calculates comprehensive performance metrics for trading strategies.
    
    Provides methods for:
    - Standard metrics (Sharpe, Sortino, Calmar ratios)
    - Drawdown and drawup analysis
    - Rolling statistics
    - Statistical significance testing (bootstrap and t-test)
    - Subperiod analysis
    """
    
    def __init__(self, portfolio_values: pd.Series, benchmark_values: pd.Series) -> None:
        """
        Initialize the performance analyzer.
        
        Parameters
        ----------
        portfolio_values : pd.Series
            Strategy portfolio value series
        benchmark_values : pd.Series
            Benchmark value series for comparison
        """
        self.portfolio = portfolio_values
        self.benchmark = benchmark_values
    
    def calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """
        Calculate maximum drawdown from peak to trough.
        
        Parameters
        ----------
        price_series : pd.Series
            Price or portfolio value series
            
        Returns
        -------
        float
            Maximum drawdown as decimal (negative value)
        """
        returns = price_series.pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    def calculate_max_drawup(self, price_series: pd.Series) -> float:
        """
        Calculate maximum drawup (gain from trough to peak).
        
        Parameters
        ----------
        price_series : pd.Series
            Price or portfolio value series
            
        Returns
        -------
        float
            Maximum drawup as decimal
        """
        returns = price_series.pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_min = cumulative.expanding().min()
        drawup = (cumulative - running_min) / running_min
        return float(drawup.max())
    
    def calculate_metrics(self, rf_rate: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        rf_rate : float, optional
            Risk-free rate for Sharpe and Sortino calculations.
            If None, auto-detects based on currency (TRY=0.25, USD=0.03)
            
        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics
        """
        returns = self.portfolio.pct_change().dropna()
        
        if len(returns) == 0:
            return {key: 0.0 for key in [
                'Total Return', 'Annualized Return', 'Annualized Volatility',
                'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown',
                'Maximum Drawup', 'Calmar Ratio', 'Win Rate'
            ]}
        
        # Determine periods per year based on data frequency
        date_diff = (returns.index[-1] - returns.index[0]).days / len(returns)
        periods_per_year = 52 if date_diff > 3 else 252
        
        # Return metrics
        total_return = float(self.portfolio.iloc[-1] / self.portfolio.iloc[0] - 1)
        years = len(returns) / periods_per_year
        ann_return = float((1 + total_return) ** (1 / years) - 1 if years > 0 else 0)
        
        # Risk metrics
        ann_vol = float(returns.std() * np.sqrt(periods_per_year))
        
        # Risk-free rate
        if rf_rate is None:
            if hasattr(self.portfolio, 'name') and 'USD' in str(self.portfolio.name):
                rf_rate = USD_RISK_FREE_RATE
            else:
                rf_rate = TRY_RISK_FREE_RATE
        
        # Sharpe ratio
        excess_return = float(returns.mean() * periods_per_year - rf_rate)
        sharpe = excess_return / ann_vol if ann_vol > 0 else 0
        
        # Drawdown and drawup
        max_dd = self.calculate_max_drawdown(self.portfolio)
        max_drawup = self.calculate_max_drawup(self.portfolio)
        
        # Win rate (percentage of positive periods)
        win_rate = float((returns > 0).sum() / len(returns))
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = float(downside_returns.std() * np.sqrt(periods_per_year))
            sortino = (ann_return - rf_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino = 0
        
        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Maximum Drawdown': max_dd,
            'Maximum Drawup': max_drawup,
            'Win Rate': win_rate,
            'Calmar Ratio': calmar
        }
    
    def calculate_rolling_sharpe(
        self,
        window_years: int = 3,
        rf_rate: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Parameters
        ----------
        window_years : int, default=3
            Rolling window length in years
        rf_rate : float, optional
            Risk-free rate for calculation
            
        Returns
        -------
        pd.Series
            Rolling Sharpe ratio series
        """
        returns = self.portfolio.pct_change().dropna()
        
        if len(returns) < 52:
            return pd.Series(dtype=float)
        
        date_diff = (returns.index[-1] - returns.index[0]).days / len(returns)
        periods_per_year = 52 if date_diff > 3 else 252
        window = int(window_years * periods_per_year)
        
        if len(returns) < window:
            return pd.Series(dtype=float)
        
        if rf_rate is None:
            if hasattr(self.portfolio, 'name') and 'USD' in str(self.portfolio.name):
                rf_rate = USD_RISK_FREE_RATE
            else:
                rf_rate = TRY_RISK_FREE_RATE
        
        rolling_mean = returns.rolling(window).mean() * periods_per_year
        rolling_std = returns.rolling(window).std() * np.sqrt(periods_per_year)
        rolling_sharpe = (rolling_mean - rf_rate) / rolling_std
        rolling_sharpe = rolling_sharpe.dropna()
        rolling_sharpe.index = pd.to_datetime(rolling_sharpe.index)
        
        return rolling_sharpe
    
    def calculate_subperiod_metrics(
        self,
        subperiods: List[Tuple[str, str, str]],
        rf_rate: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each subperiod.
        
        Parameters
        ----------
        subperiods : List[Tuple[str, str, str]]
            List of (period_name, start_date, end_date)
        rf_rate : float, optional
            Risk-free rate for calculations
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping from period name to metrics dictionary
        """
        subperiod_results = {}
        
        for period_name, start_date, end_date in subperiods:
            period_portfolio = self.portfolio.loc[start_date:end_date]
            if len(period_portfolio) > 0:
                period_analyzer = PerformanceAnalyzer(period_portfolio, period_portfolio)
                subperiod_results[period_name] = period_analyzer.calculate_metrics(rf_rate)
        
        return subperiod_results
    
    def bootstrap_test_fixed(self, n_iterations: int = 10000) -> Dict[str, Any]:
        """
        Bootstrap test for excess mean return > 0.
        
        Uses moving block bootstrap to account for autocorrelation.
        
        Parameters
        ----------
        n_iterations : int, default=10000
            Number of bootstrap iterations
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'excess_sharpe', 'p_value', and 'significant' keys
        """
        strategy_returns = self.portfolio.pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()
        
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        
        excess_returns = strategy_returns - benchmark_returns
        excess_returns = excess_returns.dropna()
        
        n = len(excess_returns)
        
        if n < 20:
            return {'excess_sharpe': 0.0, 'p_value': 1.0, 'significant': False}
        
        date_diff = (excess_returns.index[-1] - excess_returns.index[0]).days / n
        periods_per_year = 52 if date_diff > 3 else 252
        
        original_mean = float(excess_returns.mean())
        original_std = float(excess_returns.std())
        
        original_sharpe = (original_mean * periods_per_year /
                          (original_std * np.sqrt(periods_per_year))) if original_std > 0 else 0
        
        if original_mean <= 0:
            return {'excess_sharpe': original_sharpe, 'p_value': 1.0, 'significant': False}
        
        # Moving block bootstrap
        centered = excess_returns - original_mean
        block_size = int(np.sqrt(n))
        bootstrap_means = []
        
        for _ in range(n_iterations):
            sampled = []
            while len(sampled) < n:
                start = np.random.randint(0, n - block_size + 1)
                block = centered.iloc[start:start + block_size]
                sampled.extend(block.values)
            sampled = np.array(sampled[:n])
            bootstrap_means.append(float(sampled.mean()))
        
        bootstrap_means = np.array(bootstrap_means)
        p_value = float(np.mean(bootstrap_means >= original_mean))
        
        return {
            'excess_sharpe': original_sharpe,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def t_test_vs_benchmark(self) -> Dict[str, Any]:
        """
        T-test comparing strategy returns to benchmark returns.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with 't_statistic', 'p_value', and 'mean_difference'
        """
        strategy_returns = self.portfolio.pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()
        
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns[common_idx]
        benchmark_returns = benchmark_returns[common_idx]
        
        if len(strategy_returns) == 0:
            return {'t_statistic': 0.0, 'p_value': 1.0, 'mean_difference': 0.0}
        
        t_stat, p_value = stats.ttest_rel(strategy_returns, benchmark_returns)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_difference': float(strategy_returns.mean() - benchmark_returns.mean())
        }


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_interest_rates_from_csv(
    filepath: str = 'INTDSRTRM193N.csv',
    start_date: str = '2018-01-01',
    end_date: str = '2025-12-31'
) -> Optional[pd.Series]:
    """
    Load interest rate data from FRED CSV file.
    
    Parameters
    ----------
    filepath : str, default='INTDSRTRM193N.csv'
        Path to CSV file
    start_date : str, default='2018-01-01'
        Start date for data selection
    end_date : str, default='2025-12-31'
        End date for data selection
        
    Returns
    -------
    Optional[pd.Series]
        Interest rate series (as decimals) or None if loading fails
    """
    try:
        int_df = pd.read_csv(filepath, parse_dates=['observation_date'], index_col='observation_date')
        int_series = int_df.loc[start_date:end_date].squeeze() / 100
        print(f"✓ Loaded interest rates: {len(int_series)} months")
        return int_series
    except Exception as e:
        print(f"⚠ Interest rate data not loaded: {e}")
        return None


def load_cpi_from_csv(
    filepath: str = 'turkey_cpi_monthly.csv',
    start_date: str = '2018-01-01',
    end_date: str = '2025-12-31'
) -> Optional[pd.Series]:
    """
    Load monthly CPI data from CSV file.
    
    Parameters
    ----------
    filepath : str, default='turkey_cpi_monthly.csv'
        Path to CSV file with 'date' and 'cpi_yoy' columns
    start_date : str, default='2018-01-01'
        Start date for data selection
    end_date : str, default='2025-12-31'
        End date for data selection
        
    Returns
    -------
    Optional[pd.Series]
        CPI year-over-year series (as percentages) or None if loading fails
    """
    try:
        cpi_df = pd.read_csv(filepath, parse_dates=['date'])
        cpi_df = cpi_df.set_index('date')
        cpi_yoy = cpi_df.loc[start_date:end_date, 'cpi_yoy']
        
        if isinstance(cpi_yoy, pd.DataFrame):
            cpi_yoy = cpi_yoy.squeeze()
        
        print(f"✓ Loaded CPI data: {len(cpi_yoy)} months")
        return cpi_yoy
    except Exception as e:
        print(f"⚠ CPI data not loaded: {e}")
        return None


def calculate_inflation_index(cpi_yoy: pd.Series) -> pd.Series:
    """
    Calculate cumulative inflation index from YoY CPI data.
    
    Parameters
    ----------
    cpi_yoy : pd.Series
        Monthly year-over-year CPI changes (percent)
        
    Returns
    -------
    pd.Series
        Cumulative inflation index (starting at 1.0)
    """
    yoy_decimal = cpi_yoy / 100.0
    monthly_rates = (1 + yoy_decimal) ** (1 / 12) - 1
    inflation_index = (1 + monthly_rates).cumprod()
    inflation_index = inflation_index / inflation_index.iloc[0]
    return inflation_index


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def fetch_bist100_index(
    data_start: str = '2016-01-01',
    end_date: str = '2025-12-31',
    frequency: str = 'weekly'
) -> Optional[pd.Series]:
    """
    Fetch BIST100 index data (XU100.IS).
    
    Parameters
    ----------
    data_start : str, default='2016-01-01'
        Start date for data fetching
    end_date : str, default='2025-12-31'
        End date for data fetching
    frequency : str, default='weekly'
        Desired frequency ('weekly' or 'daily')
        
    Returns
    -------
    Optional[pd.Series]
        Normalized BIST100 index series or None if fetch fails
    """
    try:
        bist100 = yf.download('XU100.IS', start=data_start, end=end_date, progress=False)['Close']
        if frequency == 'weekly':
            bist100 = bist100.resample('W-FRI').last()
        if isinstance(bist100, pd.DataFrame):
            bist100 = bist100.iloc[:, 0]
        bist100_full = bist100 / bist100.loc[data_start:].iloc[0]
        print("✓ BIST100 Index data fetched")
        return bist100_full
    except Exception as e:
        print(f"⚠ BIST100 Index not available: {e}")
        return None


def fetch_tur_etf(
    data_start: str = '2016-01-01',
    end_date: str = '2025-12-31',
    frequency: str = 'weekly'
) -> Optional[pd.Series]:
    """
    Fetch iShares MSCI Turkey ETF (TUR) and normalize to analysis start.
    
    Parameters
    ----------
    data_start : str, default='2016-01-01'
        Start date for data fetching
    end_date : str, default='2025-12-31'
        End date for data fetching
    frequency : str, default='weekly'
        Desired frequency ('weekly' or 'daily')
        
    Returns
    -------
    Optional[pd.Series]
        Normalized TUR ETF series or None if fetch fails
    """
    try:
        tur = yf.download('TUR', start=data_start, end=end_date, progress=False)['Close']
        if frequency == 'weekly':
            tur = tur.resample('W-FRI').last()
        if isinstance(tur, pd.DataFrame):
            tur = tur.iloc[:, 0]
        
        analysis_start = '2018-01-01'
        tur_analysis = tur.loc[analysis_start:] / tur.loc[analysis_start:].iloc[0]
        
        print("✓ TUR ETF data fetched")
        return tur_analysis
    except Exception as e:
        print(f"⚠ TUR ETF not available: {e}")
        return None


def calculate_benchmark_robust(
    prices: pd.DataFrame,
    eligible_stocks: Dict[pd.Timestamp, List[str]]
) -> pd.Series:
    """
    Calculate equal-weighted benchmark portfolio.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data for all stocks
    eligible_stocks : Dict[pd.Timestamp, List[str]]
        Monthly eligible stock universes
        
    Returns
    -------
    pd.Series
        Equal-weighted benchmark portfolio values (normalized to 1.0 at start)
    """
    benchmark = pd.Series(index=prices.index, dtype=float)
    
    for i, date in enumerate(prices.index):
        month_start = pd.Timestamp(date.year, date.month, 1)
        eligible = eligible_stocks.get(month_start, list(prices.columns))
        
        available_prices = prices.loc[date, eligible].dropna()
        
        if len(available_prices) > 0:
            benchmark.loc[date] = float(available_prices.mean())
        else:
            benchmark.loc[date] = benchmark.iloc[i - 1] if i > 0 else 1.0
    
    benchmark = benchmark.fillna(method='ffill')
    benchmark = benchmark / benchmark.iloc[0]
    
    return benchmark


# ==============================================================================
# MACRO CORRELATION FUNCTIONS
# ==============================================================================

def calculate_macro_correlations_total_return(
    results: Dict[str, Dict[str, Any]],
    usdtry: pd.Series,
    macro_data: Dict[str, Any],
    frequency: str = 'weekly'
) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlations between strategy returns and macro variables.
    
    Aligns all data to month-end for proper comparison.
    
    Parameters
    ----------
    results : Dict
        Backtest results dictionary
    usdtry : pd.Series
        USD/TRY exchange rate series
    macro_data : Dict
        Dictionary with 'inflation_index' and 'interest_rate' keys
    frequency : str, default='weekly'
        Data frequency
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Correlations for each strategy: currency, inflation, interest rate
    """
    correlations = {}
    
    for strategy_name, strategy_data in results.items():
        portfolio = strategy_data['portfolio']
        
        # Calculate monthly returns
        portfolio_monthly = portfolio.resample('M').last()
        portfolio_monthly_returns = portfolio_monthly.pct_change().dropna()
        
        correlations[strategy_name] = {}
        
        # --- CURRENCY CORRELATION ---
        usdtry_monthly = usdtry.resample('M').last()
        usdtry_monthly_returns = usdtry_monthly.pct_change().dropna()
        
        common_dates = portfolio_monthly_returns.index.intersection(usdtry_monthly_returns.index)
        if len(common_dates) > 5:
            correlations[strategy_name]['currency_corr'] = portfolio_monthly_returns.loc[common_dates].corr(
                usdtry_monthly_returns.loc[common_dates]
            )
        else:
            correlations[strategy_name]['currency_corr'] = np.nan
        
        # --- INFLATION CORRELATION ---
        if macro_data.get('inflation_index') is not None:
            inflation_idx = macro_data['inflation_index'].copy()
            inflation_idx.index = inflation_idx.index + pd.offsets.MonthEnd(0)
            inflation_monthly_returns = inflation_idx.pct_change().dropna()
            
            common_dates = portfolio_monthly_returns.index.intersection(inflation_monthly_returns.index)
            if len(common_dates) > 5:
                correlations[strategy_name]['inflation_corr'] = portfolio_monthly_returns.loc[common_dates].corr(
                    inflation_monthly_returns.loc[common_dates]
                )
            else:
                correlations[strategy_name]['inflation_corr'] = np.nan
        else:
            correlations[strategy_name]['inflation_corr'] = np.nan
        
        # --- INTEREST RATE CORRELATION ---
        if macro_data.get('interest_rate') is not None:
            int_rate = macro_data['interest_rate'].copy()
            int_rate.index = int_rate.index + pd.offsets.MonthEnd(0)
            int_rate_changes = int_rate.diff().dropna()
            
            common_dates = portfolio_monthly_returns.index.intersection(int_rate_changes.index)
            if len(common_dates) > 5:
                correlations[strategy_name]['interest_corr'] = portfolio_monthly_returns.loc[common_dates].corr(
                    int_rate_changes.loc[common_dates]
                )
            else:
                correlations[strategy_name]['interest_corr'] = np.nan
        else:
            correlations[strategy_name]['interest_corr'] = np.nan
    
    return correlations


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def validate_data(prices: pd.DataFrame, eligible_stocks: Dict[pd.Timestamp, List[str]]) -> None:
    """Print data validation statistics."""
    print("\n" + "=" * 60)
    print("📊 DATA VALIDATION")
    print("=" * 60)
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Number of stocks: {len(prices.columns)}")
    
    date_diff = (prices.index[-1] - prices.index[0]).days / len(prices)
    freq = "WEEKLY" if date_diff > 3 else "DAILY"
    print(f"Frequency: {freq} ({len(prices)} periods)")
    
    nan_pct = prices.isna().sum().sum() / (prices.shape[0] * prices.shape[1]) * 100
    print(f"NaN percentage: {nan_pct:.2f}%")
    
    total_eligible = sum(len(stocks) for stocks in eligible_stocks.values())
    avg_eligible = total_eligible / len(eligible_stocks) if eligible_stocks else 0
    print(f"Average eligible stocks per month: {avg_eligible:.1f}")
    print("=" * 60)


def analyze_individual_stocks(
    prices_try: pd.DataFrame,
    sectors: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate individual stock statistics.
    
    Parameters
    ----------
    prices_try : pd.DataFrame
        TRY-denominated stock prices
    sectors : Dict[str, List[str]]
        Sector classification mapping
        
    Returns
    -------
    pd.DataFrame
        Stock statistics with sector, return, volatility, max drawdown
    """
    print("\n" + "=" * 70)
    print("📊 INDIVIDUAL STOCK STATISTICS")
    print("=" * 70)
    
    stock_stats = []
    
    for sector, stocks in sectors.items():
        for stock in stocks:
            if stock in prices_try.columns:
                stock_prices = prices_try[stock].dropna()
                if len(stock_prices) > 0:
                    stock_returns = stock_prices.pct_change().dropna()
                    
                    total_return = (stock_prices.iloc[-1] / stock_prices.iloc[0] - 1) * 100
                    ann_return = (1 + total_return / 100) ** (52 / len(stock_returns)) - 1 if len(stock_returns) > 0 else 0
                    volatility = stock_returns.std() * np.sqrt(52) * 100
                    max_dd = (stock_prices / stock_prices.expanding().max() - 1).min() * 100
                    
                    stock_stats.append({
                        'Sector': sector,
                        'Stock': stock,
                        'Total Return %': f"{total_return:.1f}%",
                        'Ann Return %': f"{ann_return * 100:.1f}%",
                        'Volatility %': f"{volatility:.1f}%",
                        'Max DD %': f"{max_dd:.1f}%"
                    })
    
    stock_df = pd.DataFrame(stock_stats)
    print(stock_df.to_string(index=False))
    
    # Sector averages
    print("\n" + "=" * 70)
    print("📊 SECTOR AVERAGES")
    print("=" * 70)
    
    def extract_number(x):
        return float(x.rstrip('%'))
    
    sector_avg = stock_df.groupby('Sector').agg({
        'Ann Return %': lambda x: f"{np.mean([extract_number(v) for v in x]):.1f}%",
        'Volatility %': lambda x: f"{np.mean([extract_number(v) for v in x]):.1f}%",
        'Max DD %': lambda x: f"{np.mean([extract_number(v) for v in x]):.1f}%"
    }).reset_index()
    
    print(sector_avg.to_string(index=False))
    
    return stock_df


def analyze_usdtry_by_subperiod(
    usdtry: pd.Series,
    subperiods: List[Tuple[str, str, str]]
) -> pd.DataFrame:
    """
    Analyze USD/TRY performance by subperiod.
    
    Parameters
    ----------
    usdtry : pd.Series
        USD/TRY exchange rate series
    subperiods : List[Tuple[str, str, str]]
        List of (period_name, start_date, end_date)
        
    Returns
    -------
    pd.DataFrame
        Subperiod performance statistics
    """
    print("\n" + "=" * 70)
    print("📊 USD/TRY PERFORMANCE BY SUBPERIOD")
    print("=" * 70)
    
    usdtry_returns = usdtry.pct_change().dropna()
    results = []
    
    for period_name, start_date, end_date in subperiods:
        period_usdtry = usdtry.loc[start_date:end_date]
        period_returns = usdtry_returns.loc[start_date:end_date]
        
        if len(period_usdtry) > 0:
            start_rate = period_usdtry.iloc[0]
            end_rate = period_usdtry.iloc[-1]
            total_change = (end_rate / start_rate - 1) * 100
            ann_change = (1 + total_change / 100) ** (12 / len(period_returns) * 52) - 1 if len(period_returns) > 0 else 0
            volatility = period_returns.std() * np.sqrt(52) * 100
            
            results.append({
                'Period': period_name,
                'Start TRY/USD': f"{start_rate:.2f}",
                'End TRY/USD': f"{end_rate:.2f}",
                'Total Change %': f"{total_change:.1f}%",
                'Annualized Change %': f"{ann_change * 100:.1f}%",
                'Volatility %': f"{volatility:.1f}%"
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df


def analyze_sector_performance(
    prices_try: pd.DataFrame,
    eligible_stocks: Dict[pd.Timestamp, List[str]],
    sectors: Dict[str, List[str]],
    frequency: str = 'weekly'
) -> Dict[str, Dict[str, float]]:
    """
    Analyze DMA strategy performance by sector.
    
    Parameters
    ----------
    prices_try : pd.DataFrame
        TRY-denominated stock prices
    eligible_stocks : Dict[pd.Timestamp, List[str]]
        Monthly eligible stock universes
    sectors : Dict[str, List[str]]
        Sector classification mapping
    frequency : str, default='weekly'
        Data frequency
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Performance metrics for each sector
    """
    print("\n" + "=" * 70)
    print("🏭 SECTOR PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    sector_results = {}
    
    for sector, stocks in sectors.items():
        available_stocks = [s for s in stocks if s in prices_try.columns]
        
        if available_stocks:
            print(f"\nAnalyzing {sector} sector ({len(available_stocks)} stocks)...")
            sector_prices = prices_try[available_stocks]
            
            sector_eligible = {}
            for date, eligible in eligible_stocks.items():
                sector_eligible[date] = [s for s in eligible if s in available_stocks]
            
            backtester = PortfolioBacktester(sector_prices, pd.Series(dtype=float), sector_eligible, slippage=0.001)
            
            portfolio, trade_stats = backtester.run_strategy_try(
                lambda p: MomentumStrategies.dual_moving_average_weekly(p, 21, 50),
                f"{sector}_DMA",
                debug=False
            )
            
            sector_benchmark = calculate_benchmark_robust(sector_prices, sector_eligible)
            analyzer = PerformanceAnalyzer(portfolio, sector_benchmark)
            sector_results[sector] = analyzer.calculate_metrics(rf_rate=TRY_RISK_FREE_RATE)
    
    return sector_results


def run_subperiod_analysis_enhanced(
    results: Dict[str, Dict[str, Any]],
    subperiods: List[Tuple[str, str, str]],
    benchmark_try: pd.Series,
    benchmark_usd: pd.Series
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Enhanced subperiod analysis with full metrics.
    
    Parameters
    ----------
    results : Dict
        Backtest results dictionary
    subperiods : List[Tuple[str, str, str]]
        List of (period_name, start_date, end_date)
    benchmark_try : pd.Series
        TRY benchmark series
    benchmark_usd : pd.Series
        USD benchmark series
        
    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]
        Subperiod results DataFrames and detailed metrics
    """
    print("\n" + "=" * 70)
    print("📊 DETAILED SUBPERIOD PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    subperiod_results = {}
    detailed_metrics = {}
    
    for period_name, start_date, end_date in subperiods:
        print(f"\n{'=' * 60}")
        print(f"🔵 {period_name} ({start_date} to {end_date})")
        print(f"{'=' * 60}")
        
        period_data = []
        period_detail = {}
        
        for strategy_name, strategy_data in results.items():
            try:
                period_portfolio = strategy_data['portfolio'].loc[start_date:end_date]
                if len(period_portfolio) > 0:
                    if 'TRY' in strategy_name:
                        benchmark = benchmark_try
                        rf_rate = TRY_RISK_FREE_RATE
                    else:
                        benchmark = benchmark_usd
                        rf_rate = USD_RISK_FREE_RATE
                    
                    period_benchmark = benchmark.loc[start_date:end_date]
                    analyzer = PerformanceAnalyzer(period_portfolio, period_benchmark)
                    metrics = analyzer.calculate_metrics(rf_rate=rf_rate)
                    
                    period_detail[strategy_name] = metrics
                    
                    period_data.append({
                        'Strategy': strategy_name,
                        'Ann Ret %': f"{metrics['Annualized Return'] * 100:.1f}%",
                        'Sharpe': f"{metrics['Sharpe Ratio']:.2f}",
                        'Sortino': f"{metrics['Sortino Ratio']:.2f}",
                        'Volatility %': f"{metrics['Annualized Volatility'] * 100:.1f}%",
                        'Max DD %': f"{metrics['Maximum Drawdown'] * 100:.1f}%"
                    })
                    
                    print(f"  {strategy_name}:")
                    print(f"     Ann Ret: {metrics['Annualized Return'] * 100:.1f}%")
                    print(f"     Sharpe: {metrics['Sharpe Ratio']:.2f}")
                
            except Exception as e:
                print(f"  ❌ {strategy_name} failed: {e}")
        
        subperiod_results[period_name] = pd.DataFrame(period_data)
        detailed_metrics[period_name] = period_detail
    
    # Print summary tables
    print("\n" + "=" * 100)
    print("📈 DETAILED SUBPERIOD PERFORMANCE SUMMARY BY METRIC")
    print("=" * 100)
    
    for metric_name in ['Sharpe', 'Ann Ret %', 'Sortino', 'Volatility %', 'Max DD %']:
        print(f"\n{metric_name} by Period:")
        pivot_data = []
        for strategy_name in results.keys():
            row = {'Strategy': strategy_name}
            for period_name in [p[0] for p in subperiods]:
                df = subperiod_results.get(period_name)
                if df is not None and not df.empty:
                    strategy_row = df[df['Strategy'] == strategy_name]
                    if not strategy_row.empty:
                        row[period_name] = strategy_row[metric_name].values[0]
                    else:
                        row[period_name] = 'N/A'
                else:
                    row[period_name] = 'N/A'
            pivot_data.append(row)
        
        pivot_df = pd.DataFrame(pivot_data)
        print(pivot_df.to_string(index=False))
        print()
    
    return subperiod_results, detailed_metrics


def summarize_statistical_significance(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create summary table comparing bootstrap and t-test results.
    
    Parameters
    ----------
    results : Dict
        Backtest results dictionary
        
    Returns
    -------
    pd.DataFrame
        Statistical significance comparison table
    """
    print("\n" + "=" * 80)
    print("📊 STATISTICAL SIGNIFICANCE COMPARISON")
    print("=" * 80)
    
    sig_data = []
    for strategy_name, strategy_data in results.items():
        if 'bootstrap' in strategy_data and 't_test' in strategy_data:
            bootstrap_p = strategy_data['bootstrap']['p_value']
            t_test_p = strategy_data['t_test']['p_value']
            t_stat = strategy_data['t_test']['t_statistic']
            
            sig_data.append({
                'Strategy': strategy_name,
                'Bootstrap p': f"{bootstrap_p:.4f}",
                'Bootstrap Sig': 'Yes' if bootstrap_p < 0.05 else 'No',
                'T-test p': f"{t_test_p:.4f}",
                'T-test Sig': 'Yes' if t_test_p < 0.05 else 'No',
                'T-statistic': f"{t_stat:.2f}",
                'Agreement': '✓' if (bootstrap_p < 0.05) == (t_test_p < 0.05) else '✗'
            })
    
    sig_df = pd.DataFrame(sig_data)
    print(sig_df.to_string(index=False))
    
    agreement = sum(1 for d in sig_data if d['Agreement'] == '✓') / len(sig_data) if sig_data else 0
    print(f"\n✅ Bootstrap and t-test agree on {agreement * 100:.1f}% of strategies")
    
    return sig_df


def interpret_rolling_sharpe(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print interpretation of rolling Sharpe ratios.
    
    Parameters
    ----------
    results : Dict
        Backtest results dictionary
    """
    print("\n" + "=" * 70)
    print("📈 ROLLING SHARPE RATIO INTERPRETATION")
    print("=" * 70)
    
    for strategy_name, strategy_data in results.items():
        if 'TRY' in strategy_name and 'rolling_sharpe' in strategy_data:
            rolling_sharpe = strategy_data['rolling_sharpe']
            if rolling_sharpe is not None and len(rolling_sharpe) > 0:
                print(f"\n🔵 {strategy_name}:")
                print(f"  • First available: {rolling_sharpe.iloc[0]:.2f}")
                print(f"  • Peak: {rolling_sharpe.max():.2f}")
                print(f"  • Trough: {rolling_sharpe.min():.2f}")
                print(f"  • Latest: {rolling_sharpe.iloc[-1]:.2f}")
                print(f"  • Mean: {rolling_sharpe.mean():.2f}")
                print(f"  • Stability (Std Dev): {rolling_sharpe.std():.2f}")


# ==============================================================================
# SENSITIVITY ANALYSIS FUNCTIONS
# ==============================================================================

def run_transaction_cost_sensitivity(
    prices_try: pd.DataFrame,
    usdtry: pd.Series,
    eligible_stocks: Dict[pd.Timestamp, List[str]],
    strategies: Dict[str, Dict[str, Any]],
    benchmark_try: pd.Series,
    benchmark_usd: pd.Series
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run sensitivity analysis for different transaction cost levels.
    
    Parameters
    ----------
    prices_try : pd.DataFrame
        TRY-denominated stock prices
    usdtry : pd.Series
        USD/TRY exchange rates
    eligible_stocks : Dict
        Monthly eligible stock universes
    strategies : Dict
        Strategy dictionary from MomentumStrategies.get_strategies()
    benchmark_try : pd.Series
        TRY benchmark series
    benchmark_usd : pd.Series
        USD benchmark series
        
    Returns
    -------
    Dict[str, Dict[str, Dict[str, float]]]
        Sensitivity results for each cost level
    """
    cost_levels = [SLIPPAGE_LOW, SLIPPAGE_BASELINE, SLIPPAGE_HIGH]
    cost_names = ['Low (0.05%)', 'Baseline (0.10%)', 'High (0.20%)']
    
    sensitivity_results = {}
    
    print("\n" + "=" * 70)
    print("📊 TRANSACTION COST SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    for cost_idx, (slippage, cost_name) in enumerate(zip(cost_levels, cost_names)):
        print(f"\n{'=' * 60}")
        print(f"🔵 Running with {cost_name} slippage")
        print(f"{'=' * 60}")
        
        backtester = PortfolioBacktester(prices_try, usdtry, eligible_stocks, slippage=slippage)
        cost_results = {}
        
        # Run TRY strategies
        for strategy_key, strategy_info in strategies.items():
            strategy_name = f"TRY_{strategy_key}"
            try:
                portfolio_try, trade_stats = backtester.run_strategy_try(
                    strategy_info['func'],
                    strategy_name,
                    debug=False
                )
                
                analyzer = PerformanceAnalyzer(portfolio_try, benchmark_try)
                metrics = analyzer.calculate_metrics(rf_rate=TRY_RISK_FREE_RATE)
                
                cost_results[strategy_name] = {
                    'ann_return': metrics['Annualized Return'] * 100,
                    'sharpe': metrics['Sharpe Ratio'],
                    'sortino': metrics['Sortino Ratio']
                }
                
            except Exception as e:
                print(f"  ❌ {strategy_name} failed: {e}")
        
        # Run USD strategies
        for strategy_key, strategy_info in strategies.items():
            strategy_name = f"USD_{strategy_key}"
            try:
                portfolio_usd, trade_stats = backtester.run_strategy_usd(
                    strategy_info['func'],
                    strategy_name,
                    debug=False
                )
                
                analyzer = PerformanceAnalyzer(portfolio_usd, benchmark_usd)
                metrics = analyzer.calculate_metrics(rf_rate=USD_RISK_FREE_RATE)
                
                cost_results[strategy_name] = {
                    'ann_return': metrics['Annualized Return'] * 100,
                    'sharpe': metrics['Sharpe Ratio'],
                    'sortino': metrics['Sortino Ratio']
                }
                
            except Exception as e:
                print(f"  ❌ {strategy_name} failed: {e}")
        
        sensitivity_results[cost_name] = cost_results
    
    return sensitivity_results


def run_parameter_sensitivity(
    prices_try: pd.DataFrame,
    usdtry: pd.Series,
    eligible_stocks: Dict[pd.Timestamp, List[str]],
    benchmark_try: pd.Series,
    benchmark_usd: pd.Series,
    base_slippage: float = SLIPPAGE_BASELINE
) -> Dict[str, pd.DataFrame]:
    """
    Run sensitivity analysis for different parameter choices.
    
    Parameters
    ----------
    prices_try : pd.DataFrame
        TRY-denominated stock prices
    usdtry : pd.Series
        USD/TRY exchange rates
    eligible_stocks : Dict
        Monthly eligible stock universes
    benchmark_try : pd.Series
        TRY benchmark series
    benchmark_usd : pd.Series
        USD benchmark series
    base_slippage : float, default=0.001
        Slippage assumption for backtesting
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Parameter sensitivity results for each strategy family
    """
    param_variants = MomentumStrategies.get_parameter_variants()
    sensitivity_results = {}
    
    print("\n" + "=" * 70)
    print("📊 PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    backtester = PortfolioBacktester(prices_try, usdtry, eligible_stocks, slippage=base_slippage)
    
    for strategy_type, variants in param_variants.items():
        print(f"\n{'=' * 60}")
        print(f"🔵 Testing {strategy_type} Strategy Variants")
        print(f"{'=' * 60}")
        
        strategy_results = []
        
        for variant in variants:
            try:
                portfolio_try, trade_stats = backtester.run_strategy_try(
                    variant['func'],
                    variant['name'],
                    debug=False
                )
                
                analyzer = PerformanceAnalyzer(portfolio_try, benchmark_try)
                metrics = analyzer.calculate_metrics(rf_rate=TRY_RISK_FREE_RATE)
                
                strategy_results.append({
                    'Variant': variant['desc'],
                    'Ann Ret %': f"{metrics['Annualized Return'] * 100:.1f}%",
                    'Sharpe': f"{metrics['Sharpe Ratio']:.2f}",
                    'Sortino': f"{metrics['Sortino Ratio']:.2f}",
                    'Max DD %': f"{metrics['Maximum Drawdown'] * 100:.1f}%",
                    'Win Rate': f"{trade_stats['win_rate'] * 100:.1f}%"
                })
                
                print(f"  {variant['desc']}: Ann Ret={metrics['Annualized Return'] * 100:.1f}%, "
                      f"Sharpe={metrics['Sharpe Ratio']:.2f}")
                
            except Exception as e:
                print(f"  ❌ {variant['desc']} failed: {e}")
        
        sensitivity_results[strategy_type] = pd.DataFrame(strategy_results)
        print(f"\n{strategy_type} Parameter Sensitivity Results:")
        print(sensitivity_results[strategy_type].to_string(index=False))
    
    return sensitivity_results


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_parameter_sensitivity(param_results: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Create visualization of parameter sensitivity.
    
    Parameters
    ----------
    param_results : Dict[str, pd.DataFrame]
        Parameter sensitivity results from run_parameter_sensitivity()
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    
    strategy_types = ['DMA', 'Donchian', 'RSI', 'BB']
    axes_flat = axes.flatten()
    
    for idx, strategy_type in enumerate(strategy_types):
        ax = axes_flat[idx]
        df = param_results.get(strategy_type)
        
        if df is not None and not df.empty:
            variants = df['Variant'].tolist()
            sharpes = [float(s) for s in df['Sharpe'].tolist()]
            
            colors = [
                'lightblue' if 'Baseline' in v else 'lightgreen' if 'Fast' in v else 'gold'
                for v in variants
            ]
            bars = ax.bar(range(len(variants)), sharpes, color=colors, edgecolor='navy')
            
            ax.set_title(f'{strategy_type} Strategy Variants')
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels([v.split(' - ')[0] for v in variants], rotation=45, ha='right')
            ax.set_ylabel('Sharpe Ratio')
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, sharpes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_comprehensive_results(
    results: Dict[str, Dict[str, Any]],
    prices_try: pd.DataFrame,
    prices_usd: pd.DataFrame,
    benchmark_try: pd.Series,
    benchmark_usd: pd.Series,
    spy_benchmark: Optional[pd.Series] = None,
    bist100_try: Optional[pd.Series] = None,
    bist100_usd: Optional[pd.Series] = None,
    tur_etf: Optional[pd.Series] = None,
    frequency: str = 'weekly'
) -> plt.Figure:
    """
    Generate comprehensive visualization with six subplots.
    
    Parameters
    ----------
    results : Dict
        Backtest results dictionary
    prices_try, prices_usd : pd.DataFrame
        Price data (not used directly, kept for API compatibility)
    benchmark_try, benchmark_usd : pd.Series
        Benchmark series
    spy_benchmark, bist100_try, bist100_usd, tur_etf : pd.Series, optional
        Additional benchmarks for comparison
    frequency : str, default='weekly'
        Data frequency for title
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f'BIST-100 Momentum Strategies Performance Analysis (2018-2025) - {frequency.upper()} Frequency',
        fontsize=16, fontweight='bold'
    )
    
    start_date = '2018-01-01'
    end_date = '2025-12-31'
    
    # 1. TRY Equity Curves
    ax = axes[0, 0]
    try_keys = ['TRY_DMA_21_50', 'TRY_Donchian_20', 'TRY_Combo_Filtered', 'TRY_RSI_14', 'TRY_BB_20_2']
    
    for key in try_keys:
        if key in results:
            portfolio = results[key]['portfolio'].loc[start_date:end_date]
            label = key.replace('TRY_', '').replace('_', ' ')
            ax.plot(portfolio.index, portfolio.values, linewidth=2, label=label)
    
    ax.plot(benchmark_try.loc[start_date:end_date].index,
            benchmark_try.loc[start_date:end_date].values,
            'k--', linewidth=2, alpha=0.7, label='EW Benchmark (TRY)')
    
    if bist100_try is not None:
        ax.plot(bist100_try.loc[start_date:end_date].index,
                bist100_try.loc[start_date:end_date].values,
                'b-.', linewidth=2, alpha=0.7, label='BIST100 Index')
    
    ax.set_title('TRY Denominated Returns', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (TRY)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # 2. USD Equity Curves
    ax = axes[0, 1]
    usd_keys = ['USD_DMA_21_50', 'USD_Donchian_20', 'USD_Combo_Filtered', 'USD_RSI_14', 'USD_BB_20_2']
    
    for key in usd_keys:
        if key in results:
            portfolio = results[key]['portfolio'].loc[start_date:end_date]
            label = key.replace('USD_', '').replace('_', ' ')
            ax.plot(portfolio.index, portfolio.values, linewidth=2, label=label)
    
    ax.plot(benchmark_usd.loc[start_date:end_date].index,
            benchmark_usd.loc[start_date:end_date].values,
            'k--', linewidth=2, alpha=0.7, label='EW Benchmark')
    
    if bist100_usd is not None:
        ax.plot(bist100_usd.loc[start_date:end_date].index,
                bist100_usd.loc[start_date:end_date].values,
                'b-.', linewidth=2, alpha=0.7, label='BIST100')
    
    if spy_benchmark is not None:
        ax.plot(spy_benchmark.loc[start_date:end_date].index,
                spy_benchmark.loc[start_date:end_date].values,
                'r-', linewidth=2, alpha=0.7, label='S&P 500')
    
    if tur_etf is not None:
        ax.plot(tur_etf.loc[start_date:end_date].index,
                tur_etf.loc[start_date:end_date].values,
                'g--', linewidth=2, alpha=0.7, label='TUR ETF')
    
    ax.set_title('USD Denominated Returns', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # 3. Drawdown Analysis
    ax = axes[0, 2]
    
    for key in try_keys:
        if key in results:
            portfolio = results[key]['portfolio'].loc[start_date:end_date]
            returns = portfolio.pct_change().dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            label = key.replace('TRY_', '').replace('_', ' ')
            ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.2, label=label)
            ax.plot(drawdown.index, drawdown.values, linewidth=1.5, alpha=0.8)
    
    ax.set_title('Drawdown Analysis (TRY)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # 4. Rolling 3-Year Sharpe
    ax = axes[1, 0]
    
    for key in try_keys:
        if key in results and 'rolling_sharpe' in results[key]:
            rolling_sharpe = results[key]['rolling_sharpe']
            if rolling_sharpe is not None and len(rolling_sharpe) > 0:
                label = key.replace('TRY_', '').replace('_', ' ')
                ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, label=label)
    
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='g', linestyle='--', alpha=0.3, label='Good (1.0)')
    ax.set_title('Rolling 3-Year Sharpe Ratio (TRY)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Performance Metrics Comparison
    ax = axes[1, 1]
    
    strategies_list = []
    ann_returns = []
    sharpes = []
    sortinos = []
    max_dds = []
    
    for strategy_name, strategy_data in results.items():
        if 'metrics' in strategy_data and 'TRY' in strategy_name:
            metrics = strategy_data['metrics']
            short_name = strategy_name.replace('TRY_', '').replace('_', '\n')
            strategies_list.append(short_name)
            ann_returns.append(metrics['Annualized Return'])
            sharpes.append(metrics['Sharpe Ratio'])
            sortinos.append(metrics['Sortino Ratio'])
            max_dds.append(metrics['Maximum Drawdown'])
    
    if strategies_list:
        x = np.arange(len(strategies_list))
        width = 0.2
        
        ax.bar(x - 1.5 * width, ann_returns, width, label='Ann Return', color='skyblue')
        ax.bar(x - 0.5 * width, sharpes, width, label='Sharpe', color='lightgreen')
        ax.bar(x + 0.5 * width, sortinos, width, label='Sortino', color='gold')
        ax.bar(x + 1.5 * width, max_dds, width, label='Max DD', color='salmon')
        
        ax.set_title('Performance Metrics by Strategy', fontweight='bold', fontsize=12)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies_list, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. Bootstrap P-values
    ax = axes[1, 2]
    
    p_strategies = []
    p_values = []
    colors = []
    
    for strategy_name, strategy_data in results.items():
        if 'bootstrap' in strategy_data:
            short_name = strategy_name.replace('TRY_', '').replace('USD_', '').replace('_', '\n')
            p_strategies.append(short_name)
            p_val = strategy_data['bootstrap']['p_value']
            p_values.append(p_val)
            colors.append('green' if p_val < 0.05 else 'red')
    
    if p_strategies:
        ax.bar(p_strategies, p_values, color=colors)
        ax.axhline(y=0.05, color='blue', linestyle='--', linewidth=2, label='p=0.05')
        ax.set_title('Statistical Significance', fontweight='bold', fontsize=12)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('P-value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(p_strategies, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_strategy_correlation_matrix(
    results: Dict[str, Dict[str, Any]],
    benchmark_try: pd.Series,
    benchmark_usd: pd.Series,
    bist100_try: Optional[pd.Series],
    bist100_usd: Optional[pd.Series],
    spy_benchmark: Optional[pd.Series],
    tur_etf: Optional[pd.Series]
) -> plt.Figure:
    """
    Plot correlation matrix of all strategies and benchmarks.
    
    Parameters
    ----------
    results : Dict
        Backtest results dictionary
    benchmark_try, benchmark_usd : pd.Series
        Benchmark series
    bist100_try, bist100_usd, spy_benchmark, tur_etf : pd.Series, optional
        Additional benchmarks
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    returns_dict = {}
    start_date = '2018-01-01'
    end_date = '2025-12-31'
    
    for strategy_name, strategy_data in results.items():
        returns = strategy_data['portfolio'].loc[start_date:end_date].pct_change().dropna()
        returns_dict[strategy_name] = returns
    
    returns_dict['EW BIST-100 (TRY)'] = benchmark_try.loc[start_date:end_date].pct_change().dropna()
    returns_dict['EW BIST-100 (USD)'] = benchmark_usd.loc[start_date:end_date].pct_change().dropna()
    
    if bist100_try is not None:
        returns_dict['BIST100 (TRY)'] = bist100_try.loc[start_date:end_date].pct_change().dropna()
    if bist100_usd is not None:
        returns_dict['BIST100 (USD)'] = bist100_usd.loc[start_date:end_date].pct_change().dropna()
    if spy_benchmark is not None:
        returns_dict['S&P 500'] = spy_benchmark.loc[start_date:end_date].pct_change().dropna()
    if tur_etf is not None:
        returns_dict['TUR ETF'] = tur_etf.loc[start_date:end_date].pct_change().dropna()
    
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()
    
    corr_matrix = returns_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=True, fmt='.2f', annot_kws={'size': 8}, ax=ax)
    
    ax.set_title('Strategy & Benchmark Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_sector_analysis(sector_results: Dict[str, Dict[str, float]]) -> plt.Figure:
    """
    Plot sector performance comparison with capped drawup values.
    
    Parameters
    ----------
    sector_results : Dict[str, Dict[str, float]]
        Sector performance metrics from analyze_sector_performance()
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if not sector_results:
        print("No sector results to plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Sector Performance Analysis (Weekly DMA Strategy)', fontsize=14, fontweight='bold')
    
    sectors = list(sector_results.keys())
    returns = [float(sector_results[s]['Annualized Return']) for s in sectors]
    sharpes = [float(sector_results[s]['Sharpe Ratio']) for s in sectors]
    sortinos = [float(sector_results[s]['Sortino Ratio']) for s in sectors]
    max_drawups = [min(float(sector_results[s]['Maximum Drawup']), 20.0) for s in sectors]
    max_dds = [float(sector_results[s]['Maximum Drawdown']) for s in sectors]
    
    short_sectors = [s.replace(' ', '\n') if len(s) > 10 else s for s in sectors]
    
    # Returns and Drawup by sector
    ax1 = axes[0]
    x = np.arange(len(sectors))
    width = 0.35
    
    bars1 = ax1.bar(x - width / 2, returns, width, label='Return', color='skyblue', edgecolor='navy')
    bars2 = ax1.bar(x + width / 2, max_drawups, width, label='Max Drawup', color='gold', edgecolor='navy', alpha=0.7)
    
    ax1.set_title('Returns & Maximum Drawup by Sector', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Sector')
    ax1.set_ylabel('Return / Drawup')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_sectors, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax1.set_ylim(-1, 25)
    
    for bar, val in zip(bars1, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val * 100:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    # Sharpe, Sortino and Drawdown by sector
    ax2 = axes[1]
    width2 = 0.25
    bars3 = ax2.bar(x - width2, sharpes, width2, label='Sharpe', color='lightgreen', edgecolor='navy')
    bars4 = ax2.bar(x, sortinos, width2, label='Sortino', color='gold', edgecolor='navy')
    bars5 = ax2.bar(x + width2, max_dds, width2, label='Max DD', color='salmon', edgecolor='navy')
    
    ax2.set_title('Sharpe, Sortino & Max Drawdown by Sector', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Sector')
    ax2.set_ylabel('Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_sectors, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax2.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def main() -> Optional[Dict[str, Any]]:
    """
    Main execution function for the backtesting framework.
    
    This function orchestrates the entire backtesting process:
    1. Data fetching and preparation
    2. Strategy backtesting (TRY and USD perspectives)
    3. Performance analysis and robustness checks
    4. Visualization generation
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Results dictionary containing all backtest data
    """
    print("=" * 60)
    print("BIST-100 MOMENTUM & MEAN REVERSION STRATEGIES BACKTESTING")
    print("=" * 60)
    print("Analysis Period: 2018-2025")
    print("Warm-up Period: 2016 (for signal calculation)")
    print("Frequency: WEEKLY")
    print("=" * 60)
    
    np.random.seed(42)
    
    # ==========================================================================
    # DATA FETCHING
    # ==========================================================================
    
    print("\n📥 FETCHING DATA")
    print("-" * 60)
    
    bist = BISTMomentumStrategy(
        data_start='2016-01-01',
        analysis_start='2018-01-01',
        end_date='2025-12-31',
        frequency='weekly'
    )
    
    try:
        prices_try, prices_usd, usdtry = bist.fetch_bist_data()
        eligible_stocks = bist.create_monthly_universe()
        print(f"✓ Data fetched successfully")
        print(f"  Stocks in analysis: {len(prices_try.columns)}")
        print(f"  Trading periods: {len(prices_try)}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    validate_data(prices_try, eligible_stocks)
    
    # ==========================================================================
    # BENCHMARK CALCULATION
    # ==========================================================================
    
    print("\n📊 CALCULATING BENCHMARKS")
    print("-" * 60)
    
    benchmark_try = calculate_benchmark_robust(prices_try, eligible_stocks)
    benchmark_usd = benchmark_try / usdtry
    benchmark_usd = benchmark_usd / benchmark_usd.iloc[0]
    
    # Fetch additional benchmarks
    print("\n📊 FETCHING ADDITIONAL BENCHMARKS")
    print("-" * 60)
    
    spy_benchmark = None
    try:
        spy = yf.download('SPY', start='2018-01-01', end='2025-12-31', progress=False)['Close']
        spy = spy.resample('W-FRI').last()
        if isinstance(spy, pd.DataFrame):
            spy = spy.iloc[:, 0]
        spy_benchmark = spy / spy.iloc[0]
        print("✓ S&P 500 data fetched")
    except Exception as e:
        print(f"⚠ S&P 500 not available: {e}")
    
    bist100_try = fetch_bist100_index('2016-01-01', '2025-12-31', 'weekly')
    
    bist100_usd = None
    if bist100_try is not None and usdtry is not None:
        common_dates = bist100_try.index.intersection(usdtry.index)
        if len(common_dates) > 0:
            bist100_try_aligned = bist100_try.loc[common_dates]
            usdtry_aligned = usdtry.loc[common_dates]
            bist100_usd = bist100_try_aligned / usdtry_aligned
            if isinstance(bist100_usd, pd.DataFrame):
                bist100_usd = bist100_usd.iloc[:, 0]
            bist100_usd = bist100_usd / bist100_usd.loc['2018-01-01':].iloc[0]
            print("✓ BIST100 USD calculated")
    
    tur_etf = fetch_tur_etf('2016-01-01', '2025-12-31', 'weekly')
    
    # ==========================================================================
    # MACRO DATA LOADING
    # ==========================================================================
    
    print("\n📊 LOADING MACRO DATA")
    print("-" * 60)
    
    macro_data = {}
    macro_data['interest_rate'] = load_interest_rates_from_csv(
        filepath='INTDSRTRM193N.csv',
        start_date='2018-01-01',
        end_date='2025-12-31'
    )
    
    cpi_yoy = load_cpi_from_csv(
        filepath='turkey_cpi_monthly.csv',
        start_date='2018-01-01',
        end_date='2025-12-31'
    )
    
    if cpi_yoy is not None:
        macro_data['inflation_index'] = calculate_inflation_index(cpi_yoy)
        print(f"✓ Inflation index calculated: {len(macro_data['inflation_index'])} months")
        print(f"  Total inflation multiple: {macro_data['inflation_index'].iloc[-1]:.2f}x")
    else:
        macro_data['inflation_index'] = None
    
    # ==========================================================================
    # STRATEGY BACKTESTING
    # ==========================================================================
    
    backtester = PortfolioBacktester(prices_try, usdtry, eligible_stocks, slippage=SLIPPAGE_BASELINE)
    strategies = MomentumStrategies.get_strategies('weekly')
    
    results = {}
    strategy_returns_dict = {}
    
    # TRY strategies
    print(f"\n{'=' * 50}")
    print("🔵 TRY STRATEGIES (Local Investor)")
    print(f"{'=' * 50}")
    
    for strategy_key, strategy_info in strategies.items():
        strategy_name = f"TRY_{strategy_key}"
        print(f"\n▶ Running {strategy_info['description']}...")
        
        try:
            portfolio_try, trade_stats = backtester.run_strategy_try(
                strategy_info['func'],
                strategy_name,
                debug=False
            )
            
            analyzer = PerformanceAnalyzer(portfolio_try, benchmark_try)
            metrics = analyzer.calculate_metrics(rf_rate=TRY_RISK_FREE_RATE)
            bootstrap = analyzer.bootstrap_test_fixed(n_iterations=10000)
            t_test = analyzer.t_test_vs_benchmark()
            rolling_sharpe = analyzer.calculate_rolling_sharpe(window_years=3, rf_rate=TRY_RISK_FREE_RATE)
            
            results[strategy_name] = {
                'portfolio': portfolio_try,
                'metrics': metrics,
                'bootstrap': bootstrap,
                't_test': t_test,
                'rolling_sharpe': rolling_sharpe,
                'trade_stats': trade_stats
            }
            
            strategy_returns_dict[strategy_name] = portfolio_try.pct_change().dropna()
            
            print(f"  ✓ Complete")
            print(f"    Sharpe: {metrics['Sharpe Ratio']:.2f}")
            print(f"    Bootstrap p: {bootstrap['p_value']:.4f}")
            print(f"    T-test p: {t_test['p_value']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # USD strategies
    print(f"\n{'=' * 50}")
    print("🔵 USD STRATEGIES (International Investor)")
    print(f"{'=' * 50}")
    
    for strategy_key, strategy_info in strategies.items():
        strategy_name = f"USD_{strategy_key}"
        print(f"\n▶ Running {strategy_info['description']}...")
        
        try:
            portfolio_usd, trade_stats = backtester.run_strategy_usd(
                strategy_info['func'],
                strategy_name,
                debug=False
            )
            
            analyzer = PerformanceAnalyzer(portfolio_usd, benchmark_usd)
            metrics = analyzer.calculate_metrics(rf_rate=USD_RISK_FREE_RATE)
            bootstrap = analyzer.bootstrap_test_fixed(n_iterations=10000)
            t_test = analyzer.t_test_vs_benchmark()
            rolling_sharpe = analyzer.calculate_rolling_sharpe(window_years=3, rf_rate=USD_RISK_FREE_RATE)
            
            results[strategy_name] = {
                'portfolio': portfolio_usd,
                'metrics': metrics,
                'bootstrap': bootstrap,
                't_test': t_test,
                'rolling_sharpe': rolling_sharpe,
                'trade_stats': trade_stats
            }
            
            strategy_returns_dict[strategy_name] = portfolio_usd.pct_change().dropna()
            
            print(f"  ✓ Complete")
            print(f"    Sharpe: {metrics['Sharpe Ratio']:.2f}")
            print(f"    Bootstrap p: {bootstrap['p_value']:.4f}")
            print(f"    T-test p: {t_test['p_value']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # ==========================================================================
    # MACRO CORRELATIONS
    # ==========================================================================
    
    print("\n📊 CALCULATING MACRO CORRELATIONS")
    print("-" * 60)
    macro_correlations = calculate_macro_correlations_total_return(results, usdtry, macro_data, 'weekly')
    
    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================
    
    print("\n" + "=" * 180)
    print("📈 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 180)
    
    # Calculate benchmark metrics
    benchmark_analyzer_try = PerformanceAnalyzer(benchmark_try, benchmark_try)
    benchmark_metrics_try = benchmark_analyzer_try.calculate_metrics(rf_rate=TRY_RISK_FREE_RATE)
    
    benchmark_analyzer_usd = PerformanceAnalyzer(benchmark_usd, benchmark_usd)
    benchmark_metrics_usd = benchmark_analyzer_usd.calculate_metrics(rf_rate=USD_RISK_FREE_RATE)
    
    bist100_metrics_try = None
    if bist100_try is not None:
        bist100_analyzer_try = PerformanceAnalyzer(
            bist100_try.loc['2018-01-01':],
            bist100_try.loc['2018-01-01':]
        )
        bist100_metrics_try = bist100_analyzer_try.calculate_metrics(rf_rate=TRY_RISK_FREE_RATE)
    
    bist100_metrics_usd = None
    if bist100_usd is not None:
        bist100_analyzer_usd = PerformanceAnalyzer(
            bist100_usd.loc['2018-01-01':],
            bist100_usd.loc['2018-01-01':]
        )
        bist100_metrics_usd = bist100_analyzer_usd.calculate_metrics(rf_rate=USD_RISK_FREE_RATE)
    
    tur_metrics = None
    if tur_etf is not None:
        tur_analyzer = PerformanceAnalyzer(tur_etf, tur_etf)
        tur_metrics = tur_analyzer.calculate_metrics(rf_rate=USD_RISK_FREE_RATE)
    
    spy_metrics = None
    if spy_benchmark is not None:
        spy_analyzer = PerformanceAnalyzer(spy_benchmark, spy_benchmark)
        spy_metrics = spy_analyzer.calculate_metrics(rf_rate=USD_RISK_FREE_RATE)
    
    # Build results table
    results_table = []
    
    for strategy_name, strategy_data in results.items():
        m = strategy_data['metrics']
        b = strategy_data['bootstrap']
        t_test = strategy_data['t_test']
        t = strategy_data['trade_stats']
        results_table.append({
            'Strategy': strategy_name,
            'Tot Ret %': f"{m['Total Return'] * 100:.1f}%",
            'Ann Ret %': f"{m['Annualized Return'] * 100:.1f}%",
            'Sharpe': f"{m['Sharpe Ratio']:.2f}",
            'Sortino': f"{m['Sortino Ratio']:.2f}",
            'Max DD %': f"{m['Maximum Drawdown'] * 100:.1f}%",
            'Win Rate': f"{t['win_rate'] * 100:.1f}%",
            'Win/Loss': f"{t['win_loss_ratio']:.2f}x",
            'Bootstrap p': f"{b['p_value']:.3f}",
            'T-test p': f"{t_test['p_value']:.3f}"
        })
    
    # Add benchmarks
    results_table.append({
        'Strategy': 'EW BIST-100 (TRY)',
        'Tot Ret %': f"{benchmark_metrics_try['Total Return'] * 100:.1f}%",
        'Ann Ret %': f"{benchmark_metrics_try['Annualized Return'] * 100:.1f}%",
        'Sharpe': f"{benchmark_metrics_try['Sharpe Ratio']:.2f}",
        'Sortino': f"{benchmark_metrics_try['Sortino Ratio']:.2f}",
        'Max DD %': f"{benchmark_metrics_try['Maximum Drawdown'] * 100:.1f}%",
        'Win Rate': 'N/A',
        'Win/Loss': 'N/A',
        'Bootstrap p': 'N/A',
        'T-test p': 'N/A'
    })
    
    if bist100_metrics_try is not None:
        results_table.append({
            'Strategy': 'BIST100 (TRY)',
            'Tot Ret %': f"{bist100_metrics_try['Total Return'] * 100:.1f}%",
            'Ann Ret %': f"{bist100_metrics_try['Annualized Return'] * 100:.1f}%",
            'Sharpe': f"{bist100_metrics_try['Sharpe Ratio']:.2f}",
            'Sortino': f"{bist100_metrics_try['Sortino Ratio']:.2f}",
            'Max DD %': f"{bist100_metrics_try['Maximum Drawdown'] * 100:.1f}%",
            'Win Rate': 'N/A',
            'Win/Loss': 'N/A',
            'Bootstrap p': 'N/A',
            'T-test p': 'N/A'
        })
    
    results_table.append({
        'Strategy': 'EW BIST-100 (USD)',
        'Tot Ret %': f"{benchmark_metrics_usd['Total Return'] * 100:.1f}%",
        'Ann Ret %': f"{benchmark_metrics_usd['Annualized Return'] * 100:.1f}%",
        'Sharpe': f"{benchmark_metrics_usd['Sharpe Ratio']:.2f}",
        'Sortino': f"{benchmark_metrics_usd['Sortino Ratio']:.2f}",
        'Max DD %': f"{benchmark_metrics_usd['Maximum Drawdown'] * 100:.1f}%",
        'Win Rate': 'N/A',
        'Win/Loss': 'N/A',
        'Bootstrap p': 'N/A',
        'T-test p': 'N/A'
    })
    
    if tur_metrics is not None:
        results_table.append({
            'Strategy': 'TUR ETF',
            'Tot Ret %': f"{tur_metrics['Total Return'] * 100:.1f}%",
            'Ann Ret %': f"{tur_metrics['Annualized Return'] * 100:.1f}%",
            'Sharpe': f"{tur_metrics['Sharpe Ratio']:.2f}",
            'Sortino': f"{tur_metrics['Sortino Ratio']:.2f}",
            'Max DD %': f"{tur_metrics['Maximum Drawdown'] * 100:.1f}%",
            'Win Rate': 'N/A',
            'Win/Loss': 'N/A',
            'Bootstrap p': 'N/A',
            'T-test p': 'N/A'
        })
    
    if spy_metrics is not None:
        results_table.append({
            'Strategy': 'S&P 500',
            'Tot Ret %': f"{spy_metrics['Total Return'] * 100:.1f}%",
            'Ann Ret %': f"{spy_metrics['Annualized Return'] * 100:.1f}%",
            'Sharpe': f"{spy_metrics['Sharpe Ratio']:.2f}",
            'Sortino': f"{spy_metrics['Sortino Ratio']:.2f}",
            'Max DD %': f"{spy_metrics['Maximum Drawdown'] * 100:.1f}%",
            'Win Rate': 'N/A',
            'Win/Loss': 'N/A',
            'Bootstrap p': 'N/A',
            'T-test p': 'N/A'
        })
    
    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))
    
    # ==========================================================================
    # MACRO CORRELATIONS OUTPUT
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 MACRO VARIABLE CORRELATIONS")
    print("=" * 80)
    
    macro_table = []
    for strategy_name, corr in macro_correlations.items():
        macro_table.append({
            'Strategy': strategy_name,
            'Currency': f"{corr.get('currency_corr', np.nan):.3f}",
            'Inflation': f"{corr.get('inflation_corr', np.nan):.3f}",
            'Interest Rate': f"{corr.get('interest_corr', np.nan):.3f}"
        })
    
    macro_df = pd.DataFrame(macro_table)
    print(macro_df.to_string(index=False))
    
    # ==========================================================================
    # STATISTICAL SIGNIFICANCE
    # ==========================================================================
    
    sig_summary = summarize_statistical_significance(results)
    
    # ==========================================================================
    # STOCK AND SECTOR ANALYSIS
    # ==========================================================================
    
    stock_stats = analyze_individual_stocks(prices_try, bist.sectors)
    
    subperiod_results, detailed_subperiod = run_subperiod_analysis_enhanced(
        results, SUBPERIODS, benchmark_try, benchmark_usd
    )
    
    usdtry_subperiod = analyze_usdtry_by_subperiod(usdtry, SUBPERIODS)
    
    print("\n🏭 RUNNING SECTOR ANALYSIS")
    print("-" * 60)
    sector_results = None
    try:
        sector_results = analyze_sector_performance(
            prices_try, eligible_stocks, bist.sectors,
            frequency='weekly'
        )
        if sector_results:
            print("\nSector Performance:")
            for sector, metrics in sector_results.items():
                print(f"  {sector}: Return={metrics['Annualized Return'] * 100:.1f}%, "
                      f"Sharpe={metrics['Sharpe Ratio']:.2f}")
    except Exception as e:
        print(f"⚠ Could not complete sector analysis: {e}")
    
    # ==========================================================================
    # SENSITIVITY ANALYSIS
    # ==========================================================================
    
    transaction_cost_results = run_transaction_cost_sensitivity(
        prices_try, usdtry, eligible_stocks, strategies, benchmark_try, benchmark_usd
    )
    
    parameter_sensitivity_results = run_parameter_sensitivity(
        prices_try, usdtry, eligible_stocks, benchmark_try, benchmark_usd
    )
    
    # Plot parameter sensitivity
    if parameter_sensitivity_results:
        fig_param = plot_parameter_sensitivity(parameter_sensitivity_results)
        plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # ==========================================================================
    # ROLLING SHARPE INTERPRETATION
    # ==========================================================================
    
    interpret_rolling_sharpe(results)
    
    # ==========================================================================
    # VISUALIZATIONS
    # ==========================================================================
    
    print("\n📊 GENERATING VISUALIZATIONS")
    print("-" * 60)
    
    try:
        fig1 = plot_comprehensive_results(
            results, prices_try, prices_usd,
            benchmark_try, benchmark_usd,
            spy_benchmark, bist100_try, bist100_usd, tur_etf,
            frequency='weekly'
        )
        plt.savefig('bist_momentum_analysis_weekly_final.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved as 'bist_momentum_analysis_weekly_final.png'")
        plt.show()
        
        fig2 = plot_strategy_correlation_matrix(
            results, benchmark_try, benchmark_usd,
            bist100_try, bist100_usd, spy_benchmark, tur_etf
        )
        plt.savefig('strategy_correlation_weekly.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved as 'strategy_correlation_weekly.png'")
        plt.show()
        
        if sector_results:
            fig3 = plot_sector_analysis(sector_results)
            plt.savefig('sector_analysis_weekly_final.png', dpi=300, bbox_inches='tight')
            print("✓ Plot saved as 'sector_analysis_weekly_final.png'")
            plt.show()
        
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📝 THESIS SUMMARY STATISTICS")
    print("=" * 80)
    
    if results:
        strategies_by_sharpe = sorted(
            [(n, d['metrics']['Sharpe Ratio']) for n, d in results.items()],
            key=lambda x: x[1], reverse=True
        )
        print(f"\n🏆 Best Sharpe: {strategies_by_sharpe[0][0]} - {strategies_by_sharpe[0][1]:.2f}")
        print(f"📉 Worst Sharpe: {strategies_by_sharpe[-1][0]} - {strategies_by_sharpe[-1][1]:.2f}")
        
        significant = sum(1 for d in results.values() if d['bootstrap']['significant'])
        print(f"\n📊 Statistically significant strategies (p<0.05): {significant}/{len(results)}")
        
        try_sharpes = [d['metrics']['Sharpe Ratio'] for n, d in results.items() if 'TRY' in n]
        usd_sharpes = [d['metrics']['Sharpe Ratio'] for n, d in results.items() if 'USD' in n]
        
        if try_sharpes and usd_sharpes:
            print(f"\n💱 Currency Impact:")
            print(f"   Avg TRY Sharpe: {np.mean(try_sharpes):.2f}")
            print(f"   Avg USD Sharpe: {np.mean(usd_sharpes):.2f}")
            print(f"   Difference: {np.mean(try_sharpes) - np.mean(usd_sharpes):.2f}")
        
        print(f"\n📈 Trade Statistics:")
        try_win_rate = np.mean([d['trade_stats']['win_rate'] for n, d in results.items() if 'TRY' in n])
        try_win_loss = np.mean([d['trade_stats']['win_loss_ratio'] for n, d in results.items() if 'TRY' in n])
        try_profit_factor = np.mean([d['trade_stats']['profit_factor'] for n, d in results.items() if 'TRY' in n])
        try_expectancy = np.mean([d['trade_stats']['expectancy'] for n, d in results.items() if 'TRY' in n])
        print(f"   Avg TRY Win Rate: {try_win_rate * 100:.1f}%")
        print(f"   Avg TRY Win/Loss Ratio: {try_win_loss:.2f}x")
        print(f"   Avg TRY Profit Factor: {try_profit_factor:.2f}")
        print(f"   Avg TRY Expectancy: {try_expectancy:.2f}% per trade")
        
        print(f"\n🌍 Benchmark Comparison:")
        if spy_metrics:
            print(f"   S&P 500 Sharpe: {spy_metrics['Sharpe Ratio']:.2f}")
            print(f"   Best TRY Strategy vs S&P 500: +{strategies_by_sharpe[0][1] - spy_metrics['Sharpe Ratio']:.2f}")
        
        if tur_metrics:
            print(f"   TUR ETF Sharpe: {tur_metrics['Sharpe Ratio']:.2f}")
            usd_best = max([d['metrics']['Sharpe Ratio'] for n, d in results.items() if 'USD' in n])
            print(f"   Best USD Strategy vs TUR: +{usd_best - tur_metrics['Sharpe Ratio']:.2f}")
        
        if bist100_metrics_try:
            print(f"   BIST100 Index (TRY) Sharpe: {bist100_metrics_try['Sharpe Ratio']:.2f}")
            print(f"   Best TRY Strategy vs BIST100: +{strategies_by_sharpe[0][1] - bist100_metrics_try['Sharpe Ratio']:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()

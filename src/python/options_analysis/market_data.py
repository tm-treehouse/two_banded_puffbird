#!/usr/bin/env python3
"""
Market data providers for fetching stock and options data.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import yfinance as yf

from options_analysis.treasury_helper import get_treasury_yield

logger = logging.getLogger("options_analysis")


class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.

    This class defines the interface that all market data providers must implement.
    Concrete implementations can fetch data from different sources (Yahoo Finance,
    broker APIs, etc.) but must provide the same interface.
    """

    @abstractmethod
    def get_ticker_data(self, ticker: str) -> Any:
        """
        Get basic ticker data for the specified symbol

        Parameters:
            ticker (str): The ticker symbol

        Returns:
            Any: Object representing the ticker data
        """
        pass

    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """
        Get the current price for the ticker

        Parameters:
            ticker (str): The ticker symbol

        Returns:
            float: Current price
        """
        pass

    @abstractmethod
    def get_option_chain(self, ticker: str, expiration_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get option chain data for a specific expiration date

        Parameters:
            ticker (str): The ticker symbol
            expiration_date (str): Option expiration date string

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (calls_df, puts_df)
        """
        pass

    @abstractmethod
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free interest rate

        Returns:
            float: Current risk-free rate as a decimal (e.g., 0.045 for 4.5%)
        """
        pass

    @abstractmethod
    def get_available_expiration_dates(self, ticker: str) -> List[str]:
        """
        Get available option expiration dates for the ticker

        Parameters:
            ticker (str): The ticker symbol

        Returns:
            List[str]: List of available expiration dates
        """
        pass


class YahooFinanceProvider(MarketDataProvider):
    """
    Yahoo Finance implementation of the MarketDataProvider.

    This class uses the yfinance library to fetch market data from Yahoo Finance.
    """

    def __init__(self, out_dir: Path = None):
        """
        Initialize the Yahoo Finance data provider

        Parameters:
            out_dir (Path): Path to the output directory (default: Path("out"))
        """
        self.out_dir = out_dir if out_dir is not None else Path("out")
        self.out_dir.mkdir(exist_ok=True)

    def get_ticker_data(self, ticker: str) -> yf.Ticker:
        """
        Get yfinance Ticker object for the specified symbol

        Parameters:
            ticker (str): The ticker symbol

        Returns:
            yf.Ticker: Yahoo Finance ticker object
        """
        try:
            logger.info(f"Fetching data for {ticker}")
            return yf.Ticker(ticker)
        except Exception as e:
            logger.error(f"Failed to get ticker data for {ticker}: {e}")
            raise

    def get_current_price(self, ticker: str) -> float:
        """
        Get the current price for the ticker

        Parameters:
            ticker (str): The ticker symbol

        Returns:
            float: Current price
        """
        try:
            stock = self.get_ticker_data(ticker)
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            logger.info(f"Current price for {ticker}: ${current_price:.2f}")
            return current_price
        except Exception as e:
            logger.error(f"Failed to get price history for {ticker}: {str(e)}")
            raise

    def get_option_chain(self, ticker: str, expiration_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get option chain data for a specific expiration date

        Parameters:
            ticker (str): The ticker symbol
            expiration_date (str): Option expiration date string

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing (calls_df, puts_df)
        """
        try:
            stock = self.get_ticker_data(ticker)
            options = stock.option_chain(expiration_date)
            return options.calls, options.puts
        except Exception as e:
            logger.error(f"Failed to get option chain for {ticker} {expiration_date}: {e}")
            raise

    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free interest rate from US Treasury website

        Returns:
            float: Current risk-free rate as a decimal (e.g., 0.045 for 4.5%)
        """
        try:
            # The treasury_helper.get_treasury_yield function now handles all sources
            # including Treasury site, regex parsing, CNBC, and fallback to known rate
            yield_10y = get_treasury_yield()
            if yield_10y is not None:
                logger.info(f"Current Treasury yield: {yield_10y}")
                return yield_10y

            # If all sources fail, use the default value
            logger.warning("All sources failed to provide risk-free rate")
            return 0.045  # Default fallback 4.5%

        except Exception as e:
            logger.error(f"Failed to fetch treasury yield: {e}")
            logger.warning("Using default risk-free rate of 0.045 (4.5%)")
            return 0.045  # Default fallback

    def get_available_expiration_dates(self, ticker: str) -> List[str]:
        """
        Get available option expiration dates for the ticker

        Parameters:
            ticker (str): The ticker symbol

        Returns:
            List[str]: List of available expiration dates
        """
        try:
            stock = self.get_ticker_data(ticker)
            expiration_dates = stock.options

            if not expiration_dates or len(expiration_dates) == 0:
                logger.warning(f"No option data available for {ticker}")
                return []

            logger.info(f"{ticker} has {len(expiration_dates)} expiration dates")
            return list(expiration_dates)
        except Exception as e:
            logger.error(f"Failed to get option dates for {ticker}: {e}")
            return []

    def get_sp500_tickers(self, filepath=None, refresh=False) -> List[str]:
        """
        Fetches S&P 500 tickers from file if it exists. Otherwise scrapes from Wikipedia.
        Use `refresh=True` to force update from web.

        Parameters:
            filepath (str): Optional path to cached ticker file
            refresh (bool): Force refresh from web if True

        Returns:
            List[str]: List of S&P 500 ticker symbols
        """
        # Default filepath is in the out directory
        if filepath is None:
            filepath = self.out_dir / "sp500_tickers.csv"

        # Also check the root directory for backward compatibility
        root_filepath = "sp500_tickers.csv"

        if filepath.exists() and not refresh:
            logger.info(f"Loading S&P 500 tickers from cached file: {filepath}")
            return pd.read_csv(filepath)["ticker"].tolist()
        elif os.path.exists(root_filepath) and not refresh:
            # For backward compatibility, check the root directory
            logger.info(f"Loading S&P 500 tickers from cached file: {root_filepath}")
            return pd.read_csv(root_filepath)["ticker"].tolist()

        logger.info("Fetching S&P 500 tickers from Wikipedia...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers_df = tables[0][["Symbol"]].copy()
        tickers_df["ticker"] = tickers_df["Symbol"].str.replace(".", "-", regex=False)
        tickers_df[["ticker"]].to_csv(filepath, index=False)

        return tickers_df["ticker"].tolist()

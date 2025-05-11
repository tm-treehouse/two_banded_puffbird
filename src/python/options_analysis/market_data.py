#!/usr/bin/env python3
"""
Market data providers for fetching stock and options data.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import yfinance as yf

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
            # Try the official Treasury website approach first
            yield_10y = self._get_treasury_yield()
            if yield_10y is not None:
                logger.info(f"Current 10y Treasury yield from Treasury website: {yield_10y}")
                return yield_10y

            # Fall back to alternative sources when Treasury website parsing fails
            yield_10y = self._get_wsj_yield()
            if yield_10y is not None:
                logger.info(f"Current 10y Treasury yield from WSJ: {yield_10y}")
                return yield_10y

            # If all sources fail, use the default value
            logger.warning("All sources failed to provide risk-free rate")
            return 0.045  # Default fallback 4.5%

        except Exception as e:
            logger.error(f"Failed to fetch treasury yield: {e}")
            logger.warning("Using default risk-free rate of 0.045 (4.5%)")
            return 0.045  # Default fallback

    def _get_treasury_yield(self) -> float:
        """
        Get current risk-free rate from Treasury website.

        Returns:
            float or None: Current 10-year Treasury yield as a decimal, or None if unavailable
        """
        try:
            url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=all&page=0"
            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                logger.warning(f"Treasury API returned status code: {response.status_code}")
                return None

            root = ET.fromstring(response.content)

            # Try to find the date of the most recent data
            updated = root.find(".//{http://www.w3.org/2005/Atom}updated")
            if updated is not None and updated.text:
                logger.info(f"Treasury data last updated: {updated.text}")

            # Look for content that might contain the yield information
            # Search for BC_10YEAR which is the 10-year benchmark Treasury yield
            for elem in root.iter():
                if elem.text and any(
                    term in str(elem.text).lower() for term in ["bc_10year", "10 yr", "10-year", "gs10"]
                ):
                    logger.debug(f"Found promising element: {elem.tag} with text: {elem.text[:30]}...")

                    # Look for a value format that appears to be a percentage
                    import re

                    value_match = re.search(r"(\d+\.\d+)\s*%?", elem.text)
                    if value_match:
                        value_str = value_match.group(1)
                        logger.info(f"Found 10-year yield value: {value_str}")
                        return float(value_str) / 100  # Convert percentage to decimal

            logger.warning("Could not extract 10-year yield from Treasury website")
            return None

        except Exception as e:
            logger.error(f"Error in Treasury yield extraction: {str(e)}")
            return None

    def _get_wsj_yield(self) -> float:
        """
        Get current risk-free rate from Wall Street Journal as fallback.

        Returns:
            float or None: Current 10-year Treasury yield as a decimal, or None if unavailable
        """
        try:
            url = "https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD10Y?mod=md_bond_overview"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
                # Using simple string search for the yield value
                start_marker = '"last_price":"'
                end_marker = '"'

                if start_marker in response.text:
                    start = response.text.find(start_marker) + len(start_marker)
                    end = response.text.find(end_marker, start)

                    if start > 0 and end > start:
                        value_str = response.text[start:end]
                        logger.info(f"WSJ 10-year yield value: {value_str}")
                        return float(value_str) / 100  # Convert percentage to decimal

            logger.warning("Could not extract 10-year yield from WSJ")
            return None

        except Exception as e:
            logger.error(f"Error in WSJ yield extraction: {str(e)}")
            return None

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

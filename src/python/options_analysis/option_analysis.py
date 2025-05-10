#!/usr/bin/env python3
"""
Options analysis module for evaluating option contracts.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import logging
from pathlib import Path
import time
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .black_scholes import BlackScholesCalculator
from .market_data import MarketDataProvider

logger = logging.getLogger("options_analysis")


class OptionAnalyzer:
    """
    Option analyzer class for processing and analyzing option contracts.

    This class handles the analysis of option contracts, calculating
    risk metrics, filtering based on criteria, and scoring contracts.
    """

    def __init__(self, market_data_provider: MarketDataProvider, out_dir: Path = None):
        """
        Initialize the OptionAnalyzer with a market data provider

        Parameters:
            market_data_provider (MarketDataProvider): Provider for market data
            out_dir (Path): Output directory path (default: Path("out"))
        """
        self.market_data = market_data_provider
        self.out_dir = out_dir if out_dir is not None else Path("out")
        self.out_dir.mkdir(exist_ok=True)
        self.bs_calculator = BlackScholesCalculator()

        # Default analysis parameters
        self.percentage_range = 15
        self.min_risk_score = 0.04
        self.max_delta_threshold = 0.75
        self.min_delta_threshold = 0.30
        self.min_projected_return_pct = 2
        self.min_annual_return_pct = 30

    def set_analysis_parameters(
        self,
        percentage_range: float = None,
        min_risk_score: float = None,
        max_delta_threshold: float = None,
        min_delta_threshold: float = None,
        min_projected_return_pct: float = None,
        min_annual_return_pct: float = None,
    ):
        """
        Set analysis parameters for filtering options

        Parameters:
            percentage_range (float): Price range percentage from current price
            min_risk_score (float): Minimum risk-adjusted score
            max_delta_threshold (float): Maximum absolute delta value
            min_delta_threshold (float): Minimum absolute delta value
            min_projected_return_pct (float): Minimum return on capital percentage
            min_annual_return_pct (float): Minimum annualized return percentage
        """
        if percentage_range is not None:
            self.percentage_range = percentage_range
        if min_risk_score is not None:
            self.min_risk_score = min_risk_score
        if max_delta_threshold is not None:
            self.max_delta_threshold = max_delta_threshold
        if min_delta_threshold is not None:
            self.min_delta_threshold = min_delta_threshold
        if min_projected_return_pct is not None:
            self.min_projected_return_pct = min_projected_return_pct
        if min_annual_return_pct is not None:
            self.min_annual_return_pct = min_annual_return_pct

    def process_option_chain(
        self, ticker: str, current_price: float, exp_date: str, risk_free_rate: float, today: dt.datetime
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Process a single option chain for a ticker and expiration date

        Parameters:
            ticker (str): Ticker symbol
            current_price (float): Current stock price
            exp_date (str): Expiration date string
            risk_free_rate (float): Risk-free rate
            today (dt.datetime): Current date for calculations

        Returns:
            Tuple[List[pd.DataFrame], List[pd.DataFrame]]: Processed put and call data
        """
        put_data = []
        call_data = []

        # Price bounds for filtering options by strike price
        put_lower_bound = current_price * (1 - self.percentage_range / 100)
        put_upper_bound = current_price * (1 - 5 / 100)
        call_upper_bound = current_price * (1 + self.percentage_range / 100)

        retry_count = 0
        max_retries = 2

        while retry_count <= max_retries:
            try:
                # Fetch the option chain
                logger.info(f"Processing {ticker} options for {exp_date} (attempt {retry_count + 1})")
                calls, puts = self.market_data.get_option_chain(ticker, exp_date)

                # Create copies to avoid modifying the original data
                calls = calls.copy()
                puts = puts.copy()

                # Basic filtering for liquidity
                puts = puts[(puts["volume"] > 0) & (puts["openInterest"] >= 100)]

                calls = calls[(calls["volume"] > 0) & (calls["openInterest"] >= 100)]

                # If we have no valid options after filtering, skip
                if puts.empty and calls.empty:
                    logger.debug(f"{ticker} {exp_date}: No options passed volume/OI filters")
                    break

                # Calculate midpoint premiums
                calls["midpoint"] = (calls["bid"] + calls["ask"]) / 2
                puts["midpoint"] = (puts["bid"] + puts["ask"]) / 2

                # Filter for OTM puts and calls within percentage range
                otm_puts = puts[
                    (puts["strike"] < current_price)
                    & (puts["strike"] >= put_lower_bound)
                    & (puts["strike"] <= put_upper_bound)
                ]
                otm_calls = calls[(calls["strike"] > current_price) & (calls["strike"] <= call_upper_bound)]

                if otm_puts.empty and otm_calls.empty:
                    logger.debug(f"{ticker} {exp_date}: No options in target price range")
                    break

                logger.debug(f"{ticker} {exp_date}: Found {len(otm_puts)} puts and {len(otm_calls)} calls")

                # Process each option type
                if not otm_puts.empty:
                    processed_puts = self._process_put_options(
                        otm_puts, ticker, current_price, exp_date, risk_free_rate, today
                    )
                    put_data.append(processed_puts)

                if not otm_calls.empty:
                    processed_calls = self._process_call_options(
                        otm_calls, ticker, current_price, exp_date, risk_free_rate, today
                    )
                    call_data.append(processed_calls)

                # Successfully processed, exit retry loop
                break

            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                logger.warning(f"Attempt {retry_count} failed for {ticker} {exp_date}: {error_msg}")

                if retry_count <= max_retries:
                    # Wait before retry with exponential backoff
                    wait_time = 2**retry_count
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed for {ticker} {exp_date}")

        return put_data, call_data

    def _process_put_options(
        self,
        puts_df: pd.DataFrame,
        ticker: str,
        current_price: float,
        exp_date: str,
        risk_free_rate: float,
        today: dt.datetime,
    ) -> pd.DataFrame:
        """Process and enhance put options data"""
        df = puts_df.copy()

        # Add expiration info
        df["expiration"] = exp_date
        df["ticker"] = ticker

        # Calculate premiums and returns
        df["premium_collected"] = df["midpoint"] * 100
        df["capital_required"] = df["strike"] * 100

        # We skew the return by 90% to account for slippage
        df["return_on_capital_%"] = (df["premium_collected"] / df["capital_required"]) * 90

        # Calculate days to expiration and annualized returns
        df["days_to_exp"] = (pd.to_datetime(exp_date) - today).days
        df["T"] = df["days_to_exp"] / 365  # time in years

        # Avoid division by zero
        df["return_on_capital_per_anum_%"] = np.where(
            df["days_to_exp"] > 0, (df["return_on_capital_%"] / df["days_to_exp"]) * 365, 0
        )

        # Include IV as percentage
        df["implied_volatility"] = df["impliedVolatility"] * 100

        # Calculate risk-adjusted score (ROC / IV)
        df["risk_adjusted_score"] = df["return_on_capital_%"] / df["implied_volatility"]

        # Calculate delta using Black-Scholes
        df["delta"] = df.apply(
            lambda row: self.bs_calculator.put_delta(
                S=current_price,
                K=row["strike"],
                T=row["T"],
                r=risk_free_rate,
                sigma=row["implied_volatility"] / 100,
            ),
            axis=1,
        )

        # Calculate delta × implied volatility
        df["delta_x_iv"] = df["delta"] * df["implied_volatility"]

        # Keep relevant columns
        columns = [
            "ticker",
            "strike",
            "expiration",
            "midpoint",
            "premium_collected",
            "capital_required",
            "return_on_capital_%",
            "return_on_capital_per_anum_%",
            "implied_volatility",
            "risk_adjusted_score",
            "delta",
            "delta_x_iv",
            "days_to_exp",
            "T",
        ]

        return df[columns]

    def _process_call_options(
        self,
        calls_df: pd.DataFrame,
        ticker: str,
        current_price: float,
        exp_date: str,
        risk_free_rate: float,
        today: dt.datetime,
    ) -> pd.DataFrame:
        """Process and enhance call options data"""
        df = calls_df.copy()

        # Add expiration info
        df["expiration"] = exp_date
        df["ticker"] = ticker

        # Calculate premiums and returns
        df["premium_collected"] = df["midpoint"] * 100
        df["capital_required"] = df["strike"] * 100

        # We skew the return by 90% to account for slippage
        df["return_on_capital_%"] = (df["premium_collected"] / df["capital_required"]) * 90

        # Calculate days to expiration and annualized returns
        df["days_to_exp"] = (pd.to_datetime(exp_date) - today).days
        df["T"] = df["days_to_exp"] / 365  # time in years

        # Avoid division by zero
        df["return_on_capital_per_anum_%"] = np.where(
            df["days_to_exp"] > 0, (df["return_on_capital_%"] / df["days_to_exp"]) * 365, 0
        )

        # Include IV as percentage
        df["implied_volatility"] = df["impliedVolatility"] * 100

        # Calculate risk-adjusted score (ROC / IV)
        df["risk_adjusted_score"] = df["return_on_capital_%"] / df["implied_volatility"]

        # Calculate delta using Black-Scholes
        df["delta"] = df.apply(
            lambda row: self.bs_calculator.call_delta(
                S=current_price,
                K=row["strike"],
                T=row["T"],
                r=risk_free_rate,
                sigma=row["implied_volatility"] / 100,
            ),
            axis=1,
        )

        # Calculate delta × implied volatility
        df["delta_x_iv"] = df["delta"] * df["implied_volatility"]

        # Keep relevant columns
        columns = [
            "ticker",
            "strike",
            "expiration",
            "midpoint",
            "premium_collected",
            "capital_required",
            "return_on_capital_%",
            "return_on_capital_per_anum_%",
            "implied_volatility",
            "risk_adjusted_score",
            "delta",
            "delta_x_iv",
            "days_to_exp",
            "T",
        ]

        return df[columns]

    def assign_composite_score(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a composite score based on normalized values of key metrics.
        Only normalize within the filtered DataFrame to ensure relevance.

        Parameters:
            filtered_df (pd.DataFrame): DataFrame with option data

        Returns:
            pd.DataFrame: DataFrame with added composite scores
        """

        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if min_val == max_val:
                return pd.Series([1] * len(series), index=series.index)  # All same values
            return (series - min_val) / (max_val - min_val)

        df = filtered_df.copy()

        # Normalize key metrics (within filtered subset only)
        df["normalized_delta"] = normalize(df["delta"].abs())  # closer to 0 is better
        df["normalized_premium"] = normalize(df["premium_collected"])
        df["normalized_roc"] = normalize(df["return_on_capital_%"])
        df["normalized_iv"] = 1 - normalize(df["implied_volatility"])  # inverted

        # Composite score with adjustable weights
        df["composite_score"] = (
            0.2 * df["normalized_delta"]
            + 0.3 * df["normalized_premium"]
            + 0.2 * df["normalized_roc"]
            + 0.15 * df["normalized_iv"]
        )

        return df

    def filter_puts(self, put_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter put options based on analysis parameters

        Parameters:
            put_df (pd.DataFrame): DataFrame with put option data

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if put_df.empty:
            return put_df

        logger.info(f"Starting with {len(put_df)} put options")

        # Apply filters one by one with logging to track data loss
        filtered_df = put_df[put_df["risk_adjusted_score"] >= self.min_risk_score]
        logger.info(f"After risk score filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["delta"] <= -self.min_delta_threshold]
        logger.info(f"After min delta filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["delta"] >= -self.max_delta_threshold]
        logger.info(f"After max delta filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["return_on_capital_%"] >= self.min_projected_return_pct]
        logger.info(f"After return filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["return_on_capital_per_anum_%"] >= self.min_annual_return_pct]
        logger.info(f"After annualized return filter: {len(filtered_df)} rows")

        return filtered_df

    def filter_calls(self, call_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter call options based on analysis parameters

        Parameters:
            call_df (pd.DataFrame): DataFrame with call option data

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if call_df.empty:
            return call_df

        logger.info(f"Starting with {len(call_df)} call options")

        # Apply filters one by one with logging to track data loss
        filtered_df = call_df[call_df["risk_adjusted_score"] >= self.min_risk_score]
        logger.info(f"After risk score filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["delta"] >= self.min_delta_threshold]
        logger.info(f"After min delta filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["delta"] <= self.max_delta_threshold]
        logger.info(f"After max delta filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["return_on_capital_%"] >= self.min_projected_return_pct]
        logger.info(f"After return filter: {len(filtered_df)} rows")

        filtered_df = filtered_df[filtered_df["return_on_capital_per_anum_%"] >= self.min_annual_return_pct]
        logger.info(f"After annualized return filter: {len(filtered_df)} rows")

        return filtered_df

    def analyze_with_relaxed_filters(
        self, put_df: pd.DataFrame, filename: str = "relaxed_otm_puts.csv"
    ) -> pd.DataFrame:
        """
        Apply relaxed filters for options analysis and save results

        Parameters:
            put_df (pd.DataFrame): DataFrame with put option data
            filename (str): Output filename

        Returns:
            pd.DataFrame: DataFrame with relaxed filter results
        """
        if put_df.empty:
            return pd.DataFrame()

        # Apply relaxed filters
        relaxed_filters = put_df[
            (put_df["risk_adjusted_score"] >= self.min_risk_score / 2)
            & (put_df["delta"] <= -self.min_delta_threshold / 2)
            & (put_df["delta"] >= -1.0)
            & (put_df["return_on_capital_%"] >= self.min_projected_return_pct / 2)
            & (put_df["return_on_capital_per_anum_%"] >= 15)
        ]

        if not relaxed_filters.empty:
            relaxed_scored = self.assign_composite_score(relaxed_filters)
            top_relaxed = relaxed_scored.sort_values(by="composite_score", ascending=False).head(25)
            output_file = self.out_dir / filename
            top_relaxed.to_csv(output_file, index=False)
            logger.info(f"Saved {len(top_relaxed)} rows with relaxed filters to {output_file}")
            return top_relaxed

        return pd.DataFrame()

    def save_analysis_results(
        self, filtered_df: pd.DataFrame, filename: str = "otm_puts.csv", top_n: int = 25
    ) -> pd.DataFrame:
        """
        Save analysis results to CSV file

        Parameters:
            filtered_df (pd.DataFrame): DataFrame with filtered option data
            filename (str): Output filename
            top_n (int): Number of top results to save

        Returns:
            pd.DataFrame: DataFrame with top results
        """
        if filtered_df.empty:
            return pd.DataFrame()

        scored_df = self.assign_composite_score(filtered_df)
        top_results = scored_df.sort_values(by="composite_score", ascending=False).head(top_n)

        output_file = self.out_dir / filename
        top_results.to_csv(output_file, index=False)
        logger.info(f"Exported {len(top_results)} filtered options to {output_file}")

        return top_results


class OptionsAnalysisRunner:
    """
    Runner class to orchestrate options analysis across multiple tickers.

    This class handles the parallel processing of tickers, collects
    the results, and delegates the analysis to the OptionAnalyzer.
    """

    def __init__(self, market_data_provider: MarketDataProvider, option_analyzer: OptionAnalyzer):
        """
        Initialize the runner with providers and analyzers

        Parameters:
            market_data_provider (MarketDataProvider): Provider for market data
            option_analyzer (OptionAnalyzer): Analyzer for options
        """
        self.market_data = market_data_provider
        self.analyzer = option_analyzer

    def run_analysis(self, tickers: List[str] = None, max_workers: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Run options analysis on multiple tickers

        Parameters:
            tickers (List[str]): List of ticker symbols to analyze
            max_workers (int): Maximum number of parallel worker threads

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of analysis results
        """
        if tickers is None:
            # Get S&P 500 tickers if none provided
            tickers = self.market_data.get_sp500_tickers()

        logger.info(f"Processing {len(tickers)} tickers")

        # Fetch risk-free rate once for all analyses
        risk_free_rate = self.market_data.get_risk_free_rate()
        logger.info(f"Using risk-free rate: {risk_free_rate}")

        today = dt.datetime.today()

        # Store data for all tickers
        all_put_data = []
        all_call_data = []

        # Status counters
        successful_tickers = 0
        empty_tickers = 0
        failed_tickers = 0

        logger.info(f"Processing {len(tickers)} tickers in parallel with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each ticker
            future_to_ticker = {
                executor.submit(self._process_ticker, ticker, risk_free_rate, today): ticker
                for ticker in tickers
            }

            # Process results as they complete
            completed = 0
            total = len(future_to_ticker)

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1

                try:
                    put_data, call_data = future.result()

                    # Check if both are empty lists
                    if not put_data and not call_data:
                        empty_tickers += 1
                        logger.debug(f"No data returned for {ticker}")
                        continue

                    successful_tickers += 1
                    options_count = 0

                    if put_data:
                        # Verify each DataFrame has actual rows
                        valid_dfs = []
                        for df in put_data:
                            if not df.empty:
                                valid_dfs.append(df)
                                options_count += len(df)

                        all_put_data.extend(valid_dfs)

                    if call_data:
                        # Similar check for call_data
                        valid_dfs = []
                        for df in call_data:
                            if not df.empty:
                                valid_dfs.append(df)
                                options_count += len(df)

                        all_call_data.extend(valid_dfs)

                    logger.info(
                        f"Completed processing for {ticker} ({completed}/{total}), found {options_count} total options"
                    )

                except Exception as e:
                    failed_tickers += 1
                    logger.error(f"Error processing {ticker}: {e}")

        logger.info(
            f"Ticker processing summary: {successful_tickers} successful, {empty_tickers} empty, {failed_tickers} failed"
        )

        # Process the aggregated data
        results = {}

        # Process PUT options
        if all_put_data:
            try:
                put_df = pd.concat(all_put_data, ignore_index=True)
                logger.info(f"Combined put dataframe has {len(put_df)} rows")

                # Save raw data for inspection
                put_df.to_csv(self.analyzer.out_dir / "raw_put_data.csv", index=False)

                # Apply filters
                filtered_put_df = self.analyzer.filter_puts(put_df)

                if filtered_put_df.empty:
                    logger.warning("All put options filtered out! Trying relaxed filters.")
                    relaxed_results = self.analyzer.analyze_with_relaxed_filters(put_df)
                    if not relaxed_results.empty:
                        results["relaxed_puts"] = relaxed_results
                else:
                    # Save standard filtered results
                    top_puts = self.analyzer.save_analysis_results(filtered_put_df, "otm_puts.csv")
                    results["puts"] = top_puts

            except Exception as e:
                logger.error(f"Error processing put data: {e}", exc_info=True)

        # Process CALL options - similar to puts
        if all_call_data:
            try:
                call_df = pd.concat(all_call_data, ignore_index=True)
                logger.info(f"Combined call dataframe has {len(call_df)} rows")

                # Save raw data
                call_df.to_csv(self.analyzer.out_dir / "raw_call_data.csv", index=False)

                # Apply filters
                filtered_call_df = self.analyzer.filter_calls(call_df)

                if not filtered_call_df.empty:
                    # Save standard filtered results
                    top_calls = self.analyzer.save_analysis_results(filtered_call_df, "otm_calls.csv")
                    results["calls"] = top_calls

            except Exception as e:
                logger.error(f"Error processing call data: {e}", exc_info=True)

        return results

    def _process_ticker(
        self, ticker: str, risk_free_rate: float, today: dt.datetime
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Process a single ticker to analyze its option chains.

        Parameters:
            ticker (str): Ticker symbol to analyze
            risk_free_rate (float): Current risk-free interest rate
            today (dt.datetime): Current date for calculations

        Returns:
            Tuple[List[pd.DataFrame], List[pd.DataFrame]]: Tuple of (put_data, call_data)
        """
        put_data = []
        call_data = []

        try:
            # Try to get the current price
            try:
                current_price = self.market_data.get_current_price(ticker)
            except Exception as e:
                logger.error(f"Failed to get price for {ticker}: {str(e)}")
                return put_data, call_data

            # Get expiration dates
            try:
                expiration_dates = self.market_data.get_available_expiration_dates(ticker)

                if not expiration_dates:
                    logger.warning(f"No option data available for {ticker}")
                    return put_data, call_data

            except Exception as e:
                logger.error(f"Failed to get option dates for {ticker}: {str(e)}")
                return put_data, call_data

            # Limit to a few near-term expirations
            available_exp_count = len(expiration_dates)
            if available_exp_count >= 4:
                # Take 2-6 (skip the very short-term)
                expiration_dates = expiration_dates[1:5]
            elif available_exp_count >= 2:
                # Take all but the first (shortest-term)
                expiration_dates = expiration_dates[1:]

            logger.debug(f"{ticker} using expiration dates: {expiration_dates}")

            # Track how many expiration dates were processed
            processed_exp_count = 0

            for exp_date in expiration_dates:
                # Process option chain for this expiration date
                put_chains, call_chains = self.analyzer.process_option_chain(
                    ticker, current_price, exp_date, risk_free_rate, today
                )

                # Add valid chains to our results
                if put_chains:
                    put_data.extend(put_chains)
                    processed_exp_count += 1

                if call_chains:
                    call_data.extend(call_chains)

            # Log summary for this ticker
            logger.info(
                f"{ticker} summary: processed {processed_exp_count}/{len(expiration_dates)} expirations, collected {len(put_data)} put chains and {len(call_data)} call chains"
            )

        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
            logger.debug(traceback.format_exc())

        return put_data, call_data

#!/root/venv/bin/python3
"""
----------------------------------------------------
Title:      analysis.py
License:    agpl-3.0
Author:     TM, SC
Created on: 2025-05-06
----------------------------------------------------

Description:
Financial options analysis tool that calculates risk-adjusted returns
for put and call options based on current market data. The script
fetches stock prices, treasury yields, and option chains, then applies
Black-Scholes modeling to identify potentially favorable option trades.

----------------------------------------------------

Arguments:
None (ATM)

----------------------------------------------------
How to run this script:
cd two_banded_puffbird
uv run src/python/analysis.py
----------------------------------------------------
"""

import datetime as dt
import logging
import xml.etree.ElementTree as ET
from logging.handlers import RotatingFileHandler
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from rich.console import Console
from rich.logging import RichHandler
from scipy.stats import norm

pd.options.mode.chained_assignment = None  # default='warn'

# Initialize logger
logger = logging.getLogger("options_analysis")


def setup_logging(
    log_level=logging.INFO, log_file="options_analysis.log", max_file_size_mb=1, backup_count=3
):
    """
    Configure logging with both file and console handlers.

    Parameters:
        log_level (int): Logging level (default: logging.INFO)
        log_file (str): Path to log file
        max_file_size_mb (int): Maximum log file size in MB before rotating
        backup_count (int): Number of backup logs to keep
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / log_file

    # Configure logger
    logger.setLevel(log_level)

    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler - rotating log files
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=max_file_size_mb * 1024 * 1024, backupCount=backup_count
    )
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Rich console handler
    console_handler = RichHandler(console=Console())
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"Logging configured. Log file: {log_file_path}")
    return logger


def get_sp500_tickers_from_file_or_web(filepath='sp500_tickers.csv', refresh=False):
    """
    Fetches S&P 500 tickers from file if it exists. Otherwise scrapes from Wikipedia.
    Use `refresh=True` to force update from web.
    """
    if os.path.exists(filepath) and not refresh:
        print(f"Loading S&P 500 tickers from cached file: {filepath}")
        return pd.read_csv(filepath)['ticker'].tolist()

    print("Fetching S&P 500 tickers from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    tickers_df = tables[0][['Symbol']].copy()
    tickers_df['ticker'] = tickers_df['Symbol'].str.replace('.', '-', regex=False)
    tickers_df[['ticker']].to_csv(filepath, index=False)

    return tickers_df['ticker'].tolist()

def get_treasury_yield():
    """
    Fetch the latest 10-year Treasury yield from the US Treasury website.

    Returns:
        float: The current 10-year Treasury yield as a decimal (e.g., 0.045 for 4.5%)
    """
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=all&page=0"
    response = requests.get(url)
    root = ET.fromstring(response.content)

    # Extract the most recent date and yield
    latest_entry = root.findall(".//entry")[-1]
    yield_10y = latest_entry.find(".//td[@class='GS10']").text
    logger.info(f"Current 10y Treasury yield: {yield_10y}")
    return float(yield_10y)


def bs_put_delta(S, K, T, r, sigma):
    """
    Calculate Black-Scholes delta for European put options.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free rate (as decimal, e.g., 0.045 for 4.5%)
        sigma (float): Implied volatility (as decimal, e.g., 0.25 for 25%)

    Returns:
        float: Put delta (between -1 and 0)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def bs_call_delta(S, K, T, r, sigma):
    """
    Calculate Black-Scholes delta for European call options.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free rate (as decimal, e.g., 0.045 for 4.5%)
        sigma (float): Implied volatility (as decimal, e.g., 0.25 for 25%)

    Returns:
        float: Call delta (between 0 and 1)
    """
    # if T <= 0 | sigma <= 0 | S <= 0 | K <= 0:
    #    logger.error("Invalid parameters for bs_call_delta")
    #    return np.nan  # invalid parameters

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    call_delta = norm.cdf(d1)
    logger.debug(f"Call delta: {call_delta}")
    return call_delta


def assign_composite_score(filtered_df):
    """
    Assign a composite score based on normalized values of key metrics.
    Only normalize within the filtered DataFrame to ensure relevance.
    """
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if min_val == max_val:
            return pd.Series([1] * len(series), index=series.index)  # All same values
        return (series - min_val) / (max_val - min_val)

    df = filtered_df.copy()

    # Normalize key metrics (within filtered subset only)
    df['normalized_delta'] = normalize(df['delta'].abs())  # closer to 0 is better
    df['normalized_premium'] = normalize(df['premium_collected'])
    df['normalized_roc'] = normalize(df['return_on_capital_%'])
    df['normalized_iv'] = 1 - normalize(df['implied_volatility'])  # inverted
    #df['normalized_breakeven_margin'] = normalize(df['breakeven_margin_%'])

    # Composite score with adjustable weights
    df['composite_score'] = (
        0.2 * df['normalized_delta'] +
        0.3 * df['normalized_premium'] +
        0.2 * df['normalized_roc'] +
        0.15 * df['normalized_iv']
    #    0.15 * df['normalized_breakeven_margin']
    )

    return df


def process_ticker(ticker, risk_free_rate, percentage_range, today, min_risk_score, max_delta_threshold, min_delta_threshold, min_projected_return_pct):
    """
    Process a single ticker to analyze its option chains.
    
    Returns:
        tuple: (put_data, call_data) DataFrames for the ticker's options
    """
    put_data = []
    call_data = []
    
    try:
        # Fetch the ticker data
        logger.info(f"Fetching data for {ticker}")
        stock = yf.Ticker(ticker)

        # Get current stock price
        current_price = stock.history(period="1d")["Close"].iloc[-1]
        logger.info(f"Current price for {ticker}: ${current_price:.2f}")

        # Get available option expiration dates
        expiration_dates = stock.options
        
        if not expiration_dates:
            logger.warning(f"No option data available for {ticker}")
            return put_data, call_data

        logger.info(f"Analyzing {len(expiration_dates)} expiration dates: {expiration_dates}")
        # Price bounds for filtering
        put_lower_bound = current_price * (1 - percentage_range / 100)
        put_upper_bound = current_price * (1 - 5 / 100)

        call_upper_bound = current_price * (1 + percentage_range / 100)
        logger.info(f"Price bounds - Put lower: ${put_lower_bound:.2f}, Call upper: ${call_upper_bound:.2f}")

        # Limit to a few near-term expirations for speed (e.g., first 5)
        expiration_dates = expiration_dates[2:6]
        logger.debug(f"Available expiration dates: {expiration_dates}")
        for exp_date in expiration_dates:
            try:
                logger.info(f"Processing options for {ticker}, expiration: {exp_date}")
                # Fetch the option chain
                options = stock.option_chain(exp_date)
                calls = options.calls.copy()
                puts = options.puts.copy()

                # ...existing code for processing options...
                puts = puts[
                    (puts['volume'] > 0) &
                    (puts['openInterest'] >= 100)
                ]

                calls = calls[
                    (calls['volume'] > 0) &
                    (calls['openInterest'] >= 100)
                ]

                # Calculate midpoint premiums
                calls["midpoint"] = (calls["bid"] + calls["ask"]) / 2
                puts["midpoint"] = (puts["bid"] + puts["ask"]) / 2

                # Filter for OTM and within percentage range
                otm_puts = puts[(puts["strike"] < current_price) & (puts["strike"] >= put_lower_bound)& (puts["strike"] <= put_upper_bound)]
                otm_calls = calls[(calls["strike"] > current_price) & (calls["strike"] <= call_upper_bound)]

                logger.debug(f"Found {len(otm_puts)} OTM puts and {len(otm_calls)} OTM calls for {exp_date}")

                # Add expiration info
                otm_puts["expiration"] = exp_date
                otm_calls["expiration"] = exp_date

                # Calculate premium and return on capital
                otm_puts["premium_collected"] = otm_puts["midpoint"] * 100
                otm_calls["premium_collected"] = otm_calls["midpoint"] * 100

                otm_puts["capital_required"] = otm_puts["strike"] * 100
                otm_calls["capital_required"] = otm_calls["strike"] * 100

                #We skew the data to account for slippage
                otm_puts["return_on_capital_%"] = (
                    otm_puts["premium_collected"] / otm_puts["capital_required"]
                ) * 90
                otm_calls["return_on_capital_%"] = (
                    otm_calls["premium_collected"] / otm_calls["capital_required"]
                ) * 90

                days_to_expiration = (pd.to_datetime(exp_date) - pd.Timestamp.today()).days

                otm_puts["return_on_capital_per_anum_%"] = (
                    otm_puts["return_on_capital_%"] / days_to_expiration
                ) * 365
                otm_calls["return_on_capital_per_anum_%"] = (
                    otm_calls["return_on_capital_%"] / days_to_expiration
                ) * 365

                # Include IV
                otm_puts["implied_volatility"] = otm_puts["impliedVolatility"] * 100  # Convert to %
                otm_calls["implied_volatility"] = otm_calls["impliedVolatility"] * 100

                # Calculate risk-adjusted score (ROC / IV)
                otm_puts["risk_adjusted_score"] = otm_puts["return_on_capital_%"] / otm_puts["implied_volatility"]
                otm_calls["risk_adjusted_score"] = (
                    otm_calls["return_on_capital_%"] / otm_calls["implied_volatility"]
                )

                # For puts
                otm_puts["days_to_exp"] = (pd.to_datetime(otm_puts["expiration"]) - today).dt.days
                otm_puts["T"] = otm_puts["days_to_exp"] / 365  # time in years
                otm_puts["delta"] = bs_put_delta(
                    S=current_price,
                    K=otm_puts["strike"],
                    T=otm_puts["T"],
                    r=risk_free_rate,
                    sigma=otm_puts["implied_volatility"] / 100,
                )
                otm_puts["ticker"] = ticker
                otm_puts["delta_x_iv"] = otm_puts["delta"] * otm_puts["implied_volatility"]

                otm_calls["ticker"] = ticker
                otm_calls["days_to_exp"] = (pd.to_datetime(otm_calls["expiration"]) - today).dt.days
                otm_calls["T"] = otm_calls["days_to_exp"] / 365  # time in years
                otm_calls["delta"] = 0
                otm_calls["delta_x_iv"] = otm_calls["delta"] * otm_calls["implied_volatility"]

                # Keep relevant columns
                put_columns = [
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
                    "delta_x_iv"
                ]
                call_columns = [
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
                    "delta_x_iv"
                ]

                # Before appending, check and log the data size
                if not otm_puts.empty:
                    filtered_puts = otm_puts[put_columns]
                    logger.debug(f"{ticker} {exp_date}: Adding {len(filtered_puts)} put options")
                    put_data.append(filtered_puts)
                
                if not otm_calls.empty:
                    filtered_calls = otm_calls[call_columns]
                    logger.debug(f"{ticker} {exp_date}: Adding {len(filtered_calls)} call options")
                    call_data.append(filtered_calls)

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}, {exp_date}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing ticker {ticker}: {e}", exc_info=True)
    
    return put_data, call_data


def main():
    """
    Main function that executes the option analysis workflow.

    Process:
    1. Fetches current treasury yield for risk-free rate
    2. Gets stock price data for the specified ticker
    3. Analyzes option chains for multiple expiration dates
    4. Calculates metrics including return on capital and risk-adjusted scores
    5. Filters options based on risk-adjusted score and delta thresholds
    6. Exports results to CSV files
    """
    # Setup logging
    setup_logging()
    logger.info("Starting options analysis")

    # TODO
    # risk_free_rate = get_treasury_yield
    risk_free_rate = 0.045
    logger.info(f"Using risk-free rate: {risk_free_rate}")
    today = dt.datetime.today()

    # Parameters for AAPL (Example)
    tickers = get_sp500_tickers_from_file_or_web()
    percentage_range = 15

    #logger.info(f"Analyzing ticker: {ticker} with {percentage_range}% range")

    # Filter puts and calls by minimum risk-adjusted score threshold
    min_risk_score = 0.04
    max_delta_threshold = 0.75
    min_delta_threshold = 0.30

    min_projected_return_pct = 2

    # Store premiums for plotting
    all_put_data = []
    all_call_data = []

    # Use ThreadPoolExecutor to process multiple tickers in parallel
    max_workers = 5  # Reduced from 10 to avoid rate limiting
    
    logger.info(f"Processing {len(tickers)} tickers in parallel with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each ticker
        future_to_ticker = {
            executor.submit(
                process_ticker, 
                ticker, 
                risk_free_rate, 
                percentage_range, 
                today, 
                min_risk_score, 
                max_delta_threshold, 
                min_delta_threshold, 
                min_projected_return_pct
            ): ticker for ticker in tickers
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
                    logger.debug(f"No data returned for {ticker}")
                    continue
                    
                if put_data:  # Check if not empty
                    all_put_data.extend(put_data)
                if call_data:  # Check if not empty
                    all_call_data.extend(call_data)
                logger.info(f"Completed processing for {ticker} ({completed}/{total})")
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")

    # Rest of the analysis with the collected data
    if not all_put_data and not all_call_data:
        logger.error("No valid option data collected")
        return
        
    # Combine into DataFrames
    put_df = pd.DataFrame()
    call_df = pd.DataFrame()
    
    if all_put_data:
        try:
            # Count actual dataframes with rows
            non_empty_dfs = sum(1 for df in all_put_data if not df.empty)
            total_rows = sum(len(df) for df in all_put_data)
            logger.info(f"Combining {len(all_put_data)} put dataframes ({non_empty_dfs} non-empty with {total_rows} total rows)")
            
            # Debug: Look at a sample of dataframes before combining
            if len(all_put_data) > 0:
                for i, df in enumerate(all_put_data[:5]):
                    logger.debug(f"Sample df {i}: shape={df.shape}, columns={list(df.columns)}")
                    if not df.empty:
                        logger.debug(f"First row: {df.iloc[0].to_dict()}")
                        
            # Combine the dataframes
            put_df = pd.concat(all_put_data, ignore_index=True)
            logger.info(f"Combined put dataframe has {len(put_df)} rows")
            
            # Check for column issues or missing data
            if len(put_df) < total_rows:
                logger.warning(f"Data loss during concat! Expected {total_rows} rows but got {len(put_df)}")
                # Check for column mismatches which could cause issues
                column_sets = [set(df.columns) for df in all_put_data if not df.empty]
                if len(column_sets) > 1:
                    all_columns = set.union(*column_sets)
                    column_differences = [all_columns - cols for cols in column_sets]
                    if any(column_differences):
                        logger.warning(f"Column mismatches detected: {column_differences}")
        except Exception as e:
            logger.error(f"Error combining put data: {e}", exc_info=True)
            # Try a different approach to combine
            logger.info("Attempting alternative combination approach...")
            try:
                # Alternative approach: combine dataframes one by one
                put_df = pd.DataFrame()
                for df in all_put_data:
                    if not df.empty:
                        if put_df.empty:
                            put_df = df.copy()
                        else:
                            # Check columns match before combining
                            if set(df.columns) == set(put_df.columns):
                                put_df = pd.concat([put_df, df], ignore_index=True)
                            else:
                                logger.warning(f"Skipping dataframe with mismatched columns: {set(df.columns)} vs {set(put_df.columns)}")
                logger.info(f"Alternative combination resulted in {len(put_df)} rows")
            except Exception as e2:
                logger.error(f"Alternative combination also failed: {e2}", exc_info=True)
    
    if all_call_data:
        try:
            call_df = pd.concat(all_call_data, ignore_index=True)
        except Exception as e:
            logger.error(f"Error combining call data: {e}", exc_info=True)

    if put_df.empty:
        logger.warning("No valid put option data available after concat")
    else:
        # Apply filters one by one with logging to track data loss
        logger.info(f"Starting with {len(put_df)} put options")
        
        filtered_put_df = put_df[put_df["risk_adjusted_score"] >= min_risk_score]
        logger.info(f"After risk score filter: {len(filtered_put_df)} rows")
        
        filtered_put_df = filtered_put_df[filtered_put_df["delta"] <= -min_delta_threshold]
        logger.info(f"After min delta filter: {len(filtered_put_df)} rows")
        
        filtered_put_df = filtered_put_df[filtered_put_df["delta"] >= -max_delta_threshold]
        logger.info(f"After max delta filter: {len(filtered_put_df)} rows")
        
        filtered_put_df = filtered_put_df[filtered_put_df["return_on_capital_%"] >= min_projected_return_pct]
        logger.info(f"After return filter: {len(filtered_put_df)} rows")
        
        filtered_put_df = filtered_put_df[filtered_put_df["return_on_capital_per_anum_%"] >= 30]
        logger.info(f"After annualized return filter: {len(filtered_put_df)} rows")

        if filtered_put_df.empty:
            logger.warning("All rows filtered out! No data remains after applying filters.")
            # Create a sample row to analyze what's happening
            if not put_df.empty:
                sample = put_df.iloc[0:5]
                logger.info(f"Sample data:\n{sample[['ticker', 'strike', 'delta', 'risk_adjusted_score', 'return_on_capital_%', 'return_on_capital_per_anum_%']]}")
        else:
            filtered_puts = assign_composite_score(filtered_put_df)
            top_25_puts = filtered_puts.sort_values(by='composite_score', ascending=False).head(25)

            # Export to CSV
            output_file = "otm_puts.csv"
            top_25_puts.to_csv(output_file, index=False)
            logger.info(f"Exported {len(top_25_puts)} filtered put options to {output_file}")

    if call_df.empty:
        logger.warning("No valid call option data available")
    else:
        filtered_call_df = call_df[call_df["risk_adjusted_score"] >= min_risk_score].reset_index(drop=True)
        # Process call options as needed


if __name__ == "__main__":
    main()

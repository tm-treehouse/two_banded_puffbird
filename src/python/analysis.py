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
    ticker = "AAPL"
    percentage_range = 10
    logger.info(f"Analyzing ticker: {ticker} with {percentage_range}% range")

    # Filter puts and calls by minimum risk-adjusted score threshold
    min_risk_score = 0.04
    delta_threshold = 0.50

    # Fetch the ticker data
    logger.info(f"Fetching data for {ticker}")
    stock = yf.Ticker(ticker)

    # Get current stock price
    current_price = stock.history(period="1d")["Close"].iloc[-1]
    logger.info(f"Current price for {ticker}: ${current_price:.2f}")

    # Get available option expiration dates
    expiration_dates = stock.options
    logger.debug(f"Available expiration dates: {expiration_dates}")

    # Price bounds for filtering
    put_lower_bound = current_price * (1 - percentage_range / 100)
    call_upper_bound = current_price * (1 + percentage_range / 100)
    logger.info(f"Price bounds - Put lower: ${put_lower_bound:.2f}, Call upper: ${call_upper_bound:.2f}")

    # Limit to a few near-term expirations for speed (e.g., first 5)
    expiration_dates = expiration_dates[:5]

    # Store premiums for plotting
    all_put_data = []
    all_call_data = []

    logger.info(f"Analyzing {len(expiration_dates)} expiration dates: {expiration_dates}")

    for exp_date in expiration_dates:
        try:
            logger.info(f"Processing options for expiration: {exp_date}")
            # Fetch the option chain
            options = stock.option_chain(exp_date)
            calls = options.calls.copy()
            puts = options.puts.copy()
            # Calculate midpoint premiums
            calls["midpoint"] = (calls["bid"] + calls["ask"]) / 2
            puts["midpoint"] = (puts["bid"] + puts["ask"]) / 2

            # Filter for OTM and within percentage range
            otm_puts = puts[(puts["strike"] < current_price) & (puts["strike"] >= put_lower_bound)]
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

            otm_puts["return_on_capital_%"] = (
                otm_puts["premium_collected"] / otm_puts["capital_required"]
            ) * 100
            otm_calls["return_on_capital_%"] = (
                otm_calls["premium_collected"] / otm_calls["capital_required"]
            ) * 100

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

            otm_calls["days_to_exp"] = (pd.to_datetime(otm_calls["expiration"]) - today).dt.days
            otm_calls["T"] = otm_calls["days_to_exp"] / 365  # time in years
            otm_calls["delta"] = 0
            # Unused
            # bs_call_delta(
            #    S=current_price,
            #    K=otm_puts['strike'],
            #    T=otm_puts['T'],
            #    r=risk_free_rate,
            #    sigma=otm_calls['implied_volatility'] / 100
            # )

            # Keep relevant columns
            put_columns = [
                "strike",
                "expiration",
                "midpoint",
                "premium_collected",
                "capital_required",
                "return_on_capital_%",
                "implied_volatility",
                "risk_adjusted_score",
                "delta",
                "return_on_capital_per_anum_%",
            ]
            call_columns = [
                "strike",
                "expiration",
                "midpoint",
                "premium_collected",
                "capital_required",
                "return_on_capital_%",
                "implied_volatility",
                "risk_adjusted_score",
                "delta",
                "return_on_capital_per_anum_%",
            ]

            all_put_data.append(otm_puts[put_columns])
            all_call_data.append(otm_calls[call_columns])

        except Exception as e:
            logger.error(f"Error fetching data for {exp_date}: {e}", exc_info=True)

    # Combine into DataFrames
    put_df = pd.concat(all_put_data, ignore_index=True)
    call_df = pd.concat(all_call_data, ignore_index=True)

    filtered_put_df = put_df[put_df["risk_adjusted_score"] >= min_risk_score].reset_index(drop=True)
    filtered_put_df = put_df[put_df["delta"] >= -delta_threshold].reset_index(drop=True)

    filtered_call_df = call_df[call_df["risk_adjusted_score"] >= min_risk_score].reset_index(drop=True)

    # Export to CSV
    output_file = "otm_puts.csv"
    filtered_put_df.to_csv(output_file, index=False)
    logger.info(f"Exported {len(filtered_put_df)} filtered put options to {output_file}")


if __name__ == "__main__":
    main()

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
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm

pd.options.mode.chained_assignment = None  # default='warn'


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
    # date = latest_entry.find(".//td[@class='date']").text # Unused
    yield_10y = latest_entry.find(".//td[@class='GS10']").text
    print(yield_10y)  # Added print statement to display yield
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
    #    print("Error")
    #    return np.nan  # invalid parameters

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    call_delta = norm.cdf(d1)
    print(call_delta)
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
    # TODO
    # risk_free_rate = get_treasury_yield
    risk_free_rate = 0.045
    today = dt.datetime.today()

    # Parameters for AAPL (Example)
    ticker = "AAPL"
    percentage_range = 10

    # Filter puts and calls by minimum risk-adjusted score threshold
    min_risk_score = 0.04

    delta_threshold = 0.50

    # Fetch the ticker data
    stock = yf.Ticker(ticker)

    # Get current stock price
    current_price = stock.history(period="1d")["Close"].iloc[-1]

    # Get available option expiration dates
    expiration_dates = stock.options

    # Price bounds for filtering
    put_lower_bound = current_price * (1 - percentage_range / 100)
    call_upper_bound = current_price * (1 + percentage_range / 100)

    # Limit to a few near-term expirations for speed (e.g., first 5)
    expiration_dates = expiration_dates[:5]

    # Store premiums for plotting
    all_put_data = []
    all_call_data = []

    print(expiration_dates)

    for exp_date in expiration_dates:
        try:
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

            print(otm_puts)

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

        ##
        ##        # Append to combined data
        ##        all_put_data.append(otm_puts[['strike', 'midpoint', 'expiration']])
        ##        all_call_data.append(otm_calls[['strike', 'midpoint', 'expiration']])
        ##
        except Exception as e:
            print(f"Error fetching data for {exp_date}: {e}")
    ##
    ### Combine into DataFrames
    put_df = pd.concat(all_put_data, ignore_index=True)
    call_df = pd.concat(all_call_data, ignore_index=True)

    filtered_put_df = put_df[put_df["risk_adjusted_score"] >= min_risk_score].reset_index(drop=True)
    filtered_put_df = put_df[put_df["delta"] >= -delta_threshold].reset_index(drop=True)

    filtered_call_df = call_df[call_df["risk_adjusted_score"] >= min_risk_score].reset_index(drop=True)

    ##
    ### Optional: display a preview
    ##print("Put Options Data:")
    ##print(put_df.head())
    ##
    ##print("\nCall Options Data:")
    ##print(call_df.head())
    ##
    ### Optional: export to CSV
    filtered_put_df.to_csv("otm_puts.csv", index=False)
    ### call_df.to_csv('otm_calls.csv', index=False)


if __name__ == "__main__":
    main()

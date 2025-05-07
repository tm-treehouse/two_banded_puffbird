#!/root/venv/bin/python3
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for AAPL (Example)
ticker = "AAPL"
percentage_range = 10

# Fetch the ticker data
stock = yf.Ticker(ticker)

# Get current stock price
current_price = stock.history(period='1d')['Close'].iloc[-1]

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
        calls['midpoint'] = (calls['bid'] + calls['ask']) / 2
        puts['midpoint'] = (puts['bid'] + puts['ask']) / 2

        # Filter for OTM and within percentage range
        otm_puts = puts[(puts['strike'] < current_price) & (puts['strike'] >= put_lower_bound)]
        otm_calls = calls[(calls['strike'] > current_price) & (calls['strike'] <= call_upper_bound)]

        print(otm_puts)
##
##        # Add expiration info
##        otm_puts['expiration'] = exp_date
##        otm_calls['expiration'] = exp_date
##
##        # Append to combined data
##        all_put_data.append(otm_puts[['strike', 'midpoint', 'expiration']])
##        all_call_data.append(otm_calls[['strike', 'midpoint', 'expiration']])
##
    except Exception as e:
        print(f"Error fetching data for {exp_date}: {e}")
##
### Combine into DataFrames
##put_df = pd.concat(all_put_data, ignore_index=True)
##call_df = pd.concat(all_call_data, ignore_index=True)
##
### Optional: display a preview
##print("Put Options Data:")
##print(put_df.head())
##
##print("\nCall Options Data:")
##print(call_df.head())
##
### Optional: export to CSV
### put_df.to_csv('otm_puts.csv', index=False)
### call_df.to_csv('otm_calls.csv', index=False)

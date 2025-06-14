#!/usr/bin/env python3

from options_analysis.option_simulator import option_simulator

# Create an instance of the AmericanOptionSimulator for a Sell Put Option
simulator = option_simulator(
    ticker="MSFT",             # Stock ticker
    option_type="put",         # Option type: "call" or "put"
    side="sell",               # Side: "buy" or "sell"
    strike=112,                # Option strike price
    expiry="2025-06-22",       # Expiry date in "YYYY-MM-DD"
    risk_free_rate=0.03,       # Risk-free rate (default 3%)
    n_jobs=4                   # Number of CPU cores for parallel processing
)

## Price the option using Monte Carlo simulation
    #option_price = 100 #simulator.price(n_paths=100_000, n_jobs=4)
    #print(f"Price of the Sell Put Option: ${option_price:.2f}")

# Simulate the return distribution for the sell-side of the option
#returns = simulator._simulate_paths(n_paths=1)

## Estimate the Greeks (Delta, Gamma, Vega, Theta, Rho)
#greeks = simulator.estimate_greeks()
#print(f"Estimated Greeks: {greeks}")
#
## Plot the results and save all plots to a PDF
#simulator.plot_all_to_pdf("MSFT_option_report.pdf")

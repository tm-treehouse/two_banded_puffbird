import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

class AmericanOptionSimulator:
    def __init__(self, ticker, option_type, side, strike, expiry, risk_free_rate=0.05, n_jobs=-1):
        """
        Initialize the simulator with option parameters.

        Args:
            ticker (str): Stock ticker symbol.
            option_type (str): 'call' or 'put'.
            side (str): 'buy' or 'sell'.
            strike (float): Option strike price.
            expiry (str): Expiration date in YYYY-MM-DD.
            risk_free_rate (float): Annualized risk-free rate.
            n_jobs (int): Number of CPU cores to use for parallel processing.
                          Use -1 to use all available cores.
        """
        # Store initial parameters
        self.ticker = ticker.upper()
        self.option_type = option_type.lower()
        self.side = side.lower()
        self.K = strike
        self.expiry = pd.to_datetime(expiry)
        self.r = risk_free_rate
        self.n_jobs = n_jobs  # Number of jobs for parallel processing

        # Get historical stock data and calibrate volatility
        self._fetch_data()
        self._calibrate_sigma()

        # Calculate time to maturity in trading days and years
        self.T_days = np.busday_count(datetime.today().date(), self.expiry.date())
        self.T = self.T_days / 252  # Time to maturity in years (assumes 252 trading days per year)
        self.dt = 1 / 252  # Daily step for simulation

    def _fetch_data(self):
        """Download and store historical stock prices."""
        # Fetch adjusted closing prices from Yahoo Finance
        data = yf.download(self.ticker, period="6mo", progress=False)
        df = pd.DataFrame(data)

        #Get closing price, convert to list
        close_price = df['Close']
        price_series = close_price["MSFT"]
        self.prices = price_series.tolist()


        self.S0 = self.prices[-1]  # Current stock price

    def _calibrate_sigma(self):
        """Estimate volatility (sigma) from historical log returns."""
        # Calculate log returns of stock prices
        log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        # Estimate volatility (annualized)
        self.sigma = log_returns.std() * np.sqrt(252)

    def _simulate_paths(self, S0=None, sigma=None, r=None, n_paths=100_000, n_steps=None, seed=None):
        """Simulate stock price paths using geometric Brownian motion."""
        if seed:
            np.random.seed(seed)
        if n_steps is None:
            n_steps = self.T_days  # Default to number of trading days in option's life
        S0 = S0 or self.S0
        sigma = sigma or self.sigma
        r = r or self.r

        # Generate random standard normal variables for the paths
        Z = np.random.standard_normal((n_paths, n_steps))

        # Calculate drift and diffusion for stock price movement
        drift = (r - 0.5 * sigma ** 2) * self.dt
        diffusion = sigma * np.sqrt(self.dt) * Z

        # Log-normal random walk for stock price paths
        log_returns = drift + diffusion
        log_paths = np.cumsum(log_returns, axis=1)
        log_paths = np.hstack((np.zeros((n_paths, 1)), log_paths))

        # Return simulated paths in terms of actual stock price
        return S0 * np.exp(log_paths)

    def _intrinsic(self, paths):
        """Calculate intrinsic value of the option for given paths."""
        if self.option_type == "call":
            return np.maximum(paths - self.K, 0)  # Call option payoff
        elif self.option_type == "put":
            return np.maximum(self.K - paths, 0)  # Put option payoff
        else:
            raise ValueError("Invalid option type")

    def _least_squares_valuation(self, paths_chunk):
        """Least Squares Monte Carlo (LSM) valuation for early exercise."""
        n_paths, n_steps = paths_chunk.shape
        discount = np.exp(-self.r * self.dt)  # Discount factor per step
        intrinsic = self._intrinsic(paths_chunk)  # Intrinsic values at expiry
        cashflows = intrinsic[:, -1]  # Initial cashflows (at maturity)

        # Backward induction for early exercise
        for t in range(n_steps - 2, 0, -1):
            itm = intrinsic[:, t] > 0  # In-the-money paths
            X = paths_chunk[itm, t].reshape(-1, 1)  # Stock price at time t
            Y = cashflows[itm] * discount  # Discounted cashflows at time t
            if len(X) == 0:
                continue
            model = LinearRegression().fit(X, Y)  # Least squares regression
            continuation = model.predict(X)  # Continuation value (no exercise)
            exercise = intrinsic[itm, t] > continuation  # Decide whether to exercise
            cashflows[itm] = np.where(exercise, intrinsic[itm, t], cashflows[itm] * discount)  # Exercise if profitable

        # Return the final cashflows (after discounting back to the present)
        return cashflows * discount

    def price(self, n_paths=100_000, n_jobs=None):
        """Calculate the Monte Carlo price of the option."""
        # Simulate stock price paths
        paths = self._simulate_paths(n_paths=n_paths)

        # Split the paths into chunks for parallel processing
        chunks = np.array_split(paths, max(1, n_jobs if n_jobs is not None else self.n_jobs))

        # Perform the least squares Monte Carlo valuation in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._least_squares_valuation)(chunk) for chunk in chunks
        )

        # Combine results from all chunks
        values = np.concatenate(results)

        # Option price is the average value of all simulated paths
        raw_price = np.mean(values)

        # If selling the option, return the negative of the price
        return raw_price if self.side == "buy" else -raw_price

    def simulate_returns(self, n_paths=100_000, margin=None, show_plot=True):
        """Simulate return distribution for option position."""
        paths = self._simulate_paths(n_paths=n_paths)  # Simulate paths

        # Calculate final stock prices and intrinsic values at expiry
        ST = paths[:, -1]
        intrinsic = self._intrinsic(paths)
        payoffs = intrinsic[:, -1]

        # Calculate option premium (price paid for option)
        premium = self.price(n_paths=100_000, n_jobs=1)

        # Profit or loss for the option (buy/sell side)
        pnl = payoffs - premium if self.side == "buy" else premium - payoffs

        # Calculate returns relative to margin (strike price or underlying price)
        margin = margin or (self.K if self.option_type == "put" else self.S0)
        returns = pnl / margin

        # Calculate statistics on the returns
        stats = {
            "Mean Return": np.mean(returns),
            "Std Dev": np.std(returns),
            "VaR (5%)": np.percentile(returns, 5),
            "CVaR (5%)": returns[returns <= np.percentile(returns, 5)].mean()
        }

        # Plot the return distribution if show_plot is True
        if show_plot:
            plt.hist(returns, bins=100, color="skyblue", edgecolor="black")
            plt.axvline(stats["VaR (5%)"], color="red", linestyle="--", label="VaR (5%)")
            plt.title(f"Return Distribution: {self.side.capitalize()} {self.option_type.capitalize()}")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True)
            plt.show()

        return {"stats": stats, "returns": returns}

    def estimate_greeks(self, bump=1.0, sigma_bump=0.01):
        """Estimate Greeks using finite differences."""
        base = self.price(n_paths=50_000, n_jobs=1)

        # Delta and Gamma (first and second derivatives with respect to stock price)
        up = self._simulate_bumped_price(self.S0 + bump)
        down = self._simulate_bumped_price(self.S0 - bump)
        delta = (up - down) / (2 * bump)
        gamma = (up - 2 * base + down) / (bump ** 2)

        # Vega (derivative with respect to volatility)
        vega_up = self._simulate_bumped_sigma(self.sigma + sigma_bump)
        vega = (vega_up - base) / sigma_bump

        # Theta (1 day decay)
        self.T_days -= 1
        self.T = self.T_days / 252
        theta = (self.price(n_paths=50_000, n_jobs=1) - base)
        self.T_days += 1
        self.T = self.T_days / 252

        # Rho (derivative with respect to interest rate)
        r_up = self._simulate_bumped_r(self.r + 0.01)
        rho = (r_up - base) / 0.01

        # Return Greeks as a dictionary
        return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

    def _simulate_bumped_price(self, S_bumped):
        """Simulate price for bumped underlying value (for Delta/Gamma)."""
        paths = self._simulate_paths(S0=S_bumped, n_paths=30_000)
        return np.mean(np.concatenate(
            Parallel(n_jobs=4)(delayed(self._least_squares_valuation)(chunk) for chunk in np.array_split(paths, 4))
        ))

    def _simulate_bumped_sigma(self, sigma_bumped):
        """Simulate price for bumped volatility value (for Vega)."""
        paths = self._simulate_paths(sigma=sigma_bumped, n_paths=30_000)
        return np.mean(np.concatenate(
            Parallel(n_jobs=4)(delayed(self._least_squares_valuation)(chunk) for chunk in np.array_split(paths, 4))
        ))

    def _simulate_bumped_r(self, r_bumped):
        """Simulate price for bumped interest rate value (for Rho)."""
        paths = self._simulate_paths(r=r_bumped, n_paths=30_000)
        return np.mean(np.concatenate(
            Parallel(n_jobs=4)(delayed(self._least_squares_valuation)(chunk) for chunk in np.array_split(paths, 4))
        ))

    def plot_all_to_pdf(self, filename="option_simulation.pdf"):
        """Export all plots to a PDF file."""
        with PdfPages(filename) as pdf:
            # Plot the return distribution graph
            plt.hist(self.simulate_returns()["returns"], bins=100, color="skyblue", edgecolor="black")
            plt.title(f"Return Distribution: {self.side.capitalize()} {self.option_type.capitalize()}")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            pdf.savefig()
            plt.close()

            # Include Greek estimation graph
            greek_values = self.estimate_greeks()
            labels = list(greek_values.keys())
            values = list(greek_values.values())

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_title("Greek Estimation")
            ax.set_ylabel("Value")
            pdf.savefig()
            plt.close()



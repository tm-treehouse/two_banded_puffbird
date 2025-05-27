import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json


class option_simulator:
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
        self.strike_price = strike
        self.expiry = pd.to_datetime(expiry)
        self.risk_free_rate = risk_free_rate
        self.n_jobs = n_jobs  # Number of jobs for parallel processing

        # Get historical stock data and calibrate volatility
        self._fetch_data()
        self._calculate_sigma()

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
        price_series_list = price_series.tolist()
        price_data_frame = pd.DataFrame(price_series.tolist())e
        self.prices = price_data_frame.to_numpy()
        self.current_stock_price = price_series_list[-1]  # Current stock price


    def _calculate_sigma(self,data_set):
        # Calculate mean
        data_set_mean = np.mean(data_set)/len(data_set)


        deviation = np.array(data_set)-data_set_mean
        squared = np.square(deviation)
        total_sum = sum(squared)
        variance = total_sum / len(data_set)
        sigma = numpy.sqrt(variance)
        return sigma


        """Estimate volatility (sigma) from historical log returns."""
        # Calculate log returns of stock prices
        log_returns = np.log(self.prices / self.prices.shift(1)).dropna()

        # Estimate volatility (annualized)
        #std deviation across num trading days
        self.sigma = log_returns.std() * np.sqrt(252)


    def _simulate_stock_paths(self, n_paths=100_000):
        """Simulate stock price paths using geometric Brownian motion."""
        current_stock_price = self.current_stock_price
        n_steps = self.T_days  # Default to number of trading days in option's life

        #The Wiener process.
        #Creating random fluctuations within the stock price.
        # Z = np.random.standard_normal((n_paths, n_steps))
        Z = np.random.standard_normal(n_steps)

        # Calculate drift and diffusion for stock price movement
        #Stock drive is the average expected rate of return of the stock.
        drift = (self.risk_free_rate - 0.5 * self.sigma ** 2) * self.dt

        #Volatility of the stock using sigma
        predicted_volatility = self.sigma * np.sqrt(self.dt)
        diffusion = predicted_volatility * Z

        # Log-normal random walk for stock price paths
        log_returns = drift + diffusion

        log_paths = np.cumsum(log_returns, axis=1)

        #log_paths = np.cumsum(log_returns)
        #log_paths = np.hstack((np.zeros((n_paths, 1)), log_paths))

        foo = current_stock_price * np.exp(log_paths)
        np.savetxt("foo.csv",foo,delimiter=",")
        # Return simulated paths in terms of actual stock price
        return current_stock_price * np.exp(log_paths)




    #Takes the simulated stock movement, and estimates the what the option price should be.
    def simulate_option_returns(self, n_paths=100_000, margin=None, show_plot=True):
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
        margin = margin or (self.strike_price if self.option_type == "put" else self.S0)
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

    def _intrinsic(self, paths):
        """Calculate intrinsic value of the option for given paths."""
        if self.option_type == "call":
            return np.maximum(paths - self.strike_price, 0)  # Call option payoff
        elif self.option_type == "put":
            return np.maximum(self.strike_price - paths, 0)  # Put option payoff
        else:
            raise ValueError("Invalid option type")

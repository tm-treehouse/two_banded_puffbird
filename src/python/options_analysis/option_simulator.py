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
        paths = self.geometric_brownian_motion(self.current_stock_price,self.historical_returns_mean,self.historical_returns_std_deviation,1,252,1)
        paths = paths.flatten()
        plt.plot(paths)
        plt.savefig('random_walks.png')
        print(paths)

    def _fetch_data(self):
        """Download and store historical stock prices."""
        # Fetch adjusted closing prices from Yahoo Finance
        data = yf.download(self.ticker, period="6mo", progress=False)
        df = pd.DataFrame(data)

        #Get closing price, convert to list
        close_price = df['Close']
        price_series = close_price["MSFT"]
        price_series_list = price_series.tolist()
        price_data_frame = pd.DataFrame(price_series.tolist())
        self.prices = price_data_frame.to_numpy()
        self.current_stock_price = price_series_list[-1]  # Current stock price



    #Calculates the statistical sigma of a data set
    #Aka std devitaion

    def _calculate_sigma(self):
        self.prices = self.prices.reshape(1,-1)
        self.prices = self.prices.flatten()

        data_set = pd.Series(self.prices)
        data_set = data_set.pct_change(1).dropna()
        print(data_set)

        # Calculate mean
        self.historical_returns_mean = np.mean(data_set)

        self.historical_returns_std_deviation = np.std(data_set)


    def geometric_brownian_motion(self,S0, mu, sigma, T, N, num_simulations):
        """
        Simulates Geometric Brownian Motion paths.

        Args:
            S0: Initial price.
            mu: Mean return (drift).
            sigma: Volatility.
            T: Time horizon.
            N: Number of time steps.
            num_simulations: Number of simulation paths.

        Returns:
            A numpy array of shape (N + 1, num_simulations) containing the simulated paths.
        """
        dt = T / N
        paths = np.zeros((N + 1, num_simulations))
        paths[0] = S0
        for t in range(1, N + 1):
            rand = np.random.normal(0,1,num_simulations)
            paths[t] = paths[t - 1] * np.exp((mu - (0.5 * sigma ** 2)) * dt + sigma * np.sqrt(dt) * rand)
        return paths

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
    def __init__(self, ticker, n_jobs=-1):

        # Store initial parameters
        self.ticker = ticker.upper()
        self.n_jobs = n_jobs  # Number of jobs for parallel processing

        # Get historical stock data and calibrate volatility
        self._fetch_stock_close_data(ticker)
        self._calculate_sigma()

        # Calculate time to maturity in trading days and years
        paths = self.geometric_brownian_motion(self.current_stock_price,self.historical_returns_mean,self.historical_returns_std_deviation,1,252,1000)
        paths = paths.T
        for rows in paths:
            plt.plot(rows)
        plt.savefig('random_walks.png')

    def _fetch_stock_close_data(self,ticker):
        """Download and store historical stock prices."""
        # Fetch adjusted closing prices from Yahoo Finance
        data = yf.download(self.ticker, period="6mo", progress=False)
        df = pd.DataFrame(data)

        #Get closing price, convert to list
        close_price = df['Close']
        price_series = close_price[ticker]
        price_series_list = price_series.tolist()
        price_data_frame = pd.DataFrame(price_series.tolist())
        print(price_data_frame.size)
        self.prices = price_data_frame.to_numpy()
        self.prices = self.prices.reshape(1,-1)
        self.prices = self.prices.flatten()
        self.prices = pd.Series(self.prices)
        self.current_stock_price = price_series_list[-1]  # Current stock price



    #Calculates the statistical sigma of a data set
    #Aka std devitaion

    def _calculate_sigma(self):

        data_set = self.prices.pct_change(1).dropna()

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

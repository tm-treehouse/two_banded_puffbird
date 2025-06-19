import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
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
        self._calculate_model_params()

        # Calculate time to maturity in trading days and years
        paths = self.geometric_brownian_motion(self.current_stock_price,self.historical_returns_mean,self.historical_returns_std_deviation,1,252,500000)
        paths = paths.T
        pct = self.calculate_price_value_probability(paths,"put", 19,"2025 7 21")
        #for rows in paths:
        #    plt.plot(rows)
        #plt.savefig('random_walks.png')

    def _fetch_stock_close_data(self,ticker):
        """Download and store historical stock prices."""
        # Fetch adjusted closing prices from Yahoo Finance
        data = yf.download(self.ticker, period="1mo", progress=False)
        df = pd.DataFrame(data)

        #Get closing price, convert to list
        close_price = df['Close']
        price_series = close_price[ticker]
        price_series_list = price_series.tolist()
        price_data_frame = pd.DataFrame(price_series.tolist())
        self.prices = price_data_frame.to_numpy()
        self.prices = self.prices.reshape(1,-1)
        self.prices = self.prices.flatten()
        self.prices = pd.Series(self.prices)
        self.current_stock_price = price_series_list[-1]  # Current stock price



    #Calculates the statistical sigma of a data set
    #Aka std devitaion

    def _calculate_model_params(self):

        self.returns = self.prices.pct_change(1).dropna()
        self.returns_pct = self.returns * 100

        # Calculate mean
        self.historical_returns_mean = np.mean(returns)
        self.historical_returns_std_deviation = np.std(returns)

    def _calculate_time_based_volatility(self):
        model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
        res = model.fit(disp='off')
        print(res.summary())

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



    def calculate_price_value_probability(self, dataset, option_type, strike_price, target_date_string):
            option_type.lower()
            target_date = datetime.strptime(target_date_string, "%Y %m %d")
        #Find the column associated w/ the date
            today = date.today()
            day_delta = np.busday_count(today.strftime('%Y-%m-%d'),
                      target_date.strftime('%Y-%m-%d'))
        #Consumme
            slice_at_date = np.take(dataset,indices=day_delta,axis=1)
            at_or_greater = 0
            for number in slice_at_date:
                if option_type == "call":
                    if number >= strike_price :
                        at_or_greater += 1
                elif option_type == "put":
                    if number >= strike_price :
                        at_or_greater += 1
            total_elements = len(slice_at_date)
            percentage = (at_or_greater / total_elements) * 100
            print(percentage)
            return percentage

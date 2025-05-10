#!/usr/bin/env python3
"""
Black-Scholes model implementation for options pricing and greeks calculation.
"""

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger("options_analysis")


class BlackScholesCalculator:
    """
    Black-Scholes calculator for option pricing and greeks calculation.

    This class provides methods to calculate option prices and greeks
    using the Black-Scholes model for European options.
    """

    @staticmethod
    def put_delta(S, K, T, r, sigma):
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
        # Input validation
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            logger.error("Invalid parameters for put_delta")
            return np.nan  # invalid parameters

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1

    @staticmethod
    def call_delta(S, K, T, r, sigma):
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
        # Input validation
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            logger.error("Invalid parameters for call_delta")
            return np.nan  # invalid parameters

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        call_delta = norm.cdf(d1)
        logger.debug(f"Call delta: {call_delta}")
        return call_delta

    @staticmethod
    def put_price(S, K, T, r, sigma):
        """
        Calculate Black-Scholes price for European put options.

        Parameters:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (in years)
            r (float): Risk-free rate (as decimal)
            sigma (float): Implied volatility (as decimal)

        Returns:
            float: Put option price
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            logger.error("Invalid parameters for put_price")
            return np.nan

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        """
        Calculate Black-Scholes price for European call options.

        Parameters:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration (in years)
            r (float): Risk-free rate (as decimal)
            sigma (float): Implied volatility (as decimal)

        Returns:
            float: Call option price
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            logger.error("Invalid parameters for call_price")
            return np.nan

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

#!/usr/bin/env python3
"""
----------------------------------------------------
Title:      main.py
License:    agpl-3.0
Author:     TM, SC
Created on: 2025-05-10
----------------------------------------------------

Description:
Entry point for the financial options analysis tool.
This script provides a command-line interface to the options_analysis package.
The tool calculates risk-adjusted returns for put and call options based on
current market data, fetches stock prices, treasury yields, and option chains,
then applies Black-Scholes modeling to identify potentially favorable option trades.

----------------------------------------------------

Usage Examples:

1. Show help and available options:
   $ uv run src/python/main.py --help

2. Analyze specific ticker symbols:
   $ uv run src/python/main.py --tickers AAPL,MSFT,GOOGL

3. Analyze S&P 500 stocks with custom parameters:
   $ uv run src/python/main.py --max-workers 4 --percentage-range 20 --min-delta 0.35

4. Run with detailed logging:
   $ uv run src/python/main.py --tickers AAPL --log-level DEBUG

5. Force refresh of S&P 500 tickers list:
   $ uv run src/python/main.py --refresh-tickers

Parameters:
  --tickers: Comma-separated list of ticker symbols (default: S&P 500)
  --refresh-tickers: Force refresh of S&P 500 tickers from web
  --max-workers: Number of parallel threads (default: 3)
  --percentage-range: Price range percentage from current price (default: 15)
  --min-delta: Minimum absolute delta threshold (default: 0.30)
  --max-delta: Maximum absolute delta threshold (default: 0.75)
  --min-return: Minimum return on capital percentage (default: 2)
  --min-annual-return: Minimum annualized return percentage (default: 30)
  --min-risk-score: Minimum risk-adjusted score (default: 0.04)
  --log-level: Logging level (default: INFO)

----------------------------------------------------
How to run this script:
cd two_banded_puffbird
uv run src/python/main.py [options]
----------------------------------------------------
"""

import argparse
import logging

from options_analysis.market_data import YahooFinanceProvider
from options_analysis.option_analysis import OptionAnalyzer, OptionsAnalysisRunner

# Import modules from the options_analysis package
from options_analysis.utils import ensure_output_directory, setup_logging


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Options Analysis Tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--tickers",
        help="Comma-separated list of ticker symbols to analyze. If not provided, will use S&P 500 stocks.",
        type=str,
    )

    parser.add_argument(
        "--refresh-tickers", help="Force refresh of S&P 500 tickers from web", action="store_true"
    )

    parser.add_argument(
        "--max-workers", help="Maximum number of parallel worker threads", type=int, default=3
    )

    parser.add_argument(
        "--percentage-range", help="Price range percentage from current price", type=float, default=15
    )

    parser.add_argument("--min-delta", help="Minimum absolute delta threshold", type=float, default=0.30)

    parser.add_argument("--max-delta", help="Maximum absolute delta threshold", type=float, default=0.75)

    parser.add_argument("--min-return", help="Minimum return on capital percentage", type=float, default=2)

    parser.add_argument(
        "--min-annual-return", help="Minimum annualized return percentage", type=float, default=30
    )

    parser.add_argument("--min-risk-score", help="Minimum risk-adjusted score", type=float, default=0.04)

    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    return parser.parse_args()


def main():
    """
    Main function that executes the option analysis workflow.

    Process:
    1. Fetches current treasury yield for risk-free rate
    2. Gets stock price data for the specified tickers
    3. Analyzes option chains for multiple expiration dates
    4. Calculates metrics including return on capital and risk-adjusted scores
    5. Filters options based on risk-adjusted score and delta thresholds
    6. Exports results to CSV files
    """
    # Parse command-line arguments
    args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level)
    logger.info("Starting options analysis")

    # Create output directory
    out_dir = ensure_output_directory()

    # Create market data provider
    market_data = YahooFinanceProvider(out_dir=out_dir)

    # Create option analyzer and set parameters
    option_analyzer = OptionAnalyzer(market_data, out_dir=out_dir)
    option_analyzer.set_analysis_parameters(
        percentage_range=args.percentage_range,
        min_risk_score=args.min_risk_score,
        max_delta_threshold=args.max_delta,
        min_delta_threshold=args.min_delta,
        min_projected_return_pct=args.min_return,
        min_annual_return_pct=args.min_annual_return,
    )

    # Create runner
    runner = OptionsAnalysisRunner(market_data, option_analyzer)

    # Get tickers from arguments or use S&P 500
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(",")]
        logger.info(f"Analyzing specified tickers: {tickers}")
    else:
        tickers = market_data.get_sp500_tickers(refresh=args.refresh_tickers)
        logger.info(f"Analyzing {len(tickers)} S&P 500 tickers")

    # Run the analysis
    results = runner.run_analysis(tickers=tickers, max_workers=args.max_workers)

    # Summary of results
    if "puts" in results and not results["puts"].empty:
        logger.info(f"Successfully found {len(results['puts'])} put options meeting criteria")
    elif "relaxed_puts" in results and not results["relaxed_puts"].empty:
        logger.info(f"Found {len(results['relaxed_puts'])} put options meeting relaxed criteria")
    else:
        logger.warning("No suitable put options found")

    if "calls" in results and not results["calls"].empty:
        logger.info(f"Successfully found {len(results['calls'])} call options meeting criteria")
    else:
        logger.info("No suitable call options found")

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()

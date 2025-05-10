#!/usr/bin/env python3
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

This file now serves as a compatibility wrapper around the refactored
options_analysis package. For new projects, consider using main.py directly.

----------------------------------------------------

Arguments:
None (ATM)

----------------------------------------------------
How to run this script:
cd two_banded_puffbird
uv run src/python/analysis.py
----------------------------------------------------
"""

import logging
from pathlib import Path
import sys

# Make sure the package is in the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from refactored package
from options_analysis.market_data import YahooFinanceProvider
from options_analysis.option_analysis import OptionAnalyzer, OptionsAnalysisRunner
from options_analysis.utils import ensure_output_directory, setup_logging


def main():
    """
    Legacy entry point for options analysis.
    Uses the refactored options_analysis package with default parameters.
    """
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("Starting options analysis (legacy entry point)")

    # Create output directory
    out_dir = ensure_output_directory()

    # Create market data provider
    market_data = YahooFinanceProvider(out_dir=out_dir)

    # Create option analyzer with default parameters
    option_analyzer = OptionAnalyzer(market_data, out_dir=out_dir)

    # Create runner
    runner = OptionsAnalysisRunner(market_data, option_analyzer)

    # Get S&P 500 tickers
    tickers = market_data.get_sp500_tickers()
    logger.info(f"Analyzing {len(tickers)} S&P 500 tickers")

    # Run the analysis with default parameters (3 workers)
    results = runner.run_analysis(tickers=tickers)

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
    logger.info("For more options and control, consider using main.py directly.")


if __name__ == "__main__":
    main()

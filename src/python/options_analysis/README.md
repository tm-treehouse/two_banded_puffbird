# Options Analysis Package

This package provides tools for financial options analysis, calculating risk-adjusted returns
for put and call options based on current market data.

## Features

- Fetch market data from Yahoo Finance
- Calculate option metrics using Black-Scholes model
- Analyze multiple stocks in parallel
- Filter options by risk-adjusted score and other metrics
- Export results to CSV files

## Architecture

The package follows object-oriented design principles with the following components:

- **MarketDataProvider**: Abstract base class for retrieving market data
  - **YahooFinanceProvider**: Implementation using the Yahoo Finance API
- **OptionAnalyzer**: Handles options analysis and filtering
- **OptionsAnalysisRunner**: Orchestrates the analysis process across multiple tickers
- **BlackScholesCalculator**: Implements options pricing and Greeks calculations 
- **Models**: Data classes for options contracts and other data structures

## Usage

```python
from options_analysis.utils import setup_logging, ensure_output_directory
from options_analysis.market_data import YahooFinanceProvider
from options_analysis.option_analysis import OptionAnalyzer, OptionsAnalysisRunner

# Setup logging and output directory
logger = setup_logging()
out_dir = ensure_output_directory()

# Create data provider and analyzer
market_data = YahooFinanceProvider()
analyzer = OptionAnalyzer(market_data)
runner = OptionsAnalysisRunner(market_data, analyzer)

# Run analysis
results = runner.run_analysis(tickers=['AAPL', 'MSFT', 'GOOG'])
```

## Entry Points

- `main.py`: Main entry point with command-line argument parsing
- `analysis.py`: Backward compatibility adapter that calls main.py

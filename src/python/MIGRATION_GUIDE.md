# Migration Guide: From analysis.py to Object-Oriented Structure

This guide explains how the code has been refactored from a monolithic `analysis.py` file to an object-oriented structure using the `options_analysis` package.

## Overview of Changes

The original `analysis.py` script has been refactored into a well-structured package with separate modules for each responsibility:

- **Market Data**: Fetching stock prices, option chains, and treasury rates
- **Option Analysis**: Processing and filtering option contracts
- **Black-Scholes Calculations**: Computing option Greeks and pricing
- **Models**: Data structures for representing options and related data
- **Utilities**: Logging and other helper functions

## Directory Structure

```
src/python/
├── main.py                   # New entry point with command-line interface
├── analysis.py               # Compatibility wrapper for backward compatibility
└── options_analysis/         # Main package
    ├── __init__.py           # Package exports
    ├── black_scholes.py      # Black-Scholes model implementation
    ├── market_data.py        # Data providers (Yahoo Finance, etc.)
    ├── models.py             # Data models for options
    ├── option_analysis.py    # Option analysis logic
    └── utils.py              # Utility functions (logging, etc.)
```

## How to Use the New Structure

### Command-Line Usage

Instead of directly running `analysis.py`, use `main.py` with command-line arguments:

```bash
# Old way
uv run src/python/analysis.py

# New way (with options)
PYTHONPATH=/path/to/two_banded_puffbird uv run src/python/main.py --tickers AAPL,MSFT --max-workers 4
```

### Package Usage

The package can be imported and used programmatically:

```python
from options_analysis.market_data import YahooFinanceProvider
from options_analysis.option_analysis import OptionAnalyzer, OptionsAnalysisRunner

# Create components
market_data = YahooFinanceProvider()
analyzer = OptionAnalyzer(market_data)
runner = OptionsAnalysisRunner(market_data, analyzer)

# Run analysis with custom parameters
runner.run_analysis(tickers=['AAPL', 'MSFT'], max_workers=4)
```

## Key Benefits of the New Structure

1. **Extensibility**: Easy to add new functionality (like call option analysis)
2. **Testability**: Modules can be tested in isolation
3. **Maintainability**: Cleaner code organization and separation of concerns
4. **Flexibility**: Different data sources can be swapped by implementing the `MarketDataProvider` interface
5. **Configurability**: Analysis parameters can be adjusted programmatically

## Backward Compatibility

The original `analysis.py` file is maintained as a compatibility layer that internally uses the new structure. This ensures that existing scripts and workflows will continue to work.

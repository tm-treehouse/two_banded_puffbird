# Two Banded Puffbird

Financial options analysis tool that calculates risk-adjusted returns for put and call options based on current market data.

## Features

- Fetches stock prices, treasury yields, and option chains
- Applies Black-Scholes modeling for option analysis
- Runs automatically via GitHub Actions
- Generates CSV reports with analysis results
- Object-oriented design with clean separation of concerns
- Support for both put and call option analysis

## Project Structure

The project has been refactored into an object-oriented structure:

```
src/python/
├── main.py                   # New entry point with command-line interface
├── analysis.py               # Compatibility wrapper for backward compatibility
├── MIGRATION_GUIDE.md        # Guide for transitioning to the new structure
└── options_analysis/         # Main package
    ├── __init__.py           # Package exports
    ├── black_scholes.py      # Black-Scholes model implementation
    ├── market_data.py        # Data providers (Yahoo Finance, etc.)
    ├── models.py             # Data models for options
    ├── option_analysis.py    # Option analysis logic
    ├── utils.py              # Utility functions (logging, etc.)
    └── README.md             # Package documentation
```

## Usage

### Command Line Interface

```bash
# Show help and available options
uv run src/python/main.py --help

# Analyze specific ticker symbols
uv run src/python/main.py --tickers AAPL,MSFT,GOOGL

# Run with custom parameters
uv run src/python/main.py --max-workers 4 --percentage-range 20 --min-delta 0.35
```

For backward compatibility, you can still use the old entry point:
```bash
uv run src/python/analysis.py
```

## Development

### Code Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting.

The formatting is automatically enforced through:
- Git pre-commit hook (formats code before each commit)
- GitHub Actions workflow (formats code weekly)

Both mechanisms use a common script located at `scripts/format_code.sh`.

### Setup Pre-commit Hook

The pre-commit hook is automatically installed in `.git/hooks/pre-commit`. If you need to install it manually:

```bash
# Must run:
uv run pre-commit install
chmod +x scripts/format_code.sh
chmod +x .git/hooks/pre-commit
```

Alternatively, use the pre-commit tool:

```bash
pip install pre-commit
pre-commit install
```

## Workflow

The project includes two GitHub Actions workflows:
1. `main.yml` - Runs the financial analysis and sends results via email
2. `format_code.yml` - Formats the code on a weekly basis
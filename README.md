# Two Banded Puffbird

Financial options analysis tool that calculates risk-adjusted returns for put and call options based on current market data.

## Features

- Fetches stock prices, treasury yields, and option chains
- Applies Black-Scholes modeling for option analysis
- Runs automatically via GitHub Actions
- Generates CSV reports with analysis results

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
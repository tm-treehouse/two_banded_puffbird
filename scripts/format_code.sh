#!/bin/bash
# Common script to run Ruff formatter on Python code
# Used by both Git pre-commit hook and GitHub Actions

# Exit on any error
set -e

echo "Running Ruff formatter..."

# Check if uv is available (which is used in the project)
if command -v uv &> /dev/null; then
    # Check if Ruff is installed
    if ! uv pip list | grep -q ruff; then
        echo "Installing Ruff..."
        uv pip install ruff
    fi
    
    # Run Ruff formatter using uv
    uv run -m ruff format ./src
else
    # Fallback to direct ruff command if available
    if command -v ruff &> /dev/null; then
        ruff format ./src
    else
        echo "Error: Neither uv nor ruff is available. Please install them."
        exit 1
    fi
fi

echo "Code formatting completed successfully."

#!/bin/bash
# Common script to run Ruff formatter on Python code
# Used by both Git pre-commit hook and GitHub Actions

# Exit on any error
set -e

echo "Running Ruff formatter..."

# Check if uv is available (which is used in the project)
if command -v uv &> /dev/null; then
#     # Check if Ruff is installed
#     if ! uv pip list | grep -q ruff; then
#         echo "Installing Ruff..."
#         uv pip install ruff
#     fi
    
    # Run Ruff formatter and sort imports using uv
    echo "Formatting code..."
    uvx ruff format ./src
    echo "Sorting imports..."
    uvx ruff check --select I --fix ./src
else
    # Fallback to direct ruff command if available
    if command -v ruff &> /dev/null; then
        echo "Formatting code..."
        ruff format ./src
        echo "Sorting imports..."
        ruff check --select I --fix ./src
    else
        echo "Error: Neither uv nor ruff is available. Please install them."
        exit 1
    fi
fi

echo "Code formatting completed successfully."

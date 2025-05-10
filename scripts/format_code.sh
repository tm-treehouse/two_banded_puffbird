#!/bin/bash
# Common script to run Ruff formatter on Python code
# Used by both Git pre-commit hook and GitHub Actions
#
# Usage:
#   format_code.sh [directory1] [directory2] ...
#   If no directories are provided, defaults to "./src ./scripts ./tests"

# Exit on any error
set -e

# Define directories to format (from arguments, config file, or defaults)
if [ $# -eq 0 ]; then
    # Check if the config file exists
    CONFIG_FILE="$(dirname "$0")/format_dirs.txt"
    if [ -f "$CONFIG_FILE" ]; then
        # Read directories from config file, ignoring comments and empty lines
        DIRS_TO_FORMAT=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | tr '\n' ' ')
    else
        # Default directories - only include existing ones
        DIRS_TO_FORMAT=""
        for dir in "./src" "./scripts"; do
            if [ -d "$dir" ]; then
                DIRS_TO_FORMAT="$DIRS_TO_FORMAT $dir"
            fi
        done
    fi
else
    # Use directories from command line arguments
    DIRS_TO_FORMAT="$@"
fi

echo "Running Ruff formatter on: $DIRS_TO_FORMAT"

# Check if uv is available (which is used in the project)
if command -v uv &> /dev/null; then
#     # Check if Ruff is installed
#     if ! uv pip list | grep -q ruff; then
#         echo "Installing Ruff..."
#         uv pip install ruff
#     fi
    
    # Run Ruff formatter and sort imports using uv
    echo "Formatting code..."
    # Using DIRS_TO_FORMAT from command line arguments or defaults
    uvx ruff format $DIRS_TO_FORMAT
    echo "Sorting imports..."
    uvx ruff check --select I --fix $DIRS_TO_FORMAT
    echo "Removing unused imports..."
    uvx ruff check --select F401 --fix $DIRS_TO_FORMAT
else
    # Fallback to direct ruff command if available
    if command -v ruff &> /dev/null; then
        echo "Formatting code..."
        # Using DIRS_TO_FORMAT from command line arguments or defaults
        ruff format $DIRS_TO_FORMAT
        echo "Sorting imports..."
        ruff check --select I --fix $DIRS_TO_FORMAT
        echo "Removing unused imports..."
        ruff check --select F401 --fix $DIRS_TO_FORMAT
    else
        echo "Error: Neither uv nor ruff is available. Please install them."
        exit 1
    fi
fi

echo "Code formatting completed successfully."

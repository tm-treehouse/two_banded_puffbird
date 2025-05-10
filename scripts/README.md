# Code Formatting Scripts
This directory contains scripts for code formatting using [Ruff](https://github.com/astral-sh/ruff), a fast Python linter and formatter.

## Setup Git Pre-commit Hook
```bash
# Must run:
uv run pre-commit install

# Optional run manually on all files:
uv run pre-commit run --all-files

# Run specific hook:
uv run pre-commit run ruff-format --all-files

# Update all hooks to latest version:
uv run pre-commit autoupdate

# Run pre-commit on staged files:
uv run pre-commit run
```

## Available Scripts
- `format_code.sh` - Common script used both by git pre-commit hook and GitHub Actions workflow

## How It Works
The formatting functionality is implemented in two ways:

1. **Git Pre-commit Hook**: 
   - Automatically runs every time you try to commit code
   - Formats all Python files in the commit
   - Adds the formatted files back to the staging area

2. **GitHub Actions Workflow**: 
   - Runs on a schedule (every Monday at midnight)
   - Formats all Python code in the repository
   - Commits and pushes the changes automatically

## Configuration
Formatting rules are configured in the `pyproject.toml` file at the root of the repository.

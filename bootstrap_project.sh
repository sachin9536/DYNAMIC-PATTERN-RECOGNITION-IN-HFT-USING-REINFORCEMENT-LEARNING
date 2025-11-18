#!/usr/bin/env bash
set -e

echo "Creating PR_project directory structure..."

# Create main directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p scripts
mkdir -p configs
mkdir -p tests

# Create src module directories with __init__.py files
mkdir -p src/data
mkdir -p src/features
mkdir -p src/rl_agent
mkdir -p src/rules
mkdir -p src/fusion
mkdir -p src/explainability
mkdir -p src/dashboard
mkdir -p src/utils

# Create __init__.py files for all modules
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/rl_agent/__init__.py
touch src/rules/__init__.py
touch src/fusion/__init__.py
touch src/explainability/__init__.py
touch src/dashboard/__init__.py
touch src/utils/__init__.py

# Create main project files if they don't exist
touch requirements.txt
touch .gitignore
touch README.md

echo "Project structure created successfully!"
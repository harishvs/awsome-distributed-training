#!/bin/bash

# NCCL Bad Node Finder - Unit Test Runner
# This script sets up a virtual environment and runs the unit tests

set -e

echo "Setting up test environment for NCCL Bad Node Finder..."

# Create virtual environment
python3 -m venv .testenv

# Activate virtual environment
source .testenv/bin/activate

echo "Installing test dependencies..."
# Install test dependencies
pip install -r test_requirements.txt

echo "Running unit tests..."
# Run the tests
pytest

echo "Unit tests completed successfully!"
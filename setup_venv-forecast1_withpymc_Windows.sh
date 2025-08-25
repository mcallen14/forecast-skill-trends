#!/bin/bash

set -e

VENV_DIR="venv-forecast1"
REQUIREMENTS_FILE="requirements.txt"

if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python and try again."
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip and try again."
    exit 1
fi

echo "Creating virtual environment in $VENV_DIR..."
python -m venv "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/Scripts/activate"

echo "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Requirements file not found: $REQUIREMENTS_FILE"
    deactivate
    exit 1
fi

echo "Installing packages from $REQUIREMENTS_FILE..."
pip install --no-cache-dir --only-binary=:all: -r "$REQUIREMENTS_FILE"

deactivate

echo "Setup complete. To activate the virtual environment, run:"
echo "source $VENV_DIR/Scripts/activate"

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Name of the virtual environment directory
VENV_DIR="venv-forecast1"

# Path to the requirements file
REQUIREMENTS_FILE="requirements.txt"

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Requirements file not found: $REQUIREMENTS_FILE"
    deactivate
    exit 1
fi

# Install the required packages
echo "Installing packages from $REQUIREMENTS_FILE..."
#pip install -r $REQUIREMENTS_FILE
#Note that this version of the virtual environment set up forces it to redownload everything to avoid conflicts from previously downloaded versions
pip install --no-cache-dir -r $REQUIREMENTS_FILE

# Deactivate the virtual environment
deactivate

echo "Setup complete. To activate the virtual environment, run:"
echo "source $VENV_DIR/bin/activate"

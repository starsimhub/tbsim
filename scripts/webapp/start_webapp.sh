#!/bin/bash

# TB Simulation Web App Startup Script
# This script sets up and starts the TB simulation web application

echo "TB Simulation Script Catalog Web App"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the webapp directory."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version ✓"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if tbsim is available
echo "Checking TB simulation dependencies..."
python3 -c "import tbsim, starsim" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: TB simulation modules (tbsim, starsim) not found."
    echo "Please ensure they are installed in your environment."
    echo "You may need to install them from the parent directory:"
    echo "  pip install -e /Users/mine/anaconda_projects/newgit/newtbsim"
    echo ""
    echo "Continuing anyway - some features may not work properly."
fi

# Start the web application
echo "Starting TB Simulation Web App..."
echo "Access the application at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py


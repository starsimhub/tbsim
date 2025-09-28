#!/bin/bash

# TBsim Shiny App Startup Script
# This script sets up and runs the TBsim Shiny application

echo "TBsim Shiny Web Application"
echo "==========================="
echo ""

# Check if R is installed
if ! command -v R &> /dev/null; then
    echo "Error: R is not installed. Please install R 4.0+ and try again."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

echo "✓ R and Python are available"

# Check if setup has been run
if [ ! -f "setup_complete.flag" ]; then
    echo "Running initial setup..."
    Rscript setup.R
    if [ $? -eq 0 ]; then
        touch setup_complete.flag
        echo "✓ Setup completed successfully"
    else
        echo "✗ Setup failed. Please check the error messages above."
        exit 1
    fi
else
    echo "✓ Setup already completed"
fi

# Test the setup
echo "Testing setup..."
Rscript test_setup.R

# Start the application
echo ""
echo "Starting TBsim Shiny Application..."
echo "The application will open in your web browser."
echo "To stop the application, press Ctrl+C"
echo ""

Rscript run_app.R

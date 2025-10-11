#!/bin/bash
# TBsim Shiny App Stop Script
# Gracefully stops the running Shiny app

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}    Stopping TBsim Shiny App${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

# Check if app is running
if ! lsof -i :3838 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} App is not running on port 3838"
    exit 0
fi

# Get PID
PID=$(lsof -ti :3838)

# Kill the process
echo "Stopping app (PID: $PID)..."
pkill -f "shiny::runApp.*app.R" || true

# Wait for process to stop
sleep 2

# Verify it stopped
if ! lsof -i :3838 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} App stopped successfully"
else
    echo -e "${YELLOW}⚠${NC} App may still be running. Forcing shutdown..."
    kill -9 $PID 2>/dev/null || true
    sleep 1
    if ! lsof -i :3838 > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} App forcefully stopped"
    else
        echo "Failed to stop app. Please manually kill process $PID"
        exit 1
    fi
fi


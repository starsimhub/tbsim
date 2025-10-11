#!/bin/bash
# TBsim Shiny App Startup Script
# Starts the app cleanly with minimal console output

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}    TBsim Shiny Web Application${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

# Check if already running
if lsof -i :3838 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} App is already running on port 3838"
    echo -e "\n  Access at: ${GREEN}http://localhost:3838${NC}"
    exit 0
fi

# Change to app directory
cd "$(dirname "$0")"

# Start the app in the background
echo "Starting app..."
nohup Rscript -e "shiny::runApp('app.R', port=3838, host='0.0.0.0', launch.browser=FALSE)" > app.log 2>&1 &
APP_PID=$!

# Wait for app to start
echo "Initializing (this may take a few seconds)..."
sleep 5

# Check if app is running
if lsof -i :3838 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} App started successfully!"
    echo -e "\n${BLUE}Server Details:${NC}"
    echo "  • Process ID: $APP_PID"
    echo "  • Port: 3838"
    echo "  • Log file: app.log"
    echo -e "\n${BLUE}Access the app:${NC}"
    echo -e "  • Local:   ${GREEN}http://localhost:3838${NC}"
    echo -e "  • Network: ${GREEN}http://0.0.0.0:3838${NC}"
    echo -e "\n${BLUE}To stop the server:${NC}"
    echo "  ./stop_app.sh"
    echo "  or: kill $APP_PID"
    echo -e "\n${GREEN}Happy modeling! 🦠📊${NC}"
else
    echo "Failed to start app. Check app.log for details."
    cat app.log
    exit 1
fi


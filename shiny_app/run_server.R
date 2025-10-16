# TBsim Shiny Server Launcher with Library Path Fix

# Set environment variables before loading any libraries
Sys.setenv(
  RETICULATE_PYTHON = "/Users/mine/git/tbsim/venv/bin/python",
  DYLD_FALLBACK_LIBRARY_PATH = "/opt/homebrew/lib:/usr/lib",
  DYLD_LIBRARY_PATH = "/opt/homebrew/lib:/usr/lib"
)

# Suppress warnings
options(warn = -1)
options(shiny.trace = FALSE)

# Print startup message
cat("TBsim Shiny Server\n")
cat("==================\n")
cat("Python:", Sys.getenv("RETICULATE_PYTHON"), "\n")
cat("Library Path:", Sys.getenv("DYLD_FALLBACK_LIBRARY_PATH"), "\n")
cat("\nStarting server on port 3838...\n\n")

# Load and run the app
shiny::runApp(
  "app.R", 
  port = 3838, 
  launch.browser = FALSE, 
  host = "0.0.0.0"
)


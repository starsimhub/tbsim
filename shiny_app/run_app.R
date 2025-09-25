# TBsim Shiny App Launcher
# Simple script to launch the TBsim web application

# Check if required packages are installed
required_packages <- c("shiny", "plotly", "DT", "reticulate")

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  cat("Missing required packages:", paste(missing_packages, collapse = ", "), "\n")
  cat("Please run setup.R first to install dependencies.\n")
  quit(status = 1)
}

# Check if Python and tbsim are available
library(reticulate)

if (!py_available()) {
  cat("Python not available. Please install Python 3.8+ and run setup.R\n")
  quit(status = 1)
}

# Try to import tbsim
tryCatch({
  tbsim <- import("tbsim")
  cat("TBsim package found. Starting application...\n")
}, error = function(e) {
  cat("TBsim package not found. Please run setup.R to install dependencies.\n")
  cat("Error:", e$message, "\n")
  quit(status = 1)
})

# Launch the Shiny app
cat("Launching TBsim Shiny Application...\n")
cat("The application will open in your default web browser.\n")
cat("To stop the application, press Ctrl+C in the terminal.\n\n")

shiny::runApp("app.R", launch.browser = TRUE)

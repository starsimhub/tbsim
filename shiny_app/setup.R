# Setup script for TBsim Shiny App
# This script installs all required R and Python dependencies

# Install R packages if not already installed
install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
  }
}

# Install required R packages
required_packages <- c(
  "shiny",
  "plotly", 
  "DT",
  "reticulate"
)

cat("Installing R packages...\n")
for (pkg in required_packages) {
  install_if_missing(pkg)
}

# Set up Python environment
cat("Setting up Python environment...\n")
library(reticulate)

# Try to use the virtual environment Python first
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  cat("Using virtual environment Python:", venv_python, "\n")
  use_python(venv_python, required = TRUE)
} else {
  # Fall back to system Python
  cat("Virtual environment not found, using system Python\n")
  use_python("python3", required = TRUE)
}

# Check if Python is available
if (!py_available()) {
  cat("Python not found. Please install Python 3.8+ and try again.\n")
  quit(status = 1)
}

# Install Python packages
cat("Installing Python packages...\n")
py_install(c(
  "numpy>=1.21.0",
  "scipy>=1.7.0", 
  "pandas>=2.0.0",
  "sciris>=3.1.0",
  "matplotlib>=3.5.0",
  "numba>=0.56.0",
  "starsim==2.3.2",
  "seaborn>=0.11.0",
  "plotly>=5.0.0",
  "lifelines>=0.27.0",
  "tqdm>=4.64.0",
  "networkx>=2.8.0"
))

# Install tbsim package
cat("Installing tbsim package...\n")
py_install("tbsim", pip = TRUE)

cat("Setup complete! You can now run the Shiny app.\n")
cat("To start the app, run: shiny::runApp('app.R')\n")

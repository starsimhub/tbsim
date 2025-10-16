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

# Try to use environment variable or common virtual environment Python locations
python_path <- Sys.getenv("TBSIM_PYTHON_PATH", unset = NA)
if (!is.na(python_path) && file.exists(python_path)) {
  cat("Using Python from TBSIM_PYTHON_PATH environment variable:", python_path, "\n")
  use_python(python_path, required = TRUE)
} else {
  # Check for common virtual environment locations
  venv_paths <- c("venv/bin/python", ".venv/bin/python")
  found_venv <- FALSE
  for (venv in venv_paths) {
    if (file.exists(venv)) {
      cat("Using detected virtual environment Python:", venv, "\n")
      use_python(venv, required = TRUE)
      found_venv <- TRUE
      break
    }
  }
  if (!found_venv) {
    # Fall back to system Python
    cat("No virtual environment found, using system Python\n")
    use_python("python3", required = TRUE)
  }
}

# Check if Python is available
if (!py_available()) {
  cat("Python not found. Please install Python 3.8+ and try again.\n")
  quit(status = 1)
}

# Install Python packages with pinned versions to avoid library loading issues
cat("Installing Python packages...\n")
py_install(c(
  "numpy==2.3.3",
  "scipy>=1.7.0", 
  "pandas>=2.0.0",
  "sciris>=3.1.0",
  "matplotlib>=3.5.0",
  "llvmlite==0.45.1",
  "numba==0.62.1",
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

# Verify installation
cat("\nVerifying Python environment...\n")
tryCatch({
  test_tbsim <- import("tbsim")
  cat("✓ tbsim imported successfully\n")
}, error = function(e) {
  cat("✗ Failed to import tbsim:", e$message, "\n")
  cat("If you see library loading errors, try reinstalling llvmlite and numba:\n")
  cat("  pip install --force-reinstall --no-cache-dir llvmlite==0.45.1 numba==0.62.1\n")
})

cat("Setup complete! You can now run the Shiny app.\n")
cat("To start the app, run: shiny::runApp('app.R')\n")

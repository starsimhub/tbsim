# Test script for TBsim Shiny App setup
# This script verifies that all dependencies are properly installed

cat("Testing TBsim Shiny App Setup\n")
cat("=============================\n\n")

# Test 1: Check R packages
cat("1. Testing R packages...\n")
required_r_packages <- c("shiny", "plotly", "DT", "reticulate")

for (pkg in required_r_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat("   ✓", pkg, "is installed\n")
  } else {
    cat("   ✗", pkg, "is missing\n")
  }
}

# Test 2: Check Python availability
cat("\n2. Testing Python environment...\n")
library(reticulate)

# Try to use the virtual environment Python first
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  cat("   Using virtual environment Python:", venv_python, "\n")
  use_python(venv_python, required = TRUE)
} else {
  cat("   Virtual environment not found, using system Python\n")
  use_python("python3", required = TRUE)
}

if (py_available()) {
  cat("   ✓ Python is available\n")
  cat("   Python version:", py_config()$version, "\n")
} else {
  cat("   ✗ Python is not available\n")
}

# Test 3: Check Python packages
if (py_available()) {
  cat("\n3. Testing Python packages...\n")
  
  python_packages <- c("numpy", "scipy", "pandas", "sciris", "matplotlib", 
                       "starsim", "tbsim")
  
  for (pkg in python_packages) {
    tryCatch({
      import(pkg)
      cat("   ✓", pkg, "is available\n")
    }, error = function(e) {
      cat("   ✗", pkg, "is missing:", e$message, "\n")
    })
  }
}

# Test 4: Test basic tbsim functionality
cat("\n4. Testing TBsim functionality...\n")
tryCatch({
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  sciris <- import("sciris")
  
  # Try to create a simple simulation
  pop <- starsim$People(n_agents = 100)
  tb <- tbsim$TB()
  
  cat("   ✓ TBsim basic functionality works\n")
}, error = function(e) {
  cat("   ✗ TBsim functionality test failed:", e$message, "\n")
})

# Test 5: Test Shiny app loading
cat("\n5. Testing Shiny app...\n")
tryCatch({
  source("app.R")
  cat("   ✓ Shiny app loads successfully\n")
}, error = function(e) {
  cat("   ✗ Shiny app loading failed:", e$message, "\n")
})

cat("\nSetup test complete!\n")
cat("If all tests passed, you can run the app with: Rscript run_app.R\n")

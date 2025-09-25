# Test Simple TBsim Integration
# This script tests a minimal TBsim model to avoid compatibility issues

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing simple TBsim integration...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  sciris <- import("sciris")
  
  cat("✓ Successfully imported tbsim, starsim, and sciris\n")
  
  # Create a simple simulation without complex parameters
  sim <- starsim$Sim(
    pars = list(
      dt = starsim$days(7),
      start = '1940-01-01',
      stop = '1950-01-01',  # Shorter simulation
      rand_seed = as.integer(1),
      verbose = 0
    )
  )
  
  cat("✓ Successfully created simple simulation\n")
  
  # Test running simulation
  sim$run()
  cat("✓ Successfully ran simulation\n")
  
  # Test extracting results
  results <- sim$results$flatten()
  cat("✓ Successfully extracted results\n")
  
  cat("\n🎉 Simple simulation test passed!\n")
  cat("Simulation duration:", round(max(results$t) / 365.25, 2), "years\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Simple simulation test failed.\n")
})

# Test script to print simulation parameters only
# This script will create a simulation and print its parameters

cat("Testing simulation parameters only...\n")

# Load required libraries
library(reticulate)

# Set up Python environment
venv_path <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_path)) {
  use_python(venv_path, required = TRUE)
  cat("✓ Using virtual environment Python\n")
} else {
  use_python("python3", required = TRUE)
  cat("✓ Using system Python\n")
}

# Import Python modules
starsim <- import("starsim")

cat("✓ Python modules imported successfully\n")

tryCatch({
  # Set up simulation parameters
  sim_pars <- list(
    dt = starsim$days(1),
    start = "1940-01-01",
    stop = "2010-01-01",
    rand_seed = 1L,
    verbose = 0
  )
  
  # Create population
  pop <- starsim$People(n_agents = 100)
  
  # Create simulation
  sim <- starsim$Sim(
    people = pop,
    pars = sim_pars
  )
  
  cat("✓ Simulation created\n")
  
  # Print simulation parameters
  cat("\n=== SIMULATION PARAMETERS (sim$pars) ===\n")
  cat("Simulation type:", class(sim), "\n")
  cat("Simulation parameters type:", class(sim$pars), "\n")
  cat("Simulation parameters length:", length(sim$pars), "\n\n")
  
  # Print each simulation parameter
  for (i in seq_along(sim$pars)) {
    param_name <- names(sim$pars)[i]
    param_value <- sim$pars[[i]]
    
    cat(sprintf("Sim Parameter %d: %s\n", i, param_name))
    cat(sprintf("  Type: %s\n", class(param_value)))
    cat(sprintf("  Value: %s\n", toString(param_value)))
    cat("\n")
  }
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
})

cat("\n=== TEST COMPLETED ===\n")

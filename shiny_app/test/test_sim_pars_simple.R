# Simple test script to print sim$pars contents
# This script will create a simulation object and print its parameters

cat("Testing sim$pars contents (simple version)...\n")

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
tbsim <- import("tbsim")

cat("✓ Python modules imported successfully\n")

# Create a simple simulation object
tryCatch({
  # Set up basic simulation parameters
  sim_pars <- list(
    dt = starsim$days(1),
    start = "1940-01-01",
    stop = "2010-01-01",
    rand_seed = 1L,
    verbose = 0
  )
  
  # Create population
  pop <- starsim$People(n_agents = 100)
  
  # Create a simple simulation object
  sim <- starsim$Sim(
    people = pop,
    pars = sim_pars
  )
  
  cat("✓ Simulation object created\n")
  
  # Print sim$pars contents
  cat("\n=== SIMULATION PARAMETERS (sim$pars) ===\n")
  cat("Type:", class(sim$pars), "\n")
  cat("Length:", length(sim$pars), "\n\n")
  
  # Print parameter names
  cat("Parameter names:\n")
  param_names <- names(sim$pars)
  for (i in seq_along(param_names)) {
    cat(sprintf("  %d. %s\n", i, param_names[i]))
  }
  
  cat("\n=== PARAMETER VALUES ===\n")
  
  # Try to access specific parameters safely
  safe_get_param <- function(param_name) {
    tryCatch({
      value <- sim$pars[[param_name]]
      cat(sprintf("%s: %s (type: %s)\n", param_name, toString(value), class(value)))
    }, error = function(e) {
      cat(sprintf("%s: ERROR - %s\n", param_name, e$message))
    })
  }
  
  # Print each parameter
  for (param_name in param_names) {
    safe_get_param(param_name)
  }
  
  cat("\n=== SPECIFIC PARAMETER DETAILS ===\n")
  
  # Try to get more details for key parameters
  key_params <- c("dt", "start", "stop", "rand_seed", "verbose")
  for (param in key_params) {
    if (param %in% param_names) {
      cat(sprintf("\n--- %s ---\n", param))
      tryCatch({
        value <- sim$pars[[param]]
        cat(sprintf("Value: %s\n", toString(value)))
        cat(sprintf("Type: %s\n", class(value)))
        
        # Try to get Python representation
        if (inherits(value, "python.builtin.object")) {
          tryCatch({
            cat(sprintf("Python type: %s\n", py_type(value)))
          }, error = function(e) {
            cat("Could not get Python type\n")
          })
        }
      }, error = function(e) {
        cat(sprintf("Error accessing %s: %s\n", param, e$message))
      })
    }
  }
  
}, error = function(e) {
  cat("❌ Error creating simulation object:", e$message, "\n")
  cat("This might be due to missing dependencies or configuration issues.\n")
})

cat("\n=== TEST COMPLETED ===\n")

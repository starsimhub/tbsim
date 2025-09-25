# Simple test script to print TB disease parameters
# This script will create a TB model and print its parameters

cat("Testing TB disease parameters (simple version)...\n")

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

tryCatch({
  # Set up TB parameters
  tb_pars <- list(
    dt = starsim$days(1),
    start = "1940-01-01",
    stop = "2010-01-01",
    init_prev = starsim$bernoulli(p = 0.1),
    beta = starsim$peryear(0.5),
    p_latent_fast = starsim$bernoulli(p = 0.1),
    rate_LS_to_presym = starsim$perday(0.001),
    rate_LF_to_presym = starsim$perday(0.01),
    rate_presym_to_active = starsim$perday(0.1),
    rate_active_to_clear = starsim$perday(0.01),
    rate_treatment_to_clear = starsim$peryear(0.8),
    rate_exptb_to_dead = starsim$perday(0.05),
    rate_smpos_to_dead = starsim$perday(0.02),
    rate_smneg_to_dead = starsim$perday(0.01),
    rel_trans_presymp = 0.5,
    rel_trans_smpos = 1.0,
    rel_trans_smneg = 0.3,
    rel_trans_exptb = 0.1,
    rel_trans_treatment = 0.05,
    rel_sus_latentslow = 0.1,
    cxr_asymp_sens = 0.7,
    reltrans_het = starsim$constant(v = 1.0)
  )
  
  # Create TB model
  tb <- tbsim$TB(pars = tb_pars)
  
  cat("✓ TB model created\n")
  
  # Print TB disease parameters
  cat("\n=== TB DISEASE PARAMETERS (tb$pars) ===\n")
  cat("TB object type:", class(tb), "\n")
  cat("TB parameters type:", class(tb$pars), "\n")
  cat("TB parameters length:", length(tb$pars), "\n\n")
  
  # Print each TB parameter
  for (i in seq_along(tb$pars)) {
    param_name <- names(tb$pars)[i]
    param_value <- tb$pars[[i]]
    
    cat(sprintf("TB Parameter %d: %s\n", i, param_name))
    cat(sprintf("  Type: %s\n", class(param_value)))
    cat(sprintf("  Value: %s\n", toString(param_value)))
    cat("\n")
  }
  
  # Now create a simple simulation
  cat("\n=== CREATING SIMULATION ===\n")
  
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
    diseases = tb,
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
  
  # Try to access diseases specifically
  cat("\n=== DISEASES MODULE ACCESS ===\n")
  tryCatch({
    diseases <- sim$diseases
    cat("Diseases type:", class(diseases), "\n")
    cat("Diseases length:", length(diseases), "\n")
    
    if (length(diseases) > 0) {
      cat("Disease names:\n")
      for (i in seq_along(diseases)) {
        disease_name <- names(diseases)[i]
        disease_obj <- diseases[[i]]
        cat(sprintf("  %d. %s (type: %s)\n", i, disease_name, class(disease_obj)))
      }
    }
  }, error = function(e) {
    cat("Could not access diseases:", e$message, "\n")
  })
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
})

cat("\n=== TEST COMPLETED ===\n")

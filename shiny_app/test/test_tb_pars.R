# Test script to print TB disease parameters and simulation parameters
# This script will create a complete TBsim simulation and print the parameters

cat("Testing TB disease parameters and simulation parameters...\n")

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
  # Set up simulation parameters
  sim_pars <- list(
    dt = starsim$days(1),
    start = "1940-01-01",
    stop = "2010-01-01",
    rand_seed = 1L,
    verbose = 0
  )
  
  # Create population
  pop <- starsim$People(n_agents = 1000)
  
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
  
  # Create network
  net <- starsim$RandomNet(list(
    n_contacts = starsim$poisson(lam = 10),
    dur = 0
  ))
  
  # Create demographics
  births <- starsim$Births(pars = list(birth_rate = 0.02))
  deaths <- starsim$Deaths(pars = list(death_rate = 0.01))
  
  # Create simulation
  sim <- starsim$Sim(
    people = pop,
    networks = net,
    diseases = tb,
    demographics = list(deaths, births),
    pars = sim_pars
  )
  
  cat("✓ Complete TBsim simulation created\n")
  
  # Print TB disease parameters
  cat("\n=== TB DISEASE PARAMETERS (diseases['tb']) ===\n")
  cat("TB object type:", class(tb), "\n")
  cat("TB object length:", length(tb), "\n\n")
  
  # Print TB parameters
  cat("TB Parameters (tb$pars):\n")
  if (!is.null(tb$pars)) {
    cat("Type:", class(tb$pars), "\n")
    cat("Length:", length(tb$pars), "\n\n")
    
    # Print each TB parameter
    for (i in seq_along(tb$pars)) {
      param_name <- names(tb$pars)[i]
      param_value <- tb$pars[[i]]
      
      cat(sprintf("TB Parameter %d: %s\n", i, param_name))
      cat(sprintf("  Type: %s\n", class(param_value)))
      cat(sprintf("  Value: %s\n", toString(param_value)))
      cat("\n")
    }
  } else {
    cat("TB parameters not accessible\n")
  }
  
  # Print simulation parameters
  cat("\n=== SIMULATION PARAMETERS (sim$pars) ===\n")
  cat("Simulation object type:", class(sim), "\n")
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
  
  # Try to access TB specifically from diseases
  cat("\n=== TB DISEASE ACCESS ===\n")
  tryCatch({
    tb_disease <- sim$diseases$tb
    cat("TB disease type:", class(tb_disease), "\n")
    cat("TB disease value:", toString(tb_disease), "\n")
  }, error = function(e) {
    cat("Could not access TB disease:", e$message, "\n")
  })
  
}, error = function(e) {
  cat("❌ Error creating simulation:", e$message, "\n")
  cat("This might be due to missing dependencies or configuration issues.\n")
})

cat("\n=== TEST COMPLETED ===\n")

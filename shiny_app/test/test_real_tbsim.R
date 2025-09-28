# Test Real TBsim Integration
# This script tests the real TBsim model integration

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing real TBsim integration...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  sciris <- import("sciris")
  
  cat("✓ Successfully imported tbsim, starsim, and sciris\n")
  
  # Set random seed
  set.seed(1)
  
  # Build TBsim simulation using the real model
  sim_pars <- list(
    dt = starsim$days(7),
    start = "1940-01-01",
    stop = "1950-01-01",  # Shorter simulation for testing
    rand_seed = as.integer(1),
    verbose = 0
  )
  
  # Create population
  pop <- starsim$People(n_agents = 1000)
  cat("✓ Successfully created population\n")
  
  # Create TB disease model with custom parameters
  tb_pars <- list(
    dt = starsim$days(7),
    start = "1940-01-01",
    stop = "1950-01-01",
    init_prev = starsim$bernoulli(p = 0.01),
    beta = starsim$peryear(0.0025),
    p_latent_fast = starsim$bernoulli(p = 0.1),
    
    # State transition rates
    rate_LS_to_presym = starsim$perday(3e-5),
    rate_LF_to_presym = starsim$perday(6e-3),
    rate_presym_to_active = starsim$perday(3e-2),
    rate_active_to_clear = starsim$perday(2.4e-4),
    rate_treatment_to_clear = starsim$peryear(6),
    
    # Mortality rates
    rate_exptb_to_dead = starsim$perday(0.15 * 4.5e-4),
    rate_smpos_to_dead = starsim$perday(4.5e-4),
    rate_smneg_to_dead = starsim$perday(0.3 * 4.5e-4),
    
    # Transmissibility
    rel_trans_presymp = 0.1,
    rel_trans_smpos = 1.0,
    rel_trans_smneg = 0.3,
    rel_trans_exptb = 0.05,
    rel_trans_treatment = 0.5,
    
    # Susceptibility
    rel_sus_latentslow = 0.20,
    
    # Diagnostics
    cxr_asymp_sens = 1.0,
    
    # Heterogeneity
    reltrans_het = starsim$constant(v = 1.0)
  )
  
  tb <- tbsim$TB(pars = tb_pars)
  cat("✓ Successfully created TB model\n")
  
  # Create social network
  net <- starsim$RandomNet(list(
    n_contacts = starsim$poisson(lam = 5),
    dur = 0
  ))
  cat("✓ Successfully created network\n")
  
  # Create demographic processes
  births <- starsim$Births(pars = list(birth_rate = 20))
  deaths <- starsim$Deaths(pars = list(death_rate = 15))
  cat("✓ Successfully created demographic processes\n")
  
  # Create simulation
  sim <- starsim$Sim(
    people = pop,
    networks = net,
    diseases = tb,
    demographics = list(deaths, births),
    pars = sim_pars
  )
  cat("✓ Successfully created simulation\n")
  
  # Run simulation
  sim$run()
  cat("✓ Successfully ran simulation\n")
  
  # Extract results
  results <- sim$results$flatten()
  cat("✓ Successfully extracted results\n")
  
  # Convert time to years
  time_years <- results$t / 365.25
  cat("✓ Successfully converted time to years\n")
  
  cat("\n🎉 Real TBsim integration test passed!\n")
  cat("Simulation duration:", round(max(time_years), 2), "years\n")
  cat("Final infected count:", max(results$n_infected, na.rm = TRUE), "\n")
  cat("Final latent count:", max(results$n_latent, na.rm = TRUE), "\n")
  cat("Final active count:", max(results$n_active, na.rm = TRUE), "\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Real TBsim integration failed.\n")
  
  # Try to get more detailed error information
  tryCatch({
    py_error <- reticulate::py_last_error()
    if (!is.null(py_error)) {
      cat("Python error details:", py_error$message, "\n")
    }
  }, error = function(e2) {
    cat("Could not get Python error details\n")
  })
})

# Test Working TBsim Integration
# This script tests a working TBsim model with corrected parameters

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing working TBsim integration...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  
  cat("✓ Successfully imported tbsim and starsim\n")
  
  # Set random seed
  set.seed(1)
  
  # Build TBsim simulation using corrected parameters
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
  
  # Create TB disease model with corrected parameters
  tb_pars <- list(
    dt = starsim$days(7),
    start = "1940-01-01",
    stop = "1950-01-01",
    init_prev = starsim$bernoulli(p = 0.01),
    beta = starsim$peryear(0.0025),
    p_latent_fast = starsim$bernoulli(p = 0.1),
    # Use corrected rate parameters that work with starsim 3.0.2
    rate_LS_to_presym = starsim$perday(0.001),  # Increased from 3e-5
    rate_LF_to_presym = starsim$perday(0.01),   # Increased from 6e-3
    rate_presym_to_active = starsim$perday(0.1),
    rate_active_to_clear = starsim$perday(0.001),
    rate_treatment_to_clear = starsim$peryear(0.5)
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
  
  # Extract results from sim.results
  results <- sim$results$flatten()
  cat("✓ Successfully extracted results from sim.results\n")
  
  # Check what's available in results
  cat("\nAvailable result fields:\n")
  cat(paste(names(results), collapse = ", "), "\n")
  
  # Create time vector based on simulation parameters
  start_date <- as.Date("1940-01-01")
  end_date <- as.Date("1950-01-01")
  time_days <- seq(0, as.numeric(end_date - start_date), by = 7)
  time_years <- time_days / 365.25
  cat("✓ Successfully converted time to years\n")
  
  # Check if we have the expected fields
  expected_fields <- c("tb_n_infected", "tb_n_latent_slow", "tb_n_latent_fast", "tb_n_susceptible", "tb_n_active")
  available_fields <- intersect(expected_fields, names(results))
  
  cat("\nAvailable population fields:", paste(available_fields, collapse = ", "), "\n")
  
  if (length(available_fields) > 0) {
    cat("✓ Found population data in results\n")
    
    # Show some sample data
    cat("\nSample data (first 5 time points):\n")
    for (field in available_fields) {
      cat(field, ":", paste(head(results[[field]], 5), collapse = ", "), "\n")
    }
    
    cat("\n🎉 Working TBsim integration test passed!\n")
    cat("Simulation duration:", round(max(time_years), 2), "years\n")
    cat("Final infected count:", max(unlist(results$tb_n_infected), na.rm = TRUE), "\n")
    cat("Final latent slow count:", max(unlist(results$tb_n_latent_slow), na.rm = TRUE), "\n")
    cat("Final latent fast count:", max(unlist(results$tb_n_latent_fast), na.rm = TRUE), "\n")
    cat("Final active count:", max(unlist(results$tb_n_active), na.rm = TRUE), "\n")
  } else {
    cat("❌ No expected population fields found in results\n")
  }
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Working TBsim integration failed.\n")
  
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

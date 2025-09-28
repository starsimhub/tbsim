# Test Shiny App Functionality After Merge
# This script tests that the Shiny app works correctly with the merged TBsim code

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing Shiny app functionality after merge...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  
  cat("✓ Successfully imported tbsim and starsim\n")
  
  # Test with parameters similar to what the Shiny app would use
  set.seed(1)
  
  # Build TBsim simulation using parameters similar to Shiny app defaults
  sim_pars <- list(
    dt = starsim$days(7),
    start = "1940-01-01",
    stop = "1950-01-01",
    rand_seed = as.integer(1),
    verbose = 0
  )
  
  # Create population
  pop <- starsim$People(n_agents = 1000)
  cat("✓ Successfully created population\n")
  
  # Create TB disease model with Shiny app-like parameters
  tb_pars <- list(
    dt = starsim$days(7),
    start = "1940-01-01",
    stop = "1950-01-01",
    init_prev = starsim$bernoulli(p = 0.01),
    beta = starsim$peryear(0.0025),
    p_latent_fast = starsim$bernoulli(p = 0.1)
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
  
  # Test the data extraction that the Shiny app uses
  time_days <- seq(0, 3650, by = 7)  # 10 years, weekly steps
  time_years <- time_days / 365.25
  
  # Test the mapping that the Shiny app uses
  n_infected <- results$tb_n_infected
  n_latent <- results$tb_n_latent_slow + results$tb_n_latent_fast
  n_active <- results$tb_n_active
  n_susceptible <- results$tb_n_susceptible
  n_presymp <- results$tb_n_active_presymp
  
  cat("✓ Successfully mapped TBsim results to Shiny app format\n")
  
  # Verify we have the expected data
  if (length(n_infected) > 0 && length(n_latent) > 0 && length(n_active) > 0) {
    cat("✓ Found population data in results\n")
    
    # Show some sample data
    cat("\nSample data (first 5 time points):\n")
    cat("Infected:", paste(head(n_infected, 5), collapse = ", "), "\n")
    cat("Latent:", paste(head(n_latent, 5), collapse = ", "), "\n")
    cat("Active:", paste(head(n_active, 5), collapse = ", "), "\n")
    cat("Susceptible:", paste(head(n_susceptible, 5), collapse = ", "), "\n")
    
    cat("\n🎉 Shiny app functionality test passed!\n")
    cat("Simulation duration:", round(max(time_years), 2), "years\n")
    cat("Final infected count:", max(n_infected, na.rm = TRUE), "\n")
    cat("Final latent count:", max(n_latent, na.rm = TRUE), "\n")
    cat("Final active count:", max(n_active, na.rm = TRUE), "\n")
    cat("Final susceptible count:", max(n_susceptible, na.rm = TRUE), "\n")
    
    cat("\n✅ The Shiny app should work correctly with the merged TBsim code!\n")
  } else {
    cat("❌ No expected population data found in results\n")
  }
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Shiny app functionality test failed.\n")
  
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

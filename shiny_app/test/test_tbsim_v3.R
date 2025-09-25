# Test TBsim Integration with Starsim 3.0.2
# This script tests if the TBsim model can be run from R using the correct API

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing TBsim integration with Starsim 3.0.2...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  sciris <- import("sciris")
  
  cat("✓ Successfully imported tbsim, starsim, and sciris\n")
  
  # Test basic TBsim model creation using correct API
  tb_pars <- list(
    dt = starsim$days(7),
    start = '1940-01-01',
    stop = '2010-01-01',
    init_prev = starsim$bernoulli(p = 0.01),
    beta = starsim$peryear(0.025),
    p_latent_fast = starsim$bernoulli(p = 0.1)
  )
  
  tb <- tbsim$TB(pars = tb_pars)
  cat("✓ Successfully created TB model\n")
  
  # Test population creation
  pop <- starsim$People(n_agents = 1000)
  cat("✓ Successfully created population\n")
  
  # Test network creation
  net <- starsim$RandomNet(list(
    n_contacts = starsim$poisson(lam = 5),
    dur = 0
  ))
  cat("✓ Successfully created network\n")
  
  # Test demographic processes
  births <- starsim$Births(pars = list(birth_rate = 20))
  deaths <- starsim$Deaths(pars = list(death_rate = 15))
  cat("✓ Successfully created demographic processes\n")
  
  # Test simulation creation
  sim <- starsim$Sim(
    people = pop,
    networks = net,
    diseases = tb,
    demographics = list(deaths, births),
    pars = list(
      dt = starsim$days(7),
      start = '1940-01-01',
      stop = '2010-01-01',
      rand_seed = as.integer(1),
      verbose = 0
    )
  )
  cat("✓ Successfully created simulation\n")
  
  # Test running simulation
  sim$run()
  cat("✓ Successfully ran simulation\n")
  
  # Test extracting results
  results <- sim$results$flatten()
  cat("✓ Successfully extracted results\n")
  
  # Test time conversion
  time_years <- results$t / 365.25
  cat("✓ Successfully converted time to years\n")
  
  cat("\n🎉 All tests passed! TBsim integration is working correctly.\n")
  cat("Simulation duration:", round(max(time_years), 2), "years\n")
  cat("Final infected count:", max(results$n_infected, na.rm = TRUE), "\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("TBsim integration failed.\n")
  
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

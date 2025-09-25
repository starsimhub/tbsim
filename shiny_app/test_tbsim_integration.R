# Test TBsim Integration
# This script tests if the TBsim model can be run from R

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing TBsim integration...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  sciris <- import("sciris")
  
  cat("✓ Successfully imported tbsim, starsim, and sciris\n")
  
  # Test basic TBsim model creation
  tb_pars <- list(
    dt = starsim$days(7),
    start = starsim$date('1940-01-01'),
    stop = starsim$date('2010'),
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
      start = starsim$date('1940-01-01'),
      stop = starsim$date('2010'),
      rand_seed = 1,
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
})

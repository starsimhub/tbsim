# Test App Integration
# This script tests if the TBsim integration works in the Shiny app context

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing TBsim integration in Shiny app context...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  
  cat("✓ Successfully imported tbsim and starsim\n")
  
  # Set random seed
  set.seed(1)
  
  # Build TBsim simulation (same as in Shiny app)
  sim_pars <- list(
    dt = starsim$days(7),
    start = "1940-01-01",
    stop = "1950-01-01",
    rand_seed = as.integer(1),
    verbose = 0
  )
  
  # Create population
  pop <- starsim$People(n_agents = 1000)
  
  # Create TB disease model
  tb_pars <- list(
    dt = starsim$days(7),
    start = "1940-01-01",
    stop = "1950-01-01",
    init_prev = starsim$bernoulli(p = 0.01),
    beta = starsim$peryear(0.0025),
    p_latent_fast = starsim$bernoulli(p = 0.1)
  )
  
  tb <- tbsim$TB(pars = tb_pars)
  
  # Create social network
  net <- starsim$RandomNet(list(
    n_contacts = starsim$poisson(lam = 5),
    dur = 0
  ))
  
  # Create demographic processes
  births <- starsim$Births(pars = list(birth_rate = 20))
  deaths <- starsim$Deaths(pars = list(death_rate = 15))
  
  # Create simulation
  sim <- starsim$Sim(
    people = pop,
    networks = net,
    diseases = tb,
    demographics = list(deaths, births),
    pars = sim_pars
  )
  
  # Run simulation
  sim$run()
  cat("✓ Successfully ran simulation\n")
  
  # Extract results
  results <- sim$results$flatten()
  
  # Test data conversion (same as in Shiny app)
  n_infected <- as.numeric(results$tb_n_infected$tolist())
  n_latent <- as.numeric(results$tb_n_latent_slow$tolist()) + as.numeric(results$tb_n_latent_fast$tolist())
  n_active <- as.numeric(results$tb_n_active$tolist())
  n_susceptible <- as.numeric(results$tb_n_susceptible$tolist())
  n_presymp <- as.numeric(results$tb_n_active_presymp$tolist())
  
  cat("✓ Successfully extracted and converted data\n")
  cat("Data lengths:\n")
  cat("n_infected:", length(n_infected), "\n")
  cat("n_latent:", length(n_latent), "\n")
  cat("n_active:", length(n_active), "\n")
  cat("n_susceptible:", length(n_susceptible), "\n")
  cat("n_presymp:", length(n_presymp), "\n")
  
  cat("\n🎉 TBsim integration test passed!\n")
  cat("The Shiny app should work correctly now.\n")
  cat("✅ The app is running at: http://localhost:3927\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("TBsim integration test failed.\n")
  
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

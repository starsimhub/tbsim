# Debug Plot Data
# This script debugs the data structure issues with plotting

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Debugging plot data structure...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  
  # Set random seed
  set.seed(1)
  
  # Build TBsim simulation
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
  
  # Extract results
  results <- sim$results$flatten()
  
  # Create time vector
  time_days <- seq(0, 3650, by = 7)
  time_years <- time_days / 365.25
  
  # Extract data and check types
  cat("Checking data types...\n")
  
  n_infected <- results$tb_n_infected
  n_latent_slow <- results$tb_n_latent_slow
  n_latent_fast <- results$tb_n_latent_fast
  n_latent <- n_latent_slow + n_latent_fast
  n_active <- results$tb_n_active
  n_susceptible <- results$tb_n_susceptible
  n_presymp <- results$tb_n_active_presymp
  
  cat("Data types:\n")
  cat("time_years:", class(time_years), "length:", length(time_years), "\n")
  cat("n_infected:", class(n_infected), "length:", length(n_infected), "\n")
  cat("n_latent:", class(n_latent), "length:", length(n_latent), "\n")
  cat("n_active:", class(n_active), "length:", length(n_active), "\n")
  cat("n_susceptible:", class(n_susceptible), "length:", length(n_susceptible), "\n")
  cat("n_presymp:", class(n_presymp), "length:", length(n_presymp), "\n")
  
  # Check if data is in environment format
  cat("\nChecking if data is in environment format:\n")
  cat("n_infected is environment:", is.environment(n_infected), "\n")
  cat("n_latent is environment:", is.environment(n_latent), "\n")
  cat("n_active is environment:", is.environment(n_active), "\n")
  
  # Try to convert to proper format
  cat("\nConverting data to proper format...\n")
  
  # Convert to numeric vectors
  n_infected_vec <- as.numeric(unlist(n_infected))
  n_latent_vec <- as.numeric(unlist(n_latent))
  n_active_vec <- as.numeric(unlist(n_active))
  n_susceptible_vec <- as.numeric(unlist(n_susceptible))
  n_presymp_vec <- as.numeric(unlist(n_presymp))
  
  cat("Converted data types:\n")
  cat("n_infected_vec:", class(n_infected_vec), "length:", length(n_infected_vec), "\n")
  cat("n_latent_vec:", class(n_latent_vec), "length:", length(n_latent_vec), "\n")
  cat("n_active_vec:", class(n_active_vec), "length:", length(n_active_vec), "\n")
  cat("n_susceptible_vec:", class(n_susceptible_vec), "length:", length(n_susceptible_vec), "\n")
  cat("n_presymp_vec:", class(n_presymp_vec), "length:", length(n_presymp_vec), "\n")
  
  # Show sample data
  cat("\nSample data (first 5 values):\n")
  cat("n_infected_vec:", paste(head(n_infected_vec, 5), collapse = ", "), "\n")
  cat("n_latent_vec:", paste(head(n_latent_vec, 5), collapse = ", "), "\n")
  cat("n_active_vec:", paste(head(n_active_vec, 5), collapse = ", "), "\n")
  cat("n_susceptible_vec:", paste(head(n_susceptible_vec, 5), collapse = ", "), "\n")
  cat("n_presymp_vec:", paste(head(n_presymp_vec, 5), collapse = ", "), "\n")
  
  cat("\n✅ Data conversion successful!\n")
  cat("The issue is that the TBsim results are in environment format and need to be converted to numeric vectors.\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Debug failed.\n")
})

# Test Final Plot Functionality
# This script tests the final fixed plotting functionality

library(reticulate)
library(plotly)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing final fixed plot functionality...\n")

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
  
  # Extract data with the correct conversion method
  n_infected <- as.numeric(results$tb_n_infected$tolist())
  n_latent <- as.numeric(results$tb_n_latent_slow$tolist()) + as.numeric(results$tb_n_latent_fast$tolist())
  n_active <- as.numeric(results$tb_n_active$tolist())
  n_susceptible <- as.numeric(results$tb_n_susceptible$tolist())
  n_presymp <- as.numeric(results$tb_n_active_presymp$tolist())
  
  cat("✓ Successfully extracted and converted data\n")
  cat("Data types:\n")
  cat("n_infected:", class(n_infected), "length:", length(n_infected), "\n")
  cat("n_latent:", class(n_latent), "length:", length(n_latent), "\n")
  cat("n_active:", class(n_active), "length:", length(n_active), "\n")
  cat("n_susceptible:", class(n_susceptible), "length:", length(n_susceptible), "\n")
  cat("n_presymp:", class(n_presymp), "length:", length(n_presymp), "\n")
  
  # Test the exact plotting code from the fixed Shiny app
  p <- plot_ly() %>%
    add_trace(
      x = time_years,
      y = n_susceptible,
      type = 'scatter',
      mode = 'lines',
      name = 'Susceptible',
      line = list(color = 'green')
    ) %>%
    add_trace(
      x = time_years,
      y = n_infected,
      type = 'scatter',
      mode = 'lines',
      name = 'Total Infected',
      line = list(color = 'red')
    ) %>%
    add_trace(
      x = time_years,
      y = n_latent,
      type = 'scatter',
      mode = 'lines',
      name = 'Latent TB',
      line = list(color = 'orange')
    ) %>%
    add_trace(
      x = time_years,
      y = n_presymp,
      type = 'scatter',
      mode = 'lines',
      name = 'Pre-symptomatic',
      line = list(color = 'purple')
    ) %>%
    add_trace(
      x = time_years,
      y = n_active,
      type = 'scatter',
      mode = 'lines',
      name = 'Active TB',
      line = list(color = 'darkred')
    ) %>%
    layout(
      title = "TB Simulation Results (Real TBsim Model)",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Number of Individuals"),
      hovermode = 'x unified'
    )
  
  cat("✓ Successfully created plotly object\n")
  
  # Test if we can print the plot
  print(p)
  cat("✓ Successfully displayed plot\n")
  
  cat("\n🎉 Final plot functionality test passed!\n")
  cat("The plots should now work correctly in the Shiny app!\n")
  cat("✅ The issue has been fixed - plots should now display properly!\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Final plot functionality test failed.\n")
})

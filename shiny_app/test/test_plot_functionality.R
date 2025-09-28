# Test Plot Functionality
# This script tests if the plotting functionality works correctly

library(reticulate)
library(plotly)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing plot functionality...\n")

tryCatch({
  # Import required Python modules
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  
  cat("✓ Successfully imported tbsim and starsim\n")
  
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
  cat("✓ Successfully ran simulation\n")
  
  # Extract results
  results <- sim$results$flatten()
  
  # Create time vector
  time_days <- seq(0, 3650, by = 7)
  time_years <- time_days / 365.25
  
  # Extract data like the Shiny app does
  n_infected <- results$tb_n_infected
  n_latent <- results$tb_n_latent_slow + results$tb_n_latent_fast
  n_active <- results$tb_n_active
  n_susceptible <- results$tb_n_susceptible
  n_presymp <- results$tb_n_active_presymp
  
  cat("✓ Successfully extracted data\n")
  
  # Test plotly functionality
  cat("Testing plotly functionality...\n")
  
  # Create a simple test plot
  test_plot <- plot_ly() %>%
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
    layout(
      title = "Test TB Simulation Results",
      xaxis = list(title = "Time (years)"),
      yaxis = list(title = "Number of Individuals"),
      hovermode = 'x unified'
    )
  
  cat("✓ Successfully created plotly object\n")
  
  # Test if we can print the plot
  print(test_plot)
  cat("✓ Successfully displayed plot\n")
  
  # Test the exact plotting code from the Shiny app
  cat("Testing Shiny app plotting code...\n")
  
  # This is the exact code from the Shiny app
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
  
  cat("✓ Successfully created Shiny app plot\n")
  print(p)
  cat("✓ Successfully displayed Shiny app plot\n")
  
  cat("\n🎉 Plot functionality test passed!\n")
  cat("The plotting should work in the Shiny app.\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Plot functionality test failed.\n")
  
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

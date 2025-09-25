# Test Data Conversion
# This script tests different ways to convert TBsim results to numeric vectors

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing data conversion methods...\n")

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
  
  cat("Testing different conversion methods...\n")
  
  # Method 1: Try to access the underlying numpy array
  tryCatch({
    n_infected_1 <- results$tb_n_infected$values
    cat("Method 1 (values):", class(n_infected_1), "length:", length(n_infected_1), "\n")
  }, error = function(e) {
    cat("Method 1 failed:", e$message, "\n")
  })
  
  # Method 2: Try to convert to list first
  tryCatch({
    n_infected_2 <- as.list(results$tb_n_infected)
    n_infected_2 <- as.numeric(n_infected_2)
    cat("Method 2 (list):", class(n_infected_2), "length:", length(n_infected_2), "\n")
  }, error = function(e) {
    cat("Method 2 failed:", e$message, "\n")
  })
  
  # Method 3: Try to access the data directly
  tryCatch({
    n_infected_3 <- results$tb_n_infected$data
    cat("Method 3 (data):", class(n_infected_3), "length:", length(n_infected_3), "\n")
  }, error = function(e) {
    cat("Method 3 failed:", e$message, "\n")
  })
  
  # Method 4: Try to convert using reticulate
  tryCatch({
    n_infected_4 <- reticulate::py_to_r(results$tb_n_infected)
    cat("Method 4 (py_to_r):", class(n_infected_4), "length:", length(n_infected_4), "\n")
  }, error = function(e) {
    cat("Method 4 failed:", e$message, "\n")
  })
  
  # Method 5: Try to access the numpy array directly
  tryCatch({
    n_infected_5 <- results$tb_n_infected$numpy()
    cat("Method 5 (numpy):", class(n_infected_5), "length:", length(n_infected_5), "\n")
  }, error = function(e) {
    cat("Method 5 failed:", e$message, "\n")
  })
  
  # Method 6: Try to convert using as.vector
  tryCatch({
    n_infected_6 <- as.vector(results$tb_n_infected)
    cat("Method 6 (as.vector):", class(n_infected_6), "length:", length(n_infected_6), "\n")
  }, error = function(e) {
    cat("Method 6 failed:", e$message, "\n")
  })
  
  # Method 7: Try to access the underlying data
  tryCatch({
    n_infected_7 <- results$tb_n_infected$tolist()
    n_infected_7 <- as.numeric(n_infected_7)
    cat("Method 7 (tolist):", class(n_infected_7), "length:", length(n_infected_7), "\n")
  }, error = function(e) {
    cat("Method 7 failed:", e$message, "\n")
  })
  
  cat("\nTrying to find the working method...\n")
  
  # Check what attributes are available
  cat("Available attributes:\n")
  attrs <- reticulate::py_get_attr_names(results$tb_n_infected)
  cat(paste(attrs, collapse = ", "), "\n")
  
}, error = function(e) {
  cat("❌ Error:", e$message, "\n")
  cat("Data conversion test failed.\n")
})

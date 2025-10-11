#!/usr/bin/env Rscript
# Validation script for TBsim Shiny App
# Tests that all components work together

cat("==============================================\n")
cat("TBsim Shiny App Validation Script\n")
cat("==============================================\n\n")

# Test 1: Load required packages
cat("1. Testing R package loading...\n")
tryCatch({
  library(shiny)
  library(plotly)
  library(DT)
  library(reticulate)
  library(shinydashboard)
  library(shinyBS)
  cat("   ✓ All R packages loaded successfully\n\n")
}, error = function(e) {
  cat("   ✗ Error loading R packages:", e$message, "\n\n")
  quit(status = 1)
})

# Test 2: Configure Python environment
cat("2. Testing Python environment configuration...\n")
tryCatch({
  # Use the same configure_python_env function from app.R
  configure_python_env <- function() {
    venv <- Sys.getenv("VIRTUAL_ENV", unset = NA)
    if (!is.na(venv) && nzchar(venv)) {
      python_bin <- file.path(venv, "bin", "python")
      if (file.exists(python_bin)) {
        use_python(python_bin, required = TRUE)
        return(invisible())
      }
    }
    
    workspace_venv <- "/Users/mine/gitweb/tbsim/venv/bin/python"
    if (file.exists(workspace_venv)) {
      use_python(workspace_venv, required = TRUE)
      return(invisible())
    }
    
    relative_paths <- c(
      "../venv/bin/python",
      "venv/bin/python",
      "../.venv/bin/python",
      ".venv/bin/python"
    )
    
    for (rel_path in relative_paths) {
      if (file.exists(rel_path)) {
        use_python(rel_path, required = TRUE)
        return(invisible())
      }
    }
    
    use_python("python3", required = TRUE)
  }
  
  configure_python_env()
  
  # Force Python initialization by accessing config
  config <- py_config()
  
  if (py_available()) {
    cat("   ✓ Python environment configured successfully\n")
    cat("   Python path:", config$python, "\n\n")
  } else {
    cat("   ✗ Python not available\n\n")
    quit(status = 1)
  }
}, error = function(e) {
  cat("   ✗ Error configuring Python:", e$message, "\n\n")
  quit(status = 1)
})

# Test 3: Import Python modules
cat("3. Testing Python module imports...\n")
tryCatch({
  tbsim <- import("tbsim")
  starsim <- import("starsim")
  sciris <- import("sciris")
  matplotlib <- import("matplotlib")
  plt <- import("matplotlib.pyplot")
  np <- import("numpy")
  pd <- import("pandas")
  cat("   ✓ All Python modules imported successfully\n\n")
}, error = function(e) {
  cat("   ✗ Error importing Python modules:", e$message, "\n\n")
  quit(status = 1)
})

# Test 4: Test basic TBsim functionality
cat("4. Testing basic TBsim functionality...\n")
tryCatch({
  # Create a minimal simulation
  sim_pars <- list(
    dt = starsim$days(7),
    start = "2020-01-01",
    stop = "2020-12-31",
    rand_seed = 1L,
    verbose = 0
  )
  
  pop <- starsim$People(n_agents = 100L)
  
  tb_pars <- list(
    dt = starsim$days(7),
    start = "2020-01-01",
    stop = "2020-12-31",
    init_prev = starsim$bernoulli(p = 0.05),
    beta = starsim$peryear(0.1),
    p_latent_fast = starsim$bernoulli(p = 0.1)
  )
  
  tb <- tbsim$TB(pars = tb_pars)
  
  net <- starsim$RandomNet(list(
    n_contacts = starsim$poisson(lam = 5),
    dur = 0
  ))
  
  births <- starsim$Births(pars = list(birth_rate = 20))
  deaths <- starsim$Deaths(pars = list(death_rate = 15))
  
  sim <- starsim$Sim(
    people = pop,
    networks = list(net),
    diseases = tb,
    demographics = list(deaths, births),
    pars = sim_pars
  )
  
  sim$run()
  
  results <- sim$results$flatten()
  
  if (!is.null(results$tb_n_infected)) {
    cat("   ✓ TBsim simulation completed successfully\n")
    cat("   Simulation ran for", length(results$tb_n_infected$tolist()), "time steps\n\n")
  } else {
    cat("   ✗ Simulation results missing\n\n")
    quit(status = 1)
  }
}, error = function(e) {
  cat("   ✗ Error running TBsim:", e$message, "\n\n")
  quit(status = 1)
})

# Test 5: Check app.R syntax
cat("5. Testing app.R syntax...\n")
tryCatch({
  # Just parse the file, don't run it
  parse("app.R")
  cat("   ✓ app.R has valid R syntax\n\n")
}, error = function(e) {
  cat("   ✗ Syntax error in app.R:", e$message, "\n\n")
  quit(status = 1)
})

cat("==============================================\n")
cat("✓ ALL TESTS PASSED!\n")
cat("The Shiny app is ready to run.\n")
cat("==============================================\n\n")
cat("To start the app, run:\n")
cat("  Rscript -e \"shiny::runApp('app.R', port=8080, host='0.0.0.0')\"\n")
cat("or simply:\n")
cat("  R -e \"shiny::runApp('app.R')\"\n\n")


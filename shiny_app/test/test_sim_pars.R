# Test script to print sim$pars contents
# This script will run a simulation and print the parameters

cat("Testing sim$pars contents...\n")

# Load required libraries
library(reticulate)

# Set up Python environment
venv_path <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_path)) {
  use_python(venv_path, required = TRUE)
  cat("✓ Using virtual environment Python\n")
} else {
  use_python("python3", required = TRUE)
  cat("✓ Using system Python\n")
}

# Import Python modules
starsim <- import("starsim")
tbsim <- import("tbsim")

cat("✓ Python modules imported successfully\n")

# Set up simulation parameters
sim_pars <- list(
  dt = starsim$days(1),
  start = "1940-01-01",
  stop = "2010-01-01",
  rand_seed = 1L,
  verbose = 0
)

# Create population
pop <- starsim$People(n_agents = 1000)

# Set up TB parameters
tb_pars <- list(
  dt = starsim$days(1),
  start = "1940-01-01",
  stop = "2010-01-01",
  init_prev = starsim$bernoulli(p = 0.1),
  beta = starsim$peryear(0.5),
  p_latent_fast = starsim$bernoulli(p = 0.1),
  rate_LS_to_presym = starsim$perday(0.001),
  rate_LF_to_presym = starsim$perday(0.01),
  rate_presym_to_active = starsim$perday(0.1),
  rate_active_to_clear = starsim$perday(0.01),
  rate_treatment_to_clear = starsim$peryear(0.8),
  rate_exptb_to_dead = starsim$perday(0.05),
  rate_smpos_to_dead = starsim$perday(0.02),
  rate_smneg_to_dead = starsim$perday(0.01),
  rel_trans_presymp = 0.5,
  rel_trans_smpos = 1.0,
  rel_trans_smneg = 0.3,
  rel_trans_exptb = 0.1,
  rel_trans_treatment = 0.05,
  rel_sus_latentslow = 0.1,
  cxr_asymp_sens = 0.7,
  reltrans_het = starsim$constant(v = 1.0)
)

# Create TB model
tb <- tbsim$TB(pars = tb_pars)

# Create network
net <- starsim$RandomNet(list(
  n_contacts = starsim$poisson(lam = 10),
  dur = 0
))

# Create demographics
births <- starsim$Births(pars = list(birth_rate = 0.02))
deaths <- starsim$Deaths(pars = list(death_rate = 0.01))

# Create simulation
sim <- starsim$Sim(
  people = pop,
  networks = net,
  diseases = tb,
  demographics = list(deaths, births),
  pars = sim_pars
)

cat("✓ Simulation object created\n")

# Print sim$pars contents
cat("\n=== SIMULATION PARAMETERS (sim$pars) ===\n")
cat("Type:", class(sim$pars), "\n")
cat("Length:", length(sim$pars), "\n\n")

# Print each parameter
for (i in seq_along(sim$pars)) {
  param_name <- names(sim$pars)[i]
  param_value <- sim$pars[[i]]
  
  cat(sprintf("Parameter %d: %s\n", i, param_name))
  cat(sprintf("  Type: %s\n", class(param_value)))
  cat(sprintf("  Value: %s\n", toString(param_value)))
  
  # Try to get more details for specific types
  if (inherits(param_value, "python.builtin.object")) {
    tryCatch({
      cat(sprintf("  Python type: %s\n", py_type(param_value)))
      if (py_has_attr(param_value, "__str__")) {
        cat(sprintf("  String representation: %s\n", py_str(param_value)))
      }
    }, error = function(e) {
      cat("  Could not get Python details\n")
    })
  }
  cat("\n")
}

# Also try to access specific parameters
cat("=== SPECIFIC PARAMETER ACCESS ===\n")
tryCatch({
  cat("dt:", sim$pars$dt, "\n")
}, error = function(e) {
  cat("Could not access dt:", e$message, "\n")
})

tryCatch({
  cat("start:", sim$pars$start, "\n")
}, error = function(e) {
  cat("Could not access start:", e$message, "\n")
})

tryCatch({
  cat("stop:", sim$pars$stop, "\n")
}, error = function(e) {
  cat("Could not access stop:", e$message, "\n")
})

tryCatch({
  cat("rand_seed:", sim$pars$rand_seed, "\n")
}, error = function(e) {
  cat("Could not access rand_seed:", e$message, "\n")
})

cat("\n=== TEST COMPLETED ===\n")

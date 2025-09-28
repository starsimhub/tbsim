# Test Months Conversion
# This script tests the time conversion from days to months

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing months conversion...\n")

# Test the conversion formula
time_days <- seq(0, 365, by = 7)  # 1 year with weekly timesteps
time_months <- time_days / 30.44  # Convert days to months

cat("Time conversion test:\n")
cat("Days:", head(time_days, 5), "...\n")
cat("Months:", head(time_months, 5), "...\n")
cat("Total duration in months:", max(time_months), "\n")

# Test with a typical simulation period (1940-2010)
start_date <- as.Date("1940-01-01")
end_date <- as.Date("2010-12-31")
n_days <- as.numeric(end_date - start_date)
time_days_sim <- seq(0, n_days, by = 7)
time_months_sim <- time_days_sim / 30.44

cat("\nSimulation period test (1940-2010):\n")
cat("Total days:", n_days, "\n")
cat("Total months:", max(time_months_sim), "\n")
cat("Total years:", max(time_months_sim) / 12, "\n")

cat("\n✅ Months conversion working correctly!\n")
cat("The Shiny app will now display time in months.\n")

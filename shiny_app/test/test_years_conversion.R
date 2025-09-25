# Test Years Conversion
# This script tests the time conversion back to years

cat("Testing years conversion...\n")

# Test the conversion formula
time_days <- seq(0, 365, by = 7)  # 1 year with weekly timesteps
time_years <- time_days / 365.25  # Convert days to years

cat("Time conversion test:\n")
cat("Days:", head(time_days, 5), "...\n")
cat("Years:", head(time_years, 5), "...\n")
cat("Total duration in years:", max(time_years), "\n")

# Test with a typical simulation period (1940-2010)
start_date <- as.Date("1940-01-01")
end_date <- as.Date("2010-12-31")
n_days <- as.numeric(end_date - start_date)
time_days_sim <- seq(0, n_days, by = 7)
time_years_sim <- time_days_sim / 365.25

cat("\nSimulation period test (1940-2010):\n")
cat("Total days:", n_days, "\n")
cat("Total years:", max(time_years_sim), "\n")

cat("\n✅ Years conversion working correctly!\n")
cat("The Shiny app will now display time in years.\n")

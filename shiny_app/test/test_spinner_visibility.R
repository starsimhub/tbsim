# Test Spinner Visibility
# This script tests if the loading spinner is visible and working

library(reticulate)

# Set up Python environment for tbsim
venv_python <- "/Users/mine/gitweb/FORK-tbsim/venv/bin/python"
if (file.exists(venv_python)) {
  use_python(venv_python, required = TRUE)
} else {
  use_python("python3", required = TRUE)
}

cat("Testing spinner visibility...\n")

# Check if the app is accessible
app_url <- "http://localhost:3927"
response <- tryCatch({
  curl::curl_fetch_memory(app_url)
}, error = function(e) {
  cat("❌ Could not connect to app:", e$message, "\n")
  return(NULL)
})

if (!is.null(response) && response$status_code == 200) {
  cat("✓ App is accessible\n")
  
  # Check if spinner elements exist in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("spinner-border", html_content)) {
    cat("✓ Loading spinner HTML found\n")
  } else {
    cat("❌ Loading spinner HTML not found\n")
  }
  
  if (grepl("shiny-panel-conditional", html_content)) {
    cat("✓ Conditional panel HTML found\n")
  } else {
    cat("❌ Conditional panel HTML not found\n")
  }
  
  if (grepl("simulation_running", html_content)) {
    cat("✓ Simulation running reactive found\n")
  } else {
    cat("❌ Simulation running reactive not found\n")
  }
  
  # Check if the conditional panel has the right condition
  if (grepl("output.simulation_running == true", html_content)) {
    cat("✓ Conditional panel condition found\n")
  } else {
    cat("❌ Conditional panel condition not found\n")
  }
  
  cat("\n🔍 Debugging spinner visibility...\n")
  cat("The spinner should appear when 'Run Simulation' is clicked.\n")
  cat("If it's not showing, the issue might be:\n")
  cat("  1. The reactive output isn't updating properly\n")
  cat("  2. The conditional panel condition isn't working\n")
  cat("  3. The spinner is hidden by CSS\n")
  
  cat("\n✅ App is running at:", app_url, "\n")
  cat("Try clicking 'Run Simulation' to see if the spinner appears.\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

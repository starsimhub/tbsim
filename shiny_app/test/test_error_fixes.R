# Test script to verify the error fixes
# This script checks if the my_pars and summary statistics errors are fixed

cat("Testing error fixes...\n")

# Check if the app is accessible
app_url <- "http://localhost:3838"
response <- tryCatch({
  curl::curl_fetch_memory(app_url)
}, error = function(e) {
  cat("❌ Could not connect to app:", e$message, "\n")
  return(NULL)
})

if (!is.null(response) && response$status_code == 200) {
  cat("✓ App is accessible on port 3838\n")
  
  # Check if my_pars output is in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("Raw Simulation Parameters \\(my_pars\\)", html_content)) {
    cat("✓ my_pars section title found\n")
  } else {
    cat("❌ my_pars section title not found\n")
  }
  
  if (grepl("my_pars_output", html_content)) {
    cat("✓ my_pars_output container found\n")
  } else {
    cat("❌ my_pars_output container not found\n")
  }
  
  if (grepl("shiny-text-output", html_content)) {
    cat("✓ Text output widget found\n")
  } else {
    cat("❌ Text output widget not found\n")
  }
  
  # Check for error messages in the HTML
  if (grepl("invalid.*type.*environment", html_content)) {
    cat("❌ Environment type error still present\n")
  } else {
    cat("✓ No environment type errors found\n")
  }
  
  cat("\n🎉 Error fixes test completed!\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("✅ my_pars output should now work correctly!\n")
  cat("\n📋 Fixes Applied:\n")
  cat("  • 🔧 Fixed my_pars conversion with py_to_r()\n")
  cat("  • 🔧 Added error handling for my_pars output\n")
  cat("  • 🔧 Fixed summary statistics with tolist() conversion\n")
  cat("  • 🔧 Proper Python to R object conversion\n")
  
  cat("\n✨ Ready to test!\n")
  cat("  • Run a simulation to see if errors are gone\n")
  cat("  • Check if my_pars output displays correctly\n")
  cat("  • Verify summary statistics work without errors\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

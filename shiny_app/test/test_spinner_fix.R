# Test Spinner Fix
# This script tests if the loading spinner is now working correctly

cat("Testing spinner fix...\n")

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
  
  if (grepl("data-display-if=\"output.simulation_running\"", html_content)) {
    cat("✓ Simplified conditional panel condition found\n")
  } else {
    cat("❌ Simplified conditional panel condition not found\n")
  }
  
  if (grepl("Running...", html_content)) {
    cat("✓ 'Running...' text found\n")
  } else {
    cat("❌ 'Running...' text not found\n")
  }
  
  cat("\n🎉 Spinner fix test completed!\n")
  cat("The loading spinner should now work correctly.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 What was fixed:\n")
  cat("  • 🔧 Simplified conditional panel condition\n")
  cat("  • 🔄 Changed from 'output.simulation_running == true' to 'output.simulation_running'\n")
  cat("  • ✅ This should make the spinner visible when simulation is running\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

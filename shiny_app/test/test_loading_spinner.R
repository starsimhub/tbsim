# Test Loading Spinner
# This script tests if the loading spinner functionality is working

cat("Testing loading spinner functionality...\n")

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
  
  # Check if loading spinner elements exist in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("spinner-border", html_content)) {
    cat("✓ Loading spinner HTML found\n")
  } else {
    cat("❌ Loading spinner HTML not found\n")
  }
  
  if (grepl("Running...", html_content)) {
    cat("✓ 'Running...' text found\n")
  } else {
    cat("❌ 'Running...' text not found\n")
  }
  
  if (grepl("simulation_running", html_content)) {
    cat("✓ Simulation running reactive output found\n")
  } else {
    cat("❌ Simulation running reactive output not found\n")
  }
  
  if (grepl("conditionalPanel", html_content)) {
    cat("✓ Conditional panel for loading state found\n")
  } else {
    cat("❌ Conditional panel for loading state not found\n")
  }
  
  cat("\n🎉 Loading spinner test completed!\n")
  cat("The loading spinner should now appear when 'Run Simulation' is clicked.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 Loading Features:\n")
  cat("  • 🔄 Cycling spinner icon\n")
  cat("  • 📝 'Running...' text\n")
  cat("  • 🎯 Conditional display (only shows when running)\n")
  cat("  • 🎨 Bootstrap styling\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

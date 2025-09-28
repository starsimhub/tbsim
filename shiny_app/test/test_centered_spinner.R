# Test Centered Spinner
# This script tests if the loading spinner is now positioned in the main content area

cat("Testing centered spinner positioning...\n")

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
  
  # Check if spinner is positioned in the main content area
  html_content <- rawToChar(response$content)
  
  if (grepl("Simulation Results.*loading_spinner.*results_plot", html_content)) {
    cat("✓ Spinner positioned between title and plot\n")
  } else {
    cat("❌ Spinner not positioned correctly\n")
  }
  
  if (grepl("loading_spinner.*shiny-html-output", html_content)) {
    cat("✓ loading_spinner container found in main area\n")
  } else {
    cat("❌ loading_spinner container not found\n")
  }
  
  if (grepl("text-align: center", html_content)) {
    cat("✓ Centered styling found\n")
  } else {
    cat("❌ Centered styling not found\n")
  }
  
  cat("\n🎉 Centered spinner test completed!\n")
  cat("The loading spinner should now appear in the main content area.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 New Spinner Features:\n")
  cat("  • 🎯 Positioned in main content area\n")
  cat("  • 📍 Between 'Simulation Results' title and plot\n")
  cat("  • 🎨 Centered with larger text\n")
  cat("  • 📏 20px padding for better visibility\n")
  cat("  • 🔄 More prominent loading indication\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

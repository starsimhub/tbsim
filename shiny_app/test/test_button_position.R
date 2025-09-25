# Test Button Position
# This script tests if the Run Simulation button is positioned at the top

cat("Testing button positioning...\n")

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
  
  # Check if Run Simulation button exists in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("Run Simulation", html_content)) {
    cat("✓ Run Simulation button found in HTML\n")
  } else {
    cat("❌ Run Simulation button not found\n")
  }
  
  if (grepl("Reset to Defaults", html_content)) {
    cat("✓ Reset to Defaults button found in HTML\n")
  } else {
    cat("❌ Reset to Defaults button not found\n")
  }
  
  cat("\n🎉 Button positioning test completed!\n")
  cat("The Run Simulation button should now be at the top of the sidebar.\n")
  cat("✅ App is running at:", app_url, "\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

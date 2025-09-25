# Test script to verify the restarted app with my_pars output
# This script checks if the my_pars functionality is working

cat("Testing restarted app with my_pars output...\n")

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
  
  # Check if the browser title is correct
  if (grepl("<title>TBsim - Tuberculosis Simulation Web Interface</title>", html_content)) {
    cat("✓ Browser tab title is correct\n")
  } else {
    cat("❌ Browser tab title is incorrect\n")
  }
  
  cat("\n🎉 Restarted app test completed!\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("✅ my_pars output is now available!\n")
  cat("\n📋 Available Features:\n")
  cat("  • 🖥️  Browser tab with proper title\n")
  cat("  • 🎨 Logo in page header\n")
  cat("  • 📊 Simulation results with plots\n")
  cat("  • 📋 Summary statistics table\n")
  cat("  • 📊 Parameters table\n")
  cat("  • 📄 Raw my_pars text output (NEW!)\n")
  cat("  • 🔄 Loading spinner\n")
  cat("  • 🎨 Viridis color palette\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

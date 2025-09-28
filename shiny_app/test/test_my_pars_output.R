# Test script to verify my_pars output functionality
# This script checks if the my_pars text output is working

cat("Testing my_pars output functionality...\n")

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
  
  if (grepl("verbatimTextOutput", html_content)) {
    cat("✓ verbatimTextOutput widget found\n")
  } else {
    cat("❌ verbatimTextOutput widget not found\n")
  }
  
  cat("\n🎉 my_pars output test completed!\n")
  cat("The my_pars text output should now be available at the bottom of the Results tab.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 New my_pars Output Features:\n")
  cat("  • 📄 Raw simulation parameters display\n")
  cat("  • 🔍 Verbatim text output for debugging\n")
  cat("  • 📊 Complete parameter inspection\n")
  cat("  • 🎯 Located at bottom of Results tab\n")
  cat("\n✨ Benefits:\n")
  cat("  • 🔍 Debug simulation parameters\n")
  cat("  • 📝 Inspect raw parameter values\n")
  cat("  • 🎯 Verify parameter settings\n")
  cat("  • 📊 Complete parameter transparency\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

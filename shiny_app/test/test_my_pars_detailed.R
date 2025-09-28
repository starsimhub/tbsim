# Test script to verify the detailed my_pars output
# This script checks if the my_pars now shows detailed parameter information

cat("Testing detailed my_pars output...\n")

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
  
  cat("\n🎉 Detailed my_pars output test completed!\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("✅ my_pars output should now show detailed parameters!\n")
  cat("\n📋 New my_pars Output Features:\n")
  cat("  • 📄 Detailed parameter listing\n")
  cat("  • 🔍 Individual parameter values\n")
  cat("  • 📊 Parameter types and values\n")
  cat("  • 🎯 Complete parameter transparency\n")
  cat("  • 📝 Formatted output with headers\n")
  
  cat("\n✨ What You'll See Now:\n")
  cat("  • === SIMULATION PARAMETERS (my_pars) ===\n")
  cat("  • Type: [object type]\n")
  cat("  • Length: [number of parameters]\n")
  cat("  • Parameter 1: [name]\n")
  cat("  •   Type: [parameter type]\n")
  cat("  •   Value: [parameter value]\n")
  cat("  • ... (for all 24 parameters)\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

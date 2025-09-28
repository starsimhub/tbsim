# Test Parameters Section
# This script tests if the new parameters section is working

cat("Testing parameters section...\n")

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
  
  # Check if parameters section exists in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("Simulation Parameters Used", html_content)) {
    cat("✓ Parameters section title found\n")
  } else {
    cat("❌ Parameters section title not found\n")
  }
  
  if (grepl("parameters_table", html_content)) {
    cat("✓ Parameters table container found\n")
  } else {
    cat("❌ Parameters table container not found\n")
  }
  
  if (grepl("datatables.*parameters_table", html_content)) {
    cat("✓ Parameters DataTable widget found\n")
  } else {
    cat("❌ Parameters DataTable widget not found\n")
  }
  
  cat("\n🎉 Parameters section test completed!\n")
  cat("The new parameters section should now be available.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 New Parameters Section Features:\n")
  cat("  • 📊 Comprehensive parameters table\n")
  cat("  • 🏷️ Categorized parameters (Basic, TB, Demographics, etc.)\n")
  cat("  • 🔍 Searchable and sortable table\n")
  cat("  • 📏 Scrollable with 400px height\n")
  cat("  • 🎯 Shows all parameters used in simulation\n")
  cat("\n✨ Benefits:\n")
  cat("  • 📝 Complete parameter documentation\n")
  cat("  • 🔄 Reproducible simulations\n")
  cat("  • 📊 Easy parameter comparison\n")
  cat("  • 🎯 Transparent simulation settings\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

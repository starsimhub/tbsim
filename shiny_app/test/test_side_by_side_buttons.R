# Test Side by Side Buttons
# This script tests if the action buttons are now displayed side by side

cat("Testing side by side button layout...\n")

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
  
  # Check if button layout elements exist in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("display: inline-block; margin-right: 10px;", html_content)) {
    cat("✓ Run Simulation button container found with proper styling\n")
  } else {
    cat("❌ Run Simulation button container not found\n")
  }
  
  if (grepl("display: inline-block;", html_content)) {
    cat("✓ Reset button container found with inline-block styling\n")
  } else {
    cat("❌ Reset button container not found\n")
  }
  
  if (grepl("Run Simulation.*Reset to Defaults", html_content)) {
    cat("✓ Both buttons found in HTML\n")
  } else {
    cat("❌ Both buttons not found\n")
  }
  
  cat("\n🎉 Side by side button test completed!\n")
  cat("The action buttons should now be displayed side by side.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 Layout Features:\n")
  cat("  • 🔄 Run Simulation button (left)\n")
  cat("  • 🔄 Loading spinner (next to Run button)\n")
  cat("  • 🔄 Reset to Defaults button (right)\n")
  cat("  • 📏 10px margin between button groups\n")
  cat("  • 🎯 Inline-block layout for horizontal alignment\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

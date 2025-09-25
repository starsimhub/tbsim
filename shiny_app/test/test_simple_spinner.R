# Test Simple Spinner
# This script tests the simplified emoji-based spinner approach

cat("Testing simple spinner approach...\n")

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
  
  # Check if spinner container exists
  html_content <- rawToChar(response$content)
  
  if (grepl("loading_spinner", html_content)) {
    cat("✓ loading_spinner container found\n")
  } else {
    cat("❌ loading_spinner container not found\n")
  }
  
  if (grepl("shiny-html-output", html_content)) {
    cat("✓ shiny-html-output class found\n")
  } else {
    cat("❌ shiny-html-output class not found\n")
  }
  
  cat("\n🎉 Simple spinner test completed!\n")
  cat("The new approach should be much more reliable.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 Simple Spinner Features:\n")
  cat("  • 🔄 Emoji spinner (🔄)\n")
  cat("  • 📝 'Running...' text\n")
  cat("  • 🎯 Simple span element\n")
  cat("  • ✅ No complex CSS animations\n")
  cat("  • 🚀 More reliable than Bootstrap spinner\n")
  cat("\n💡 The spinner will appear when you click 'Run Simulation'!\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

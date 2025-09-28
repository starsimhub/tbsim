# Test New Spinner Approach
# This script tests the new emoji-based spinner approach

cat("Testing new spinner approach...\n")

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
  
  # Check if new spinner elements exist in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("loading_spinner", html_content)) {
    cat("✓ loading_spinner uiOutput found\n")
  } else {
    cat("❌ loading_spinner uiOutput not found\n")
  }
  
  if (grepl("animation: pulse", html_content)) {
    cat("✓ Pulse animation found\n")
  } else {
    cat("❌ Pulse animation not found\n")
  }
  
  if (grepl("Running...", html_content)) {
    cat("✓ 'Running...' text found\n")
  } else {
    cat("❌ 'Running...' text not found\n")
  }
  
  cat("\n🎉 New spinner approach test completed!\n")
  cat("The new emoji-based spinner should be more reliable.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 New Spinner Features:\n")
  cat("  • 🔄 Emoji-based spinner (🔄)\n")
  cat("  • 📝 'Running...' text\n")
  cat("  • 🎨 CSS pulse animation\n")
  cat("  • 🎯 Simpler, more reliable approach\n")
  cat("  • ✅ No complex Bootstrap dependencies\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

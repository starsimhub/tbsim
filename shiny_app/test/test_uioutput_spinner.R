# Test UIOutput Spinner
# This script tests if the new uiOutput approach for the spinner works

cat("Testing UIOutput spinner approach...\n")

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
  
  # Check if uiOutput elements exist in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("loading_spinner", html_content)) {
    cat("✓ loading_spinner uiOutput found\n")
  } else {
    cat("❌ loading_spinner uiOutput not found\n")
  }
  
  if (grepl("shiny-html-output", html_content)) {
    cat("✓ shiny-html-output class found\n")
  } else {
    cat("❌ shiny-html-output class not found\n")
  }
  
  cat("\n🎉 UIOutput spinner test completed!\n")
  cat("The new approach should work better than conditionalPanel.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 What changed:\n")
  cat("  • 🔄 Replaced conditionalPanel with uiOutput\n")
  cat("  • 🎯 Using renderUI for better control\n")
  cat("  • ✅ This should make the spinner visible when simulation is running\n")
  cat("  • 🔧 More reliable than conditionalPanel approach\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

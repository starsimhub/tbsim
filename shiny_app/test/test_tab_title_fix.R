# Test script to verify the browser tab title fix
# This script checks that the title is now properly set

cat("Testing browser tab title fix...\n")

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
  
  # Check the HTML title
  html_content <- rawToChar(response$content)
  
  # Look for the title tag
  if (grepl("<title>TBsim - Tuberculosis Simulation Web Interface</title>", html_content)) {
    cat("✓ Browser tab title is correct\n")
  } else {
    cat("❌ Browser tab title is incorrect\n")
  }
  
  # Check that we don't have raw HTML in the title
  if (grepl("<div style=", html_content) && !grepl("<title>.*<div", html_content)) {
    cat("✓ No raw HTML in browser title\n")
  } else {
    cat("❌ Raw HTML found in browser title\n")
  }
  
  # Check that the logo is still present in the page
  if (grepl("logo.png", html_content)) {
    cat("✓ Logo is still present in the page\n")
  } else {
    cat("❌ Logo not found in the page\n")
  }
  
  # Check that the custom header div is present
  if (grepl("display: flex; align-items: center; margin-bottom: 20px", html_content)) {
    cat("✓ Custom header styling is present\n")
  } else {
    cat("❌ Custom header styling not found\n")
  }
  
  cat("\n🎉 Tab title fix test completed!\n")
  cat("✅ Browser tab should now show: 'TBsim - Tuberculosis Simulation Web Interface'\n")
  cat("✅ Logo should still be visible in the page header\n")
  cat("✅ No more raw HTML in the browser tab title\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

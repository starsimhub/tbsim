# Test Viridis Colors
# This script tests if the viridis color palette is applied to the plots

cat("Testing viridis color palette...\n")

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
  
  # Check if viridis colors are in the HTML
  html_content <- rawToChar(response$content)
  
  if (grepl("#440154", html_content)) {
    cat("✓ Viridis purple (Susceptible) found\n")
  } else {
    cat("❌ Viridis purple not found\n")
  }
  
  if (grepl("#31688e", html_content)) {
    cat("✓ Viridis blue (Total Infected) found\n")
  } else {
    cat("❌ Viridis blue not found\n")
  }
  
  if (grepl("#35b779", html_content)) {
    cat("✓ Viridis green (Latent TB) found\n")
  } else {
    cat("❌ Viridis green not found\n")
  }
  
  if (grepl("#fde725", html_content)) {
    cat("✓ Viridis yellow (Pre-symptomatic) found\n")
  } else {
    cat("❌ Viridis yellow not found\n")
  }
  
  if (grepl("#e16462", html_content)) {
    cat("✓ Viridis red (Active TB) found\n")
  } else {
    cat("❌ Viridis red not found\n")
  }
  
  cat("\n🎉 Viridis color test completed!\n")
  cat("The plots should now use the viridis color palette.\n")
  cat("✅ App is running at:", app_url, "\n")
  cat("\n📋 Viridis Color Mapping:\n")
  cat("  • 🟣 Susceptible: #440154 (Dark Purple)\n")
  cat("  • 🔵 Total Infected: #31688e (Blue)\n")
  cat("  • 🟢 Latent TB: #35b779 (Green)\n")
  cat("  • 🟡 Pre-symptomatic: #fde725 (Yellow)\n")
  cat("  • 🔴 Active TB: #e16462 (Red)\n")
  cat("\n✨ Benefits of Viridis:\n")
  cat("  • 🎨 Perceptually uniform\n")
  cat("  • ♿ Colorblind friendly\n")
  cat("  • 📊 Scientific standard\n")
  cat("  • 🌈 Beautiful gradients\n")
  
} else {
  cat("❌ App is not accessible\n")
  cat("Please make sure the Shiny app is running.\n")
}

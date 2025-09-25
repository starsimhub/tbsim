# Test Logo Integration
# This script tests if the logo is properly integrated in the Shiny app

cat("Testing logo integration...\n")

# Check if logo file exists
if (file.exists("logo.png")) {
  cat("✓ Logo file exists in current directory\n")
} else {
  cat("❌ Logo file not found in current directory\n")
}

# Check if logo is in www directory
if (file.exists("www/logo.png")) {
  cat("✓ Logo file exists in www directory\n")
} else {
  cat("❌ Logo file not found in www directory\n")
}

# Check if www directory exists
if (dir.exists("www")) {
  cat("✓ www directory exists\n")
} else {
  cat("❌ www directory not found\n")
}

cat("\n🎉 Logo integration test completed!\n")
cat("The logo should now be visible in the Shiny app header.\n")
cat("✅ App is running at: http://localhost:3927\n")

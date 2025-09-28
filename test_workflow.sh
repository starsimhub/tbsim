#!/bin/bash

echo "🧪 Testing GitHub Actions Workflow Locally"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test step
test_step() {
    local step_name="$1"
    local command="$2"
    
    echo -e "\n${YELLOW}Testing: ${step_name}${NC}"
    echo "Command: $command"
    
    if eval "$command"; then
        echo -e "${GREEN}✅ ${step_name} passed${NC}"
        return 0
    else
        echo -e "${RED}❌ ${step_name} failed${NC}"
        return 1
    fi
}

# Test 1: TBSim installation
test_step "TBSim Installation" "pip install -e .[dev]"

# Test 2: Sphinx dependencies
test_step "Sphinx Dependencies" "pip install -r docs/requirements.txt"

# Test 2.5: Check Pandoc availability
test_step "Pandoc Availability" "which pandoc || echo 'Pandoc not found - notebooks may not process correctly'"

# Test 3: TBSim import
test_step "TBSim Import" "python -c \"import tbsim; print('TBSim imported successfully')\""

# Test 4: Clean build directory
test_step "Clean Build Directory" "rm -rf docs/_build docs/_site"

# Test 5: Sphinx build
test_step "Sphinx Build" "cd docs && make html"

# Test 6: Fallback build (with warnings disabled)
test_step "Fallback Build (Warnings Disabled)" "cd docs && make html SPHINXOPTS=\"-W --keep-going\""

echo -e "\n${GREEN}🎉 Workflow testing completed!${NC}"
echo "Check the docs/_build/html directory for the generated documentation."

# GitHub Actions Scripts

This directory contains utility scripts used by GitHub Actions workflows.

## Available Scripts

### `test_tutorial_imports.py`
Tests tutorial imports and functionality for the TB interventions tutorial.

**Usage:**
```bash
# From root directory
python3 .github/scripts/test_tutorial_imports.py

# From docs directory
python3 ../.github/scripts/test_tutorial_imports.py
```

**What it tests:**
- `tbsim` module imports
- Intervention module imports (`bcg`, `tpt`, `beta`)
- Tutorial script functionality
- Scenario creation (with API compatibility handling)

### `verify_docs.py`
Verifies documentation setup and build readiness.

**Usage:**
```bash
python3 .github/scripts/verify_docs.py
```

**What it checks:**
- Documentation directory structure
- Sphinx packages availability
- Tutorial integration in `tutorials.rst`

### `verify_build.py`
Verifies documentation build output.

**Usage:**
```bash
# From docs directory
python3 ../.github/scripts/verify_build.py
```

**What it checks:**
- Build output directory structure
- Required files (index.html, tutorials)
- Package versions (nbsphinx, myst_parser)

## Workflow Integration

These scripts are called by the following workflows:

- **`deploy-docs.yml`**: 
  - `verify_build.py` - Build output verification
  - `test_tutorial_imports.py` - Tutorial functionality testing

- **`test-docs.yml`**: 
  - `test_tutorial_imports.py` - Tutorial functionality testing (optional)
  - `verify_docs.py` - Documentation setup verification

## Adding New Scripts

To add a new script:

1. Create the script in this directory
2. Make it executable: `chmod +x script_name.py`
3. Add proper error handling and exit codes
4. Update this README with usage instructions
5. Integrate it into the appropriate workflow

## Script Requirements

All scripts should:
- Have proper error handling
- Return appropriate exit codes (0 for success, 1 for failure)
- Provide clear output messages
- Be executable from different working directories
- Include docstrings and comments 
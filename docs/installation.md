# Installation Guide

This guide covers different ways to install and set up TBsim for your tuberculosis modeling projects.

## Prerequisites

- Python 3.12 or higher
- pip package manager
- Git (for development installation)

## Local Installation

Clone and install TBsim locally:

```bash
git clone https://github.com/starsimhub/tbsim.git
cd tbsim
pip install -e .
```

## Development Installation

For developers who need to build documentation and tests:

```bash
git clone https://github.com/starsimhub/tbsim.git
cd tbsim
pip install -e .[dev]
```

## Starsim library imports

TBsim requires Starsim >=3.4.0. Household networks live in the Starsim library (not core ``starsim``):

```python
import starsim.library.networks as ssln

net = ssln.HouseholdNet(dhs_data=dhs_data, dynamic=False)
```

## Troubleshooting

Common installation issues:

**Import Errors**: Ensure you're using the correct Python environment and that all dependencies are installed.

**Starsim Compatibility**: Requires Starsim >=3.4.0 (`pip install "starsim>=3.4.0"`). See [Starsim library imports](#starsim-library-imports) above.

**Permission Errors**: On some systems, you may need to use `pip install --user` or run with appropriate permissions.

For additional help, please open an issue on the GitHub repository.

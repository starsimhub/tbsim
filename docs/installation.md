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

## Troubleshooting

Common installation issues:

**Import Errors**: Ensure you're using the correct Python environment and that all dependencies are installed.

**Starsim Compatibility**: Make sure you have a compatible version of Starsim installed.

**Permission Errors**: On some systems, you may need to use `pip install --user` or run with appropriate permissions.

For additional help, please open an issue on the GitHub repository.

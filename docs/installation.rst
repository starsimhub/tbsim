Installation Guide
=================

This guide covers different ways to install and set up TBsim for your tuberculosis modeling projects.

Prerequisites
------------

- Python 3.11 or higher
- pip package manager
- Git (for development installation)

Option 1: Online Environment (Google Colab)
------------------------------------------

You can run the tutorials online using Google Colab (links are included at the top of each tutorial notebook).

For Google Colab, run the install cell to install TBsimV2 from GitHub:

.. code-block:: python

   %pip install -q git+https://github.com/starsimhub/tbsimV2.git

Option 2: Local Installation
----------------------------

Clone and install TBsim locally:

.. code-block:: bash

   git clone https://github.com/starsimhub/tbsimV2.git
   cd tbsimV2
   pip install -e .

Option 3: Development Installation
---------------------------------

For developers who want to work with the latest features:

1. Install TBsim first:
   
   .. code-block:: bash
   
      git clone https://github.com/starsimhub/tbsimV2.git
      cd tbsimV2
      pip install -e .

2. Install the latest Starsim development version:
   
   .. code-block:: bash
   
      git clone https://github.com/starsimhub/starsim.git
      cd starsim
      pip install -e .

Dependencies
-----------

TBsim requires several key dependencies:

- **Starsim**: Core simulation framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting and visualization
- **SciPy**: Scientific computing utilities

All dependencies are automatically installed when you install TBsim.

Verification
-----------

To verify your installation, run:

.. code-block:: bash

   python -c "import tbsim; print('TBsim installed successfully')"

If you encounter any issues, check the troubleshooting section below.

Troubleshooting
--------------

Common installation issues:

**Import Errors**: Ensure you're using the correct Python environment and that all dependencies are installed.

**Starsim Compatibility**: Make sure you have a compatible version of Starsim installed.

**Permission Errors**: On some systems, you may need to use `pip install --user` or run with appropriate permissions.

For additional help, please open an issue on the GitHub repository.

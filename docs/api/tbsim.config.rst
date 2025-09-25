TBsim Configuration
==================

This module provides configuration utilities for TBsim simulations, including directory management and result organization.

Main Configuration Module
------------------------

.. automodule:: tbsim.config
   :members:
   :undoc-members:
   :show-inheritance:

Available Functions
-----------------

**create_res_dir**
   Creates directories for storing simulation results and figures

Key Features
-----------

- **Result Directory Management**: Automatic creation of organized result directories
- **Date-based Organization**: Optional date-based directory naming
- **Flexible Path Structure**: Configurable base paths and postfixes
- **Automatic Creation**: Ensures directories exist before use

Usage Examples
-------------

Basic result directory creation:

.. code-block:: python

   from tbsim.config import create_res_dir
   
   # Create results directory in current working directory
   results_dir = create_res_dir()
   
   # Create results directory with custom base path
   results_dir = create_res_dir(base='/path/to/project')
   
   # Create results directory with postfix and no date
   results_dir = create_res_dir(postfix='experiment_1', append_date=False)

Directory Structure
-----------------

The function creates directories with the following structure:
- Base directory (default: current working directory)
- 'results' subdirectory
- Optional postfix directory
- Optional date-based directory (format: MM-DD_HH-MM)

This organization helps keep simulation results organized and easily accessible for analysis and comparison. 
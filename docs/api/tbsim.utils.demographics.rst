Demographics Utilities
=====================

This module provides utilities for managing population demographics and age structures in TBsim simulations.

Main Demographics Module
------------------------

.. automodule:: tbsim.utils.demographics
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
-----------

- **Age Structure Management**: Create and manage population age distributions
- **Demographic Parameters**: Age-specific parameters and characteristics
- **Population Modeling**: Tools for realistic population structures
- **Age Group Utilities**: Common age group definitions and filtering

Usage Examples
-------------

Creating age structures:

.. code-block:: python

   from tbsim.utils.demographics import create_age_structure
   
   # Create age structure for 10,000 people
   age_structure = create_age_structure(pop_size=10000)
   
   # Create age structure with custom parameters
   age_structure = create_age_structure(
       pop_size=5000,
       age_distribution='uniform',
       max_age=80
   )

Age group utilities:

.. code-block:: python

   from tbsim.utils.demographics import get_age_group
   
   # Get age group for specific age
   group = get_age_group(25)  # Returns 'adult'
   
   # Get age group distribution
   distribution = get_age_group_distribution(people)

Common Age Groups
-----------------

The module defines standard age groups:
- **Infants**: 0-1 years
- **Children**: 1-5 years
- **School Age**: 5-15 years
- **Adults**: 15-65 years
- **Elderly**: 65+ years

These utilities help create realistic population structures for TBsim simulations. 
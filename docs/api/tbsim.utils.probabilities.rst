Probability Utilities
====================

This module provides utilities for probability calculations and statistical operations in TBsim simulations.

Main Probabilities Module
-------------------------

.. automodule:: tbsim.utils.probabilities
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**Range**
   Class for managing probability ranges with min/max values and distributions

**Probability**
   Class for handling probability values with various input formats

Key Features
-----------

- **Probability Management**: Handle probability values and ranges
- **Distribution Support**: Various probability distributions
- **Range Operations**: Min/max probability bounds
- **Input Flexibility**: Multiple input formats (dict, JSON, CSV)
- **Validation**: Ensure probability values are valid

Usage Examples
-------------

Creating probability ranges:

.. code-block:: python

   from tbsim.utils.probabilities import Range
   
   # Create range with min/max values
   prob_range = Range(min=0.1, max=0.5)
   
   # Create range with distribution
   prob_range = Range(min=0.0, max=1.0, dist='uniform')

Working with probabilities:

.. code-block:: python

   from tbsim.utils.probabilities import Probability
   
   # Create from dictionary
   prob = Probability.from_dict({
       'value': 0.3,
       'uncertainty': 0.1
   })
   
   # Create from JSON file
   prob = Probability.from_json('probabilities.json')
   
   # Create from CSV
   prob = Probability.from_csv('prob_data.csv')

Probability Operations
---------------------

**Range Properties**
   - `min`: Minimum probability value
   - `max`: Maximum probability value
   - `dist`: Probability distribution type

**Probability Methods**
   - `from_dict()`: Create from dictionary input
   - `from_json()`: Create from JSON file
   - `from_csv()`: Create from CSV data
   - `validate()`: Check probability validity

**Common Distributions**
   - Uniform distribution
   - Normal distribution
   - Beta distribution
   - Custom distributions

These utilities help manage uncertainty and probability parameters in TBsim simulations. 
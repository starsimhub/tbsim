TBsim Miscellaneous Utilities
============================

This module provides miscellaneous utilities and helper functions for TBsim simulations.

Main Misc Module
----------------

.. automodule:: tbsim.misc
   :members:
   :undoc-members:
   :show-inheritance:

Available Submodules
-------------------

**tbsim.misc.tbterms**
   TB terminology and definitions for consistent language usage

Subpackages
----------

.. toctree::
   :maxdepth: 4

   tbsim.misc.tbterms

Key Features
-----------

- **Terminology Management**: Consistent TB-related language and definitions
- **Utility Functions**: Helper functions for common operations
- **Data Processing**: Miscellaneous data manipulation utilities
- **Export Functions**: Tools for data export and sharing

Usage Examples
-------------

Accessing TB terminology:

.. code-block:: python

   from tbsim.misc.tbterms import TBTerms, TermsExport
   
   # Get TB terms definitions
   terms = TBTerms()
   
   # Export terms to various formats
   exporter = TermsExport()
   exporter.export_terms(terms, 'tb_terms.csv')

For detailed information about specific submodules, see the individual documentation above.

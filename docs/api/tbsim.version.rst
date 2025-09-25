TBsim Version Information
========================

This module provides version and license information for the TBsim package.

Main Version Module
------------------

.. automodule:: tbsim.version
   :members:
   :undoc-members:
   :show-inheritance:

Available Variables
-----------------

**__version__**
   Current version string (e.g., '0.5.0')

**__versiondate__**
   Release date in YYYY-MM-DD format

**__license__**
   Full license information with version details

Version Information
------------------

The current version of TBsim is **0.5.0**, released on **January 7, 2025**.

License Details
--------------

TBsim is licensed under the MIT License, with copyright held by IDM from 2023-2025.

Usage Examples
-------------

Accessing version information:

.. code-block:: python

   from tbsim.version import __version__, __versiondate__, __license__
   
   print(f"TBsim version: {__version__}")
   print(f"Release date: {__versiondate__}")
   print(f"License: {__license__}")

Programmatic version checking:

.. code-block:: python

   import tbsim.version
   
   if tbsim.version.__version__ >= '0.5.0':
       print("Using TBsim 0.5.0 or higher")
   else:
       print("Consider upgrading to latest version")

Version History
--------------

- **0.5.0** (2025-01-07): Current stable release
- **0.1.0** (2024): Initial release

For detailed changelog information, see the :doc:`changelog` section. 
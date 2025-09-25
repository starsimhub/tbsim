TB Terms and Definitions
========================

This module provides comprehensive tuberculosis terminology definitions, abbreviations, and export functionality for documentation and user interface support.

TB Terms Enumeration
--------------------

.. automodule:: tbsim.misc.tbterms.TBTerms
   :members:
   :undoc-members:
   :show-inheritance:

Terms Export Utility
-------------------

.. automodule:: tbsim.misc.tbterms.TermsExport
   :members:
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

The TB Terms module provides a comprehensive collection of tuberculosis-related terminology and abbreviations with the following key features:

**Terminology Coverage**
   - **Treatment Regimens**: 1HP, 3HP, 3HR, 4R, 6H, 6Lfx, 9H
   - **Diagnostic Methods**: ACF, CAD, CXR, ELISA, IGRA, mWRD, PPD, TBST
   - **Drug Types**: ARV, FDC, INH, Lfx, RIF, TDF
   - **Disease States**: LF, LS, MDR-TB, RR-TB, XDR-TB, XPTB
   - **Organizations**: GDG, HMIS, NGO, WHO
   - **Clinical Terms**: ART, BCG, CRP, IFN-γ, IPT, PMTPT, TPT

**Abbreviation Management**
   - Safe, structured access to common TB-related abbreviations
   - Reverse lookup using original terms
   - Integration with tools for tooltips, autocomplete, and documentation
   - Friendly string representations for use in UIs or logs

**Export Functionality**
   - CSV export for external analysis
   - Markdown export for documentation
   - Structured data access
   - Custom formatting options

Key Features
-----------

**Comprehensive Coverage**
   - All major TB treatment regimens
   - Diagnostic and screening methods
   - Drug and treatment terminology
   - Clinical and research terms
   - Organizational and guideline references

**Safe Access Methods**
   - Enum-based access for type safety
   - Dictionary-like access for flexibility
   - Error handling for missing terms
   - Default value support

**Export Capabilities**
   - CSV export with custom formatting
   - Markdown export for documentation
   - Structured data access
   - Integration with documentation systems

**Integration Support**
   - Tooltip and autocomplete support
   - Documentation generation
   - User interface integration
   - Data validation and consistency

Usage Examples
-------------

Basic term access:

.. code-block:: python

   from tbsim.misc.tbterms import TBTerms
   
   # Access terms directly
   dots = TBTerms.DOTS
   print(f"DOTS: {dots.value}")  # 'Directly Observed Treatment, Short-course'
   
   # Get term by key
   tpt = TBTerms.get("TPT")
   print(f"TPT: {tpt.value}")    # 'tuberculosis preventive treatment'

Reverse lookup:

.. code-block:: python

   # Get original abbreviation from term
   dots_term = TBTerms.DOTS
   original = dots_term.orig()
   print(f"Original: {original}")  # 'DOTS'
   
   # Get help information
   help_text = dots_term.help()
   print(f"Help: {help_text}")     # Detailed description

Dictionary access:

.. code-block:: python

   # Get all terms as dictionary
   all_terms = TBTerms.as_dict()
   print(f"Total terms: {len(all_terms)}")
   
   # Access specific terms
   if "MDR_TB" in all_terms:
       print(f"MDR-TB: {all_terms['MDR_TB']}")
   
   # Get all keys
   keys = TBTerms.keys()
   print(f"Available keys: {keys}")

Export functionality:

.. code-block:: python

   from tbsim.misc.tbterms import TermsExport
   
   # Export to CSV
   TermsExport.export_tbterms_to_csv('tb_terms.csv')
   
   # Export to markdown
   markdown_content = TermsExport.export_tbterms_to_markdown()
   print(markdown_content)

Integration examples:

.. code-block:: python

   # Create tooltip dictionary
   tooltips = {}
   for key in TBTerms.keys():
       term = TBTerms.get(key)
       tooltips[key] = term.value
   
   # Use in UI components
   def show_tooltip(term_key):
       if term_key in tooltips:
           return tooltips[term_key]
       return "Term not found"

Error handling:

.. code-block:: python

   # Safe term access with error handling
   try:
       term = TBTerms.get("UNKNOWN_TERM")
       print(f"Term: {term.value}")
   except ValueError as e:
       print(f"Term not found: {e}")
   
   # Check if term exists
   if TBTerms.get("TPT", None):
       print("TPT term exists")
   else:
       print("TPT term not found")

Data Structure
-------------

**Term Organization**
   - Alphabetical ordering by abbreviation
   - Grouped by category (treatment, diagnostic, etc.)
   - Consistent naming conventions
   - Cross-referenced terms

**Export Formats**
   - CSV: Comma-separated values with headers
   - Markdown: Formatted documentation
   - Dictionary: Python dictionary format
   - JSON: Structured data format

**Access Methods**
   - Direct attribute access (TBTerms.DOTS)
   - Dictionary-style access (TBTerms.get("DOTS"))
   - Key enumeration (TBTerms.keys())
   - Value enumeration (TBTerms.values())

Integration Points
-----------------

**Documentation Systems**
   - Sphinx documentation integration
   - Markdown documentation generation
   - API documentation support
   - User guide generation

**User Interfaces**
   - Tooltip and help system integration
   - Autocomplete functionality
   - Search and filtering
   - Consistent terminology display

**Data Analysis**
   - Term frequency analysis
   - Category-based grouping
   - Export for external tools
   - Consistency validation

For detailed information about specific methods and parameters, see the individual class documentation above. All methods include comprehensive functionality and implementation details in their docstrings. 
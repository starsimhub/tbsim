TB Drug Types and Parameters
============================

This module provides comprehensive drug type definitions, parameter management, and drug regimen configuration for tuberculosis treatment interventions.

TB Drug Type Enumeration
------------------------

.. automodule:: tbsim.interventions.tb_drug_types.TBDrugType
   :members:
   :undoc-members:
   :show-inheritance:

TB Drug Parameters
-----------------

.. automodule:: tbsim.interventions.tb_drug_types.TBDrugParameters
   :members:
   :undoc-members:
   :show-inheritance:

TB Drug Type Parameters
-----------------------

.. automodule:: tbsim.interventions.tb_drug_types.TBDrugTypeParameters
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
----------------

.. automodule:: tbsim.interventions.tb_drug_types
   :members:
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

The TB Drug Types module provides a comprehensive framework for managing tuberculosis drug regimens and treatment parameters:

**Drug Type Definitions**
   - **DOTS (1)**: Standard Directly Observed Treatment, Short-course
   - **DOTS_IMPROVED (2)**: Enhanced DOTS with better support and monitoring
   - **EMPIRIC_TREATMENT (3)**: Treatment without confirmed drug sensitivity
   - **FIRST_LINE_COMBO (4)**: First-line combination therapy for drug-sensitive TB
   - **SECOND_LINE_COMBO (5)**: Second-line therapy for MDR-TB
   - **THIRD_LINE_COMBO (6)**: Third-line therapy for XDR-TB
   - **LATENT_TREATMENT (7)**: Treatment for latent TB infection

**Parameter Management**
   - Drug-specific effectiveness curves
   - Time-dependent treatment outcomes
   - Configurable treatment parameters
   - Standardized parameter sets

**Integration Capabilities**
   - Treatment intervention integration
   - Dynamic parameter adjustment
   - Drug resistance modeling
   - Treatment outcome tracking

Key Features
-----------

**Comprehensive Drug Coverage**
   - All WHO-recommended TB drug regimens
   - Drug resistance level specific treatments
   - Latent TB infection treatment options
   - Empiric treatment protocols

**Parameter Standardization**
   - Pre-configured parameter sets for each drug type
   - Evidence-based effectiveness curves
   - Configurable treatment parameters
   - Validation and error checking

**Treatment Effectiveness Modeling**
   - Time-dependent effectiveness curves
   - Drug-specific outcome patterns
   - Adherence consideration
   - Resistance development modeling

**Flexible Configuration**
   - Custom parameter modification
   - Drug type switching
   - Dynamic parameter updates
   - Batch parameter management

Usage Examples
-------------

Basic drug type usage:

.. code-block:: python

   from tbsim.interventions.tb_drug_types import TBDrugType
   
   # Get drug type information
   dots_type = TBDrugType.DOTS
   print(f"DOTS value: {dots_type.value}")
   print(f"DOTS name: {dots_type.name}")
   
   # Get all available drug types
   all_types = TBDrugType.get_all_types()
   print(f"Available types: {all_types}")

Creating drug parameters:

.. code-block:: python

   from tbsim.interventions.tb_drug_types import TBDrugParameters, TBDrugType
   
   # Create custom drug parameters
   custom_drug = TBDrugParameters(
       drug_name="Custom Regimen",
       drug_type=TBDrugType.FIRST_LINE_COMBO
   )
   
   # Configure parameters
   custom_drug.configure({
       'base_effectiveness': 0.85,
       'time_to_peak': 30,
       'peak_effectiveness': 0.95,
       'decay_rate': 0.02
   })

Using pre-configured parameters:

.. code-block:: python

   from tbsim.interventions.tb_drug_types import get_dots_parameters, get_drug_parameters
   
   # Get standard DOTS parameters
   dots_params = get_dots_parameters()
   
   # Get parameters for specific drug type
   first_line_params = get_drug_parameters(TBDrugType.FIRST_LINE_COMBO)
   
   # Get all parameter sets
   from tbsim.interventions.tb_drug_types import get_all_drug_parameters
   all_params = get_all_drug_parameters()

Drug effectiveness analysis:

.. code-block:: python

   # Analyze treatment effectiveness over time
   dots_params = get_dots_parameters()
   
   # Get effectiveness at different time points
   effectiveness_30d = dots_params.get_effectiveness(30)   # 30 days
   effectiveness_90d = dots_params.get_effectiveness(90)   # 90 days
   effectiveness_180d = dots_params.get_effectiveness(180) # 180 days
   
   print(f"Effectiveness at 30 days: {effectiveness_30d:.3f}")
   print(f"Effectiveness at 90 days: {effectiveness_90d:.3f}")
   print(f"Effectiveness at 180 days: {effectiveness_180d:.3f}")

Integration with Treatment Interventions
-------------------------------------

**Enhanced TB Treatment Integration**
   - Direct integration with EnhancedTBTreatment class
   - Automatic parameter loading
   - Drug type validation
   - Treatment outcome tracking

**Parameter Validation**
   - Range checking for all parameters
   - Type validation
   - Consistency checking
   - Error reporting and handling

**Dynamic Parameter Management**
   - Runtime parameter updates
   - Drug type switching
   - Effectiveness curve modification
   - Resistance pattern updates

Mathematical Framework
---------------------

**Effectiveness Curves**
   - Time-dependent effectiveness modeling
   - Peak effectiveness timing
   - Decay rate calculations
   - Adherence effects

**Treatment Outcomes**
   - Success probability calculations
   - Failure rate modeling
   - Resistance development
   - Relapse probability

**Parameter Relationships**
   - Drug type specific relationships
   - Cross-drug interactions
   - Resistance patterns
   - Treatment sequencing

For detailed information about specific methods and parameters, see the individual class documentation above. All methods include comprehensive mathematical models and implementation details in their docstrings.

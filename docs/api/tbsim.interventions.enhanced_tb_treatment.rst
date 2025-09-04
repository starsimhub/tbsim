Enhanced TB Treatment Interventions
=================================

This module provides comprehensive TB treatment interventions including DOTS implementation, enhanced treatment protocols, and drug regimen management.

Enhanced TB Treatment Class
--------------------------

.. automodule:: tbsim.interventions.enhanced_tb_treatment
   :members:
   :undoc-members:
   :show-inheritance:

Treatment Factory Functions
--------------------------

.. automodule:: tbsim.interventions.enhanced_tb_treatment
   :members:
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

The Enhanced TB Treatment module implements comprehensive treatment interventions for tuberculosis with the following key features:

**Treatment Protocols**
   - **DOTS Implementation**: Directly Observed Treatment, Short-course
   - **Enhanced DOTS**: Improved treatment with better support and monitoring
   - **First Line Treatment**: Standard combination therapy for drug-sensitive TB
   - **Custom Regimens**: Configurable treatment parameters and protocols

**Treatment Management**
   - Individual treatment assignment and tracking
   - Treatment duration and adherence modeling
   - Treatment outcome monitoring
   - Drug resistance considerations

**Intervention Features**
   - Age and risk factor targeting
   - Dynamic treatment initiation
   - Treatment effectiveness tracking
   - Integration with diagnostic systems

Key Features
-----------

**DOTS Implementation**
   - Standard WHO-recommended DOTS protocol
   - Treatment observation and support
   - Standardized drug regimens
   - Outcome monitoring and reporting

**Enhanced Treatment Options**
   - Improved DOTS with additional support
   - First-line combination therapy
   - Customizable treatment parameters
   - Risk-stratified treatment approaches

**Treatment Tracking**
   - Individual treatment status
   - Treatment duration monitoring
   - Adherence tracking
   - Outcome assessment

**Integration Capabilities**
   - Diagnostic system integration
   - Comorbidity consideration
   - Network-based targeting
   - Dynamic parameter adjustment

Usage Examples
-------------

Basic DOTS implementation:

.. code-block:: python

   from tbsim.interventions.enhanced_tb_treatment import create_dots_treatment
   
   # Create standard DOTS treatment
   dots = create_dots_treatment()
   sim.add_intervention(dots)
   sim.run()

Enhanced DOTS treatment:

.. code-block:: python

   from tbsim.interventions.enhanced_tb_treatment import create_dots_improved_treatment
   
   # Create enhanced DOTS with better support
   enhanced_dots = create_dots_improved_treatment()
   sim.add_intervention(enhanced_dots)
   sim.run()

First-line combination therapy:

.. code-block:: python

   from tbsim.interventions.enhanced_tb_treatment import create_first_line_treatment
   
   # Create first-line combination treatment
   first_line = create_first_line_treatment()
   sim.add_intervention(first_line)
   sim.run()

Custom treatment parameters:

.. code-block:: python

   from tbsim.interventions.enhanced_tb_treatment import EnhancedTBTreatment
   
   # Create custom treatment intervention
   custom_treatment = EnhancedTBTreatment(pars={
       'treatment_duration': 180,      # 6 months
       'success_rate': 0.85,           # 85% success rate
       'target_age_min': 15,           # Minimum age 15
       'target_age_max': 65,           # Maximum age 65
       'start_date': '2020-01-01',    # Start date
       'stop_date': '2030-12-31'      # Stop date
   })
   
   sim.add_intervention(custom_treatment)
   sim.run()

Treatment Monitoring
-------------------

**Individual Tracking**
   - Treatment initiation dates
   - Current treatment status
   - Treatment duration
   - Adherence patterns

**Population Level Metrics**
   - Treatment coverage rates
   - Treatment success rates
   - Treatment completion rates
   - Drug resistance patterns

**Outcome Assessment**
   - Cure rates
   - Treatment failure rates
   - Relapse rates
   - Mortality rates

Integration with Other Modules
-----------------------------

**Diagnostic Integration**
   - Treatment initiation based on diagnostic results
   - Treatment type selection based on diagnostic findings
   - Integration with enhanced diagnostic systems

**Comorbidity Considerations**
   - HIV status consideration in treatment selection
   - Nutritional status effects on treatment outcomes
   - Age-specific treatment protocols

**Network Effects**
   - Household-based treatment targeting
   - Community treatment programs
   - Contact tracing integration

For detailed information about specific methods and parameters, see the individual class documentation above. All methods include comprehensive mathematical models and implementation details in their docstrings.

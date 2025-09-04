Enhanced TB Diagnostic Interventions
===================================

This module provides advanced TB diagnostic capabilities with stratified parameters and multiple testing methods.

Main Enhanced TB Diagnostic Module
---------------------------------

.. automodule:: tbsim.interventions.enhanced_tb_diagnostic
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**EnhancedTBDiagnostic**
   Advanced TB diagnostic intervention with age, state, and HIV-stratified parameters

Key Features
-----------

- **Stratified Parameters**: Age and TB state-specific sensitivity/specificity
- **Multiple Diagnostic Methods**: Xpert, Oral Swab, FujiLAM, CAD CXR
- **HIV Integration**: HIV-stratified diagnostic parameters
- **Advanced Testing**: Support for various TB testing technologies
- **Comprehensive Tracking**: Detailed diagnostic method and outcome monitoring

Available Diagnostic Methods
---------------------------

**Xpert MTB/RIF**
   - Adult smear-positive: 90.9% sensitivity, 96.6% specificity
   - Adult smear-negative: 77.5% sensitivity, 95.8% specificity
   - Adult extra-pulmonary: 77.5% sensitivity, 95.8% specificity
   - Children: 73% sensitivity, 95% specificity

**Oral Swab Testing**
   - Adult smear-positive: 80% sensitivity, 90% specificity
   - Adult smear-negative: 30% sensitivity, 98% specificity
   - Children: 25% sensitivity, 95% specificity

**FujiLAM (HIV-focused)**
   - HIV-positive adults: 75% sensitivity, 90% specificity
   - HIV-negative adults: 58% sensitivity, 98% specificity
   - HIV-positive children: 57.9% sensitivity, 87.7% specificity
   - HIV-negative children: 51% sensitivity, 89.5% specificity

**CAD CXR (Computer-Aided Detection)**
   - Children: 66% sensitivity, 79% specificity

Usage Examples
-------------

Basic enhanced diagnostic:

.. code-block:: python

   from tbsim.interventions.enhanced_tb_diagnostic import EnhancedTBDiagnostic
   from tbsim import TB
   
   sim = ss.Sim()
   
   # Add TB module
   tb = TB()
   sim.add_module(tb)
   
   # Add enhanced diagnostic testing
   diagnostic = EnhancedTBDiagnostic()
   sim.add_module(diagnostic)
   
   sim.run()

Advanced diagnostic with multiple methods:

.. code_block:: python

   diagnostic = EnhancedTBDiagnostic(
       use_oral_swab=True,      # Enable oral swab testing
       use_fujilam=True,        # Enable FujiLAM for HIV
       use_cadcxr=True,         # Enable CAD CXR for children
       coverage=0.9             # 90% testing coverage
   )

Key Methods
-----------

**Diagnostic Management**
   - `step()`: Execute enhanced diagnostic testing each time step
   - `_get_diagnostic_parameters()`: Get stratified sensitivity/specificity
   - `init_results()`: Initialize diagnostic result tracking
   - `update_results()`: Update results during simulation

**Parameter Stratification**
   - Age-based parameters (children vs. adults)
   - TB state-specific parameters (smear-positive, smear-negative, EPTB)
   - HIV-stratified parameters for LAM testing
   - Method-specific parameters for different technologies

Advanced Features
----------------

**Stratified Testing**
   - Different parameters for different population groups
   - Age-specific diagnostic accuracy
   - HIV status consideration for LAM testing
   - TB state-specific testing approaches

**Multiple Technologies**
   - Xpert MTB/RIF as primary method
   - Oral swab for enhanced case detection
   - FujiLAM for HIV-positive individuals
   - CAD CXR for pediatric cases

**Result Tracking**
   - Diagnostic method used for each test
   - Stratified outcome analysis
   - Technology-specific performance metrics
   - Comprehensive diagnostic pathway analysis

For basic diagnostic capabilities, see the :doc:`tbsim.interventions.tb_diagnostic` module.

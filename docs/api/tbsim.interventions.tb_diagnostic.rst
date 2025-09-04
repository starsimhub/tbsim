TB Diagnostic Interventions
==========================

This module provides TB diagnostic testing capabilities for TBsim simulations.

Main TB Diagnostic Module
-------------------------

.. automodule:: tbsim.interventions.tb_diagnostic
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**TBDiagnostic**
   TB diagnostic testing intervention with configurable sensitivity and specificity

Key Features
-----------

- **Diagnostic Testing**: Simulate TB testing with realistic parameters
- **Test Accuracy**: Configurable sensitivity and specificity
- **False Negative Handling**: Allow retesting with increased care-seeking
- **Coverage Control**: Manage testing coverage rates
- **Result Tracking**: Comprehensive diagnostic outcome monitoring

Usage Examples
-------------

Basic TB diagnostic:

.. code-block:: python

   from tbsim.interventions.tb_diagnostic import TBDiagnostic
   from tbsim import TB
   
   sim = ss.Sim()
   
   # Add TB module
   tb = TB()
   sim.add_module(tb)
   
   # Add diagnostic testing
   diagnostic = TBDiagnostic()
   sim.add_module(diagnostic)
   
   sim.run()

Custom diagnostic parameters:

.. code-block:: python

   diagnostic = TBDiagnostic(
       coverage=0.8,                    # 80% testing coverage
       sensitivity=0.85,                # 85% test sensitivity
       specificity=0.95,                # 95% test specificity
       care_seeking_multiplier=2.0      # 2x care-seeking after false negative
   )

Key Methods
-----------

**Diagnostic Management**
   - `step()`: Execute diagnostic testing each time step
   - `init_results()`: Initialize diagnostic result tracking
   - `update_results()`: Update results during simulation

**Testing Logic**
   - Identifies eligible individuals (sought care, not diagnosed)
   - Applies coverage filters for testing selection
   - Determines TB status and applies test accuracy
   - Updates person states based on test results
   - Handles false negatives with retesting support

Test Parameters
--------------

**Coverage**: Fraction of care-seeking individuals who get tested
**Sensitivity**: Probability of positive test given TB infection
**Specificity**: Probability of negative test given no TB infection
**Care-seeking Multiplier**: Increased health-seeking after false negatives

Diagnostic Outcomes
------------------

The module tracks:
- **Testing Volume**: Number of tests performed
- **Test Results**: Positive and negative outcomes
- **False Results**: Inaccurate test results
- **Retesting**: False negative handling and retesting
- **Diagnosis**: Confirmed TB cases

For enhanced diagnostic capabilities, see the :doc:`tbsim.interventions.enhanced_tb_diagnostic` module.

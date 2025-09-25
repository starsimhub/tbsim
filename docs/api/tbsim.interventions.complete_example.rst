Complete Intervention Example
=============================

This example shows how to properly combine multiple interventions that require specific attributes.

.. code-block:: python

   from tbsim import TB
   from tbsim.interventions.healthseeking import HealthSeekingBehavior
   from tbsim.interventions.tb_diagnostic import TBDiagnostic
   from tbsim.interventions.tb_treatment import TBTreatment
   from tbsim.interventions.tpt import TPTInitiation
   import starsim as ss
   
   # Create TB disease model
   tb = TB()
   
   # Create intervention sequence (order matters!)
   # 1. Health seeking creates 'sought_care' attribute
   health_seeking = HealthSeekingBehavior()
   
   # 2. Diagnostic requires 'sought_care' and creates 'diagnosed' attribute
   diagnostic = TBDiagnostic()
   
   # 3. Treatment requires 'diagnosed' attribute
   treatment = TBTreatment()
   
   # 4. TPT requires 'sought_care', 'non_symptomatic', 'symptomatic' attributes
   tpt = TPTInitiation()
   
   # Create simulation with all interventions
   sim = ss.Sim(
       diseases=tb,
       interventions=[health_seeking, diagnostic, treatment, tpt]
   )
   
   sim.run()

Key Points:
-----------

1. **Attribute Dependencies**: Some interventions require attributes created by other interventions
2. **Order Matters**: Health-seeking should come before diagnostics, diagnostics before treatment
3. **Required Attributes**:
   - `sought_care`: Created by health-seeking interventions
   - `diagnosed`: Created by diagnostic interventions  
   - `non_symptomatic`, `symptomatic`: Required for TPT eligibility
4. **Household Structure**: TPT requires household IDs (`hhid` attribute) for contact tracing

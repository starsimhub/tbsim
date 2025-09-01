DOTS Implementation Tutorial
============================

This tutorial demonstrates how to use the enhanced TB treatment system in TBsim, which provides detailed drug-specific parameters matching EMOD-Generic's approach.

Overview
--------

The enhanced TB treatment system allows you to:

- Use different drug regimens (DOTS, first-line, second-line, etc.)
- Access drug-specific parameters (cure rates, costs, durations)
- Track treatment outcomes with detailed metrics
- Compare different treatment strategies

Prerequisites
------------

Make sure you have TBsim installed and the enhanced TB treatment module available:

.. code-block:: python

    from tbsim.interventions import (
        TBDrugType, 
        EnhancedTBTreatment, 
        create_dots_treatment,
        get_drug_parameters
    )

Basic DOTS Treatment
-------------------

The simplest way to add DOTS treatment to your simulation:

.. code-block:: python

    import starsim as ss
    from tbsim import TB
    from tbsim.interventions import create_dots_treatment
    
    # Create simulation with TB
    sim = ss.Sim(
        modules=[TB()],
        interventions=[create_dots_treatment()]
    )
    
    # Run simulation
    sim.run()
    
    # Check results
    print(f"Total treated: {sim.results['n_treated'][-1]}")
    print(f"Treatment success: {sim.results['n_treatment_success'][-1]}")

Advanced Treatment Configuration
-------------------------------

You can customize treatment parameters for different scenarios:

.. code-block:: python

    from tbsim.interventions import EnhancedTBTreatment, TBDrugType
    
    # Create first-line combination therapy
    first_line_treatment = EnhancedTBTreatment(
        drug_type=TBDrugType.FIRST_LINE_COMBO,
        treatment_success_rate=0.95,  # Override default
        reseek_multiplier=2.0
    )
    
    # Create MDR-TB treatment
    mdr_treatment = EnhancedTBTreatment(
        drug_type=TBDrugType.SECOND_LINE_COMBO,
        treatment_success_rate=0.75
    )
    
    sim = ss.Sim(
        modules=[TB()],
        interventions=[first_line_treatment, mdr_treatment]
    )

Comparing Drug Parameters
------------------------

You can compare different drug regimens:

.. code-block:: python

    from tbsim.interventions import get_all_drug_parameters
    
    # Get all drug parameters
    all_params = get_all_drug_parameters()
    
    # Compare cure rates and costs
    print("Drug Type | Cure Rate | Cost per Course")
    print("----------|-----------|----------------")
    for drug_type, params in all_params.items():
        print(f"{drug_type.name:10} | {params.cure_rate:9.3f} | ${params.cost_per_course:13.0f}")

Drug Effectiveness Over Time
---------------------------

The system models how drug effectiveness changes during treatment:

.. code-block:: python

    from tbsim.interventions import get_dots_parameters
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get DOTS parameters
    dots_params = get_dots_parameters()
    
    # Calculate effectiveness over time
    time_points = np.arange(0, 365, 7)  # Weekly for a year
    effectiveness = [dots_params.get_effectiveness(t) for t in time_points]
    
    # Plot effectiveness
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, effectiveness)
    plt.xlabel('Days on Treatment')
    plt.ylabel('Drug Effectiveness')
    plt.title('DOTS Treatment Effectiveness Over Time')
    plt.grid(True)
    plt.show()

Treatment Outcomes Analysis
--------------------------

Track detailed treatment outcomes:

.. code-block:: python

    # Run simulation with enhanced tracking
    sim = ss.Sim(
        modules=[TB()],
        interventions=[create_dots_treatment()],
        verbose=0
    )
    sim.run()
    
    # Analyze results
    results = sim.results
    
    print("Treatment Outcomes:")
    print(f"Total diagnosed: {results['n_diagnosed'][-1]}")
    print(f"Total treated: {results['n_treated'][-1]}")
    print(f"Treatment success: {results['n_treatment_success'][-1]}")
    print(f"Treatment failure: {results['n_treatment_failure'][-1]}")
    print(f"Success rate: {results['n_treatment_success'][-1]/results['n_treated'][-1]:.3f}")

Cost-Effectiveness Analysis
--------------------------

Compare different treatment strategies:

.. code-block:: python

    from tbsim.interventions import TBDrugType, get_drug_parameters
    
    def calculate_cost_effectiveness(drug_type):
        params = get_drug_parameters(drug_type)
        cost_per_cure = params.cost_per_course / params.cure_rate
        return cost_per_cure
    
    # Compare cost-effectiveness
    drug_types = [TBDrugType.DOTS, TBDrugType.FIRST_LINE_COMBO, TBDrugType.SECOND_LINE_COMBO]
    
    print("Cost-Effectiveness Analysis:")
    print("Drug Type | Cost per Cure")
    print("----------|---------------")
    for drug_type in drug_types:
        cost_per_cure = calculate_cost_effectiveness(drug_type)
        print(f"{drug_type.name:10} | ${cost_per_cure:12.0f}")

Integration with Other Modules
-----------------------------

The enhanced treatment system works with other TBsim modules:

.. code-block:: python

    from tbsim import TB
    from tbsim.comorbidities.hiv import HIV
    from tbsim.interventions import create_dots_treatment
    
    # TB-HIV co-infection simulation
    sim = ss.Sim(
        modules=[TB(), HIV()],
        interventions=[create_dots_treatment()],
        pars={
            'n_agents': 10000,
            'end_year': 2030
        }
    )
    
    sim.run()

Testing Your Implementation
--------------------------

Run the test suite to verify everything works:

.. code-block:: bash

    python -m tbsim.test_dots_implementation

This will:
- Test all drug parameter creation
- Run sample simulations
- Generate comparison plots
- Validate results

Next Steps
----------

- Explore different drug combinations
- Analyze treatment outcomes in detail
- Integrate with diagnostic interventions
- Model treatment adherence patterns
- Study cost-effectiveness of different strategies

For more information, see the :doc:`../api/tbsim.interventions.enhanced_tb_treatment` API reference.

User Guide
==========

This guide provides comprehensive information about using TBsim for tuberculosis modeling projects.

Core Concepts
------------

TBsim is built around several key concepts:

**Individual-Based Modeling (IBM)**
   Each person in the simulation is represented as an individual agent with unique characteristics, health states, and behaviors.

**Network-Based Modeling (NBM)**
   Social connections between individuals are modeled as networks, allowing for realistic transmission dynamics.

**Modular Design**
   Components can be mixed and matched to create custom simulation scenarios.

Main Components
--------------

TB Model
~~~~~~~~

The core tuberculosis model includes:

- **Disease States**: Susceptible, Latent, Active, Treated, Recovered
- **Transmission**: Person-to-person spread through social networks
- **Natural History**: Progression from infection to disease
- **Treatment**: Drug regimens and outcomes

Interventions
~~~~~~~~~~~~

Available intervention types:

- **BCG Vaccination**: Prevention through vaccination
- **Treatment**: Various drug regimens and protocols
- **Preventive Therapy**: Isoniazid preventive therapy (IPT)
- **Diagnostic Tools**: Enhanced case detection methods

Comorbidities
~~~~~~~~~~~~

Modeling of co-occurring conditions:

- **HIV**: TB-HIV co-infection dynamics
- **Malnutrition**: Impact on TB progression and treatment
- **Other Conditions**: Extensible framework for additional comorbidities

Networks
~~~~~~~~

Social structure modeling:

- **Household Networks**: Family and household connections
- **Community Networks**: Broader social interactions
- **Transmission Networks**: Disease spread pathways

Analyzers
~~~~~~~~~

Built-in analysis tools:

- **DWT Analyzer**: Discrete wavelet transform analysis
- **Plotting Tools**: Comprehensive visualization capabilities
- **Post-Processing**: Data analysis and export functions

Configuration
-------------

TBsim uses a flexible configuration system:

- **Parameter Files**: YAML/JSON configuration files
- **Runtime Configuration**: Programmatic parameter adjustment
- **Scenario Management**: Multiple simulation scenarios

Running Simulations
------------------

Basic simulation workflow:

1. **Setup**: Configure parameters and components
2. **Initialize**: Create population and networks
3. **Run**: Execute the simulation
4. **Analyze**: Process and visualize results
5. **Export**: Save results for further analysis

Advanced Features
----------------

**Multi-Scenario Analysis**
   Run multiple parameter combinations simultaneously

**Calibration Tools**
   Automated parameter fitting to observed data

**Optimization**
   Find optimal intervention strategies

**Sensitivity Analysis**
   Assess parameter uncertainty and impact

For specific examples and tutorials, see the :doc:`examples` and :doc:`tutorials` sections.

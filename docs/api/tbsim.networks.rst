Network Structures
=================

This module provides social network structures for modeling transmission patterns, household connections, and social interactions in TBsim simulations.

Network Classes
--------------

.. automodule:: tbsim.networks
   :members:
   :undoc-members:
   :show-inheritance:

Household Network Implementation
------------------------------

.. automodule:: tbsim.networks.HouseholdNet
   :members:
   :undoc-members:
   :show-inheritance:

RATIONS Trial Network
--------------------

.. automodule:: tbsim.networks.HouseholdNetRationsTrial
   :members:
   :undoc-members:
   :show-inheritance:

Network Visualization
--------------------

.. automodule:: tbsim.networks
   :members:
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

The Networks module provides comprehensive social network modeling for tuberculosis transmission with the following key features:

**Network Types**
   - **Household Networks**: Family and household-based transmission modeling
   - **RATIONS Trial Networks**: Specialized networks for nutritional intervention studies
   - **Social Networks**: Community and social contact modeling
   - **Transmission Networks**: Disease-specific contact patterns

**Network Structure**
   - Dynamic network generation and management
   - Household assignment and management
   - Contact pattern modeling
   - Transmission probability calculations

**Integration Capabilities**
   - TB disease model integration
   - Comorbidity consideration
   - Intervention targeting
   - Population dynamics support

Key Features
-----------

**Household Network Management**
   - Automatic household generation
   - Household size distribution modeling
   - Dynamic household membership
   - Contact pattern generation

**Transmission Modeling**
   - Contact-based transmission probabilities
   - Household clustering effects
   - Social mixing patterns
   - Age and risk factor consideration

**Network Visualization**
   - Household structure plotting
   - Contact network visualization
   - Transmission pattern analysis
   - Network statistics reporting

**Dynamic Network Updates**
   - Birth and death handling
   - Household formation and dissolution
   - Contact pattern evolution
   - Intervention effects on networks

Usage Examples
-------------

Basic household network:

.. code-block:: python

   from tbsim.networks import HouseholdNet
   import starsim as ss
   
   sim = ss.Sim()
   households = HouseholdNet()
   sim.add_module(households)
   sim.run()

RATIONS trial network:

.. code-block:: python

   from tbsim.networks import HouseholdNetRationsTrial
   
   # Create RATIONS trial specific network
   rations_network = HouseholdNetRationsTrial(pars={
       'household_size_mean': 4.5,
       'household_size_std': 1.2,
       'contact_probability': 0.3
   })
   
   sim.add_module(rations_network)
   sim.run()

Custom network parameters:

.. code-block:: python

   from tbsim.networks import HouseholdNet
   
   # Customize network parameters
   households = HouseholdNet(pars={
       'household_size_mean': 3.8,
       'household_size_std': 1.5,
       'contact_probability': 0.25,
       'max_household_size': 8
   })
   
   sim.add_module(households)
   sim.run()

Network visualization:

.. code-block:: python

   from tbsim.networks import plot_household_structure
   
   # Plot household structure
   plot_household_structure(
       households=households.households,
       title="Household Network Structure",
       figsize=(15, 10)
   )

Accessing network data:

.. code-block:: python

   # Get household information
   household_list = households.households
   household_sizes = [len(hh) for hh in household_list]
   
   # Get contact patterns
   contact_matrix = households.contact_matrix
   transmission_rates = households.transmission_rates
   
   # Get network statistics
   total_households = len(household_list)
   avg_household_size = np.mean(household_sizes)
   max_household_size = max(household_sizes)

Network Analysis
---------------

**Household Statistics**
   - Household size distribution
   - Household composition analysis
   - Contact pattern statistics
   - Transmission rate calculations

**Network Properties**
   - Clustering coefficients
   - Average path lengths
   - Degree distributions
   - Community structure analysis

**Transmission Dynamics**
   - Contact-based transmission modeling
   - Household clustering effects
   - Social mixing patterns
   - Intervention impact assessment

Integration with Disease Models
-----------------------------

**TB Transmission Integration**
   - Network-based transmission modeling
   - Household clustering effects
   - Contact pattern consideration
   - Transmission probability calculations

**Comorbidity Effects**
   - HIV status consideration in contacts
   - Nutritional status effects on transmission
   - Age-specific contact patterns
   - Risk factor integration

**Intervention Targeting**
   - Household-based intervention delivery
   - Contact tracing integration
   - Community intervention programs
   - Network-based effectiveness assessment

Mathematical Framework
---------------------

**Network Generation**
   - Household size distribution modeling
   - Contact probability calculations
   - Transmission rate determination
   - Dynamic network updates

**Transmission Modeling**
   - Contact-based transmission: P(transmission) = f(contact, risk_factors)
   - Household effects: P(household) = f(household_size, contact_density)
   - Social mixing: P(social) = f(age, risk_group, intervention_status)

**Network Evolution**
   - Birth and death processes
   - Household formation and dissolution
   - Contact pattern evolution
   - Intervention effects on networks

For detailed information about specific methods and parameters, see the individual class documentation above. All methods include comprehensive mathematical models and implementation details in their docstrings. 
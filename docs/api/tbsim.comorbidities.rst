TB Comorbidities
===============

This module handles modeling of co-occurring conditions with tuberculosis, including HIV and malnutrition.

Main Comorbidities Module
-------------------------

.. automodule:: tbsim.comorbidities
   :members:
   :undoc-members:
   :show-inheritance:

Available Comorbidity Types
--------------------------

**HIV Co-infection** (`tbsim.comorbidities.hiv`)
   Modeling of TB-HIV co-infection dynamics and interactions

**Malnutrition** (`tbsim.comorbidities.malnutrition`)
   Impact of malnutrition on TB progression and treatment outcomes

**Connector Classes**
   Specialized classes that integrate TB with specific comorbidities

Subpackages
----------

.. toctree::
   :maxdepth: 4

   tbsim.comorbidities.hiv
   tbsim.comorbidities.malnutrition

Key Features
-----------

- **Bidirectional Interactions**: Model how comorbidities affect TB and vice versa
- **State Transitions**: Complex state modeling for multiple conditions
- **Treatment Interactions**: Account for drug interactions and side effects
- **Risk Stratification**: Different risk profiles for various comorbidity combinations
- **Network Effects**: Social network impacts on comorbidity spread

Usage Examples
-------------

TB-HIV co-infection modeling:

.. code-block:: python

   from tbsim.comorbidities.hiv import HIV
   from tbsim import TB
   
   sim = ss.Sim()
   tb = TB()
   hiv = HIV()
   
   sim.add_module(tb)
   sim.add_module(hiv)
   sim.run()

TB with malnutrition:

.. code-block:: python

   from tbsim.comorbidities.malnutrition import Malnutrition
   from tbsim import TB
   
   malnutrition = Malnutrition()
   sim.add_module(malnutrition)

Connector Classes
----------------

**TB_HIV_CNN**: Connects TB and HIV models for integrated simulation
**TB_Nutrition_Connector**: Links TB and malnutrition models

These connector classes ensure proper interaction between different disease models and maintain consistency in the simulation.

For detailed information about specific comorbidity types, see the individual subpackage documentation above.

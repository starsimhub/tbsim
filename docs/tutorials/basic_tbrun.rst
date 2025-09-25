Example Tutorial
===============

This tutorial demonstrates the most basic usage of tbsim to create and run a tuberculosis simulation. It shows how to set up a population, configure the TB disease module, define contact networks, and visualize results.

Overview
--------

This example creates a simple TB simulation with:
- 1000 agents in the population
- TB disease transmission with configurable parameters
- Random contact networks
- Birth and death demographics
- Basic result visualization

The simulation runs from 1940 to 2010 with weekly time steps.

Setup and Imports
-----------------

First, import the required modules:

.. code-block:: python

   import tbsim as mtb
   import starsim as ss
   import sciris as sc
   import matplotlib.pyplot as plt

Building the Simulation
-----------------------

The ``build_tbsim()`` function creates a complete simulation with all necessary components:

.. code-block:: python

   def build_tbsim(sim_pars=None):
      # Simulation parameters
      spars = dict(
         dt = ss.days(7),           # Weekly time steps
         start = ss.date('1940-01-01'),  # Simulation start date
         stop = ss.date('2010-12-31'),   # Simulation end date
         rand_seed = 1,             # Random seed for reproducibility
      )
      if sim_pars is not None:
         spars.update(sim_pars)

      # Create population of 1000 agents
      pop = ss.People(n_agents=1000)
      
      # Configure TB disease module
      tb = mtb.TB(dict(
         dt = ss.days(7),           # Disease module time step
         beta = ss.peryear(0.0025)  # Transmission rate per year
      ))
      
      # Define random contact network
      net = ss.RandomNet(dict(
         n_contacts=ss.poisson(lam=5),  # Average 5 contacts per agent
         dur=0                          # Contact duration (0 = instantaneous)
      ))
      
      # Demographics: births and deaths
      births = ss.Births(pars=dict(birth_rate=20))  # 20 births per 1000 per year
      deaths = ss.Deaths(pars=dict(death_rate=15))  # 15 deaths per 1000 per year

      # Assemble the simulation
      sim = ss.Sim(
         people=pop,
         networks=net,
         diseases=tb,
         demographics=[deaths, births],
         pars=spars,
      )

      sim.pars.verbose = 0  # Reduce output verbosity
      return sim

Running the Simulation
----------------------

Execute the simulation and visualize results:

.. code-block:: python

   if __name__ == '__main__':
      # Create and run the simulation
      sim = build_tbsim()
      sim.run()
      
      # Print simulation parameters
      print(sim.pars)
      
      # Process and visualize results
      results = sim.results.flatten()
      results = {'basic': results}
      mtb.plot_combined(
         results, 
         dark=True, 
         n_cols=3, 
         filter=mtb.FILTERS.important_metrics
      )
      
      plt.show()

Key Parameters Explained
------------------------

- **dt**: Time step duration (7 days = weekly updates)
- **beta**: TB transmission rate (0.0025 per year per contact)
- **n_contacts**: Number of contacts per agent (Poisson distribution with mean 5)
- **birth_rate/death_rate**: Demographic rates per 1000 population per year
- **rand_seed**: Ensures reproducible results

Expected Output
---------------

The simulation will generate plots showing key TB metrics over time, including:
- TB prevalence and incidence
- Disease progression through different states
- Population demographics
- Contact network statistics

This basic example provides a foundation for more complex TB simulations with additional interventions, comorbidities, or population structures.



# Overview of the People Class

The **People** module is a core component of a simulation framework that models agents (or people) over time. It manages the creation, state updates, and lifecycle events (such as births and deaths) for a population. The design emphasizes scalability, state coherence, dynamic extensibility, and tight integration with the overall simulation environment.

---

## 1. Introduction and Purpose

In many simulation environments, particularly in epidemiology, social dynamics, or agent-based modeling, managing a population efficiently is key. The **People** class serves as a container and orchestrator for all agent-related operations. Its responsibilities include:

- **Initialization:** Setting up internal arrays to uniquely identify agents and tracking important simulation states.
- **State Management:** Maintaining dynamic arrays (or *states*) for agent attributes such as age, gender, life status, and any other custom state introduced by additional modules.
- **Lifecycle Operations:** Managing growth (adding agents), death (through scheduled or immediate removal), and age progression.
- **Interoperability:** Offering mechanisms for linking with other simulation modules via a modular design, enabling users to extend functionality without altering the core People logic.

---

## 2. Architectural Overview

### Modular & Extensible Design

- **Decoupled State Handling:**  
  The People class holds a registry of states in both a user-friendly dictionary (`self.states`) and an internal registry (`self._states`). This dual mechanism ensures that while external users can access the states via intuitive attribute syntax (e.g., `people.alive` or `people.age`), the simulation engine maintains full control over state consistency and dynamic resizing.

- **Plugin Integration:**  
  Through methods like `add_module`, the design allows external modules (representing distinct simulation concerns such as disease progression or intervention strategies) to register their own states. This enables dynamic extension of the People object while keeping module-specific logic decoupled and self-contained.

- **Inter-Class Communication:**  
  The People instance serves as the backbone for simulation-wide operations, linking to external modules (like a disease module) and aggregating results for analysis.

### Core Dependencies

The code leverages several powerful Python libraries, including:
- **NumPy & Pandas:** For numerical operations and dataframe manipulations.
- **Sciris:** For object management and enhanced dictionary-like objects.
- **StarSim:** For simulation-specific functions such as random UID generation, state creation, and statistical distributions.
- **Pathlib:** For file handling when loading population age data.

---

## 3. Key Components and Concepts

### Unique Identifiers and State Arrays

- **UIDs, Slots, and AUIDs:**  
  Upon initialization, the People class generates UIDs that uniquely identify every agent. These identifiers are used to:
  - Track persistent identity.
  - Serve as indices in dynamic arrays (or states) that monitor agent properties over time.
  - Synchronize various state arrays so that the addition or removal of agents is managed uniformly across the simulation.

### Managing Agent Attributes via States

- **Primary States:**  
  The class immediately initializes essential states such as:
  - `alive` (indicating if an agent is active)
  - `female` (a Bernoulli distribution to determine gender)
  - `age` (with an age distribution derived from a provided dataset or a default uniform distribution)
  - Additional states like `ti_dead` (time index for death) and `scale` (a multiplier for simulation results).
  
- **Dynamic State Registration:**  
  External states can be added later via `extra_states` during initialization or through modules. The states are stored in an `ndict`, a flexible container that maps state names to their respective arrays.

### Age Distribution and Data Processing

The People class includes a static method, `get_age_dist`, that processes input data in multiple formats (e.g., NumPy arrays, Pandas Series or DataFrames, and even file paths to CSVs). Key points include:
- **Flexible Input:**  
  It interprets the provided age data as either counts or probabilities. The method automatically infers bin edges if only lower boundaries are provided.
- **Fallback Distribution:**  
  In the absence of age data, a uniform distribution (ages 0–100) is used.

---

## 4. Detailed Method Analysis

### Initialization and Construction

- **`__init__`:**  
  Sets up the fundamental arrays and state infrastructure. Key operations include:
  - Generating UIDs and preparing arrays (e.g., `slot`, `parent`).
  - Creating default states using StarSim helper functions.
  - Promoting any extra states for expanded functionality.
  - Linking state objects back to the People instance to ensure coordinated updates.

- **Static Method `get_age_dist`:**  
  Converts a variety of age data representations into an `ss.Dist` distribution. This allows new agents to be assigned ages based on realistic demographic distributions.

### Linking with the Simulation Environment

- **`link_sim`:**  
  Once the People object is created, it must be associated with an overarching simulation object. This method:
  - Registers all state objects with the simulation.
  - Ensures that the People instance knows about simulation-level parameters and time-step data.
  
- **`add_module`:**  
  Facilitates the integration of external modules:
  - Dynamically registers module-specific states.
  - Exposes the module states both as attributes (e.g., `people.hiv`) and in the main state dictionary.

### Agent Lifecycle Operations

- **`grow`:**  
  This method handles the addition of new agents, ensuring that:
  - UIDs are generated for newly added agents.
  - All state arrays are dynamically resized.
  - Active agent tracking (`auids`) is updated accordingly.

- **Death Handling:**
  - **`request_death`:**  
    Allows external modules to schedule an agent’s death, safeguarding that multiple modules can request death for the same agent within a time step.
  - **`step_die`:**  
    Reviews the scheduled deaths and executes state changes by setting the `alive` flag to False. It also delegates further module-specific death processing.
  - **`remove_dead`:**  
    Cleans up the population by removing dead agents from networks and active agent lists.

### Iteration, Indexing, and Serialization

- **Pythonic Interface Enhancements:**
  - Special methods like `__getitem__`, `__setitem__`, `__iter__`, and `__len__` enable intuitive indexing and iteration over agents.
  - The class provides a `person()` method that aggregates an agent’s various state attributes into a `Person` object.
  
- **`__setstate__`:**  
  This method ensures proper deserialization of People objects, re-creating internal state registries so that copied or unpickled objects maintain full functionality.

### Result Aggregation and Post-Step Operations

- **`update_results` & `finish_step`:**  
  These methods integrate People state changes with the simulation’s overall result tracking. They:
  - Update counts of alive agents and recorded deaths.
  - Handle end-of-step cleanup by removing dead agents and updating time-dependent states like age.

---

## 5. The Person Class: A Lightweight Data Wrapper

The **Person** class is implemented as a simple extension of a dictionary object:
- **Purpose:**  
  To encapsulate all attributes related to a single agent.
- **Key Functionality:**  
  It provides a `to_df` method, converting the agent’s attribute dictionary into a Pandas DataFrame. This is particularly useful for debugging, reporting, or exporting simulation data for further analysis.

---

## 6. Error Handling, Debugging, and Best Practices

- **Robust Initialization:**  
  The People module performs several checks (e.g., ensuring that states are added only once and that re-initialization is prevented) to avoid inconsistent simulation states.
- **Dynamic Resizing:**  
  Special care is taken to always update all state arrays simultaneously when agents are grown or removed. This consistency is crucial in large-scale simulations.
- **Integration Warnings:**  
  The design emphasizes that users should interact with higher-level simulation components (e.g., using `sim.init(reset=True)`) instead of directly reinitializing a People instance, thereby maintaining system integrity.

---

## 7. Usage Example

Here is a brief example of how a simulation might instantiate and interact with the People class:

```python
import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd

# Instantiate the People object with a population of 2000 agents
ppl = ss.People(2000)

# Optionally, link the people object to a simulation environment
# sim = ss.Sim(parameters, ...)
# ppl.link_sim(sim)

# Initialize the states with default or provided values
ppl.init_vals()

# Access individual agent properties using indexing:
agent = ppl[0]  # returns a Person instance
print(agent.to_df())

# Add new agents
new_uids = ppl.grow(n=100)

# Request death for some agents (simulate an event)
ppl.request_death(np.array([0, 1, 2]))

# Process death events in the current timestep
death_ids = ppl.step_die()
print("Agents that died this timestep:", death_ids)
```

This example demonstrates the core capabilities: initializing the population, interacting with individual agents, dynamically adding or removing agents, and interfacing with other simulation components.

---

# Class People 
<img width="1108" alt="image" src="https://github.com/user-attachments/assets/92d16c92-03cd-4600-9534-07c6debde71d" />


## 8. Conclusion

The People class and its associated Person class form a robust framework for managing agent-based simulations. Its design leverages best practices in modularity, dynamic data handling, and extensibility. By abstracting agent state management behind a rich set of methods and properties, the module allows simulation developers to integrate complex phenomena (such as aging, death, and multi-module interactions) seamlessly. This architecture ensures that simulations can scale efficiently, maintain data integrity, and remain extensible for future enhancements.

This document should serve as a detailed technical overview for software engineers looking to understand, utilize, and extend the People module within their simulation projects.

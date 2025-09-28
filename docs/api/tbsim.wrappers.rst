TBsim Wrappers
==============

This module provides utility wrappers and helper classes for common TBsim operations, particularly for agent filtering and selection.

Main Wrappers Module
--------------------

.. automodule:: tbsim.wrappers
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**Agents**
   Static utility class for filtering and selecting agents based on various criteria

Key Features
-----------

- **Agent Filtering**: Filter agents by age, alive status, and other criteria
- **Static Methods**: No instantiation required, use directly
- **UID-based Selection**: Returns UIDs for efficient agent selection
- **Age-based Filtering**: Common age group selections (under 5, over 5, etc.)
- **Alive Status**: Filter only currently alive individuals

Usage Examples
-------------

Basic agent filtering:

.. code-block:: python

   from tbsim.wrappers import Agents
   
   # Get all alive individuals
   alive_uids = Agents.get_alive(sim.people)
   
   # Get children under 5 years old
   children_uids = Agents.under_5(sim.people)
   
   # Get adults over 18 years old
   adults_uids = Agents.get_by_age(sim.people, min_age=18)

Age-based filtering:

.. code-block:: python

   # Get individuals between 15 and 65 years old
   working_age_uids = Agents.get_by_age(sim.people, min_age=15, max_age=65)
   
   # Get elderly individuals over 65
   elderly_uids = Agents.get_by_age(sim.people, min_age=65)

Combined filtering:

.. code-block:: python

   # Get alive children under 5
   alive_children = Agents.get_alive_by_age(sim.people, max_age=5)

Available Methods
----------------

**Age-based Methods**
   - `of_age(people, age)`: Get agents of exact age
   - `under_5(people)`: Get agents ≤ 5 years old
   - `over_5(people)`: Get agents > 5 years old
   - `get_by_age(people, max_age, min_age)`: Get agents in age range

**Status-based Methods**
   - `get_alive(people)`: Get all alive agents
   - `get_alive_by_age(people, max_age, min_age)`: Get alive agents in age range

These methods are designed to work efficiently with the Starsim framework and return UIDs for further processing.
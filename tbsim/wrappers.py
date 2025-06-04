import numpy as np
import starsim as ss

__all__ = ['Agents']

    
class Agents:
    """
    Static utility methods for filtering agents by age, alive status, etc.
    """
    @staticmethod
    def under_5(people):
        """
        Return UIDs of alive individuals <= 5 years old.

        Parameters:
            people (People): The sim.people object

        Returns:
            UIDs: The selected individuals
        """
        alive = people.auids
        age = people.age[alive]
        mask = age <= 5
        return ss.uids(alive[mask])
    
    @staticmethod
    def over_5(people):
        """
        Return UIDs of alive individuals > 5 years old.

        Parameters:
            people (People): The sim.people object

        Returns:
            UIDs: The selected individuals
        """
        alive = people.auids
        age = people.age[alive]
        mask = age > 5
        return ss.uids(alive[mask])
    
    @staticmethod
    def get_by_age(people, max_age=None, min_age=None):
        """
        Return UIDs of individuals filtered by age.

        Parameters:
            people (People): The sim.people object
            max_age (float, optional): Upper age bound (inclusive)
            min_age (float, optional): Lower age bound (exclusive)

        Returns:
            UIDs: The selected individuals
        """
        age = people.age
        mask = np.ones_like(age, dtype=bool)
        if max_age is not None:
            mask &= age <= max_age
        if min_age is not None:
            mask &= age > min_age
        return ss.uids(np.where(mask)[0])

    @staticmethod
    def get_alive(people):
        """Return UIDs of currently alive individuals."""
        return people.auids

    @staticmethod
    def get_alive_by_age(people, max_age=None, min_age=None):
        """
        Return UIDs of alive individuals filtered by age.
        """
        alive = people.auids
        age = people.age[alive]
        mask = np.ones(len(alive), dtype=bool)
        if max_age is not None:
            mask &= age <= max_age
        if min_age is not None:
            mask &= age > min_age
        return ss.uids(alive[mask])

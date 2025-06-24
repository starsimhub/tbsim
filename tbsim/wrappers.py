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
        return ss.uids(people.age <= 5)
    
    @staticmethod
    def over_5(people):
        """
        Return UIDs of alive individuals > 5 years old.

        Parameters:
            people (People): The sim.people object

        Returns:
            UIDs: The selected individuals
        """
        return ss.uids(people.age > 5)
    
    @staticmethod
    def get_by_age(people, max_age=None, min_age=None):
        """
        Return UIDs of ALIVE individuals filtered by age.

        Parameters:
            people (People): The sim.people object
            max_age (float, optional): Upper age bound (inclusive)
            min_age (float, optional): Lower age bound (exclusive)

        Returns:
            UIDs: The selected individuals
        """
        if max_age is None | min_age is None:
            raise ValueError("At least one of max_age or min_age must be specified.")
        if max_age < min_age:
            raise ValueError("max_age must be greater than min_age.")
        return ss.uids((people.age >= min_age) & (people.age < max_age))

    @staticmethod
    def get_alive(people):
        """Return UIDs of currently alive individuals."""
        return ss.uids(people.auids)

   
    @staticmethod
    def where(condition):
        """
        Return UIDs based on a condition.

        Parameters:
            condition (array-like): A boolean array or condition to filter UIDs.

        Returns:
            UIDs: Starsim-compliant UIDs of selected indices.
        """
        results = None
        if not isinstance(condition, (np.ndarray, list)):
            raise ValueError("Condition must be a numpy array or list.")
        if len(condition) != len(ss.uids()):
            raise ValueError("Condition length must match the number of agents.")
        try:
            results = ss.uids(condition)
        except Exception as e:
            
            error_message = str(e)
            error_message+= f"\nCondition: {condition}\n"
            error_message+= f"Number of agents: {len(ss.uids())}\n"
            error_message+= f"Condition length: {len(condition)}\n"
            error_message+= f"Condition type: {type(condition)}\n"
            print(error_message)
            # Log the error message or handle it as needed
            # Handle specific exceptions if needed
            
            raise ValueError(f"\n Error processing condition: {e}")

        return results
        
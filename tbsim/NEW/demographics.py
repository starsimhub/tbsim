import starsim as ss
import numpy as np

all = ['NewBornsSocialIntroduction']

class NewBornsSocialIntroduction(ss.Pregnancy):

    def generate_and_associate_offspring(self, conceive_uids, include_nutritional_states, network):
        """
        Generates offspring for the given conceive_uids and associates them within the specified network.
        Optionally includes nutritional states for the newborns if specified.

        Parameters:
        - conceive_uids (list): Unique identifiers for the individuals conceiving offspring.
        - include_nutritional_states (bool): If True, assigns nutritional states to the newborns based on their mother.
        - network (string): The name of the network to associate the newborns with. i.e. 'HarlemNet' which are the households.

        Returns:
        - list: Unique identifiers for the newborns.
        """
        newborn_uids = super().make_embryos(conceive_uids)

        if len(newborn_uids) == 0:
            return newborn_uids

        people = self.sim.people
        people.hhid[newborn_uids] = people.hhid[conceive_uids]
        people.arm[newborn_uids] = people.arm[conceive_uids]
        
        if include_nutritional_states:
            self.assign_malnutritional_states_to_newborns(newborn_uids, conceive_uids, self.sim)
        
        hn = self.sim.networks[network]

        p1s = []
        p2s = []
        for newborn_uid, mother_uid in zip(newborn_uids, conceive_uids):
            for contact in hn.find_contacts(mother_uid):
                p1s.append(contact)
                p2s.append(newborn_uid)

        hn.edges.p1 = ss.uids(np.concatenate([hn.edges.p1, p1s]))
        hn.edges.p2 = ss.uids(np.concatenate([hn.edges.p2, p2s]))
        # Beta is zero while prenatal
        hn.edges.beta = np.concatenate([hn.edges.beta, np.zeros_like(p1s)])#.astype(ss.dtypes.float)

        return newborn_uids
    
    def assign_malnutritional_states_to_newborns(newborn_uids, conceive_uids, simulation):
        """
        Assigns malnutrition states from parents to their newborns within a simulation.

        This function directly modifies the malnutrition states of newborns in the simulation
        to match those of their parents at the time of birth.

        Parameters:
        - newborn_uids (list of int): The unique identifiers of the newborns.
        - conceive_uids (list of int): The unique identifiers of the parents.
        - simulation (Simulation): The simulation object containing the malnutrition disease model
        and the population to which the newborns and parents belong.

        Returns:
        None. The function modifies the simulation's state in-place.
        """
         # Assume baby has the same micro/macro state as mom
        nut = simulation.diseases['malnutrition']
        nut.micro_state[newborn_uids] = nut.micro_state[conceive_uids]
        nut.macro_state[newborn_uids] = nut.macro_state[conceive_uids]
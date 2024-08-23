import starsim as ss
import numpy as np
import networkx as nx

__all__ = ['HouseholdUnit', 'HouseholdNet', 'HouseholdNewborns']

class HouseholdNet(ss.Network):
    def __init__(self, hhs, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.hhs = hhs
        self.default_pars()
        self.update_pars(pars, **kwargs)
        return

    """_summary_
    
    """
    def init_pre(self, sim):
        super().init_pre(sim)
        for hh in self.hhs:                 # For each household
            p1s, p2s = hh.edges()        # Get all their contacts

            self.edges.p1 = np.concatenate([self.edges.p1, p1s])
            self.edges.p2 = np.concatenate([self.edges.p2, p2s])
            self.edges.beta = np.concatenate([self.edges.beta, np.ones_like(p1s)])

        self.edges.p1 = ss.uids(self.edges.p1)
        self.edges.p2 = ss.uids(self.edges.p2)

        return
    
    """_summary_
    Updates the network based on individuals who are delivering at the current simulation time step. 
    For each delivering individual, it identifies their infant, sets the infant's household ID and 
    study arm to match the mother's, and adds the infant to the mother's contacts in the 'harlemnet' network. 
    It then updates the 'p1', 'p2', and 'beta' attributes of the 'harlemnet' network's 'contacts' object with the new contacts.
    """
    def update(self):
        super().update()

        newborns = ((self.sim.people.age > 0) & (self.sim.people.age < self.sim.dt)).uids
        if len(newborns) == 0:
            return

        # Activate household contacts by setting beta to 1
        hn = self.sim.networks['householdnet']
        for infant_uid in newborns:
            hn.edges.beta[hn.edges.p2 == infant_uid] = 1.0

        return
    
    def add_members(self, newborn_mother_dyads):
        p1s = []
        p2s = []
        for newborn_uid, mother_uid in newborn_mother_dyads:
            for contact in self.find_contacts(mother_uid):
                p1s.append(contact)
                p2s.append(newborn_uid)

        self.edges.p1 = ss.uids(np.concatenate([self.edges.p1, p1s]))
        self.edges.p2 = ss.uids(np.concatenate([self.edges.p2, p2s]))

        # Beta is zero while prenatal
        self.edges.beta = np.concatenate([self.edges.beta, np.zeros_like(p1s)])#.astype(ss.dtypes.float)


class HouseholdUnit():
    """
    _Attributes:_
    hhid:   Unique identifier for the household.
    uids:   List of unique identifiers for household members.
    n:      Number of members in the household.
    macro:  Macro nutrition status of the household.
    arm:    Group in the study that the household belongs to.
    """ 
    def __init__(self, hhid, uids, hh_macro, hh_bmi, study_arm, indexes_arr=None, properties=None):
        self.hhid = hhid
        self.uids = uids
        self.n = len(uids)
        self.macro_metric = hh_macro
        self.bmi_metric = hh_bmi
        self.arm = study_arm
        return
    
    """_summary_
    The contacts method generates a complete graph using NetworkX, 
    representing all possible pairs of contacts within a group of individuals identified by uids. 
    It then separates these pairs into two lists, p1s and p2s, and returns these lists.
    """
    def edges(self):
        g = nx.complete_graph(self.uids)
        p1s = []
        p2s = []
        for edge in g.edges():
            p1, p2 = edge
            p1s.append(p1)
            p2s.append(p2)
        return p1s, p2s
    
    def set_property(self, prop, value, uidx=None):
        setattr(self, prop, value)
        return
    
class HouseholdNewborns(ss.Pregnancy):
    """
    A class that represents the generation of newborns in a household network. Inherits from starsim.Pregnancy.
    Attributes:
        sim (Simulation): The simulation object.
    """

    def assign_nutrition_status(self, newborn_uids, conceive_uids):
        nut = self.sim.diseases['malnutrition']
        nut.micro_state[newborn_uids] = nut.micro_state[conceive_uids]
        nut.macro_state[newborn_uids] = nut.macro_state[conceive_uids]
        nut.bmi_state[newborn_uids] = nut.bmi_state[conceive_uids]
        return
            
    def make_embryos(self, conceive_uids, targetNetworkName='householdnet'):
        """
        Generates newborns based on the given conceive UIDs.
        Args:
            conceive_uids (list): A list of UIDs representing the individuals who conceived.
            targetNetworkName (str, optional): The name of the target network. Defaults to 'householdnet'.
        Returns:
            list: A list of UIDs representing the newborns.
        """

        newborn_uids = super().make_embryos(conceive_uids)

        if len(newborn_uids) == 0:
            return newborn_uids

        # Assign household ID, study arm, and nutrition status to newborns
        people = self.sim.people
        people.hhid[newborn_uids] = people.hhid[conceive_uids]
        people.arm[newborn_uids] = people.arm[conceive_uids]
        self.assign_nutrition_status(newborn_uids, conceive_uids)

        # Connect to networks
        hn = self.sim.networks[targetNetworkName]
        dyads = zip(newborn_uids, conceive_uids)
        hn.add_members(dyads)

        return newborn_uids

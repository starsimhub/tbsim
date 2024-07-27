import starsim as ss
import numpy as np
import networkx as nx

__all__ = ['HouseHold', 'HouseHoldNet']

class HouseHoldNet(ss.Network):
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
        hn = self.sim.networks['harlemnet']
        for infant_uid in newborns:
            hn.edges.beta[hn.edges.p2 == infant_uid] = 1.0

        return
    
    
"""_Summary_
The HouseHold network class.
"""
class HouseHold():
    """
    _Attributes:_
    hhid:   Unique identifier for the household.
    uids:   List of unique identifiers for household members.
    n:      Number of members in the household.
    macro:  Macro nutrition status of the household.
    arm:    Group in the study that the household belongs to.
    """ 
    def __init__(self, hhid, uids, macro_nutrition, study_arm):
        self.hhid = hhid
        self.uids = uids
        self.n = len(uids)
        self.macro = macro_nutrition
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
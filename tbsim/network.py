import starsim as ss
import numpy as np
import pandas as pd
import networkx as nx

__all__ = ['HarlemNet', 'HouseHold']

class HarlemNet(ss.Network):
    def __init__(self, hhs, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.hhs = hhs
        self.pars = ss.dictmergeleft(dict(), pars)
        #self.sim = None
        return

    """_summary_
    
    """
    def initialize(self, sim):
        self.sim = sim # for births
        for hh in self.hhs:                 # For each household
            p1s, p2s = hh.contacts()        # Get all their contacts

            self.contacts.p1 = np.concatenate([self.contacts.p1, p1s])
            self.contacts.p2 = np.concatenate([self.contacts.p2, p2s])
            self.contacts.beta = np.concatenate([self.contacts.beta, np.ones_like(p1s)])

        self.contacts.p1 = self.contacts.p1.astype(ss.dtypes.int)
        self.contacts.p2 = self.contacts.p2.astype(ss.dtypes.int)
        self.contacts.beta = self.contacts.beta.astype(ss.dtypes.float)

        return
    
    """_summary_
    Updates the network based on individuals who are delivering at the current simulation time step. 
    For each delivering individual, it identifies their infant, sets the infant's household ID and 
    study arm to match the mother's, and adds the infant to the mother's contacts in the 'harlemnet' network. 
    It then updates the 'p1', 'p2', and 'beta' attributes of the 'harlemnet' network's 'contacts' object with the new contacts.
    """
    def update(self, ppl):
        super().update(ppl)

        preg = self.sim.demographics['pregnancy']
        #deliveries = ss.true(preg.pregnant & (preg.ti_delivery <= self.sim.ti))
        deliveries = ss.true(preg.ti_delivery == self.sim.ti)

        if len(deliveries) == 0:
            return

        mn = self.sim.networks['maternalnet'].to_df()
        hn = self.sim.networks['harlemnet']

        p1s = []
        p2s = []
        for mother_uid in deliveries:
            infant_uid = mn.loc[(mn['p1'] == mother_uid) & (mn['dur'] >= 0)]['p2'].values[0] # No twins!
            ppl.hhid[infant_uid] = ppl.hhid[mother_uid]
            ppl.arm[infant_uid] = ppl.arm[mother_uid]

            for contact in hn.find_contacts(mother_uid):
                p1s.append(contact)
                p2s.append(infant_uid)

        hn.contacts.p1 = np.concatenate([hn.contacts.p1, p1s]).astype(ss.dtypes.int)
        hn.contacts.p2 = np.concatenate([hn.contacts.p2, p2s]).astype(ss.dtypes.int)
        hn.contacts.beta = np.concatenate([hn.contacts.beta, np.ones_like(p1s)]).astype(ss.dtypes.float)

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
    def contacts(self):
        g = nx.complete_graph(self.uids)
        p1s = []
        p2s = []
        for edge in g.edges():
            p1, p2 = edge
            p1s.append(p1)
            p2s.append(p2)
        return p1s, p2s
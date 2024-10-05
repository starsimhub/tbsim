import starsim as ss
import numpy as np
import networkx as nx

__all__ = ['HouseholdNet', 'HouseholdNewborns']

class HouseholdNet(ss.Network):
    def __init__(self, hhs=None, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.hhs = [] if hhs is None else hhs
        self.define_pars()
        self.update_pars(pars, **kwargs)
        return

    def add_hh(self, uids):
        g = nx.complete_graph(uids)
        p1s = []
        p2s = []
        for edge in g.edges():
            p1, p2 = edge
            p1s.append(p1)
            p2s.append(p2)

        self.edges.p1 = ss.uids(np.concatenate([self.edges.p1, p1s]))
        self.edges.p2 = ss.uids(np.concatenate([self.edges.p2, p2s]))
        self.edges.beta = np.concatenate([self.edges.beta, np.ones_like(p1s)])
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        for hh in self.hhs:        # For each household
            self.add_hh(hh)
            p1s, p2s = hh.edges()  # Get all their contacts

            self.edges.p1 = np.concatenate([self.edges.p1, p1s])
            self.edges.p2 = np.concatenate([self.edges.p2, p2s])
            self.edges.beta = np.concatenate([self.edges.beta, np.ones_like(p1s)])

        self.edges.p1 = ss.uids(self.edges.p1)
        self.edges.p2 = ss.uids(self.edges.p2)

        return
    
    def update(self):
        """
        Converts newborns added by HouseholdNewborns to full household member by
        updating beta for edges adjacent to the newborn. Only works in
        conjunction with HouseholdNewborns as a replacement for Pregnancy.
        """
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



class HouseholdNewborns(ss.Pregnancy):
    """
    A class that represents the generation of newborns in a household network. Inherits from starsim.Pregnancy.
    Attributes:
        sim (Simulation): The simulation object.
    """

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
        rations = self.sim.interventions.rationstrial
        rations.hhid[newborn_uids] = rations.hhid[conceive_uids]
        rations.arm[newborn_uids] = rations.arm[conceive_uids]

        # Connect to networks
        hn = self.sim.networks[targetNetworkName]
        dyads = zip(newborn_uids, conceive_uids)
        hn.add_members(dyads)

        return newborn_uids

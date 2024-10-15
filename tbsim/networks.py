import starsim as ss
import numpy as np
import networkx as nx

__all__ = ['HouseholdNet']

class HouseholdNet(ss.Network):
    def __init__(self, hhs=None, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.hhs = [] if hhs is None else hhs
        self.define_pars(
            add_newborns = False,
        )
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

    def step(self):
        """ Adds newborns to the trial population, including hhid, arm, and household contacts """
        super().step()

        if not self.pars.add_newborns:
            return

        newborn_uids = ss.uids((self.sim.people.age > 0) & (self.sim.people.age < self.dt))
        if len(newborn_uids) == 0:
            return

        mother_uids = self.sim.people.parent[newborn_uids]

        if self.ti == 0:
            # Filter out agents that were part of the initial population rather than born
            keep = (mother_uids >= 0)
            newborn_uids = newborn_uids[keep]
            mother_uids = mother_uids[keep]

        if len(newborn_uids) == 0:
            return # Nothing to do

        rations = self.sim.interventions.rationstrial

        # Connect to networks
        p1s, p2s = [], []
        for newborn_uid, mother_uid in zip(newborn_uids, mother_uids):
            #contacts = self.find_contacts(mother_uid) # Do not use find_contacts because mother could have died (so no contacts)
            contacts = ss.uids(rations.hhid == rations.hhid[mother_uid]) # Fortunately, we can still retrieve the hhid of the mother, even if dead
            if len(contacts) > 0:
                # Ut oh, baby might be the only agent in the house!
                p1s.append(contacts)
                p2s.append([newborn_uid] * len(contacts))

        p1 = ss.uids.cat(p1s)
        p2 = ss.uids.cat(p2s)

        self.edges.p1   = ss.uids.cat([self.edges.p1, p1])
        self.edges.p2   = ss.uids.cat([self.edges.p2, p2])
        self.edges.beta = ss.uids.cat([self.edges.beta, np.ones_like(p1)])

        # Set HHID and arm (works even if mother has died)
        rations.hhid[newborn_uids] = rations.hhid[mother_uids]
        rations.arm[newborn_uids] = rations.arm[mother_uids]

        return

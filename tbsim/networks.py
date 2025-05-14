import starsim as ss
import numpy as np
import networkx as nx

__all__ = ['HouseholdNet']

class HouseholdNet(ss.Network):
    """
    A household-level contact network for agent-based simulations using Starsim.

    This network constructs complete graphs among household members and supports 
    dynamically adding newborns to the simulation and linking them to their household 
    based on the parent-child relationship. It is especially useful in intervention 
    trials where household structure and arm assignment are important (e.g., RATIONS trial).

    Parameters
    ----------
    hhs : list of lists or arrays of int, optional
        A list of households, where each household is represented by a list or array of agent UIDs.
    pars : dict, optional
        Dictionary of network parameters. Supports:
            - `add_newborns` (bool): Whether to dynamically add newborns to households.
    **kwargs : dict
        Additional keyword arguments passed to the `Network` base class.

    Attributes
    ----------
    hhs : list
        List of household UID groups.
    pars : sc.objdict
        Dictionary-like container of network parameters.
    edges : Starsim EdgeStruct
        Container for the network's edges (p1, p2, and beta arrays).

    Methods
    -------
    add_hh(uids):
        Add a complete graph among the given UIDs to the network.
    
    init_pre(sim):
        Initialize the network prior to simulation start. Adds initial household connections.

    step():
        During simulation, adds newborns to the network by linking them to their household contacts 
        and assigning household-level attributes (e.g., hhid, trial arm).
    """
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

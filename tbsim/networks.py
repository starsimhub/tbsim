import starsim as ss
import numpy as np
import networkx as nx

__all__ = ['HouseholdNet']

class HouseholdNet(ss.Network):
    """
    **** RATIONS trial network ****
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


class HouseholdNetGeneric(ss.Network):
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

    def _generate_complete_graph_edges(self, uids):
        """
        Generate all edges for a complete graph without using NetworkX.
        More efficient than NetworkX for simple complete graphs.
        """
        n = len(uids)
        if n < 2:
            return [], []
        
        # Calculate total number of edges in complete graph
        total_edges = n * (n - 1) // 2
        
        # Pre-allocate arrays
        p1s = np.empty(total_edges, dtype=int)
        p2s = np.empty(total_edges, dtype=int)
        
        # Generate all edges efficiently
        edge_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                p1s[edge_idx] = uids[i]
                p2s[edge_idx] = uids[j]
                edge_idx += 1
        
        return p1s, p2s

    def add_hh(self, uids):
        """Add a complete graph among the given UIDs to the network."""
        if len(uids) < 2:
            return
            
        p1s, p2s = self._generate_complete_graph_edges(uids)
        
        # Efficient concatenation - only concatenate once
        self.edges.p1 = ss.uids(np.concatenate([self.edges.p1, p1s]))
        self.edges.p2 = ss.uids(np.concatenate([self.edges.p2, p2s]))
        self.edges.beta = np.concatenate([self.edges.beta, np.ones(len(p1s))])
        return

    def init_pre(self, sim):
        """Initialize the network prior to simulation start. Adds initial household connections."""
        super().init_pre(sim)
        
        # Pre-calculate total number of edges needed
        total_edges = sum(len(hh) * (len(hh) - 1) // 2 for hh in self.hhs if len(hh) >= 2)
        
        if total_edges == 0:
            return
            
        # Pre-allocate arrays for all edges
        all_p1s = np.empty(total_edges, dtype=int)
        all_p2s = np.empty(total_edges, dtype=int)
        
        # Fill arrays efficiently
        edge_idx = 0
        for hh in self.hhs:
            if len(hh) >= 2:
                p1s, p2s = self._generate_complete_graph_edges(hh)
                n_edges = len(p1s)
                all_p1s[edge_idx:edge_idx + n_edges] = p1s
                all_p2s[edge_idx:edge_idx + n_edges] = p2s
                edge_idx += n_edges
        
        # Single concatenation for all edges
        self.edges.p1 = ss.uids(np.concatenate([self.edges.p1, all_p1s]))
        self.edges.p2 = ss.uids(np.concatenate([self.edges.p2, all_p2s]))
        self.edges.beta = np.concatenate([self.edges.beta, np.ones(total_edges)])
        
        return

    def step(self): 
        """Adds newborns to the trial population, including hhid, arm, and household contacts."""
        if not self.pars.add_newborns:
            return

        # Vectorized newborn detection
        dt = self.sim.t.dt if hasattr(self.sim.t, 'dt') else 1.0
        newborn_mask = (self.sim.people.age > 0) & (self.sim.people.age < dt)
        newborn_uids = ss.uids(newborn_mask)
        
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

        # Check if rationstrial intervention exists
        if hasattr(self.sim.interventions, 'rationstrial') and self.sim.interventions.rationstrial is not None:
            rations = self.sim.interventions.rationstrial
            
            # Vectorized household contact finding
            mother_hhids = rations.hhid[mother_uids]
            
            # Create mapping from hhid to all contacts in that household
            unique_hhids, hhid_indices = np.unique(mother_hhids, return_inverse=True)
            
            # Pre-calculate total number of new edges needed
            total_new_edges = 0
            hhid_to_contacts = {}
            
            for hhid in unique_hhids:
                contacts = ss.uids(rations.hhid == hhid)
                if len(contacts) > 0:
                    hhid_to_contacts[hhid] = contacts
                    # Count newborns in this household
                    newborns_in_hh = np.sum(mother_hhids == hhid)
                    total_new_edges += len(contacts) * newborns_in_hh
            
            if total_new_edges > 0:
                # Pre-allocate arrays for new edges
                new_p1s = np.empty(total_new_edges, dtype=int)
                new_p2s = np.empty(total_new_edges, dtype=int)
                
                # Fill arrays efficiently
                edge_idx = 0
                for i, (newborn_uid, mother_uid) in enumerate(zip(newborn_uids, mother_uids)):
                    hhid = mother_hhids[i]
                    if hhid in hhid_to_contacts:
                        contacts = hhid_to_contacts[hhid]
                        n_contacts = len(contacts)
                        new_p1s[edge_idx:edge_idx + n_contacts] = contacts
                        new_p2s[edge_idx:edge_idx + n_contacts] = newborn_uid
                        edge_idx += n_contacts

                # Single concatenation for all new edges
                self.edges.p1 = ss.uids.cat([self.edges.p1, new_p1s])
                self.edges.p2 = ss.uids.cat([self.edges.p2, new_p2s])
                self.edges.beta = ss.uids.cat([self.edges.beta, np.ones(total_new_edges)])

                # Set HHID and arm (works even if mother has died)
                rations.hhid[newborn_uids] = rations.hhid[mother_uids]
                rations.arm[newborn_uids] = rations.arm[mother_uids]
        else:
            # Fallback: create simple connections to mother only
            if len(newborn_uids) > 0:
                # Connect newborns to their mothers
                self.edges.p1 = ss.uids.cat([self.edges.p1, mother_uids])
                self.edges.p2 = ss.uids.cat([self.edges.p2, newborn_uids])
                self.edges.beta = ss.uids.cat([self.edges.beta, np.ones(len(newborn_uids))])

        return
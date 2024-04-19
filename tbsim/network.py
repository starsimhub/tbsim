import starsim as ss
import numpy as np
import pandas as pd
import networkx as nx

__all__ = ['HarlemNet']

class HarlemNet(ss.Network):
    def __init__(self, hhs, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.hhs = hhs
        self.pars = ss.omerge(dict(), pars)
        return

    def initialize(self, sim):
        for hh in self.hhs:
            p1s, p2s = hh.contacts()

            self.contacts.p1 = np.concatenate([self.contacts.p1, p1s])
            self.contacts.p2 = np.concatenate([self.contacts.p2, p2s])
            self.contacts.beta = np.concatenate([self.contacts.beta, np.ones_like(p1s)])

        self.contacts.p1 = self.contacts.p1.astype(ss.dtypes.int)
        self.contacts.p2 = self.contacts.p2.astype(ss.dtypes.int)
        self.contacts.beta = self.contacts.beta.astype(ss.dtypes.float)

        return

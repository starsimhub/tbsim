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
        #self.sim = None
        return

    def initialize(self, sim):
        self.sim = sim # for births
        for hh in self.hhs:
            p1s, p2s = hh.contacts()

            self.contacts.p1 = np.concatenate([self.contacts.p1, p1s])
            self.contacts.p2 = np.concatenate([self.contacts.p2, p2s])
            self.contacts.beta = np.concatenate([self.contacts.beta, np.ones_like(p1s)])

        self.contacts.p1 = self.contacts.p1.astype(ss.dtypes.int)
        self.contacts.p2 = self.contacts.p2.astype(ss.dtypes.int)
        self.contacts.beta = self.contacts.beta.astype(ss.dtypes.float)

        return

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
            hhid = ppl.hhid[mother_uid]
            ppl.hhid[infant_uid] = hhid

            for contact in hn.find_contacts(mother_uid):
                p1s.append(contact)
                p2s.append(infant_uid)

        hn.contacts.p1 = np.concatenate([hn.contacts.p1, p1s]).astype(ss.dtypes.int)
        hn.contacts.p2 = np.concatenate([hn.contacts.p2, p2s]).astype(ss.dtypes.int)
        hn.contacts.beta = np.concatenate([hn.contacts.beta, np.ones_like(p1s)]).astype(ss.dtypes.float)

        return

import starsim as ss
import tbsim as mtb
import networkx as nx
import pandas as pd
import numpy as np

__all__ = ['Harlem', 'StudyArm']


from enum import IntEnum, auto

class StudyArm(IntEnum):
    CONTROL = auto()
    VITAMIN = auto()


class Harlem():
    def __init__(self, pars=None):#p_microdeficient_given_macro, n_hhs=194):

        self.pars = dict(
            p_microdeficient_given_macro = { # Guess values, not from data
                mtb.MacroNutrients.UNSATISFACTORY: 1.0,
                mtb.MacroNutrients.MARGINAL: 0.75,
                mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD: 0.25,
                mtb.MacroNutrients.STANDARD_OR_ABOVE: 0.0,
            },
            n_hhs = 194,
        )
        self.pars = ss.omerge(self.pars, pars)

        self.hhdat = pd.DataFrame({
            'size': np.arange(1,10),
            'p': np.array([3, 17, 24, 20, 13, 9, 7, 4, 3]) / 100
        })

        macro = mtb.MacroNutrients
        self.macrodat = pd.DataFrame({
            'habit': [ macro.STANDARD_OR_ABOVE, macro.SLIGHTLY_BELOW_STANDARD, macro.MARGINAL, macro.UNSATISFACTORY ],
            # These are the 1942 levels
            'p_control': [21.1, 28.9, 38.9, 11.1],
            'p_vitamin': [29.2, 30.3, 28.1, 12.4],
        })
        self.macrodat['p_control'] /= self.macrodat['p_control'].sum()
        self.macrodat['p_vitamin'] /= self.macrodat['p_vitamin'].sum()

        self.armdat = pd.DataFrame({
            'arm': [StudyArm.CONTROL, StudyArm.VITAMIN],
            'p': [0.5, 0.5]
        })
        self.armdat['p'] /= self.armdat['p'].sum()

        self.hhs = self.make_hhs()
        self.n_agents = np.sum([hh.n for hh in self.hhs]) # Hopefully about 579
        return

    def make_hhs(self):
        hh_sizes = np.random.choice(a=self.hhdat['size'].values, p=self.hhdat['p'].values, size=self.pars.n_hhs)
        arm = np.random.choice(a=self.armdat['arm'].values, p=self.armdat['p'].values, size=self.pars.n_hhs)

        idx = 0
        hhs = []
        for hhid, (size, arm) in enumerate(zip(hh_sizes, arm)):
            uids = np.arange(idx, idx+size)
            if arm == StudyArm.CONTROL:
                p = self.macrodat['p_control'].values
            else:
                p = self.macrodat['p_vitamin'].values
            macro = np.random.choice(a=self.macrodat['habit'].values, p=p)
            hh = mtb.HouseHold(hhid, uids, mtb.MacroNutrients(macro), StudyArm(arm))
            hhs.append(hh)
            idx += size

        return hhs

    def people(self):
        extra_states = [
            ss.State('hhid', int, ss.INT_NAN),
            ss.State('arm', int, ss.INT_NAN),
        ]
        pop = ss.People(n_agents = self.n_agents, extra_states=extra_states)
        return pop

    def net(self):
        net = mtb.HarlemNet(self.hhs)
        return net

    def set_states(self, sim):
        pop = sim.people
        nut = sim.diseases['nutrition']
        for hh in self.hhs:
            p_deficient = self.pars.p_microdeficient_given_macro[hh.macro]
            for uid in hh.uids:
                pop.hhid[uid] = hh.hhid
                pop.arm[uid] = hh.arm
                nut.macro[uid] = hh.macro
                nut.micro[uid] = mtb.MicroNutrients.DEFICIENT if np.random.rand() < p_deficient else mtb.MicroNutrients.NORMAL
        
        # Set relative LS progression after changing macro and micro states
        c = sim.connectors['tb_nutrition_connector']
        tb = sim.diseases['tb']
        tb.rel_LS_prog[sim.people.uid] = c.pars.rel_LS_prog_func(nut.macro, nut.micro)
        return

    def choose_seed_infections(self, sim, p_hh):
        tb = sim.diseases['tb']
        hh_has_seed = np.random.binomial(p=p_hh, n=1, size=len(self.hhs))
        seed_uids = []
        for hh, seed in zip(self.hhs, hh_has_seed):
            if not seed:
                continue
            seed_uid = np.random.choice(hh.uids)
            seed_uids.append(seed_uid)
        return np.array(seed_uids)


import starsim as ss
import tbsim as mtb
import sciris as sc
import pandas as pd
import numpy as np

__all__ = ['Harlem', 'StudyArm', 'HarlemPregnancy']


from enum import IntEnum, auto

class StudyArm(IntEnum):
    CONTROL = auto()
    VITAMIN = auto()


class Harlem():
    def __init__(self, pars=None):

        self.pars = sc.objdict(
            p_microdeficient_given_macro = { # Guess values, not from data
                mtb.MacroNutrients.UNSATISFACTORY: 1.0,
                mtb.MacroNutrients.MARGINAL: 0.75,
                mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD: 0.25,
                mtb.MacroNutrients.STANDARD_OR_ABOVE: 0.0,
            },
            n_hhs = 194,
            p_control = 0.5, # Household-level probability of control arm
        )
        self.pars = sc.mergedicts(self.pars, pars)
        self.pars.n_hhs = int(np.round(self.pars.n_hhs)) # Must be an integer

        self.hhdat = pd.DataFrame({
            'size': np.arange(1,10),
            'p': np.array([3, 17, 24, 20, 13, 9, 7, 4, 3]) / 100
        })

        macro = mtb.MacroNutrients
        self.macrodat = pd.DataFrame({
            'habit': [ macro.STANDARD_OR_ABOVE, macro.SLIGHTLY_BELOW_STANDARD, macro.MARGINAL, macro.UNSATISFACTORY ],
            # These are the 1942 levels from Appendix Table 3 or 7 of Downes
            'p_control': [21.1, 28.9, 38.9, 11.1],
            'p_vitamin': [29.2, 30.3, 28.1, 12.4],
        })
        self.macrodat['p_control'] /= self.macrodat['p_control'].sum()
        self.macrodat['p_vitamin'] /= self.macrodat['p_vitamin'].sum()

        self.armdat = pd.DataFrame({
            'arm': [StudyArm.CONTROL, StudyArm.VITAMIN],
            'p': [self.pars.p_control, 1-self.pars.p_control]
        })
        self.armdat['p'] /= self.armdat['p'].sum()

        self.hhs = self.make_hhs()
        self.n_agents = np.sum([hh.n for hh in self.hhs]) # Hopefully about 579 if running all of Harlem
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
             ss.FloatArr('hhid', default=np.nan),
             ss.FloatArr('arm', default=np.nan),
        ]
        pop = ss.People(n_agents = self.n_agents, extra_states=extra_states)
        return pop

    def net(self):
        net = mtb.HarlemNet(self.hhs)
        return net

    def set_states(self, sim):
        pop = sim.people
        nut = sim.diseases['malnutrition']
        for hh in self.hhs:
            p_deficient = self.pars.p_microdeficient_given_macro[hh.macro]
            for uid in hh.uids:
                pop.hhid[ss.uids(uid)] = hh.hhid
                pop.arm[ss.uids(uid)] = hh.arm
                nut.macro_state[ss.uids(uid)] = hh.macro            # We are assuming that the macro state is the same for all members of the household
                nut.micro_state[ss.uids(uid)] = mtb.MicroNutrients.DEFICIENT if np.random.rand() < p_deficient else mtb.MicroNutrients.NORMAL

        # Set relative LS progression after changing macro and micro states
        c = sim.connectors['tb_nutrition_connector']
        tb = sim.diseases['tb']
        tb.rel_LS_prog[sim.people.uid] = c.pars.rel_LS_prog_func(nut.macro_state, nut.micro_state)
        return

    def choose_seed_infections(self):
        seed_uids = []
        for hh in self.hhs:
            seed_uid = np.random.choice(hh.uids)
            seed_uids.append(seed_uid)
        return ss.uids(seed_uids)

class HarlemPregnancy(ss.Pregnancy):

    def make_embryos(self, conceive_uids):
        newborn_uids = super().make_embryos(conceive_uids)

        if len(newborn_uids) == 0:
            return newborn_uids

        people = self.sim.people
        nut = self.sim.diseases['malnutrition']
        people.hhid[newborn_uids] = people.hhid[conceive_uids]
        people.arm[newborn_uids] = people.arm[conceive_uids]
        # Assume baby has the same micro/macro state as mom
        nut.micro_state[newborn_uids] = nut.micro_state[conceive_uids]
        nut.macro_state[newborn_uids] = nut.macro_state[conceive_uids]

        hn = self.sim.networks['harlemnet']

        p1s = []
        p2s = []
        for newborn_uid, mother_uid in zip(newborn_uids, conceive_uids):
            for contact in hn.find_contacts(mother_uid):
                p1s.append(contact)
                p2s.append(newborn_uid)

        hn.edges.p1 = ss.uids(np.concatenate([hn.edges.p1, p1s]))
        hn.edges.p2 = ss.uids(np.concatenate([hn.edges.p2, p2s]))
        # Beta is zero while prenatal
        hn.edges.beta = np.concatenate([hn.edges.beta, np.zeros_like(p1s)])#.astype(ss.dtypes.float)

        return newborn_uids
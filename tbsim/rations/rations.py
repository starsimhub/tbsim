import starsim as ss
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['Rations']

'''
PLAN:
1. CREATE 2 arms: CONTROL and VITAMIN
2. Create 2800 Agents (indexes 0-2799)
3. households with sizes from 1 to 4
4. Assign each household to an arm 50/50
5. Assign each household a macro state based on the arm
  2a. set prognoses for each agent on one of the Active TB states from day 1? (maybe warm up period? do we call it warm up period? :D )
  
to be continued... 

'''














class Rations():
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
            'arm': [mtb.StudyArm.CONTROL, mtb.StudyArm.VITAMIN],
            'p': [self.pars.p_control, 1-self.pars.p_control]
        })
        self.armdat['p'] /= self.armdat['p'].sum()

        self.hhs = self.make_hhs()
        self.n_agents = np.sum([hh.n for hh in self.hhs]) # Hopefully about 579 if running all of Rations
        return

    def make_hhs(self):
        hh_sizes = np.random.choice(a=self.hhdat['size'].values, p=self.hhdat['p'].values, size=self.pars.n_hhs)
        arm = np.random.choice(a=self.armdat['arm'].values, p=self.armdat['p'].values, size=self.pars.n_hhs)

        idx = 0
        hhs = []
        for hhid, (size, arm) in enumerate(zip(hh_sizes, arm)):
            uids = np.arange(idx, idx+size)
            if arm == mtb.StudyArm.CONTROL:
                p = self.macrodat['p_control'].values
            else:
                p = self.macrodat['p_vitamin'].values
            macro = np.random.choice(a=self.macrodat['habit'].values, p=p)
            hh = mtb.HouseHold(hhid, uids, mtb.MacroNutrients(macro), mtb.StudyArm(arm))
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
        net = mtb.HouseHoldNet(self.hhs)
        
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
        tb.rel_LF_prog[sim.people.uid] = c.pars.rel_LF_prog_func(nut.macro_state, nut.micro_state)

        return

    def choose_seed_infections(self):
        seed_uids = []
        for hh in self.hhs:
            seed_uid = np.random.choice(hh.uids)
            seed_uids.append(seed_uid)
        return ss.uids(seed_uids)


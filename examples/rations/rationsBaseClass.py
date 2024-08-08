import starsim as ss
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['Rations']

'''
PLAN:
1. Create 2800 Agents (indexes 0-2799)  -  DONE
1.a. Give a TB Active status (Active smear posivite, Active Extra Pulmonary) - DONE
2. Allocate 1400 to CONTROL and and 1400 VITAMIN groups done
3. Generate households with sizes from 1 to 6 done
4. Assign each household to an arm 50/50 - done
5. Assign each household a macro state based on the arm -done
  2a. set prognoses for each agent on one of the Active TB states from day 1? (maybe warm up period? do we call it warm up period? :D )
  

TODO:
- add the bmi state to the Malnutrition class

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
            n_hhs = 2800, # Number of households (2800 participants plus their families)
            
            # Household-level probability of control arm - 50% is the default
            p_control = 0.5, # Household-level probability of control arm
            hhdat = pd.DataFrame({
                        'size': np.arange(2,6),
                        'p': np.array([10, 30, 40, 20]) / 100
                        }),
        )
        self.pars = sc.mergedicts(self.pars, pars)
        self.pars.n_hhs = int(np.round(self.pars.n_hhs)) # Must be an integer


        self.hhdat = self.pars.hhdat
        
        macro = mtb.MacroNutrients
        self.macrodat = pd.DataFrame({
            'options': [ macro.STANDARD_OR_ABOVE, macro.SLIGHTLY_BELOW_STANDARD, macro.MARGINAL, macro.UNSATISFACTORY ],
            'p_control': [35, 15, 30, 20],  # [21.1, 28.9, 38.9, 11.1],
            'p_vitamin': [34, 16, 21, 29],  #[29.2, 30.3, 28.1, 12.4],
        })
        self.macrodat['p_control'] /= self.macrodat['p_control'].sum()
        self.macrodat['p_vitamin'] /= self.macrodat['p_vitamin'].sum()

        ebmi = mtb.eBmiStatus
        self.bmidat = pd.DataFrame({
            'options': [ebmi.NORMAL_WEIGHT, ebmi.MILD_THINNESS,    ebmi.MODERATE_THINNESS, ebmi.SEVERE_THINNESS ],
            'p_control': [35, 15, 30, 20],      # Guess values, not from data - must enter values from 
            'p_vitamin': [34, 16, 21, 29],      # Guess values, not from data - must enter values
        })
        self.bmidat['p_control'] /= self.bmidat['p_control'].sum()
        self.bmidat['p_vitamin'] /= self.bmidat['p_vitamin'].sum()
        # Arm data - 
        self.armdat = pd.DataFrame({
            'arm': [mtb.StudyArm.CONTROL, mtb.StudyArm.VITAMIN],
            'p': [self.pars.p_control, 1-self.pars.p_control]
        })
        
                
        self.armdat['p'] /= self.armdat['p'].sum()

        # ---------- Create households ----------
        self.hhs = self.make_hhs()
        
        # ---------- Create people ---------------
        self.n_agents = np.sum([hh.n for hh in self.hhs]) # Hopefully around 10000 if running all of Rations
        
        # self.hhs[:0].macro_metric = mtb.MacroNutrients.STANDARD_OR_ABOVE
        # self.hhs[29].uids[0]   ---> np.int64(115)

        self.hhsIndexes = []
        for hh in self.hhs:
            self.hhsIndexes.append(hh.uids[0])
        return

    def make_hhs(self):
        # Create households
        hh_sizes = np.random.choice(a=self.hhdat['size'].values, p=self.hhdat['p'].values, size=self.pars.n_hhs)
        
        # Assign armstbsim/rations/rations.py
        arm = np.random.choice(a=self.armdat['arm'].values, p=self.armdat['p'].values, size=self.pars.n_hhs)

        idx = 0
        hhs = []
        # FOR EACH CREATED HOUSEHOLD
        for hhid, (size, arm) in enumerate(zip(hh_sizes, arm)):
            uids = np.arange(idx, idx+size)
            
            armstr = 'p_control' if arm == mtb.StudyArm.CONTROL else 'p_vitamin'

        
            pMa = self.macrodat[armstr].values
            pBmi = self.bmidat[armstr].values
            
            macro = np.random.choice(a=self.macrodat['options'].values, p=pMa)  # Randomly Choose the macro state option
            bmi = np.random.choice(a=self.bmidat['options'].values, p=pBmi)     # Randomly Choose the bmi state option
            
            # Create the household
            hh = mtb.GenericHouseHold(hhid, uids, mtb.eMacroNutrients(macro),  mtb.eBmiStatus(bmi), mtb.eStudyArm(arm))
            
            # Append the household to the list
            hhs.append(hh)
            idx += size

        return hhs

    def people(self):
        extra_states = [
             ss.FloatArr('hhid', default=np.nan),
             ss.FloatArr('arm', default=np.nan),
             ss.FloatArr('macro_state', default=np.nan),
             ss.FloatArr('micro_state', default=np.nan),
             ss.FloatArr('bmi_state', default=np.nan),
        ]
        n_agents = self.n_agents
        print(f"Current total of people: {n_agents}")
        pop = ss.People(n_agents = self.n_agents, extra_states=extra_states)
        return pop

    def net(self):
        # Create the network of households
        net = mtb.HouseHoldNet(self.hhs)
        return net

    def set_states(self, sim, target_group):
        # Set household states
        pop = sim.people
        nut = sim.diseases['malnutrition']
        
        for hh in self.hhs:
            # p_bmi_deficient = self.pars.bmi_status_modifier[hh.bmi_metric]
            p_micro_deficient = self.pars.p_microdeficient_given_macro[hh.macro_metric]
            
            for uid in hh.uids:
                pop.hhid[ss.uids(uid)] = hh.hhid
                pop.arm[ss.uids(uid)] = hh.arm
                nut.macro_state[ss.uids(uid)] = hh.macro_metric            # We are assuming that the macro state is the same for all members of the household
                nut.bmi_state[ss.uids(uid)] = hh.bmi_metric

                nut.micro_state[ss.uids(uid)] = mtb.MicroNutrients.DEFICIENT if np.random.rand() < p_micro_deficient else mtb.MicroNutrients.NORMAL
                
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


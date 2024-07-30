import starsim as ss
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['Rations']

'''
PLAN:
1. Create 2800 Agents (indexes 0-2799)
1.a. Give a TB Active status (Active smear posivite, Active Extra Pulmonary)
2. Allocate 1400 to CONTROL and and 1400 VITAMIN groups
3. Generate households with sizes from 1 to 4
4. Assign each household to an arm 50/50
5. Assign each household a macro state based on the arm
  2a. set prognoses for each agent on one of the Active TB states from day 1? (maybe warm up period? do we call it warm up period? :D )
  
to be continued... 

'''

class Rations():
    def __init__(self, pars=None):

        self.pars = sc.objdict(
            bmi_status_modifier = { # Guess values, not from data
                mtb.eBmiStatus.SEVERE_THINNESS: 1.0,
                mtb.eBmiStatus.MODERATE_THINNESS: 0.75,
                mtb.eBmiStatus.MILD_THINNESS: 0.25,
                mtb.eBmiStatus.NORMAL_WEIGHT: 0.0,
                mtb.eBmiStatus.OVERWEIGHT: 0.25,
            },
            n_hhs = 2_800, # Number of households (2800 participants plus their families)
            
            # Household-level probability of control arm - 50% is the default
            p_control = 0.5, # Household-level probability of control arm
        )
        
        self.pars = sc.mergedicts(self.pars, pars)
        self.pars.n_hhs = int(np.round(self.pars.n_hhs)) # Must be an integer

        self.hhdat = pd.DataFrame({
            'size': np.arange(1,5),
            'p': np.array([3, 30, 40, 27]) / 100
        })

        macro = mtb.MacroNutrients
        
        self.macrodat = pd.DataFrame({
            'bmi': [
                mtb.eBmiStatus.SEVERE_THINNESS,
                mtb.eBmiStatus.MODERATE_THINNESS,
                mtb.eBmiStatus.MILD_THINNESS,
                mtb.eBmiStatus.NORMAL_WEIGHT,
                mtb.eBmiStatus.OVERWEIGHT 
                ],
            # 
            'p_control': [21.1, 28.9, 38.9, 11.0, .1],
            'p_vitamin': [29.2, 30.3, 28.1, 12.0, .4],
        })
        self.macrodat['p_control'] /= self.macrodat['p_control'].sum()
        self.macrodat['p_vitamin'] /= self.macrodat['p_vitamin'].sum()

        # Arm data - 
        self.armdat = pd.DataFrame({
            'arm': [mtb.StudyArm.CONTROL, mtb.StudyArm.VITAMIN],
            'p': [self.pars.p_control, 1-self.pars.p_control]
        })
        
        
        self.armdat['p'] /= self.armdat['p'].sum()

        # ---------- Create households ----------
        self.hhs = self.make_hhs(metric='bmi', metric_options=mtb.eBmiStatus)
        
        # ---------- Create people ---------------
        self.n_agents = np.sum([hh.n for hh in self.hhs]) # Hopefully about 579 if running all of Rations
        return

    def make_hhs(self, metric, metric_options):
        # Create households
        hh_sizes = np.random.choice(a=self.hhdat['size'].values, p=self.hhdat['p'].values, size=self.pars.n_hhs)
        
        # Assign arms
        arm = np.random.choice(a=self.armdat['arm'].values, p=self.armdat['p'].values, size=self.pars.n_hhs)

        idx = 0
        hhs = []
        for hhid, (size, arm) in enumerate(zip(hh_sizes, arm)):
            uids = np.arange(idx, idx+size)
            
            
            if arm == mtb.StudyArm.CONTROL:
                p = self.macrodat['p_control'].values
            else:
                p = self.macrodat['p_vitamin'].values
                
            # Randomly choose a metric value    
            metric_choice = np.random.choice(a=self.macrodat[metric].values, p=p)
            
            # Create the household
            hh = mtb.GenericHouseHold(hhid, uids, hh_macro=1, hh_micro=1, hh_bmi=metric_options(metric_choice), study_arm=mtb.StudyArm(arm))
            
            # Append the household to the list
            hhs.append(hh)
            idx += size

        return hhs

    def people(self, n_agents=None):
        extra_states = [
             ss.FloatArr('hhid', default=np.nan),
             ss.FloatArr('arm', default=np.nan),
             ss.FloatArr('macro_state', default=np.nan),
             ss.FloatArr('micro_state', default=np.nan),
        ]
        if n_agents is None:    # If n_agents is not provided, use the number of agents in the households
            n_agents = self.n_agents
            
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
            p_deficient = self.pars.bmi_status_modifier[hh.bmi_metric]
            for uid in hh.uids:
                pop.hhid[ss.uids(uid)] = hh.hhid
                pop.arm[ss.uids(uid)] = hh.arm
                nut.macro_state[ss.uids(uid)] = hh.macro_metric            # We are assuming that the macro state is the same for all members of the household
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


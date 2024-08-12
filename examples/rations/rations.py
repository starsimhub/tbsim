#### --------------BASE CLASS FOR RATIONS SIMULATION -------------------
import starsim as starsim
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['RATIONS']



class RATIONS():
    """
    A class to represent the Rations simulation.
    
    Attributes:
    -----------
    
    pars: dict
        - p_microdeficient_given_macro: dict
            Parameters for the simulation.
        - n_hhs: int
            Number of households.
        - p_control: float
            Percentage of households in the control arm.
        - hhdat: pd.DataFrame
            DataFrame containing household data.
            size:       Represents the size of something (e.g., household size).
            p:          Represents the probability or proportion associated with each size.
            p_control:  Percentage of households in the control arm
    macrodat : pd.DataFrame
        Food Habits - DataFrame containing macro nutrient data.
    bmidat : pd.DataFrame
        DataFrame containing BMI status data.
    armdat : pd.DataFrame
        DataFrame containing study arm data.
    hhs : list
        List of households.
    n_agents : int
        Total number of agents (people).
    hhsIndexes : list
        List of household indexes.
    """
    def __init__(self, pars=None):
        #____________________________________________________
        #           Default parameters and Initial data
        self.pars = sc.objdict(
            p_microdeficient_given_macro = { # Guess values, not from data
                mtb.MacroNutrients.UNSATISFACTORY: 1.0,
                mtb.MacroNutrients.MARGINAL: 0.75,
                mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD: 0.25,
                mtb.MacroNutrients.STANDARD_OR_ABOVE: 0.0},
            num_HHs = 2800, # Number of households (2800 participants plus their families)
            p_control = 0.5, 
            hhdat = pd.DataFrame({
                        'size': np.arange(2,6),
                        'p': np.array([10, 30, 40, 20]) / 100
                        }))
        
        
        self.pars = sc.mergedicts(self.pars, pars)
        self.pars.num_HHs = int(np.round(self.pars.num_HHs)) # Making sure it is an integer
        self.hhdat = self.pars.hhdat
        # ____________________________________________________
        #      Food habits among participants: 
        
        eMacro = mtb.eMacroNutrients
        self.macrodat = pd.DataFrame({
            'habit': [eMacro.STANDARD_OR_ABOVE, 
                      eMacro.SLIGHTLY_BELOW_STANDARD, 
                      eMacro.MARGINAL, 
                      eMacro.UNSATISFACTORY ],
            'p_control': [35, 15, 30, 20],      # TODO: must enter values from RATIONS data
            'p_vitamin': [34, 16, 21, 29],      # TODO: must enter values from RATIONS data
        })
        self.macrodat['p_control'] /= self.macrodat['p_control'].sum()
        self.macrodat['p_vitamin'] /= self.macrodat['p_vitamin'].sum()

        # ____________________________________________________
        #       BMI status among population:
        eBmi = mtb.eBmiStatus
        self.bmidat = pd.DataFrame({
            'status': [eBmi.NORMAL_WEIGHT, 
                       eBmi.MILD_THINNESS, 
                       eBmi.MODERATE_THINNESS, 
                       eBmi.SEVERE_THINNESS ],
            'p_control': [35, 15, 30, 20],      # TODO: must enter values from RATIONS data
            'p_vitamin': [34, 16, 21, 29],      # TODO: must enter values from RATIONS data
        })
        self.bmidat['p_control'] /= self.bmidat['p_control'].sum()
        self.bmidat['p_vitamin'] /= self.bmidat['p_vitamin'].sum()
        
        # ____________________________________________________
        #        Arm data:
        eArm = mtb.eStudyArm
        self.armdat = pd.DataFrame({
            'arm': [eArm.CONTROL, eArm.VITAMIN],
            'p':   [self.pars.p_control, 1-self.pars.p_control]     # By default should be 50 : 50
        })               
        self.armdat['p'] /= self.armdat['p'].sum()

        # ____________________________________________________
        #       Households 
        self.hhs = self.make_hhs()

        # Calculate the total number of agents
        self.n_agents = np.sum([hh.n for hh in self.hhs]) # Hopefully around 10000 if running all of Rations
        
        # self.hhs[29].uids[0]   ---> np.int64(115) # DEBUGGING SAMPLE CODE - enter this in the debug console to see the value of the first person in the 30th household
        self.hhsIndexes = []
        for hh in self.hhs:
            self.hhsIndexes.append(hh.uids[0])
            
            
        return
    
    # ____________________________________________________
    #
    #         create people
    # ____________________________________________________
    def people(self):
        individual_properties = [
             starsim.FloatArr('hhid', default=np.nan),
             starsim.FloatArr('arm', default=np.nan),
             starsim.FloatArr('macro_state', default=np.nan),
             starsim.FloatArr('micro_state', default=np.nan),
             starsim.FloatArr('bmi_state', default=np.nan),
             starsim.FloatArr('is_index', default=np.nan),
        ]
        n_agents = self.n_agents
        print(f"Current total of people: {n_agents}")
        pop = starsim.People(n_agents = self.n_agents, extra_states=individual_properties)
        return pop

    
    # ____________________________________________________
    #
    #                   make households
    # ____________________________________________________
    
    def make_hhs(self):
        # ARRAY:  Determine the size of each household based on the distribution of household sizes
        hh_sizes = np.random.choice(a=self.hhdat['size'].values, 
                                    p=self.hhdat['p'].values, 
                                    size=self.pars.num_HHs)
        
        # Assign arm to each household based on the distribution of arms
        arm = np.random.choice(a=self.armdat['arm'].values, 
                               p=self.armdat['p'].values, 
                               size=self.pars.num_HHs)

        idx = 0
        hhs = []
        
        # ---------- FOR EACH CREATED HOUSEHOLD ------
        
        for hhid, (size, arm) in enumerate(zip(hh_sizes, arm)):
            uids = np.arange(idx, idx+size)
            
            armstr = 'p_control' if arm == mtb.StudyArm.CONTROL else 'p_vitamin'
        
            pMa = self.macrodat[armstr].values
            pBmi = self.bmidat[armstr].values
            
            macro = np.random.choice(a=self.macrodat['habit'].values, p=pMa)  # Randomly Choose the macro state option
            bmi = np.random.choice(a=self.bmidat['status'].values, p=pBmi)     # Randomly Choose the bmi state option
            
            # Create the household
            hh = mtb.GenericHouseHold(hhid, uids, mtb.eMacroNutrients(macro),  mtb.eBmiStatus(bmi), mtb.eStudyArm(arm))
            
            # Append the household to the list
            hhs.append(hh)
            idx += size

        return hhs
    
    # ____________________________________________________
    #
    #         create Household Network
    # ____________________________________________________
    def net(self):
        # Create the network of households
        net = mtb.HouseHoldNet(self.hhs)
        return net

    # ____________________________________________________
    #
    #         set initial household members states
    # ____________________________________________________
    def set_states(self, sim, target_group):
        # Set household states
        pop = sim.people
        nut = sim.diseases['malnutrition']
        
        # Set settings for the entire group of participants (all)  
        for hh in self.hhs:
            # p_bmi_deficient = self.pars.bmi_status_modifier[hh.bmi_metric]
            p_micro_deficient = self.pars.p_microdeficient_given_macro[hh.macro_metric]
            
            for uid in hh.uids:
                pop.hhid[starsim.uids(uid)] = hh.hhid
                pop.arm[starsim.uids(uid)] = hh.arm       #Possible BUG:  one group could end up with more people than the other, because of the way how the arm is assigned

                nut.macro_state[starsim.uids(uid)] = hh.macro_metric     # We are assuming that the macro state is the same for all members of the household
                nut.bmi_state[starsim.uids(uid)] = hh.bmi_metric         # We are assuming that the bmi state is the same for all members of the household
                nut.micro_state[starsim.uids(uid)] = mtb.MicroNutrients.DEFICIENT if np.random.rand() < p_micro_deficient else mtb.MicroNutrients.NORMAL
                
        # Set relative LS progression after changing macro and micro states
        c = sim.connectors['tb_nutrition_connector']
        tb = sim.diseases['tb']
        tb.rel_LS_prog[sim.people.uid] = c.pars.rel_LS_prog_func(nut.macro_state, nut.micro_state)     # this part of the code is also called as part of the "TB_Nutrition_Connector" class  (within the connectors.py file)
        tb.rel_LF_prog[sim.people.uid] = c.pars.rel_LF_prog_func(nut.macro_state, nut.micro_state)

        return

    def choose_seed_infections(self):
        seed_uids = []
        for hh in self.hhs:
            seed_uid = np.random.choice(hh.uids)
            seed_uids.append(seed_uid)
        return starsim.uids(seed_uids)


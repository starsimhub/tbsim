#### -------------- HELBER CLASS FOR CREATING RATIONS SIMULATIONS -------------------
import starsim as ss
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd

__all__ = ['RATIONS']


class RationsHouseholds(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        self.add_states(
            ss.FloatArr('hhid'),
            ss.BoolArr('is_seed'),
            ss.BoolArr('intervention_arm')
        )

        self.default_pars(
            p_sm_pos = ss.bernoulli(0.72),
            dur_active_to_dx = ss.weibull(c=2, scale=3 * 7/365),
            dur_dx_to_first_visit = ss.uniform(low=0, high=365/12),
        )
        self.update_pars(pars, **kwargs)

        return

    def initialize(self, sim):
        super().initialize(sim)
        # Pick one adult to be the source
        # At time of diagnosis, must be 18+ with microbiologically confirmed pulminary TB, so probably an adult?

        # Latent phase doesn't matter, not transmissible during that period
        # So fast forward to end of latent, beginning of active PRE-SYMPTOMATIC stage
        inds18plus = [u for u in self.uids if u.age >= 18]
        self.seed_uid = ss.choice(inds18plus)
        tb = sim.diseases['tb']
        tb.set_prognoses(self.seed_uid)

        # After set_prognoses, seed_uids will be in latent slow or fast
        # Change to ACTIVE_PRESYMP and set time of activation to current time step
        tb.ti_presymp[self.seed_uid] = sim.ti # +1?

        # All RATIONS index cases are pulmonary. Using TBsim defaults, assuming 72% are SmPos and the rest are SmNeg
        random_distribution = np.random.choice([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG], p=[0.72, 0.28], size=len(self.seed_uid))
        tb.active_tb_state[self.seed_uid] = random_distribution

        # AFTER pre-symptomatic period, the individual needs to transition to
        # one of the pulmonary stages (Sm+ or Sm-), we don't want
        # extra-pulmonary

        # The individual should be shedding active pulmonary TB for some period
        # of time before seeking care, distribution from input

        # On seeking care, the individual will be diagnosed and start treatment.
        # Only at this point does RATIONS learn of this individual/household.

        # At some additional delay, the household receives its first visit +
        # food basket for either just the index (control) or index + HH members
        # (intervention).


    def update():
        super().update()

        # Check for new active --> treatment

        # If frequency is right, do a "visit"
        return



class RATIONS():
    """
    A class to help create a simulation of the RATIONS trial.
    
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
    hhsHeadUID : list
        List of household indexes.
    """

    def __init__(self, pars=None):
        #____________________________________________________
        #           Default parameters and initial data

        # https://censusindia.gov.in/nada/index.php/catalog/712
        weights_7_10 = np.array([0.50, 0.30, 0.15, 0.05])   # Manual disaggregation
        weights_11_14 = np.array([0.25, 0.25, 0.25, 0.25])

        self.pars = sc.objdict(
            p_microdeficient_given_macro = { # Guess values, not from data
                mtb.MacroNutrients.UNSATISFACTORY: 1.0,
                mtb.MacroNutrients.MARGINAL: 0.75,
                mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD: 0.25,
                mtb.MacroNutrients.STANDARD_OR_ABOVE: 0.0},
            num_HHs = 2800, # Number of households (2800 households plus their families)
            p_control = 0.5, # Probability of a household being in the control arm of the trial

            hhdat = pd.DataFrame({
                'size': np.arange(1,16),
                'n': np.concatenate([
                        np.array([186832, 489076, 701325, 1145585, 1221054, 951857]),
                        weights_7_10 * 1_334_629,
                        weights_11_14 * 160_132,
                        np.array([46657])
                    ]),
            })
        )

        if False:
            import scipy.stats as sps
            #data = sps.rv_histogram((self.pars.hhdat['n'], np.arange(1, 17))).rvs(size=1_000_000)
            #data = sps.rv_histogram(([186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657], [1,2,3,4,5,6,7, 11, 15, 20])).rvs(size=1_000_000)
            data = sps.rv_histogram(([186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657], np.array([1,2,3,4,5,6,7, 11, 15, 16])-0.5), density=False).rvs(size=1_000_000)

            #result = sps.fit(sps.poisson, data, bounds={'mu': (0, 25)})
            #result = sps.fit(sps.lognorm, data, bounds={'s': (1, 15), 'loc':(0, 10)})
            result = sps.fit(sps.nbinom, data, bounds={'n': (1, 25), 'p': (0,1), 'loc': (0, 2)}) # --> n=19, p=0.8

            import matplotlib.pyplot as plt  # matplotlib must be installed to plot
            result.plot()
            plt.show()

        self.pars = sc.mergedicts(self.pars, pars)

        self.pars.hhdat['p'] = self.pars.hhdat['n'] / self.pars.hhdat['n'].sum() # Normalize
        self.pars.num_HHs = int(np.round(self.pars.num_HHs)) # Making sure it is an integer


        # ____________________________________________________
        #      Food habits among participants: 
        eMacro = mtb.eMacroNutrients
        self.macrodat = pd.DataFrame({
            'habit': [eMacro.STANDARD_OR_ABOVE, 
                      eMacro.SLIGHTLY_BELOW_STANDARD, 
                      eMacro.MARGINAL, 
                      eMacro.UNSATISFACTORY ],
            'ctrl': [35, 15, 30, 20],      # TODO: must enter values from RATIONS data
            'intv': [34, 16, 21, 29],      # TODO: must enter values from RATIONS data
        })
        self.macrodat['ctrl'] /= self.macrodat['ctrl'].sum()
        self.macrodat['intv'] /= self.macrodat['intv'].sum()

        # ____________________________________________________
        #       BMI status among population:
        eBmi = mtb.eBmiStatus
        self.bmidat = pd.DataFrame({
            'status': [eBmi.NORMAL_WEIGHT, 
                       eBmi.MILD_THINNESS, 
                       eBmi.MODERATE_THINNESS, 
                       eBmi.SEVERE_THINNESS ],
            'ctrl': [35, 15, 30, 20],      # TODO: must enter values from RATIONS data
            'intv': [34, 16, 21, 29],      # TODO: must enter values from RATIONS data
        })
        self.bmidat['ctrl'] /= self.bmidat['ctrl'].sum()
        self.bmidat['intv'] /= self.bmidat['intv'].sum()
        
        # ____________________________________________________
        #        Arm data:
        eArm = mtb.eStudyArm
        self.armdat = pd.DataFrame({
            'arm': [eArm.CONTROL, eArm.VITAMIN],
            'p':   [self.pars.p_control, 1-self.pars.p_control]     # By default should be 50 : 50
        })               

        # ____________________________________________________
        #       Households 
        self.hhs = self.make_hhs()

        # Calculate the total number of agents
        self.n_agents = np.sum([hh.n for hh in self.hhs]) # Hopefully around 10000 if running all of Rations
        
        return

    # ____________________________________________________
    #
    #         create people
    # ____________________________________________________
    def people(self):
        individual_properties = [
             ss.FloatArr('hhid', default=np.nan),
             ss.FloatArr('arm', default=np.nan),
             ss.FloatArr('macro_state', default=np.nan),
             ss.FloatArr('micro_state', default=np.nan),
             ss.FloatArr('bmi_state', default=np.nan),
             ss.FloatArr('is_index', default=np.nan),
        ]
        n_agents = self.n_agents
        print(f"Current total of people: {n_agents}")
        pop = ss.People(n_agents = self.n_agents, extra_states=individual_properties)
        return pop

    # ____________________________________________________
    #
    #                   make households
    # ____________________________________________________
    
    def make_hhs(self):
        # ARRAY:  Determine the size of each household based on the distribution of household sizes
        hhdat = self.pars.hhdat
        hh_sizes = np.random.choice(a=hhdat['size'].values, 
                                    p=hhdat['p'].values, 
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
            
            armstr = 'ctrl' if arm == mtb.StudyArm.CONTROL else 'intv'
        
            pMa = self.macrodat[armstr].values
            pBmi = self.bmidat[armstr].values
            
            macro = np.random.choice(a=self.macrodat['habit'].values, p=pMa)  # Randomly Choose the macro state option
            bmi = np.random.choice(a=self.bmidat['status'].values, p=pBmi)     # Randomly Choose the bmi state option
            
            # Create the household
            hh = mtb.HouseholdUnit(hhid, uids, mtb.eMacroNutrients(macro),  mtb.eBmiStatus(bmi), mtb.eStudyArm(arm))
            
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
        net = mtb.HouseholdNet(self.hhs)
        return net

    # ____________________________________________________
    #
    #         set initial household members states
    # ____________________________________________________
    def set_states(self, sim):
        # Set household states
        pop = sim.people
        nut = sim.diseases['malnutrition']
        
        # Set settings for the entire group of participants (all)  
        for hh in self.hhs:
            # p_bmi_deficient = self.pars.bmi_status_modifier[hh.bmi_metric]
            p_micro_deficient = self.pars.p_microdeficient_given_macro[hh.macro_metric]
            
            for uid in hh.uids:
                pop.hhid[uid] = hh.hhid
                pop.arm[uid] = hh.arm

                nut.macro_state[uid] = hh.macro_metric     # We are assuming that the macro state is the same for all members of the household
                nut.bmi_state[uid] = hh.bmi_metric         # We are assuming that the bmi state is the same for all members of the household
                nut.micro_state[uid] = mtb.MicroNutrients.DEFICIENT if np.random.rand() < p_micro_deficient else mtb.MicroNutrients.NORMAL
                
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
        return ss.uids(seed_uids)


#### -------------- HELBER CLASS FOR CREATING RATIONS SIMULATIONS -------------------
import starsim as ss
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd
from enum import IntEnum, auto

__all__ = ['RATIONSTrial', 'RATIONS']

# mtb.StudyArm?
class Arm(IntEnum):
    CONTROL = 0
    INTERVENTION = 1


class RATIONSTrial(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        self.add_states(
            ss.FloatArr('hhid'),
            ss.BoolArr('is_seed'),
            ss.BoolArr('arm')
        )

        self.default_pars(
            n_hhs = 2_800,
            p_intv = ss.bernoulli(0.5), # 50% randomization
            p_sm_pos = ss.bernoulli(0.72), # SmPos vs SmNeg for active pulmonary TB of index patients
            dur_active_to_dx = ss.weibull(c=2, scale=3 * 7/365),
            dur_dx_to_first_visit = ss.uniform(low=0, high=365/12),
            hhsize_ctrl = ss.histogram(
                values=[186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657][1:],
                bins=np.array([1,2,3,4,5,6,7, 11, 15, 16][1:]),
                density=False),
            hhsize_intv = ss.histogram(
                values=[186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657][1:],
                bins=np.array([1,2,3,4,5,6,7, 11, 15, 16][1:]),
                density=False),
        )
        self.update_pars(pars, **kwargs)

        return

    def init_post(self):
        super().init_post()

        ppl = self.sim.people
        tb = self.sim.diseases['tb']

        # Pick one adult to be the source. At time of diagnosis, must be 18+
        # with microbiologically confirmed pulminary TB
        over18 = ppl.age>=18
        seed_uids = ss.uids(np.random.choice(a=ppl.uid, p=over18/np.count_nonzero(over18), size=self.pars.n_hhs, replace=False))
        self.is_seed[seed_uids] = True
        non_seeds = np.setdiff1d(ppl.uid, seed_uids)

        arm = self.pars.p_intv.rvs(seed_uids)
        self.arm[seed_uids] = arm

        hhsize = np.zeros(self.pars.n_hhs, dtype=int)
        hhsize[arm == Arm.CONTROL]      = self.pars.hhsize_ctrl(seed_uids[arm == Arm.CONTROL])
        hhsize[arm == Arm.INTERVENTION] = self.pars.hhsize_intv(seed_uids[arm == Arm.INTERVENTION])

        # Map people to households
        idx = 0
        hhs = []
        for hhid, (seed_uid, size, arm) in enumerate(zip(seed_uids, hhsize, arm)):
            nonseed_uids = non_seeds[idx : idx+size-1] # -1 because the seed will be included
            hh_uids = ss.uids(np.concatenate( (np.array([seed_uid]), nonseed_uids) ))
            self.hhid[hh_uids] = hhid

            if False:
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

        # UPDATE NETWORK


        # Initialize the TB infection
        tb.set_prognoses(seed_uids)

        # After set_prognoses, seed_uids will be in latent slow or fast.  Latent
        # phase doesn't matter, not transmissible during that period.  So fast
        # forward to end of latent, beginning of active PRE-SYMPTOMATIC stage.
        # Change to ACTIVE_PRESYMP and set time of activation to current time
        # step.
        tb.ti_presymp[seed_uids] = self.sim.ti # +1?

        # All RATIONS index cases are pulmonary, choose SmPos vs SmNeg
        smpos = self.pars.p_sm_pos(seed_uids)
        tb.active_tb_state[seed_uids[smpos]] = mtb.TBS.ACTIVE_SMPOS
        tb.active_tb_state[seed_uids[~smpos]] = mtb.TBS.ACTIVE_SMNEG

        # The individual should be shedding active pulmonary TB for some period
        # of time before seeking care, distribution from input

        # On seeking care, the individual will be diagnosed and start treatment.
        # Only at this point does RATIONS learn of this individual/household.

        # At some additional delay, the household receives its first visit +
        # food basket for either just the index (control) or index + HH members
        # (intervention).
        return


    def apply(self, sim):
        super().apply()

        tb = self.sim.diseases['tb']
        ti, dt = self.sim.ti, self.sim.dt

        # SEEDS: Pre symp --> Active
        presym_uids = ( ((self.state[self.is_seed] == mtb.TBS.ACTIVE_SMPOS) | (self.state[self.is_seed] == mtb.TBS.ACTIVE_SMNEG)) & (tb.ti_active[self.id_seed] <= ti)).uids
        if len(presym_uids):
            # Newly active, figure out time to care seeking
            dur_untreated = self.pars.dur_active_to_dx(presym_uids)
            self.ti_dx[presym_uids] = ti + int(round(dur_untreated/dt))
        
        # SEEDS: Active --> Diagnosed and beginning immediate treatment
        dx_uids = ( ((self.state[self.is_seed] == mtb.TBS.ACTIVE_SMPOS) | (self.state[self.is_seed] == mtb.TBS.ACTIVE_SMNEG)) & (self.ti_dx[self.id_seed] <= ti)).uids
        if len(dx_uids):
            # Newly diagnosed. Start treatment and determine when the first RATIONS visit will occur.
            tb.ti_treated[dx_uids] = ti # TODO

            dur_dx_to_first_visit = self.pars.dur_dx_to_first_visit(dx_uids)
            self.ti_first_visit[presym_uids] = ti + int(round(dur_dx_to_first_visit/dt))

        # If frequency is right, do a "visit"
        # VISIT IS TO A HH, NOT A PERSON
        # WILL NEED UIDS OF ALL HH MEMBERS TO CHECK FOR SYMPTOMS, ETC
        # SET TIMER TO NEXT VISIT, END VISITS AT 6, 12, or 6-12MO


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


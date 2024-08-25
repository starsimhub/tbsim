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
        
        self.add_states( # For individual people
            ss.FloatArr('hhid'),
            ss.BoolArr('intv_arm'),
            ss.FloatArr('ti_dx'), # Only used for seeds, but easier here
            ss.FloatArr('ti_treatment')
        )

        self.default_pars(
            n_hhs = 2_800,
            p_intv = ss.bernoulli(0.5), # 50% randomization
            p_sm_pos = ss.bernoulli(0.72), # SmPos vs SmNeg for active pulmonary TB of index patients
            dur_active_to_dx = ss.weibull(c=2, scale=3 * 7/365),
            dur_dx_to_first_visit = ss.uniform(low=0, high=365/12),
            dur_visit_to_tx = ss.weibull(c=2, scale=3 * 7/365),
            #hhsize_ctrl = ss.histogram(
            #    values=[186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657][1:],
            #    bins=np.array([1,2,3,4,5,6,7, 11, 15, 16][1:]),
            #    density=False),
            #hhsize_intv = ss.histogram(
            #    values=[186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657][1:],
            #    bins=np.array([1,2,3,4,5,6,7, 11, 15, 16][1:]),
            #    density=False),
        )
        self.update_pars(pars, **kwargs)

        # States for seeds/households
        self.seed_uids = np.full(self.pars.n_hhs, fill_value=np.nan)
        self.ti_first_visit = np.full(self.pars.n_hhs, fill_value=np.nan)
        self.ti_visit = np.full(self.pars.n_hhs, fill_value=np.nan)
        self.uids_by_hhid = []
        self.hhs = None

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'new_hhs_enrolled', self.sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'incident_cases_ctrl', self.sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'incident_cases_intv', self.sim.npts, dtype=int)
        return

    def init_post(self):
        super().init_post()

        ppl = self.sim.people
        tb = self.sim.diseases['tb']

        # Pick one adult to be the source. At time of diagnosis, must be 18+
        # with microbiologically confirmed pulminary TB
        over18 = ppl.age>=18
        self.seed_uids = ss.uids(np.random.choice(a=ppl.uid, p=over18/np.count_nonzero(over18), size=self.pars.n_hhs, replace=False))
        non_seeds = ss.uids(np.setdiff1d(ppl.uid, self.seed_uids))

        self.intv_arm[self.seed_uids] = self.pars.p_intv.rvs(self.seed_uids)

        # Map people to households
        self.hhid[self.seed_uids] = np.arange(self.pars.n_hhs)
        self.hhid[non_seeds] = np.random.choice(np.arange(self.pars.n_hhs), len(non_seeds), replace=True)

        # Now that we know how agents map to hhs, we can update some things...
        hhn = self.sim.networks['householdnet']
        for hhid in np.arange(self.pars.n_hhs):
            uids_in_hh = (self.hhid == hhid).uids
            self.uids_by_hhid.append(uids_in_hh)

            # Set the arm for the other HH members
            self.intv_arm[uids_in_hh] = self.intv_arm[self.seed_uids[hhid]]
            
            # Add this household to the network
            hhn.add_hh(uids_in_hh)

        # Initialize the TB infection
        tb.set_prognoses(self.seed_uids)

        # After set_prognoses, seed_uids will be in latent slow or fast.  Latent
        # phase doesn't matter, not transmissible during that period.  So fast
        # forward to end of latent, beginning of active PRE-SYMPTOMATIC stage.
        # Change to ACTIVE_PRESYMP and set time of activation to current time
        # step.
        tb.ti_presymp[self.seed_uids] = self.sim.ti # +1?

        # All RATIONS index cases are pulmonary, choose SmPos vs SmNeg
        smpos = self.pars.p_sm_pos(self.seed_uids)
        tb.active_tb_state[self.seed_uids[smpos]] = mtb.TBS.ACTIVE_SMPOS
        tb.active_tb_state[self.seed_uids[~smpos]] = mtb.TBS.ACTIVE_SMNEG

        # The individual should be shedding active pulmonary TB for some period
        # of time before seeking care, distribution from input

        # On seeking care, the individual will be diagnosed and start treatment.
        # Only at this point does RATIONS learn of this individual/household.

        # At some additional delay, the household receives its first visit +
        # food basket for either just the index (control) or index + HH members
        # (intervention).
        return


    def apply(self, sim):
        super().apply(sim)

        tb = self.sim.diseases['tb']
        ti, dt = self.sim.ti, self.sim.dt

        # SEEDS: Pre symp --> Active (state change has already happend in TB on this timestep)
        active_uids = self.seed_uids[ np.isin(tb.state[self.seed_uids], [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG]) & (tb.ti_active[self.seed_uids] == ti) ]
        if len(active_uids):
            # Newly active, figure out time to care seeking
            dur_untreated = self.pars.dur_active_to_dx(active_uids)
            self.ti_dx[active_uids] = np.ceil(ti + dur_untreated / dt)

        # SEEDS: Active --> Diagnosed and beginning immediate treatment
        dx_uids = self.seed_uids[np.isin(tb.state[self.seed_uids], [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG]) & (self.ti_dx[self.seed_uids] == ti)]
        if len(dx_uids):
            # Newly diagnosed. Start treatment and determine when the first RATIONS visit will occur.
            tb.start_treatment(dx_uids)

            dur_dx_to_first_visit = self.pars.dur_dx_to_first_visit(dx_uids)
            hhids = self.hhid[dx_uids].astype(int)
            self.ti_first_visit[hhids] = np.ceil(ti + dur_dx_to_first_visit / dt)
            self.ti_visit[hhids] = self.ti_first_visit[hhids]

        # If frequency is right, do a "visit"
        #visit_hhids = np.argwhere(self.ti_visit <= ti)
        self.results['new_hhs_enrolled'][ti] += np.count_nonzero(self.ti_first_visit == ti)
        visit_hhids = np.where(self.ti_visit == ti)[0]
        if len(visit_hhids):
            visit_uids = ss.uids(np.concatenate([self.uids_by_hhid[h] for h in visit_hhids]))

            # Check for new active cases
            active = np.isin(tb.state[visit_uids], [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]) # Include extra-pulmonary here?
            not_dx = np.isnan(self.ti_dx[visit_uids]) # TODO: Doesn't allow for reinfection?
            new_active_uids = visit_uids[ active & not_dx]

            if len(new_active_uids):
                # Record cases
                self.results['incident_cases_ctrl'][ti] = np.count_nonzero(~self.intv_arm[new_active_uids] )
                self.results['incident_cases_intv'][ti] = np.count_nonzero( self.intv_arm[new_active_uids] )

                # SET TIME TO TREATMENT
                dur_visit_to_tx = self.pars.dur_visit_to_tx(new_active_uids)
                self.ti_treatment[new_active_uids] = np.ceil(ti + dur_visit_to_tx / dt)

            # PLAN NEXT VISIT
            # The basket was provided to the participants for the duration of
            # treatment—6 months for drug susceptible tuberculosis and 12 months
            # for multidrug-resistant tuberculosis. The intervention was extended
            # if the patient had a BMI of lower than 18·5 kg/m² or any household
            # contact in the intervention group fulfilled the following: an adult
            # household contact with a BMI of lower than 16 kg/m²; children (aged
            # <10 years) with a weight-for-age Z-score of lower than –2SD and
            # adolescents (aged 10–18 years) with BMI-for-age Z-scores of lower
            # than –2SD. This extension was for a period of 12 months or until
            # improvements above these cutoffs, whichever was shorter.
            # --> Connect to nutrition module HERE

            # Set timer to next visit, end visits at 6, 12, OR 6-12mo
            # "monthly for the first year and every 3 months thereafter"
            within_1y = (ti - self.ti_first_visit[visit_hhids]) * dt < 1 # year
            self.ti_visit[visit_hhids[within_1y]] = np.ceil(ti + 1/12 / dt)
            self.ti_visit[visit_hhids[~within_1y]] = np.ceil(ti + 3/12 / dt)

        treatment_uids = (self.ti_treatment == ti).uids
        if len(treatment_uids):
            # Start treatment for those diagnosed during a household visit
            tb.start_treatment(treatment_uids)

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

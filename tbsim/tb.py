import numpy as np
import sciris as sc
from sciris import randround as rr # Since used frequently
import starsim as ss
from starsim.diseases.sir import SIR

__all__ = ['TB']

class TB(SIR):
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        
        """Add TB parameters and states to the TB model"""
        pars = ss.omergeleft(pars,
            init_prev = 0.01,   # Initial prevalence - TODO: Check if there is one
            beta = 0.5,         # Transmission rate  - TODO: Check if there is one
        )

        """
        DISEASE PROGRESSION: 
        Rates can be interpreted as mean time to transition between states
        For example, for TB Fast Progression( tb_LF_to_act_pre_sym), 1 / (6e-3) = 166.67 days  ~ 5.5 months ~ 0.5 years
        """
        # Natural history according with Stewart slides (EMOD TB model):
        pars = ss.omergeleft(pars,
            # tb_susceptible_to_exposed = 0,        # May be used for susceptibles that migrate to a different zone
            # TODO: FOR ALL DISTRIBUTIONS...
            # TODO: Check on lognorm_ex vs lognorm_im in EMOD
            # TODO: Check on stdev from EMOD
            p_latent_fast = 0.1, # Probability of latent fast as opposed to latent slow
            dur_tb_LF_to_act_pre_sym = ss.expon(scale=1/6e-3),  # Latent Fast to Active Pre-Symptomatic     
            dur_tb_LS_to_act_pre_sym = ss.expon(mean=3e-5),     # Latent Slow to Active Pre-Symptomatic
            
            p_ExPTB = 0.1,
            p_smear_positive = 0.65 / (0.65+0.25), # Amongst those without extrapulminary TB

            #dur_tb_active_to_LS = ss.lognorm_ex(mean=0, stdev=0.1),               # Smear Positive to Recovered
            #dur_tb_active_to_recovered = ss.lognorm_ex(mean=2.4e-4, stdev=0.1),   # Smear Positive to Recovered
            
            #dur_tb_SP_to_dead = ss.lognorm_ex(mean=4.5e-4,  stdev=0.1),           # Smear Positive Pulmonary TB to Dead
            #dur_tb_SN_to_dead = 0.3,                # Smear Negative Pulmonary TB to Dead
            #dur_tb_ExPTB_to_dead = 0.15,            # Extra-Pulmonary TB to Dead

            # Additional parameters:
            #tb_latent_cure = 0.0,               
            #tb_active_cure = 2.4e-4,            
            #tb_presymptomatic_cure_rate = 0.0,  
            #tb_presymptomatic_rate = 3e-2,      
            #tb_inactivation = 0.0,
            #tb_active_period_distribution = ss.random, 
        )
        
        '''
        par_dists = ss.omergeleft(par_dists,
            ## Durations of states
            dur_tb_SP_to_dead = 4.5e-4,
            dur_tb_SN_to_dead = 0.3,
            dur_tb_ExPTB_to_dead = 0.15,
            )
        '''
        
        """
        INFECTIOUSNESS:
        On a scale where 0 represents no infectiousness and 1 represents the highest level of infectiousness typically observed in TB cases
        """
        
        # Infectiousness of each state < MOCK VALUES, actual ones to be determined. Also if they will be
        # relative or absolute values. Arrays are used to represent the range of values.
        pars = ss.omergeleft(pars,
            tb_latent_slow_infectiousness = 0.0,
            tb_latent_fast_infectiousness = 0.0,
            tb_active_presymptomatic_infectiousness = 0.625,  #[0.5,0.75],
            tb_smear_positive_infectiousness = 1.0,
            tb_smear_negative_infectiousness =  0.3625,  # [0.25, 0.50],
            tb_extra_pulmonary_infectiousness = .05,  #[0.0, 0.1],
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.add_states(
            # Initialize states specific to TB:
            ## Susceptible                              # Existent state part of People
            ## Dead                                     # Existent state part of People 
            ss.State('exposed', bool, False),           # Exposed to TB
            ss.State('latent_slow', bool, True),        # Latent TB, slow progression
            ss.State('latent_fast', bool, False),       # Latent TB, fast progression
            ss.State('active_pre_symp', bool, False),   # Active TB, pre-symptomatic
            ss.State('smear_positive', bool, False),    # Active TB, smear positive
            ss.State('smear_negative', bool, True),     # Active TB, smear negative
            ss.State('extra_pulmonary', bool, True),    # Active TB, extra-pulmonary
            ss.State('recovered', bool, False),         

            # Timestep of state changes          
            ss.State('ti_exposed', int, ss.INT_NAN),
            ss.State('ti_latent_slow', int, ss.INT_NAN),
            ss.State('ti_latent_fast', int, ss.INT_NAN),
            ss.State('ti_active_pre_symptomatic', int, ss.INT_NAN),
            ss.State('ti_smear_positive', int, ss.INT_NAN),
            ss.State('ti_smear_negative', int, ss.INT_NAN),
            ss.State('ti_extra_pulmonary', int, ss.INT_NAN),
            )

        # Convert the scalar numbers to a Bernoulli distribution
        self.pars.p_latent_fast = ss.bernoulli(self.pars.p_latent_fast)
        self.pars.init_prev = ss.bernoulli(self.pars.init_prev)
        self.pars.p_ExPTB = ss.bernoulli(self.pars.p_ExPTB)
        self.pars.p_smear_positive = ss.bernoulli(ss.pars.p_smear_positive)

        return
         
    # region Properties
        # TODO: Implement the properties for the model here
        @property
        def infectious(self):
            """
            Property that represents the infectious state of the TB disease.

            This property returns a boolean value indicating whether the disease is in the infectious state.
            The disease is considered infectious if it is either in the 'infected' or 'exposed' state.

            Returns:
                bool: True if the disease is in the 'infected' or 'exposed' state, False otherwise.
            """
            return self.infected | self.exposed
    #endregion

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        # self.results += ss.Result(self.name, 'tb_deaths', sim.npts, dtype=int)
        return  

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """

        
        # eligible_uids = ss.true((sim.people.age >= self.pars['init_prev']['age_range'][0]) & (sim.people.age <= self.pars['init_prev']['age_range'][1]))
        # eligible_uids = ss.true((sim.people.age >= self.pars['min_age']) & (sim.people.age <= self.pars['max_age']))
        # initial_cases = self.pars['seed_infections'].filter(eligible_uids)
        self.set_prognoses(sim, eligible_uids)
        # self.set_prognoses(sim, initial_cases)
        return   
    
    def update_pre(self, sim):
        # Make all the updates from the SIR model 
        n_deaths = super().update_pre(sim)

        # Additional updates: progress exposed -> infected
        infected = ss.true(self.exposed & (self.ti_infected <= sim.year))
        self.exposed[infected] = False
        self.infected[infected] = True

        return n_deaths

    def update_death(self, sim, uids):
        super().update_death(sim, uids)
        self.exposed[uids] = False
        return

    def set_prognoses(self, sim, uids, from_uids=None):
        # Carry out state changes associated with infection
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = sim.year

        # Calculate and schedule future outcomes
        # dur_exp = self.pars['dur_exp'].rvs(uids)
        dur_tb_exposed_to_LF = self.pars['dur_tb_exposed_to_LF'].rvs(uids)
        dur_tb_exposed_to_LS = self.pars['dur_tb_exposed_to_LS'].rvs(uids)

        # dur_smear_positive = self.pars['dur_smear_positive'].rvs(uids)
        # dur_smear_negative = self.pars['dur_smear_negative'].rvs(uids)
        # dur_extra_pulmonary = self.pars['dur_extra_pulmonary'].rvs(uids)
        
        # self.ti_infected[uids] = sim.year + dur_exp
        self.ti_latent_slow[uids] = sim.year + dur_tb_exposed_to_LF
        self.ti_latent_fast[uids] = sim.year + dur_tb_exposed_to_LS

        # self.ti_smear_positive[uids] = sim.year + dur_smear_positive
        # self.ti_smear_negative[uids] = sim.year + dur_smear_negative
        # self.ti_extra_pulmonary[uids] = sim.year + dur_extra_pulmonary
                
        
        
        dur_inf = self.pars['dur_inf'].rvs(uids)
        will_die = self.pars['p_death'].rvs(uids)        
        self.ti_recovered[uids[~will_die]] = sim.year + dur_inf[~will_die]
        self.ti_dead[uids[will_die]] = sim.year + dur_inf[will_die]

        # Update result count of new infections 
        self.results['new_infections'][sim.ti] += len(uids)
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
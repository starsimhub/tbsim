import numpy as np
import sciris as sc
from sciris import randround as rr # Since used frequently
import starsim as ss

__all__ = ['TB']

class TB(ss.Infection):
    """
    Tuberculosis (TB) disease model
    """
    def __init__(self, params=None, param_dists=None, *args, **kwargs):
        # Initialize the parent class        
        super().__init__(params, **kwargs) 

        # States
        self.add_states(
            # Initialize the states for TB
            ss.State('susceptibleTB', bool, True),  # Susceptible to TB
            ss.State('latent_slow', bool, True),  # Latent TB, slow progression
            ss.State('latent_fast', bool, False),  # Latent TB, fast progression
            ss.State('active_pre_symptomatic', bool, False), # Active TB, pre-symptomatic
            ss.State('smear_positive', bool, False), # Active TB, smear positive
            ss.State('smear_negative', bool, True), # Active TB, smear negative
            ss.State('extra_pulmonary', bool, True), # Active TB, extra-pulmonary
            ss.State('dead', bool, False),
            ss.State('recovered', bool, False),)  # Timestep of state changes
        
        # Duration of each state
        self.add_states(
            ss.State('dur_latent_slow', float, ss.INT_NAN),
            ss.State('dur_latent_fast', float, ss.INT_NAN),
            ss.State('dur_active_pre_symptomatic', float, ss.INT_NAN),
            ss.State('dur_smear_positive', float, ss.INT_NAN),
            ss.State('dur_smear_negative', float, ss.INT_NAN),
            ss.State('dur_extra_pulmonary', float, ss.INT_NAN),
            
        )
        self.add_states(            
            # Timestep of state changes
            ss.State('ti_exposed', int, ss.INT_NAN),
            ss.State('ti_latent_slow', int, ss.INT_NAN),
            ss.State('ti_latent_fast', int, ss.INT_NAN),
            ss.State('ti_active_pre_symptomatic', int, ss.INT_NAN),
            ss.State('ti_smear_positive', int, ss.INT_NAN),
            ss.State('ti_smear_negative', int, ss.INT_NAN),
            ss.State('ti_extra_pulmonary', int, ss.INT_NAN),
            )   
        
        
        
        # Parameters
        # placeholder PARAMETERS *** add here  ***
        params = ss.omergeleft(params,{
            'dur_latent_fast': 1,  # Duration from exposure to latent fast
            'dur_latent_slow': 10,  # Duration from latent fast to latent slow
            'prob_progression_fast': 0.1,  # Probability of fast progression
            'prob_progression_slow': 0.01,  # Probability of slow progression to active TB
            'treatment_success_rate': 0.85,  # Treatment success probability
        })
        
        param_networks = ss.omergeleft(params, {
            'nn_zone' : 1,   # Zone of residence, this could be 1-> Rural 2-> 'urban' 
            'nn_quality_of_care' : 1,  # Quality of care, this could be 1-> 'low' 2-> 'high'
        })
        
        # super().__init__(pars=params, par_dists=param_dists, *args, **kwargs)
        
        
        # Options:  # 1. Set default parameters according to: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4395246/ 
        # the ones below were extracted from emod
        default_pars = dict(
            tb_latent_cure_rate = 0.0005479,    
            tb_acute_duration_in_months = 2.9,
            tb_inactivation_rate = 0.00041096,  
            
            tb_presymptomatic_rate = 0.0274,
            tb_presymptomatic_cure_rate = 0.0274,
            
            #  Disease progression            
            tb_fast_progressor_rate = 0.0411,   
            tb_slow_progressor_rate = 2.05E-06, 
            tb_fast_progressor_fraction_adult = 1.0,
            tb_fast_progressor_fraction_child = 1.0,
            
            tb_active_mortality_rate = 0.0,   
            tb_active_cure_rate = 0.0,
            tb_active_period_distribution = ss.distributions.random, # Distribution of active period (lognormal)
            tb_active_presymptomatic_infectivity_multiplier = 0.0274,
            tb_active_period_std_dev = 1.0,
            
            tb_extrapulmonary_fraction_adult = 0.0,
            tb_extrapulmonary_fraction_child = 0.0,
            tb_extrapulmonary_mortality_multiplier = 0.4,   
            
            
            tb_smear_positive_fraction_adult = 1.0,
            tb_smear_positive_fraction_child = 1.0,
            
            tb_smear_negative_infectivity_multiplier = 0.0,
            tb_smear_negative_mortality_multiplier = 0.0,  
        )
        self.pars = ss.omerge(default_pars, self.pars) # NB: regular omerge rather than omergeleft

        return

         
    # region Properties
        # TODO: Implement the properties for the model here
    #endregion
               
    # region -------------- INITIALIZATION METHODS -----------
    
    def initialize(self, sim):
        super().initialize(sim)
        # self.set_immunity(sim)
        return
    
    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """

        eligible_uids = ss.true((sim.people.age >= self.pars['init_prev']['age_range'][0]) & (sim.people.age <= self.pars['init_prev']['age_range'][1]))
        initial_cases = self.pars['seed_infections'].filter(eligible_uids)
        self.set_prognoses(sim, initial_cases)
        return
    
    # endregion    
        

      
    def make_new_cases(self, sim):
        """
        Add new cases of the disease
        """
        
        # TODO: Implement the logic for adding new cases here
        pass

    
    def set_prognoses(self, sim, uids, from_uids):
        """
        Set prognoses for the disease
        """
        # TODO: Implement the logic for setting prognoses here
        return
    
    
    
        # region  --------- UPDATE METHODS -----------
    
    def update_pre(self, sim):
        """ Carry out autonomous updates at the start of the timestep (prior to transmission)  """
        # TODO: Implement the logic for updating the model states here
        print("UPDATING BEFORE", "="*20 )
        pass
    
    def update_results(self, sim):
        """ Update results """
        super().update_results(sim)
        print("UPDATING RESULTS", "="*20 )
        return
    # endregion 
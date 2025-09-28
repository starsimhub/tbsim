"""
Immigration module for Starsim TB simulations.

This module adds new people to the simulation population over time,
representing immigration or in-migration to the study population.
"""

import numpy as np
import starsim as ss
import sciris as sc

__all__ = ['Immigration', 'SimpleImmigration']

# Placeholder for default parameters
_ = None

class Immigration(ss.Demographics):
    """
    Add new people to the simulation population over time.
    
    This module simulates immigration by adding new agents to the population
    at specified rates. New immigrants can have different characteristics
    (age, TB status, etc.) than the existing population.
    
    Parameters:
    -----------
    immigration_rate : float or ss.Rate
        Rate of immigration (people per year)
    age_distribution : dict or array-like
        Age distribution for new immigrants
    tb_status_distribution : dict
        Distribution of TB status for new immigrants
    rel_immigration : float
        Relative immigration rate multiplier
    """
    
    def __init__(self, pars=None, immigration_rate=_, rel_immigration=_, 
                 age_distribution=_, tb_status_distribution=_, **kwargs):
        super().__init__()
        
        self.define_pars(
            immigration_rate=ss.peryear(10),  # 10 immigrants per year by default
            rel_immigration=1.0,  # Relative immigration rate multiplier
            age_distribution=None,  # Will use population age distribution if None
            tb_status_distribution=dict(
                susceptible=0.7,  # 70% susceptible
                latent_fast=0.15,  # 15% latent fast progressors
                latent_slow=0.10,  # 10% latent slow progressors
                active=0.05,  # 5% active TB
            ),
        )
        self.update_pars(pars, **kwargs)
        
        # Define states for immigration tracking
        self.define_states(
            ss.IntArr('hhid', default=-1),  # Household ID for immigrants
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),  # When they immigrated
            ss.FloatArr('age_at_immigration', default=np.nan),  # Age when they immigrated
            ss.StrArr('immigration_tb_status', default=''),  # TB status at immigration
        )
        
        # Initialize tracking variables
        self.n_immigrants = 0
        self.dist = ss.bernoulli(p=0)  # Will be updated each timestep
        
        return
    
    def init_pre(self, sim):
        """Initialize with simulation information."""
        super().init_pre(sim)
        
        # Set up age distribution for immigrants
        if self.pars.age_distribution is None:
            # Use a default age distribution if none provided
            self.pars.age_distribution = {
                0: 0.15,   # 15% children 0-4
                5: 0.20,   # 20% children 5-14
                15: 0.25,  # 25% young adults 15-29
                30: 0.20,  # 20% adults 30-49
                50: 0.15,  # 15% middle-aged 50-64
                65: 0.05,  # 5% elderly 65+
            }
        
        return
    
    def init_results(self):
        """Initialize results tracking."""
        # Initialize results array
        self.results = ss.Result(name='immigration')
        self.results.n_immigrants = ss.Arr(name='n_immigrants', 
                                          summarize_by='sum', 
                                          label='Number of immigrants')
        return
    
    def get_immigration_rate(self):
        """Calculate the current immigration rate."""
        sim = self.sim
        p = self.pars
        
        # Get current immigration rate
        if isinstance(p.immigration_rate, ss.Rate):
            current_rate = p.immigration_rate.value * p.rel_immigration
        else:
            current_rate = p.immigration_rate * p.rel_immigration
        
        # Convert to probability per timestep
        try:
            immigration_prob = ss.prob.array_to_prob(
                np.array([current_rate]), 
                self.t.dt
            )[0]
        except:
            # Fallback calculation
            dt_years = self.t.dt / 365.25  # Convert to years
            immigration_prob = current_rate * dt_years
        
        # Clip to valid probability range and ensure it's not NaN
        immigration_prob = np.clip(immigration_prob, 0, 1)
        if np.isnan(immigration_prob):
            immigration_prob = 0.0
        
        return immigration_prob
    
    def get_immigrant_characteristics(self, n_immigrants):
        """Generate characteristics for new immigrants."""
        if n_immigrants == 0:
            return {}
        
        # Generate ages for immigrants
        if isinstance(self.pars.age_distribution, dict):
            # Use provided age distribution
            age_groups = list(self.pars.age_distribution.keys())
            age_probs = list(self.pars.age_distribution.values())
            
            # Normalize probabilities
            age_probs = np.array(age_probs)
            age_probs = age_probs / age_probs.sum()
            
            # Generate ages within each age group
            ages = []
            for age_group in age_groups:
                # Add some randomness within the age group
                if age_group == 0:
                    ages.extend(np.random.uniform(0, 5, int(n_immigrants * self.pars.age_distribution[age_group])))
                elif age_group == 5:
                    ages.extend(np.random.uniform(5, 15, int(n_immigrants * self.pars.age_distribution[age_group])))
                elif age_group == 15:
                    ages.extend(np.random.uniform(15, 30, int(n_immigrants * self.pars.age_distribution[age_group])))
                elif age_group == 30:
                    ages.extend(np.random.uniform(30, 50, int(n_immigrants * self.pars.age_distribution[age_group])))
                elif age_group == 50:
                    ages.extend(np.random.uniform(50, 65, int(n_immigrants * self.pars.age_distribution[age_group])))
                else:  # 65+
                    ages.extend(np.random.uniform(65, 85, int(n_immigrants * self.pars.age_distribution[age_group])))
            
            # Ensure we have exactly n_immigrants ages
            ages = np.array(ages[:n_immigrants])
            if len(ages) < n_immigrants:
                # Fill remaining with random ages
                remaining = n_immigrants - len(ages)
                ages = np.concatenate([ages, np.random.uniform(0, 85, remaining)])
        else:
            # Fallback to uniform age distribution
            ages = np.random.uniform(0, 85, n_immigrants)
        
        # Generate TB status for immigrants
        tb_statuses = np.random.choice(
            list(self.pars.tb_status_distribution.keys()),
            size=n_immigrants,
            p=list(self.pars.tb_status_distribution.values())
        )
        
        return {
            'ages': ages,
            'tb_statuses': tb_statuses,
        }
    
    def step(self):
        """Add immigrants to the population."""
        # Calculate immigration probability for this timestep
        immigration_prob = self.get_immigration_rate()
        
        # Determine number of immigrants (using Poisson approximation for small probabilities)
        if immigration_prob > 0 and not np.isnan(immigration_prob):
            n_immigrants = np.random.poisson(immigration_prob)
        else:
            n_immigrants = 0
        
        if n_immigrants == 0:
            self.n_immigrants = 0
            return []
        
        # Get characteristics for new immigrants
        characteristics = self.get_immigrant_characteristics(n_immigrants)
        
        # Add new people to the population
        new_uids = self.sim.people.grow(n_immigrants)
        
        # Set ages for new immigrants
        self.sim.people.age[new_uids] = characteristics['ages']
        
        # Set TB status for new immigrants
        tb = self.sim.diseases.tb
        for i, uid in enumerate(new_uids):
            tb_status = characteristics['tb_statuses'][i]
            if tb_status == 'susceptible':
                tb.state[uid] = tb.SUSCEPTIBLE
            elif tb_status == 'latent_fast':
                tb.state[uid] = tb.LATENT_FAST
            elif tb_status == 'latent_slow':
                tb.state[uid] = tb.LATENT_SLOW
            elif tb_status == 'active':
                tb.state[uid] = tb.ACTIVE_PRESYMP  # Start as presymptomatic
        
        # Update household assignments for new immigrants
        # Assign them to existing households or create new ones
        self.assign_immigrants_to_households(new_uids)
        
        self.n_immigrants = n_immigrants
        return new_uids
    
    def assign_immigrants_to_households(self, new_uids):
        """Assign new immigrants to households."""
        # Get household network
        hh_net = None
        for net in self.sim.networks.values():
            if hasattr(net, 'hhid'):
                hh_net = net
                break
        
        if hh_net is None:
            return  # No household network found
        
        # For simplicity, assign immigrants to existing households
        # In a more sophisticated model, you might create new households
        existing_hhids = np.unique(hh_net.hhid[hh_net.hhid >= 0])
        
        if len(existing_hhids) > 0:
            # Assign each immigrant to a random existing household
            assigned_hhids = np.random.choice(existing_hhids, size=len(new_uids))
            hh_net.hhid[new_uids] = assigned_hhids
        else:
            # Create new households for immigrants
            for i, uid in enumerate(new_uids):
                hh_net.hhid[uid] = i
    
    def update_results(self):
        """Update results tracking."""
        if hasattr(self, 'results') and self.results is not None:
            self.results.n_immigrants[self.sim.ti] = self.n_immigrants
        return



class SimpleImmigration(ss.Demographics):
    """
    Simple immigration module that adds new people to the simulation.
    
    Parameters:
    -----------
    immigration_rate : float
        Number of immigrants per year
    """
    
    def __init__(self, pars=None, immigration_rate=20, **kwargs):
        super().__init__()
        
        self.define_pars(
            immigration_rate=immigration_rate,
        )
        self.update_pars(pars, **kwargs)
        
        # Define states for simple immigration tracking
        self.define_states(
            ss.IntArr('hhid', default=-1),  # Household ID for immigrants
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),  # When they immigrated
            ss.FloatArr('age_at_immigration', default=np.nan),  # Age when they immigrated
        )
        
        # Initialize tracking
        self.n_immigrants = 0
        
        return
    
    def step(self):
        """Add immigrants to the population."""
        # Calculate immigrants per timestep
        dt_years = self.t.dt / 365.25  # Convert to years
        immigrants_per_timestep = self.pars.immigration_rate * dt_years
        
        # Use Poisson distribution for number of immigrants
        n_immigrants = np.random.poisson(immigrants_per_timestep)
        
        if n_immigrants == 0:
            self.n_immigrants = 0
            return []
        
        # Add new people to the population
        new_uids = self.sim.people.grow(n_immigrants)
        
        # Set random ages for new immigrants (0-80 years)
        ages = np.random.uniform(0, 80, n_immigrants)
        self.sim.people.age[new_uids] = ages
        
        # The Starsim framework automatically handles growing all disease arrays
        # when people.grow() is called, so we don't need to manually grow TB arrays
        
        # Set TB status for new immigrants (mostly susceptible)
        tb = self.sim.diseases.tb
        import tbsim as mtb
        for uid in new_uids:
            # 90% susceptible, 5% latent slow, 3% latent fast, 2% active
            rand = np.random.random()
            if rand < 0.90:
                tb.state[uid] = mtb.TBS.NONE  # Susceptible
            elif rand < 0.95:
                tb.state[uid] = mtb.TBS.LATENT_SLOW
            elif rand < 0.98:
                tb.state[uid] = mtb.TBS.LATENT_FAST
            else:
                tb.state[uid] = mtb.TBS.ACTIVE_PRESYMP
        
        # Assign to households
        self.assign_to_households(new_uids)
        
        self.n_immigrants = n_immigrants
        return new_uids
    
    def assign_to_households(self, new_uids):
        """Assign new immigrants to existing households."""
        # Get household network
        hh_net = None
        for net in self.sim.networks.values():
            if hasattr(net, 'hhid'):
                hh_net = net
                break
        
        if hh_net is None:
            return
        
        # Get existing households
        existing_hhids = np.unique(hh_net.hhid[hh_net.hhid >= 0])
        
        if len(existing_hhids) > 0:
            # Assign each immigrant to a random existing household
            assigned_hhids = np.random.choice(existing_hhids, size=len(new_uids))
            hh_net.hhid[new_uids] = assigned_hhids
        else:
            # Create new households for immigrants
            for i, uid in enumerate(new_uids):
                hh_net.hhid[uid] = i

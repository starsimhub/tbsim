import numpy as np
import sciris as sc
from sciris import randround as rr # Since used frequently
import starsim as ss
import starsim.people as sp
import matplotlib.pyplot as plt

#from enum import Enum

__all__ = ['TB', 'TBS']

class TBS(): # Enum
    NONE            = np.nan # No TB
    LATENT_SLOW     = 0.0    # Latent TB, slow progression
    LATENT_FAST     = 1.0    # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2.0    # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3.0    # Active TB, smear positive
    ACTIVE_SMNEG    = 4.0    # Active TB, smear negative
    ACTIVE_EXPTB    = 5.0    # Active TB, extra-pulmonary
    DEAD            = 6.0    # TB death
    
class TB(ss.Infection):
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        
        """Add TB parameters and states to the TB model"""
        pars = ss.dictmergeleft(pars,
            init_prev = 0.01,   # Initial prevalence - TODO: Check if there is one
            beta = 0.25,         # Transmission rate  - TODO: Check if there is one
        )

        """
        DISEASE PROGRESSION: 
        Rates can be interpreted as mean time to transition between states
        For example, for TB Fast Progression( tb_LF_to_act_pre_sym), 1 / (6e-3) = 166.67 days  ~ 5.5 months ~ 0.5 years
        """
        # Natural history according with Stewart slides (EMOD TB model):
        pars = ss.dictmergeleft(pars,
            p_latent_fast = 0.1, # Probability of latent fast as opposed to latent slow

            rate_LS_to_presym = 3e-5,                   # Latent Slow to Active Pre-Symptomatic (per day)
            dur_LF_to_presymp = ss.expon(scale=1/6e-3), # Latent Fast to Active Pre-Symptomatic (per day)

            dur_presym = ss.expon(scale=1/3e-2),  # Pre-symptomatic to symptomatic (days)
            
            p_exptb = 0.1,
            p_smpos = 0.65 / (0.65+0.25), # Amongst those without extrapulminary TB

            rate_active_to_cure = 2.4e-4,

            rate_exptb_to_dead = 0.15 * 4.5e-4, # Extra-Pulmonary TB to Dead (days)
            rate_smpos_to_dead = 4.5e-4,        # Smear Positive Pulmonary TB to Dead (days)
            rate_smneg_to_dead = 0.3 * 4.5e-4,  # Smear Negative Pulmonary TB to Dead (days)

            # TODO: VALUES and list sources
            rel_trans_smpos     = 1.0,
            rel_trans_smneg     = 0.3,
            rel_trans_exptb     = 0.05,
            rel_trans_presymp   = 0.1,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.add_states(
            # Initialize states specific to TB:
            ss.FloatArr('state', default=TBS.NONE),             # One state to rule them all?
            # ss.FloatArr('latent_slow'),
            # ss.FloatArr('latent_fast'),
            # ss.FloatArr('active_presym'),
            # ss.FloatArr('active_smpos'),
            # ss.FloatArr('active_smneg'),
            # ss.FloatArr('active_exptb'),
            ss.FloatArr('active_state', default =TBS.NONE),     # Form of active TB (SmPos, SmNeg, or ExpTB)
            ss.FloatArr('rel_LS_prog', default=1.0),                # Multiplier on the latent-slow progression rate
            ss.FloatArr('ppf_LS_to_presymp', default= ss.random()),     # CDF samples for transition from latent slow to active pre-symptomatic

            # Timestep of state changes          
            ss.FloatArr('ti_latent'),
            ss.FloatArr('ti_presymp'),
            ss.FloatArr('ti_active'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('ti_cure'),
        )
        
        # Convert the scalar numbers to a Bernoulli distribution
        self.pars.p_latent_fast = ss.bernoulli(self.pars.p_latent_fast)
        self.pars.init_prev = ss.bernoulli(self.pars.init_prev)
        self.pars.p_exptb = ss.bernoulli(self.pars.p_exptb)
        self.pars.p_smpos = ss.bernoulli(self.pars.p_smpos)

        # Random number streams used in state flow
        self.choose_cure_or_die_ti = ss.random()
        self.will_die = ss.random()

        return

    # TODO: Implement the properties for the model here
    @property
    def infectious(self):
        """
        Infectious if in any of the active states
        """
        return (self.state==TBS.ACTIVE_PRESYMP) | (self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB)

    def set_prognoses(self, sim, uids, from_uids=None):
        super().set_prognoses(sim, uids, from_uids)

        # Carry out state changes upon new infection
        self.susceptible[uids] = False
        self.infected[uids] = True # Not needed, but useful for reporting
        self.ti_infected[uids] = sim.ti # Not needed, but useful for reporting
        self.ti_latent[uids] = sim.ti

        p = self.pars # Shortcut

        # Decide which agents go to latent fast vs slow
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.state[slow_uids] = TBS.LATENT_SLOW
        self.state[fast_uids] = TBS.LATENT_FAST

        # Determine time index to become active pre-symptomatic
        #self.ppf_LS_to_presymp[slow_uids] = p.ppf_LS_to_presymp.rvs(slow_uids)

        # Determine which agents will have extrapulminary TB
        exptb_uids, not_exptb_uids = p.p_exptb.filter(uids, both=True)
        self.active_state[exptb_uids] = TBS.ACTIVE_EXPTB

        # Of those not going exptb, choose smear positive or smear negative
        smpos_uids, smneg_uids = p.p_smpos.filter(not_exptb_uids, both=True)
        self.active_state[smpos_uids] = TBS.ACTIVE_SMPOS
        self.active_state[smneg_uids] = TBS.ACTIVE_SMNEG

        # Set ti of presymp
        rate = self.rel_LS_prog[slow_uids] * self.pars.rate_LS_to_presym
        self.ti_presymp[slow_uids] = sim.ti - np.log(1 - self.ppf_LS_to_presymp[slow_uids])/rate  / 365 / sim.dt
        self.ti_presymp[fast_uids] = sim.ti + p.dur_LF_to_presymp.rvs(fast_uids) / 365 / sim.dt

        # Update result count of new infections 
        self.results['new_infections'][sim.ti] += len(uids)
        return

    def update_pre(self, sim):
        # Make all the updates from the SIR model 
        super().update_pre(sim)
        p = self.pars

        # Latent --> active pre-symptomatic
        latents = (((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST))  & (self.ti_presymp <= sim.ti)).uids
        
        if len(latents):
            self.state[latents] = TBS.ACTIVE_PRESYMP
            self.rel_trans[latents] = p.rel_trans_presymp

            # Determine duration of presymp (before symp)
            self.ti_active[latents] = sim.ti + p.dur_presym.rvs(latents) / 365 / sim.dt

        # Pre symp --> Active
        presym = ((self.state == TBS.ACTIVE_PRESYMP) & (self.ti_active <= sim.ti)).uids
        if len(presym):
            self.state[presym] = self.active_state[presym]

            state = self.state[presym] 
            # y = presym
            exptb_uids = presym[state ==TBS.ACTIVE_EXPTB]
            smpos_uids = presym[state ==TBS.ACTIVE_SMPOS]
            smneg_uids = presym[state ==TBS.ACTIVE_SMNEG]

            # Set relative transmission rates for each Active state
            self.rel_trans[exptb_uids] = p.rel_trans_exptb
            self.rel_trans[smpos_uids] = p.rel_trans_smpos
            self.rel_trans[smneg_uids] = p.rel_trans_smneg

            # Set ti for next state, recovered or dead
            # Using Gillespie SSA
            rand_ti = self.choose_cure_or_die_ti.rvs(presym)
            rand_die = self.will_die.rvs(presym)
            uids = presym
            
            #total_rate = p.rate_active_to_cure + p.rate_exptb_to_dead
            total_rate = np.concatenate([
                p.rate_exptb_to_dead * np.ones(len(exptb_uids)),
                p.rate_smpos_to_dead * np.ones(len(smpos_uids)),
                p.rate_smneg_to_dead * np.ones(len(smneg_uids)),
            ]) + p.rate_active_to_cure
            dur_active = -np.log(rand_ti)/total_rate / 365 / sim.dt
            will_die = rand_die < p.rate_active_to_cure / total_rate

            die_uids = uids[will_die]
            cure_uids = uids[~will_die]

            # Determine duration of symp (before death)
            self.ti_dead[die_uids] = sim.ti + dur_active[will_die]
            self.ti_cure[cure_uids] = sim.ti + dur_active[~will_die]

        # Active --> Susceptible
        uids = ( (self.ti_cure <= sim.ti) & ((self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB))).uids
        if len(uids):
            # Set state and reset timers
            self.state[uids] = TBS.NONE
            self.susceptible[uids] = True
            self.infected[uids] = False
            self.ti_latent[uids] = np.nan
            self.ti_presymp[uids] = np.nan
            self.ti_active[uids] = np.nan
            self.ti_dead[uids] = np.nan
            self.ti_cure[uids] = np.nan

        # Active --> Death
        deaths = ( (self.state != TBS.DEAD) & (self.ti_dead <= sim.ti) ).uids
        if len(deaths):
            sim.people.request_death(deaths)
            self.state[deaths] = TBS.DEAD
        self.results['new_deaths'][sim.ti] = len(deaths)

        return


    def update_death(self, sim, uids):
        if len(uids) == 0:
            return # Nothing to do

        super().update_death(sim, uids)
        # Make sure these agents do not transmit or get infected after death
        self.susceptible[uids] = False
        self.infected[uids] = False
        #self.state[uids] = TBS.NONE
        self.rel_trans[uids] = 0
        return

    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        for rkey in ['latent_slow', 'latent_fast', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            self.results += ss.Result(self.name, f'n_{rkey}', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_deaths', sim.npts, dtype=int)
        return

    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti

        res.n_latent_slow[ti] = np.count_nonzero(self.state == TBS.LATENT_SLOW)
        res.n_latent_fast[ti] = np.count_nonzero(self.state == TBS.LATENT_FAST)
        res.n_active_presymp[ti] = np.count_nonzero(self.state == TBS.ACTIVE_PRESYMP)
        res.n_active_smpos[ti] = np.count_nonzero(self.state == TBS.ACTIVE_SMPOS) 
        res.n_active_smneg[ti] = np.count_nonzero(self.state == TBS.ACTIVE_SMNEG)
        res.n_active_exptb[ti] = np.count_nonzero(self.state == TBS.ACTIVE_EXPTB)
        return

    def finalize_results(self, sim):
        super().finalize_results(sim)
        self.results['cum_deaths'] = np.cumsum(self.results['new_deaths'])
        return

    def plot(self):
        fig = plt.figure()
        for rkey in ['latent_slow', 'latent_fast', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            plt.plot(self.results['n_'+rkey], label=rkey.title())
        plt.legend()
        return fig

import numpy as np
import starsim as ss
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
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.default_pars(
            init_prev = ss.bernoulli(0.01),   # Initial prevalence - TODO: Check if there is one
            beta = 0.25, # Transmission rate  - TODO: Check if there is one
            p_latent_fast = ss.bernoulli(0.1), # Probability of latent fast as opposed to latent slow

            rate_LS_to_presym = 3e-5, # Latent Slow to Active Pre-Symptomatic (per day)
            rate_LF_to_presym = 6e-3, # Latent Fast to Active Pre-Symptomatic (per day)

            dur_presym = ss.expon(scale=1/3e-2),  # Pre-symptomatic to symptomatic (days)
            
            p_exptb = ss.bernoulli(0.1),
            p_smpos = ss.bernoulli(0.65 / (0.65+0.25)), # Amongst those without extrapulminary TB

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
        self.update_pars(pars, **kwargs)
        
        self._add_states()

        # Random number streams used in state flow
        self.choose_cure_or_die_ti = ss.random()
        self.will_die = ss.random()

        ##### TEMP: Shouldn't need a separate rng for this as the ss.FloatArr default should get it... but that's not working. This is a workaround
        self.ppf_LS_to_presymp_rng = ss.random()
        self.ppf_LF_to_presymp_rng = ss.random()

        return

    def _add_states(self):
        self.add_states(
            # Initialize states specific to TB:
            ss.FloatArr('state', default=TBS.NONE),                 # One state to rule them all?
            ss.FloatArr('active_tb_state', default=TBS.NONE),         # Form of active TB (SmPos, SmNeg, or ExpTB)
        )
        
        self.add_states(            
            ss.FloatArr('rel_LS_prog', default=1.0),                # Multiplier on the latent-slow progression rate
            ss.FloatArr('rel_LF_prog', default=1.0),                # Multiplier on the latent-fast progression rate
            
            ##### TEMP: ss.FloatArr('ppf_LS_to_presymp', default=ss.random()), # CDF samples for transition from latent slow to active pre-symptomatic
            ##### TEMP: ss.FloatArr('ppf_LF_to_presymp', default=ss.random()), # CDF samples for transition from latent fast to active pre-symptomatic
            ss.FloatArr('ppf_LS_to_presymp'), # CDF samples for transition from latent slow to active pre-symptomatic
            ss.FloatArr('ppf_LF_to_presymp'), # CDF samples for transition from latent fast to active pre-symptomatic
        )
        
        self.add_states(
            # Timestep of state changes          
            ss.FloatArr('ti_latent'),
            ss.FloatArr('ti_presymp'),
            ss.FloatArr('ti_active'),
            ss.FloatArr('ti_dead'),
            ss.FloatArr('ti_cure'),
        )
        return

    def init_post(self):
        super().init_post()
        # TEMP, shouldn't need this!
        self.ppf_LS_to_presymp[self.sim.people.uid] = self.ppf_LS_to_presymp_rng(self.sim.people.uid)
        return

    @property
    def infectious(self):
        """
        Infectious if in any of the active states
        """
        return (self.state==TBS.ACTIVE_PRESYMP) | (self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB)
    
    def set_prognoses(self, uids, from_uids=None):
        super().set_prognoses(uids, from_uids)

        p = self.pars # Shortcut
        ti = self.sim.ti
        dt = self.sim.dt

        # Carry out state changes upon new infection
        self.susceptible[uids] = False
        self.infected[uids] = True # Not needed, but useful for reporting
        self.ti_infected[uids] = ti # Not needed, but useful for reporting
        self.ti_latent[uids] = ti

        # Decide which agents go to latent fast vs slow
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.state[slow_uids] = TBS.LATENT_SLOW
        self.state[fast_uids] = TBS.LATENT_FAST

        # Determine time index to become active pre-symptomatic
        #self.ppf_LS_to_presymp[slow_uids] = p.ppf_LS_to_presymp.rvs(slow_uids)

        # Determine which agents will have extrapulminary TB
        exptb_uids, not_exptb_uids = p.p_exptb.filter(uids, both=True)
        self.active_tb_state[exptb_uids] = TBS.ACTIVE_EXPTB

        # Of those not going exptb, choose smear positive or smear negative
        smpos_uids, smneg_uids = p.p_smpos.filter(not_exptb_uids, both=True)
        self.active_tb_state[smpos_uids] = TBS.ACTIVE_SMPOS
        self.active_tb_state[smneg_uids] = TBS.ACTIVE_SMNEG

        # Set ti of presymp for slow and fast progressors
        rate_slow = self.rel_LS_prog[slow_uids] * self.pars.rate_LS_to_presym
        self.ti_presymp[slow_uids] = np.ceil(ti - np.log(1 - self.ppf_LS_to_presymp[slow_uids])/rate_slow  / 365 / dt)
        
        rate_fast = self.rel_LF_prog[fast_uids] * self.pars.rate_LF_to_presym
        self.ti_presymp[fast_uids] = np.ceil(ti - np.log(1 - self.ppf_LF_to_presymp[fast_uids])/rate_fast  / 365 / dt)
        
        # Update result count of new infections 
        self.results['new_infections'][ti] += len(uids)
        return

    def update_pre(self):
        # Make all the updates from the SIR model 
        super().update_pre()
        p = self.pars
        ti = self.sim.ti
        dt = self.sim.dt

        # Latent --> active pre-symptomatic
        latents = (((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST))  & (self.ti_presymp <= ti)).uids
        
        if len(latents):
            self.state[latents] = TBS.ACTIVE_PRESYMP
            self.rel_trans[latents] = p.rel_trans_presymp

            # Determine duration of presymp (before symp)
            self.ti_active[latents] = np.ceil(ti + p.dur_presym.rvs(latents) / 365 / dt)

        # Pre symp --> Active
        presym = ((self.state == TBS.ACTIVE_PRESYMP) & (self.ti_active <= ti)).uids
        if len(presym):
            self.state[presym] = self.active_tb_state[presym]

            state = self.state[presym] 
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
            dur_active = -np.log(rand_ti)/total_rate / 365
            will_die = rand_die < p.rate_active_to_cure / total_rate

            die_uids = uids[will_die]
            cure_uids = uids[~will_die]

            # Determine duration of symp (before death)
            self.ti_dead[die_uids] = np.ceil(ti + dur_active[will_die] / dt)
            self.ti_cure[cure_uids] = np.ceil(ti + dur_active[~will_die] / dt)

        # Active --> Susceptible
        uids = ( (self.ti_cure <= ti) & ((self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB))).uids
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
        deaths = ( (self.state != TBS.DEAD) & (self.ti_dead <= ti) ).uids
        if len(deaths):
            self.sim.people.request_death(deaths)
            self.state[deaths] = TBS.DEAD
        self.results['new_deaths'][ti] = len(deaths)

        return

    def update_death(self, uids):
        if len(uids) == 0:
            return # Nothing to do

        super().update_death(uids)
        # Make sure these agents do not transmit or get infected after death
        self.susceptible[uids] = False
        self.infected[uids] = False
        #self.state[uids] = TBS.NONE
        self.rel_trans[uids] = 0
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        for rkey in ['latent_slow', 'latent_fast', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            self.results += ss.Result(self.name, f'n_{rkey}', self.sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_deaths', self.sim.npts, dtype=int)
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.sim.ti

        res.n_latent_slow[ti] = np.count_nonzero(self.state == TBS.LATENT_SLOW)
        res.n_latent_fast[ti] = np.count_nonzero(self.state == TBS.LATENT_FAST)
        res.n_active_presymp[ti] = np.count_nonzero(self.state == TBS.ACTIVE_PRESYMP)
        res.n_active_smpos[ti] = np.count_nonzero(self.state == TBS.ACTIVE_SMPOS) 
        res.n_active_smneg[ti] = np.count_nonzero(self.state == TBS.ACTIVE_SMNEG)
        res.n_active_exptb[ti] = np.count_nonzero(self.state == TBS.ACTIVE_EXPTB)
        return

    def finalize_results(self):
        super().finalize_results()
        self.results['cum_deaths'] = np.cumsum(self.results['new_deaths'])
        return

    def plot(self):
        fig = plt.figure()
        for rkey in ['latent_slow', 'latent_fast', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            plt.plot(self.results['n_'+rkey], label=rkey.title())
        plt.legend()
        return fig

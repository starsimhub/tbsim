import numpy as np
import sciris as sc
from sciris import randround as rr # Since used frequently
import starsim as ss
from starsim.diseases.sir import SIR
import matplotlib.pyplot as plt

#from enum import Enum

__all__ = ['TB', 'TBS']

class TBS(): # Enum
    NONE            = ss.INT_NAN # No TB
    LATENT_SLOW     = 0    # Latent TB, slow progression
    LATENT_FAST     = 1    # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2    # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3    # Active TB, smear positive
    ACTIVE_SMNEG    = 4    # Active TB, smear negative
    ACTIVE_EXPTB    = 5    # Active TB, extra-pulmonary

class TB(SIR):
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        
        """Add TB parameters and states to the TB model"""
        pars = ss.omergeleft(pars,
            init_prev = 0.01,   # Initial prevalence - TODO: Check if there is one
            beta = 0.25,         # Transmission rate  - TODO: Check if there is one
        )

        """
        DISEASE PROGRESSION: 
        Rates can be interpreted as mean time to transition between states
        For example, for TB Fast Progression( tb_LF_to_act_pre_sym), 1 / (6e-3) = 166.67 days  ~ 5.5 months ~ 0.5 years
        """
        # Natural history according with Stewart slides (EMOD TB model):
        pars = ss.omergeleft(pars,
            p_latent_fast = 0.1, # Probability of latent fast as opposed to latent slow

            rate_LS_to_presym = 3e-5,                   # Latent Slow to Active Pre-Symptomatic (per day)
            ppf_LS_to_presymp = ss.random(),            # To draw cumulative value for inverse cum transform
            dur_LF_to_presymp = ss.expon(scale=1/6e-3), # Latent Fast to Active Pre-Symptomatic (per day)

            dur_presym = ss.expon(scale=1/3e-2),  # Pre-symptomatic to symptomatic (days)
            
            p_exptb = 0.1,
            p_smpos = 0.65 / (0.65+0.25), # Amongst those without extrapulminary TB

            dur_smpos_to_dead = ss.expon(scale=1/4.5e-4),           # Smear Positive Pulmonary TB to Dead (days)
            dur_smneg_to_dead = ss.expon(scale=1/(0.3 * 4.5e-4)),   # Smear Negative Pulmonary TB to Dead (days)
            dur_exptb_to_dead = ss.expon(scale=1/(0.15 * 4.5e-4)),  # Extra-Pulmonary TB to Dead (days)

            # TODO: VALUES and list sources
            rel_trans_smpos     = 1.0,
            rel_trans_smneg     = 0.3,
            rel_trans_exptb     = 0.05,
            rel_trans_presymp   = 0.1,
        )
        
        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.add_states(
            # Initialize states specific to TB:
            ## Susceptible                              # Existent state part of People
            ## Dead                                     # Existent state part of People 
            ss.State('state', int, default=TBS.NONE),

            ss.State('rel_LS_prog', float, 1.0), # Multiplier on the latent-slow progression rate

            # CDF samples for transition from latent slow to active pre-symptomatic
            ss.State('ppf_LS_to_presymp', float, 0),
            ss.State('dur_LF_to_presymp', int, ss.INT_NAN),

            ss.State('active_state', int, TBS.NONE),

            # Duration of active states
            ss.State('dur_presymp', int, ss.INT_NAN),
            ss.State('dur_symp_to_dead', int, ss.INT_NAN),

            # Timestep of state changes          
            ss.State('ti_latent', int, ss.INT_NAN),
            ss.State('ti_presymp', int, ss.INT_NAN),
            ss.State('ti_active', int, ss.INT_NAN),
            )

        # Convert the scalar numbers to a Bernoulli distribution
        self.pars.p_latent_fast = ss.bernoulli(self.pars.p_latent_fast)
        self.pars.init_prev = ss.bernoulli(self.pars.init_prev)
        self.pars.p_exptb = ss.bernoulli(self.pars.p_exptb)
        self.pars.p_smpos = ss.bernoulli(self.pars.p_smpos)

        return

    # TODO: Implement the properties for the model here
    @property
    def infectious(self):
        """
        Infectious if in any of the active states
        """
        return self.state in [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]

    def update_pre(self, sim):
        # Make all the updates from the SIR model 
        super().update_pre(sim)
        p = self.pars

        # Latent --> active pre-symptomatic
        inds = ss.true((self.state == TBS.LATENT_SLOW or self.state == TBS.LATENT_FAST) & (self.ti_presymp <= sim.ti))
        if len(inds):
            self.state[inds] = TBS.ACTIVE_PRESYMP
            self.rel_trans[inds] = self.pars.rel_trans_presymp
        
        # Pre symp --> Active
        inds = ss.true(self.state == TBS.ACTIVE_PRESYMP & (self.ti_active <= sim.ti))
        if len(inds):
            self.state[inds] = self.active_state
            self.rel_trans[self.active_state[inds] == TBS.ACTIVE_EXPTB] = self.pars.rel_trans_exptb
            self.rel_trans[self.active_state[inds] == TBS.ACTIVE_SMPOS] = self.pars.rel_trans_smpos
            self.rel_trans[self.active_state[inds] == TBS.ACTIVE_SMNEG] = self.pars.rel_trans_smneg

        return

    def set_prognoses(self, sim, uids, from_uids=None):
        # Carry out state changes associated with infection
        self.susceptible[uids] = False
        self.ti_latent[uids] = sim.ti

        p = self.pars

        # Calculate and schedule future outcomes

        # Decide which agents go to latent fast vs slow
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.state[fast_uids] = TBS.LATENT_FAST
        self.state[slow_uids] = TBS.LATENT_SLOW

        # Determine time index to become active pre-symptomatic
        self.ppf_LS_to_presymp[slow_uids] = p.ppf_LS_to_presymp.rvs(slow_uids)
        self.dur_LF_to_presymp[fast_uids] = p.dur_LF_to_presymp.rvs(fast_uids) / 365 / sim.dt

        # Determine which agents will have extrapulminary TB
        exptb_uids, not_exptb_uids = p.p_exptb.filter(uids, both=True)
        self.active_state[exptb_uids] = TBS.ACTIVE_EXPTB

        # Of those not going exptb, choose smear positive or smear negative
        smpos_uids, smneg_uids = p.p_smpos.filter(not_exptb_uids, both=True)
        self.active_state[smpos_uids] = TBS.ACTIVE_SMPOS
        self.active_state[smneg_uids] = TBS.ACTIVE_SMNEG

        # Determine duration of presymp (before symp)
        self.dur_presymp[exptb_uids] = p.dur_presym.rvs(exptb_uids) / 365 / sim.dt
        #self.dur_presymp_to_exptb[exptb_uids] = p.dur_presym_to_exptb.rvs(exptb_uids) / 365 / sim.dt
        #self.dur_presymp_to_smpos[smpos_uids] = p.dur_presym_to_smpos.rvs(smpos_uids) / 365 / sim.dt
        #self.dur_presymp_to_smneg[smneg_uids] = p.dur_presym_to_smneg.rvs(smneg_uids) / 365 / sim.dt

        # Determine duration of symp (before death)
        self.dur_symp_to_dead[exptb_uids] = p.dur_exptb_to_dead.rvs(exptb_uids) / 365 / sim.dt
        self.dur_symp_to_dead[smpos_uids] = p.dur_smpos_to_dead.rvs(smpos_uids) / 365 / sim.dt
        self.dur_symp_to_dead[smneg_uids] = p.dur_smneg_to_dead.rvs(smneg_uids) / 365 / sim.dt

        self.set_ti(sim, uids)

        # Update result count of new infections 
        self.results['new_infections'][sim.ti] += len(uids)
        return

    def set_ti(self, sim, uids):
        # Set ti of presymp
        slow_uids = ss.true(self.state[uids] == TBS.LATENT_SLOW)
        rate = self.rel_LS_prog[slow_uids] * self.pars.rate_LS_to_presym
        # TODO: FACTOR IN ti_latent here:
        self.ti_presymp[slow_uids] = sim.ti - np.log(1 - self.ppf_LS_to_presymp[slow_uids])/rate  / 365 / sim.dt

        fast_uids = ss.true(self.state[uids] == TBS.LATENT_FAST)
        self.ti_presymp[fast_uids] = sim.ti + self.dur_LF_to_presymp[fast_uids]

        # Set ti of active
        self.ti_active[uids] = self.ti_presymp[uids] + self.dur_presymp[uids]

        # Set ti of dead
        exptb_uids = ss.true(self.active_state[uids] == TBS.ACTIVE_EXPTB)
        smpos_uids = ss.true(self.active_state[uids] == TBS.ACTIVE_SMPOS)
        smneg_uids = ss.true(self.active_state[uids] == TBS.ACTIVE_SMNEG)
        self.ti_dead[exptb_uids] = self.ti_active[exptb_uids] + self.dur_symp_to_dead[exptb_uids]
        self.ti_dead[smpos_uids] = self.ti_active[smpos_uids] + self.dur_symp_to_dead[smpos_uids]
        self.ti_dead[smneg_uids] = self.ti_active[smneg_uids] + self.dur_symp_to_dead[smneg_uids]

    def update_death(self, sim, uids):
        if len(uids) == 0:
            return # Nothing to do

        super().update_death(sim, uids)
        # Make sure these agents do not transmit or get infected after death
        self.susceptible[uids] = False
        self.state[uids] = TBS.NONE
        self.rel_trans[uids] = 0
        return

    def init_results(self, sim):
        """ Initialize results """
        super().init_results(sim)
        for rkey in ['latent_slow', 'latent_fast', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            self.results += ss.Result(self.name, f'n_{rkey}', sim.npts, dtype=int)
        return

    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti

        res.n_latent_slow[ti] = np.count_nonzero(self.state == TBS.LATENT_SLOW)
        res.n_latent_fast[ti] = np.count_nonzero(self.state == TBS.LATENT_FAST)
        res.n_active_presymp[ti] = np.count_nonzero(self.state == TBS.ACTIVE_PRESYMP)
        res.n_active_smpos[ti] = np.count_nonzero(self.state == TBS.ACTIVE_EXPTB)
        res.n_active_smneg[ti] = np.count_nonzero(self.state == TBS.ACTIVE_SMPOS)
        res.n_active_exptb[ti] = np.count_nonzero(self.state == TBS.ACTIVE_SMNEG)

        return

    def plot(self):
        fig = plt.figure()
        for rkey in ['latent_slow', 'latent_fast', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            plt.plot(self.results['n_'+rkey], label=rkey.title())
        plt.legend()
        return fig
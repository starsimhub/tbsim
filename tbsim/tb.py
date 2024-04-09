import numpy as np
import sciris as sc
from sciris import randround as rr # Since used frequently
import starsim as ss
from starsim.diseases.sir import SIR
import matplotlib.pyplot as plt

__all__ = ['TB']

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
            dur_LF_to_act_pre_sym = ss.expon(scale=1/6e-3),  # Latent Fast to Active Pre-Symptomatic (days)
            dur_LS_to_act_pre_sym = ss.expon(scale=1/3e-5),  # Latent Slow to Active Pre-Symptomatic (days)
            dur_presym_to_symp_exptb = ss.expon(scale=1/3e-2),     # Pre-Symptomatic to Symptomatic for exptb (days)
            dur_presym_to_symp_smpos = ss.expon(scale=1/3e-2),     # Pre-Symptomatic to Symptomatic for smpos (days)
            dur_presym_to_symp_smneg = ss.expon(scale=1/3e-2),     # Pre-Symptomatic to Symptomatic for smneg (days)
            
            p_exptb = 0.1,
            p_smpos = 0.65 / (0.65+0.25), # Amongst those without extrapulminary TB

            dur_smpos_to_dead = ss.expon(scale=1/4.5e-4), # Smear Positive Pulmonary TB to Dead (days)
            dur_smneg_to_dead = ss.expon(scale=1/(0.3 * 4.5e-4)),    # Smear Negative Pulmonary TB to Dead (days)
            dur_exptb_to_dead = ss.expon(scale=1/(0.15 * 4.5e-4)),# Extra-Pulmonary TB to Dead (days)

            # TODO: VALUES and list sources
            rel_trans_smpos = 1.0,
            rel_trans_exptb = 0.05,
            rel_trans_presymp = 0.1,
            rel_trans_smneg = 0.3,
        )
        
        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        self.add_states(
            # Initialize states specific to TB:
            ## Susceptible                              # Existent state part of People
            ## Dead                                     # Existent state part of People 
            ss.State('latent', bool, False),            # Latent TB, fast progression
            ss.State('active_presymp', bool, False),   # Active TB, pre-symptomatic
            ss.State('active_smpos', bool, False),    # Active TB, smear positive
            ss.State('active_smneg', bool, False),    # Active TB, smear negative
            ss.State('active_exptb', bool, False),   # Active TB, extra-pulmonary
            ss.State('recovered', bool, False),         

            # Timestep of state changes          
            ss.State('ti_latent', int, ss.INT_NAN),
            ss.State('ti_active_presymp', int, ss.INT_NAN),
            ss.State('ti_active_exptb', int, ss.INT_NAN),
            ss.State('ti_active_smneg', int, ss.INT_NAN),
            ss.State('ti_active_smpos', int, ss.INT_NAN),
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
        #return self.infected | self.exposed
        return self.ti_active_presymp | self.ti_active_exptb | self.ti_active_smneg | self.ti_active_smpos

    def update_pre(self, sim):
        # Make all the updates from the SIR model 
        super().update_pre(sim)

        # Latent --> active pre-symptomatic
        inds = ss.true(self.latent & (self.ti_active_presymp <= sim.ti))
        if len(inds):
            self.latent[inds] = False
            self.active_presymp[inds] = True
            self.rel_trans[inds] = self.pars.rel_trans_presymp

        # Pre symp --> Active extra pulminary
        inds = ss.true(self.active_presymp & (self.ti_active_exptb <= sim.ti))
        if len(inds):
            self.active_presymp[inds] = False
            self.active_exptb[inds] = True
            self.rel_trans[inds] = self.pars.rel_trans_exptb

        # Pre symp --> Active smear positive
        inds = ss.true(self.active_presymp & (self.ti_active_smpos <= sim.ti))
        if len(inds):
            self.active_presymp[inds] = False
            self.active_smpos[inds] = True
            self.rel_trans[inds] = self.pars.rel_trans_smpos

        # Pre symp --> Active smear negative
        inds = ss.true(self.active_presymp & (self.ti_active_smneg <= sim.ti))
        if len(inds):
            self.active_presymp[inds] = False
            self.active_smneg[inds] = True
            self.rel_trans[inds] = self.pars.rel_trans_smneg

        return

    def update_death(self, sim, uids):
        if len(uids) == 0:
            return # Nothing to do

        super().update_death(sim, uids)
        # Make sure these agents do not transmit or get infected after death
        self.susceptible[uids] = False
        self.rel_trans[uids] = 0
        self.latent[uids] = False
        self.active_presymp[uids] = False
        self.active_smpos[uids] = False
        self.active_smneg[uids] = False
        self.active_exptb[uids] = False
        self.recovered[uids] = False
        return

    def set_prognoses(self, sim, uids, from_uids=None):
        # Carry out state changes associated with infection
        self.susceptible[uids] = False
        self.ti_latent[uids] = sim.ti

        p = self.pars

        # Calculate and schedule future outcomes

        # Decide which agents go to latent fast vs slow
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.latent[fast_uids] = True
        self.latent[slow_uids] = True

        # Determine time index to become active pre-symptomatic
        self.ti_active_presymp[fast_uids] = sim.ti + p.dur_LF_to_act_pre_sym.rvs(fast_uids) / 365.0 / sim.dt
        self.ti_active_presymp[slow_uids] = sim.ti + p.dur_LS_to_act_pre_sym.rvs(slow_uids) / 365.0 / sim.dt

        # Determine which agents will have extrapulminary TB
        exptb_uids, not_exptb_uids = p.p_exptb.filter(uids, both=True)

        # Of those not going exptb, choose smear positive or smear negative
        smpos_uids, smneg_uids = p.p_smpos.filter(not_exptb_uids, both=True)

        # Determine time index to become active
        self.ti_active_exptb[exptb_uids] = self.ti_active_presymp[exptb_uids] + p.dur_presym_to_symp_exptb.rvs(exptb_uids) / 365.0 / sim.dt
        self.ti_active_smpos[smpos_uids] = self.ti_active_presymp[smpos_uids] + p.dur_presym_to_symp_smpos.rvs(smpos_uids) / 365.0 / sim.dt
        self.ti_active_smneg[smneg_uids] = self.ti_active_presymp[smneg_uids] + p.dur_presym_to_symp_smneg.rvs(smneg_uids) / 365.0 / sim.dt


        ##### dur_inf = self.pars['dur_inf'].rvs(uids) # TODO: Duration of infection?
        ##### will_die = self.pars['p_death'].rvs(uids) # TODO: Probability of death?
        #self.ti_recovered[uids[~will_die]] = sim.year + dur_inf[~will_die]

        # Determine time index of death
        self.ti_dead[smpos_uids] = self.ti_active_smpos[smpos_uids] + p.dur_smpos_to_dead.rvs(smpos_uids) / 365 / sim.dt
        self.ti_dead[smneg_uids] = self.ti_active_smneg[smneg_uids] + p.dur_smneg_to_dead.rvs(smneg_uids) / 365 / sim.dt
        self.ti_dead[exptb_uids] = self.ti_active_exptb[exptb_uids] + p.dur_exptb_to_dead.rvs(exptb_uids) / 365 / sim.dt

        # Update result count of new infections 
        self.results['new_infections'][sim.ti] += len(uids)
        return


    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti

        res.n_latent[ti] = np.count_nonzero(self.latent)
        res.n_active_presymp[ti] = np.count_nonzero(self.active_presymp)
        res.n_active_smpos[ti] = np.count_nonzero(self.active_smpos)
        res.n_active_smneg[ti] = np.count_nonzero(self.active_smneg)
        res.n_active_exptb[ti] = np.count_nonzero(self.active_exptb)

        return

    def plot(self):
        """ Default plot for SIS model """
        fig = plt.figure()
        for rkey in ['latent', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            plt.plot(self.results['n_'+rkey], label=rkey.title())
        plt.legend()
        return fig
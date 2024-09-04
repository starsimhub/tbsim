import numpy as np
import starsim as ss
import matplotlib.pyplot as plt

#from enum import Enum

__all__ = ['TB', 'TBS']

DAYS_PER_YEAR = 365

class TBS(): # Enum
    NONE            = np.nan # No TB
    LATENT_SLOW     = 0.0    # Latent TB, slow progression
    LATENT_FAST     = 1.0    # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2.0    # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3.0    # Active TB, smear positive
    ACTIVE_SMNEG    = 4.0    # Active TB, smear negative
    ACTIVE_EXPTB    = 5.0    # Active TB, extra-pulmonary
    DEAD            = 8.0    # TB death


class TB(ss.Infection):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.default_pars(
            init_prev = ss.bernoulli(0.01),   # Initial prevalence - TODO: Check if there is one
            beta = 0.25, # Transmission rate  - TODO: Check if there is one
            p_latent_fast = ss.bernoulli(0.1), # Probability of latent fast as opposed to latent slow

            rate_LS_to_presym = 3e-5,           # Latent Slow to Active Pre-Symptomatic (per day)
            rate_LF_to_presym = 6e-3,           # Latent Fast to Active Pre-Symptomatic (per day)
            rate_presym_to_active = 3e-2,       # Pre-symptomatic to symptomatic (per day)
            rate_active_to_clear = 2.4e-4,      # Active infection to natural clearance (per day)
            rate_exptb_to_dead = 0.15 * 4.5e-4, # Extra-Pulmonary TB to Dead (per day)
            rate_smpos_to_dead = 4.5e-4,        # Smear Positive Pulmonary TB to Dead (per day)
            rate_smneg_to_dead = 0.3 * 4.5e-4,  # Smear Negative Pulmonary TB to Dead (per day)
            rate_treatment_to_clear = 2/12 / DAYS_PER_YEAR, # 2 months (per day, for consistency)

            p_exptb = ss.bernoulli(0.1),
            p_smpos = ss.bernoulli(0.65 / (0.65+0.25)), # Amongst those without extrapulminary TB

            # Relative transmissibility of each state, TODO: VALUES and list sources
            rel_trans_smpos   = 1.0,
            rel_trans_smneg   = 0.3,
            rel_trans_exptb   = 0.05,
            rel_trans_presymp = 0.1,

            reltrans_dist = None,
        )
        self.update_pars(pars, **kwargs)
        
        self.add_states(
            # Initialize states specific to TB:
            ss.FloatArr('state', default=TBS.NONE),             # One state to rule them all?
            ss.FloatArr('active_tb_state', default=TBS.NONE),   # Form of active TB (SmPos, SmNeg, or ExpTB)
            ss.FloatArr('rr_activation', default=1.0),          # Multiplier on the latent-to-presymp rate
            ss.FloatArr('rr_clearance', default=1.0),           # Multiplier on the active-to-susceptible rate
            ss.FloatArr('rr_death', default=1.0),               # Multiplier on the active-to-dead rate
            ss.BoolArr('on_treatment', default=False),

            ss.FloatArr('ti_presymp'),
            ss.FloatArr('ti_active'),
        )

        self.p_latent_to_presym = ss.bernoulli(p=self.p_latent_to_presym)
        self.p_presym_to_active = ss.bernoulli(p=self.p_presym_to_active)
        self.p_active_to_clear = ss.bernoulli(p=self.p_active_to_clear)
        self.p_active_to_death = ss.bernoulli(p=self.p_active_to_death)

        return

    def init_post(self):
        super().init_post()
        if isinstance(self.pars.reltrans_dist, ss.Dist):
            uids = self.sim.people.auids
            self.rel_trans[uids] = self.pars.reltrans_dist(uids)

    @staticmethod
    def p_latent_to_presym(self, sim, uids):
        # Could be more complex function of time in state, but exponential for now
        assert np.isin(self.state[uids], [TBS.LATENT_FAST, TBS.LATENT_SLOW]).all()

        rate = np.full(len(uids), fill_value=self.pars.rate_LS_to_presym)
        rate[self.state[uids] == TBS.LATENT_FAST] = self.pars.rate_LF_to_presym
        rate *= self.rr_activation[uids]

        prob = 1-np.exp(-DAYS_PER_YEAR * rate * sim.dt) # or just rate * dt
        return prob

    @staticmethod
    def p_presym_to_active(self, sim, uids):
        # Could be more complex function of time in state, but exponential for now
        assert (self.state[uids] == TBS.ACTIVE_PRESYMP).all()
        rate = np.full(len(uids), fill_value=self.pars.rate_presym_to_active)
        prob = 1-np.exp(-DAYS_PER_YEAR * rate * sim.dt) # or just rate * dt
        return prob

    @staticmethod
    def p_active_to_clear(self, sim, uids):
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all()
        rate = np.full(len(uids), fill_value=self.pars.rate_active_to_clear)
        rate[self.on_treatment[uids]] = self.pars.rate_treatment_to_clear # Those on treatment have a different clearance rate
        rate *= self.rr_clearance[uids]

        prob = 1-np.exp(-DAYS_PER_YEAR * rate * sim.dt) # or just rate * dt
        return prob

    @staticmethod
    def p_active_to_death(self, sim, uids):
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all()
        rate = np.full(len(uids), fill_value=self.pars.rate_exptb_to_dead)
        rate[self.state[uids] == TBS.ACTIVE_SMPOS] = self.pars.rate_smpos_to_dead
        rate[self.state[uids] == TBS.ACTIVE_SMNEG] = self.pars.rate_smneg_to_dead

        rate *= self.rr_death[uids]

        prob = 1-np.exp(-DAYS_PER_YEAR * rate * sim.dt) # or just rate * dt
        return prob

    @property
    def infectious(self):
        """
        Infectious if in any of the active states
        """
        return (self.on_treatment) | (self.state==TBS.ACTIVE_PRESYMP) | (self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB)

    def set_prognoses(self, uids, from_uids=None):
        super().set_prognoses(uids, from_uids)

        p = self.pars # Shortcut
        ti = self.sim.ti
        dt = self.sim.dt

        # Carry out state changes upon new infection
        self.susceptible[uids] = False
        self.infected[uids] = True # Not needed, but useful for reporting

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

        # Update result count of new infections 
        self.results['new_infections'][ti] += len(uids)
        return

    def update_pre(self):
        # Make all the updates from the SIR model 
        super().update_pre()
        p = self.pars
        ti = self.sim.ti

        # Latent --> active pre-symptomatic
        latent_uids = (((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST))).uids
        new_presymp_uids = self.p_latent_to_presym.filter(latent_uids)
        if len(new_presymp_uids):
            self.state[new_presymp_uids] = TBS.ACTIVE_PRESYMP
            self.ti_presymp[new_presymp_uids] = ti
            self.rel_trans[new_presymp_uids] = p.rel_trans_presymp

        # Pre symp --> Active
        presym_uids = (self.state == TBS.ACTIVE_PRESYMP).uids
        new_active_uids = self.p_presym_to_active.filter(presym_uids)
        if len(new_active_uids):
            active_state = self.active_tb_state[new_active_uids] 
            self.state[new_active_uids] = active_state
            self.ti_active[new_active_uids] = ti

            exptb_uids = new_active_uids[active_state ==TBS.ACTIVE_EXPTB]
            smpos_uids = new_active_uids[active_state ==TBS.ACTIVE_SMPOS]
            smneg_uids = new_active_uids[active_state ==TBS.ACTIVE_SMNEG]

            # Set relative transmission rates for each Active state
            self.rel_trans[exptb_uids] = p.rel_trans_exptb
            self.rel_trans[smpos_uids] = p.rel_trans_smpos
            self.rel_trans[smneg_uids] = p.rel_trans_smneg

        # Active --> Susceptible via natural recovery or as accelerated by treatment
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_EXPTB))).uids
        new_clear_uids = self.p_active_to_clear.filter(active_uids)
        if len(new_clear_uids):
            # Set state and reset timers
            self.susceptible[new_clear_uids] = True
            self.infected[new_clear_uids] = False
            self.state[new_clear_uids] = TBS.NONE
            self.active_tb_state[new_clear_uids] = TBS.NONE
            self.ti_presymp[new_clear_uids] = np.nan
            self.ti_active[new_clear_uids] = np.nan

        # Active --> Death
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_EXPTB))).uids # Recompute after clear
        new_death_uids = self.p_active_to_death.filter(active_uids)
        if len(new_death_uids):
            self.sim.people.request_death(new_death_uids)
            self.state[new_death_uids] = TBS.DEAD
        self.results['new_deaths'][ti] = len(new_death_uids)

        # Reset relative rates for the next time step, they will be recalculated
        uids = self.sim.people.auids
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

        return

    def start_treatment(self, uids):
        # Begin individual on TB treatment, assuming all TB is drug susceptible

        # Only treat individuals who have active TB
        tx_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_EXPTB))).uids
        self.on_treatment[tx_uids] = True
        self.rr_death[tx_uids] = 0 # People on treatment don't die...
        return len(tx_uids)

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

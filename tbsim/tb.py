import numpy as np
import starsim as ss
import matplotlib.pyplot as plt

from enum import IntEnum

__all__ = ['TB', 'TBS']

class TBS(IntEnum):
    NONE            = -1    # No TB
    LATENT_SLOW     = 0     # Latent TB, slow progression
    LATENT_FAST     = 1     # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2     # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3     # Active TB, smear positive
    ACTIVE_SMNEG    = 4     # Active TB, smear negative
    ACTIVE_EXPTB    = 5     # Active TB, extra-pulmonary
    DEAD            = 8     # TB death


class TB(ss.Infection):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.define_pars(
            init_prev = ss.bernoulli(0.01),     # Initial seed infections
            beta = 0.25,                        # Transmission rate
            p_latent_fast = ss.bernoulli(0.1),  # Probability of latent fast as opposed to latent slow

            rate_LS_to_presym       = ss.perday(3e-5),                 # Latent Slow to Active Pre-Symptomatic (per day)
            rate_LF_to_presym       = ss.perday(6e-3),                 # Latent Fast to Active Pre-Symptomatic (per day)
            rate_presym_to_active   = ss.perday(3e-2),                 # Pre-symptomatic to symptomatic (per day)
            rate_active_to_clear    = ss.perday(2.4e-4),               # Active infection to natural clearance (per day)
            rate_exptb_to_dead      = ss.perday(0.15 * 4.5e-4),        # Extra-Pulmonary TB to Dead (per day)
            rate_smpos_to_dead      = ss.perday(4.5e-4),               # Smear Positive Pulmonary TB to Dead (per day)
            rate_smneg_to_dead      = ss.perday(0.3 * 4.5e-4),         # Smear Negative Pulmonary TB to Dead (per day)
            rate_treatment_to_clear = ss.peryear(2/12),                # 2 months

            active_state = ss.choice(a=[TBS.ACTIVE_EXPTB, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG], p=[0.1, 0.65, 0.25]),

            # Relative transmissibility of each state
            rel_trans_presymp   = 0.1,
            rel_trans_smpos     = 1.0,
            rel_trans_smneg     = 0.3,
            rel_trans_exptb     = 0.05,
            rel_trans_treatment = 0.5, # Multiplicative on smpos, smneg, or exptb rel_trans

            reltrans_het = ss.constant(v=1.0),
        )
        self.update_pars(pars, **kwargs) 

        # Validate rates
        for k, v in self.pars.items():
            if k[:5] == 'rate_':
                assert isinstance(v, ss.rate), 'Rate parameters for TB must be TimePars, e.g. ss.perday(x)'

        self.define_states(
            # Initialize states specific to TB:
            ss.FloatArr('state', default=TBS.NONE),             # One state to rule them all?
            ss.FloatArr('active_tb_state', default=TBS.NONE),   # Form of active TB (SmPos, SmNeg, or ExpTB)
            ss.FloatArr('rr_activation', default=1.0),          # Multiplier on the latent-to-presymp rate
            ss.FloatArr('rr_clearance', default=1.0),           # Multiplier on the active-to-susceptible rate
            ss.FloatArr('rr_death', default=1.0),               # Multiplier on the active-to-dead rate
            ss.State('on_treatment', default=False),

            ss.FloatArr('ti_presymp'),
            ss.FloatArr('ti_active'),

            ss.FloatArr('reltrans_het', default=1.0),           # Individual-level heterogeneity on infectiousness, acts in addition to stage-based rates
        )

        self.p_latent_to_presym = ss.bernoulli(p=self.p_latent_to_presym)
        self.p_presym_to_active = ss.bernoulli(p=self.p_presym_to_active)
        self.p_active_to_clear = ss.bernoulli(p=self.p_active_to_clear)
        self.p_active_to_death = ss.bernoulli(p=self.p_active_to_death)

        return

    @staticmethod
    def p_latent_to_presym(self, sim, uids):
        # Could be more complex function of time in state, but exponential for now
        assert np.isin(self.state[uids], [TBS.LATENT_FAST, TBS.LATENT_SLOW]).all()

        rate = np.full(len(uids), fill_value=self.pars.rate_LS_to_presym)
        rate[self.state[uids] == TBS.LATENT_FAST] = self.pars.rate_LF_to_presym
        rate *= self.rr_activation[uids]

        prob = 1-np.exp(-rate)
        return prob

    @staticmethod
    def p_presym_to_active(self, sim, uids):
        # Could be more complex function of time in state, but exponential for now
        assert (self.state[uids] == TBS.ACTIVE_PRESYMP).all()
        rate = np.full(len(uids), fill_value=self.pars.rate_presym_to_active)
        prob = 1-np.exp(-rate)
        return prob

    @staticmethod
    def p_active_to_clear(self, sim, uids):
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all()
        rate = np.full(len(uids), fill_value=self.pars.rate_active_to_clear)
        rate[self.on_treatment[uids]] = self.pars.rate_treatment_to_clear # Those on treatment have a different clearance rate
        rate *= self.rr_clearance[uids]

        prob = 1-np.exp(-rate)
        return prob

    @staticmethod
    def p_active_to_death(self, sim, uids):
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all()
        rate = np.full(len(uids), fill_value=self.pars.rate_exptb_to_dead)
        rate[self.state[uids] == TBS.ACTIVE_SMPOS] = self.pars.rate_smpos_to_dead
        rate[self.state[uids] == TBS.ACTIVE_SMNEG] = self.pars.rate_smneg_to_dead

        rate *= self.rr_death[uids]

        prob = 1-np.exp(-rate)
        return prob

    @property
    def infectious(self):
        """
        Infectious if in any of the active states
        """
        return (self.on_treatment) | (self.state==TBS.ACTIVE_PRESYMP) | (self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB)

    def set_prognoses(self, uids, from_uids=None):
        super().set_prognoses(uids, from_uids)

        p = self.pars

        # Carry out state changes upon new infection
        self.susceptible[uids] = False
        self.infected[uids] = True # Not needed, but useful for reporting

        # Set base transmission heterogeneity
        self.reltrans_het[uids] = p.reltrans_het.rvs(uids)

        # Decide which agents go to latent fast vs slow
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.state[slow_uids] = TBS.LATENT_SLOW
        self.state[fast_uids] = TBS.LATENT_FAST

        # Determine active TB state
        self.active_tb_state[uids] = self.pars.active_state.rvs(uids)

        # Update result count of new infections 
        self.results['new_infections'][self.ti] += len(uids)
        return

    def step(self):
        # Make all the updates from the SIR model 
        super().step()
        p = self.pars
        ti = self.ti

        # Latent --> active pre-symptomatic
        latent_uids = (((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST))).uids
        new_presymp_uids = self.p_latent_to_presym.filter(latent_uids)
        if len(new_presymp_uids):
            self.state[new_presymp_uids] = TBS.ACTIVE_PRESYMP
            self.ti_presymp[new_presymp_uids] = ti

        # Pre symp --> Active
        presym_uids = (self.state == TBS.ACTIVE_PRESYMP).uids
        if len(presym_uids):
            new_active_uids = self.p_presym_to_active.filter(presym_uids)
            if len(new_active_uids):
                active_state = self.active_tb_state[new_active_uids] 
                self.state[new_active_uids] = active_state
                self.ti_active[new_active_uids] = ti

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

        # Set rel_trans
        self.rel_trans[:] = 1 # Reset

        state_reltrans = [
            (TBS.ACTIVE_PRESYMP, p.rel_trans_presymp),
            (TBS.ACTIVE_EXPTB, p.rel_trans_exptb),
            (TBS.ACTIVE_SMPOS, p.rel_trans_smpos),
            (TBS.ACTIVE_SMNEG, p.rel_trans_smneg),
        ]

        for state, reltrans in state_reltrans:
            uids = self.state == state
            self.rel_trans[uids] *= reltrans

        # Transmission heterogeneity
        uids = self.infectious
        self.rel_trans[uids] *= self.reltrans_het[uids]

        # Treatment can reduce transmissibility
        uids = self.on_treatment
        self.rel_trans[uids] *= self.pars.rel_trans_treatment

        # Reset relative rates for the next time step, they will be recalculated
        uids = self.sim.people.auids
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

        return

    def start_treatment(self, uids):
        # Begin individual on TB treatment, assuming all TB is drug susceptible

        tbs = self.state[uids]

        # Only treat individuals who have active TB
        tx_uids = ss.uids(((tbs == TBS.ACTIVE_SMPOS) | (tbs == TBS.ACTIVE_SMPOS) | (tbs == TBS.ACTIVE_EXPTB)))
        self.on_treatment[tx_uids] = True
        self.rr_death[tx_uids] = 0 # People on treatment don't die...
        return len(tx_uids)

    def step_die(self, uids):
        if len(uids) == 0:
            return # Nothing to do

        super().step_die(uids)
        # Make sure these agents do not transmit or get infected after death
        self.susceptible[uids] = False
        self.infected[uids] = False
        #self.state[uids] = TBS.NONE
        self.rel_trans[uids] = 0
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        
        self.define_results(
            ss.Result('n_latent_slow',    dtype=int, label='Latent Slow'),
            ss.Result('n_latent_fast',    dtype=int, label='Latent Fast'),
            ss.Result('n_active_presymp', dtype=int, label='Active Pre-Symptomatic'), 
            ss.Result('n_active_smpos',   dtype=int, label='Active Smear Positive'),
            ss.Result('n_active_smneg',   dtype=int, label='Active Smear Negative'),
            ss.Result('n_active_exptb',   dtype=int, label='Active Extra-Pulmonary'),
            ss.Result('new_deaths',       dtype=int, label='New Deaths'),
            ss.Result('cum_deaths',       dtype=int, label='Cumulative Deaths')
        )
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti

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

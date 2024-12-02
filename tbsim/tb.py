import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
from tbsim.parametervalues import RatesByAge

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
    """
    TB model with age-specific progression rates.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()

        self.define_pars(
            init_prev = ss.bernoulli(0.01),     # Initial seed infections
            beta = 0.25,                        # Transmission rate
            p_latent_fast = ss.bernoulli(0.1),  # Probability of latent fast as opposed to latent slow
            by_age = True,                      # Whether to use age-specific rates
            rates_byage = None,
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
        self.p_presym_to_clear = ss.bernoulli(p=self.p_presym_to_clear)
        self.p_presym_to_active = ss.bernoulli(p=self.p_presym_to_active)
        self.p_active_to_clear = ss.bernoulli(p=self.p_active_to_clear)
        self.p_active_to_death = ss.bernoulli(p=self.p_active_to_death)
        
        self.rba = RatesByAge(self.t.unit, self.t.dt)
        self.pars['rates_byage'] = self.rba.RATES
        self.age_cutoffs = self.rba.AGE_CUTOFFS

        return

    @staticmethod
    def p_latent_to_presym(self, sim, uids):
        # Could be more complex function of time in state, but exponential for now
        assert np.isin(self.state[uids], [TBS.LATENT_FAST, TBS.LATENT_SLOW]).all()
        rate = np.zeros(len(uids))
        rate[self.state[uids] == TBS.LATENT_SLOW] = self.pars['rates_byage']['rate_LS_to_presym'][0]
        rate[self.state[uids] == TBS.LATENT_FAST] = self.pars['rates_byage']['rate_LF_to_presym'][0]
        
        if self.pars.by_age:
            ls_uids = np.isin(self.state[uids], [TBS.LATENT_SLOW])
            age_indexes = np.digitize(self.sim.people.age[uids[ls_uids]], bins=self.age_cutoffs) - 1
            rate[ls_uids] = self.pars['rates_byage']['rate_LS_to_presym'][age_indexes]
            
            lf_uids = np.isin(self.state[uids], [TBS.LATENT_FAST] )
            age_indexes = np.digitize(self.sim.people.age[uids[lf_uids]], bins=self.age_cutoffs) - 1
            rate[lf_uids] = self.pars['rates_byage']['rate_LF_to_presym'][age_indexes]
            
        # Apply individual activation multipliers
        rate *= self.rr_activation[uids]
        prob = 1-np.exp(-rate)
        return prob
    
    @staticmethod
    def p_presym_to_clear(self, sim, uids):
        # Could be more complex function of time in state, but exponential for now
        assert (self.state[uids] == TBS.ACTIVE_PRESYMP).all()
        rate = np.zeros(len(uids))
        rate[self.on_treatment[uids]] =self.pars['rates_byage']['rate_treatment_to_clear'][0] # Default rate
        
        if self.pars.by_age:
            mask = np.isin(self.state[uids], [TBS.ACTIVE_PRESYMP])
            age_indexes = np.digitize(self.sim.people.age[uids[mask]], bins=self.age_cutoffs) - 1
            rate[mask] = self.pars['rates_byage']['rate_treatment_to_clear'][age_indexes]
        prob = 1-np.exp(-rate)
        return prob

    @staticmethod
    def p_presym_to_active(self, sim, uids):
        # Could be more complex function of time in state, but exponential for now
        assert (self.state[uids] == TBS.ACTIVE_PRESYMP).all(), "The p_presym_to_active function should only be called for agents in the pre symptomatic state, however some agents were in a different state."
        rate = np.full(len(uids), fill_value=self.pars['rates_byage']['rate_presym_to_active'][0])
 
        if self.pars.by_age:
            mask = np.isin(self.state[uids], [TBS.ACTIVE_PRESYMP])
            age_indexes = np.digitize(self.sim.people.age[uids[mask]], bins=self.age_cutoffs) - 1
            rate[mask] = self.pars['rates_byage']['rate_presym_to_active'][age_indexes]
        prob = 1-np.exp(-rate)
        return prob

    @staticmethod
    def p_active_to_clear(self, sim, uids):
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all()
        rate = np.full(len(uids), fill_value=self.pars['rates_byage']['rate_active_to_clear'][0])
        rate[self.on_treatment[uids]] = self.pars['rates_byage']['rate_treatment_to_clear'][1]     # Default values
      
        if self.pars.by_age:
            mask = np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
            age_indexes = np.digitize(self.sim.people.age[uids[mask]], bins=self.age_cutoffs) - 1
            rate[mask] = self.pars['rates_byage']['rate_active_to_clear'][age_indexes]
            
            on_treatment_uids = ss.uids(self.on_treatment[uids])
            mask = np.isin(on_treatment_uids, [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
            age_indexes = np.digitize(self.sim.people.age[uids[mask]], bins=self.age_cutoffs) - 1
            rate[mask] = self.pars['rates_byage']['rate_treatment_to_clear'][age_indexes]
            
        rate *= self.rr_clearance[uids]
        prob = 1-np.exp(-rate)
        return prob


    @staticmethod
    def p_active_to_death(self, sim, uids):
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all()
        rate = np.zeros(len(uids))
        rate[self.state[uids] == TBS.ACTIVE_SMPOS] = self.pars['rates_byage']['rate_smpos_to_dead'][0] 
        rate[self.state[uids] == TBS.ACTIVE_SMNEG] = self.pars['rates_byage']['rate_smneg_to_dead'][0] 
        rate[self.state[uids] == TBS.ACTIVE_SMNEG] = self.pars['rates_byage']['rate_exptb_to_dead'][0] 
        
        if self.pars.by_age:
            smpos_uids = np.isin(self.state[uids], [TBS.ACTIVE_SMPOS])
            age_indexes = np.digitize(self.sim.people.age[uids[smpos_uids]], bins=self.age_cutoffs) - 1
            rate[smpos_uids] = self.pars['rates_byage']['rate_smpos_to_dead'][age_indexes]
            
            smneg_uids = np.isin(self.state[uids], [TBS.ACTIVE_SMNEG])
            age_indexes = np.digitize(self.sim.people.age[uids[smneg_uids]], bins=self.age_cutoffs) - 1
            rate[smneg_uids] = self.pars['rates_byage']['rate_smneg_to_dead'][age_indexes]
            
            exptb_uids = np.isin(self.state[uids], [TBS.ACTIVE_EXPTB])
            age_indexes = np.digitize(self.sim.people.age[uids[exptb_uids]], bins=self.age_cutoffs) - 1
            rate[exptb_uids] = self.pars['rates_byage']['rate_exptb_to_dead'][age_indexes]
            
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
        # Perform TB progression steps
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
        new_clear_presymp_uids = ss.uids()
        if len(presym_uids):
            # Pre symp --> Clear
            new_clear_presymp_uids = self.p_presym_to_clear.filter(presym_uids)

            new_active_uids = self.p_presym_to_active.filter(presym_uids)
            if len(new_active_uids):
                active_state = self.active_tb_state[new_active_uids] 
                self.state[new_active_uids] = active_state
                self.ti_active[new_active_uids] = ti

        # Active --> Susceptible via natural recovery or as accelerated by treatment (clear)
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_EXPTB))).uids
        new_clear_active_uids = self.p_active_to_clear.filter(active_uids)
        new_clear_uids = ss.uids.cat(new_clear_presymp_uids, new_clear_active_uids)
        if len(new_clear_uids):
            # Set state and reset timers
            self.susceptible[new_clear_uids] = True
            self.infected[new_clear_uids] = False
            self.state[new_clear_uids] = TBS.NONE
            self.active_tb_state[new_clear_uids] = TBS.NONE
            self.ti_presymp[new_clear_uids] = np.nan
            self.ti_active[new_clear_uids] = np.nan
            self.on_treatment[new_clear_uids] = False

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
        """ Start treatment for active TB """
        if len(uids) == 0:
            return 0  # No one to treat

        rst = self.state[uids]

        #find individuals with active TB
        is_active = np.isin(rst, [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])

        # Get the corresponding UIDs that match the active state
        tx_uids = uids[is_active]

        if len(tx_uids) == 0:
            return 0  # No one to treat

        # Mark the individuals as being on treatment
        self.on_treatment[tx_uids] = True

        # Adjust death and clearance rates for those starting treatment
        self.rr_death[tx_uids] = 0  # People on treatment have zero death rate

        # Reduce transmission rates for people on treatment
        self.rel_trans[tx_uids] *= self.pars.rel_trans_treatment

        # Return the number of individuals who started treatment
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
            ss.Result('cum_deaths',       dtype=int, label='Cumulative Deaths'),
        )
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti

        res.n_latent_slow[ti]    = np.count_nonzero(self.state == TBS.LATENT_SLOW)
        res.n_latent_fast[ti]    = np.count_nonzero(self.state == TBS.LATENT_FAST)
        res.n_active_presymp[ti] = np.count_nonzero(self.state == TBS.ACTIVE_PRESYMP)
        res.n_active_smpos[ti]   = np.count_nonzero(self.state == TBS.ACTIVE_SMPOS) 
        res.n_active_smneg[ti]   = np.count_nonzero(self.state == TBS.ACTIVE_SMNEG)
        res.n_active_exptb[ti]   = np.count_nonzero(self.state == TBS.ACTIVE_EXPTB)
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

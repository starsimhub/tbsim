import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
from enum import IntEnum

__all__ = ['TB', 'TBS']

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TBS(IntEnum):
    NONE            = -1    # No TB
    LATENT_SLOW     = 0     # Latent TB, slow progression
    LATENT_FAST     = 1     # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2     # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3     # Active TB, smear positive
    ACTIVE_SMNEG    = 4     # Active TB, smear negative
    ACTIVE_EXPTB    = 5     # Active TB, extra-pulmonary
    DEAD            = 8     # TB death

# ---------------------------------------------------------------------------
# Parameter and State Definition Mixin
# ---------------------------------------------------------------------------
class TBParameterMixin:
    def define_tb_parameters(self):
        self.define_pars(
            init_prev = ss.bernoulli(0.01),                            # Initial seed infections
            beta = ss.beta(0.25),                                      # Infection probability
            p_latent_fast = ss.bernoulli(0.1),                         # Probability of latent fast vs slow
            rate_LS_to_presym       = ss.perday(3e-5),                 # Latent Slow to Active Pre-Symptomatic            
            rate_LF_to_presym       = ss.perday(6e-3),                 # Latent Fast to Active Pre-Symptomatic            
            rate_presym_to_active   = ss.perday(3e-2),                 # Pre-symptomatic to symptomatic            
            rate_active_to_clear    = ss.perday(2.4e-4),               # Active to natural clearance            
            rate_exptb_to_dead      = ss.perday(0.15 * 4.5e-4),        # Extra-Pulmonary TB to Dead            
            rate_smpos_to_dead      = ss.perday(4.5e-4),               # Smear Positive TB to Dead            
            rate_smneg_to_dead      = ss.perday(0.3 * 4.5e-4),         # Smear Negative TB to Dead            
            rate_treatment_to_clear = ss.peryear(12/2),                # Treatment clearance rate

            active_state = ss.choice(a=[TBS.ACTIVE_EXPTB, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG],
                                     p=[0.1, 0.65, 0.25]),           # Distribution for active TB types

            # Relative transmissibility of each state
            rel_trans_presymp   = 0.1,
            rel_trans_smpos     = 1.0,
            rel_trans_smneg     = 0.3,
            rel_trans_exptb     = 0.05,
            rel_trans_treatment = 0.5,  # For those on treatment

            rel_sus_latentslow = 0.5,  # Relative susceptibility for slow progressors

            cxr_asymp_sens = 1.0,      # CXR sensitivity for asymptomatic screening

            reltrans_het = ss.constant(v=1.0),
        )

        # Validate that all rate parameters are of type TimePar
        for k, v in self.pars.items():
            if k.startswith('rate_'):
                assert isinstance(v, ss.TimePar), (
                    'Rate parameters for TB must be TimePars, e.g. ss.perday(x)'
                )

    def define_tb_states(self):
        self.define_states(
            ss.FloatArr('state', default=TBS.NONE),
            ss.FloatArr('latent_tb_state', default=TBS.NONE),
            ss.FloatArr('active_tb_state', default=TBS.NONE),
            ss.FloatArr('rr_activation', default=1.0),
            ss.FloatArr('rr_clearance', default=1.0),
            ss.FloatArr('rr_death', default=1.0),
            ss.State('on_treatment', default=False),
            ss.State('ever_infected', default=False),
            ss.FloatArr('ti_presymp'),
            ss.FloatArr('ti_active'),
            ss.FloatArr('ti_cur', default=0),
            ss.FloatArr('reltrans_het', default=1.0),
        )

# ---------------------------------------------------------------------------
# Transition and Probability Handling Mixin
# ---------------------------------------------------------------------------
class TBTransitionMixin:
    def _apply_transition(self, transition_func, sim, uids):
        """Helper: Given a transition function returning per-agent probabilities,
           sample which agents transition.
        """
        if len(uids) == 0:
            return ss.uids()
        probs = transition_func(sim, uids)
        random_values = np.random.rand(len(uids))
        return uids[random_values < probs]

    # Transition probability functions as instance methods
    def p_latent_to_presym(self, sim, uids):
        assert np.all(np.isin(self.state[uids], [TBS.LATENT_FAST, TBS.LATENT_SLOW]))
        rate = np.full(len(uids), self.pars.rate_LS_to_presym)
        rate[self.state[uids] == TBS.LATENT_FAST] = self.pars.rate_LF_to_presym
        rate *= self.rr_activation[uids]
        return 1 - np.exp(-rate)

    def p_presym_to_clear(self, sim, uids):
        assert np.all(self.state[uids] == TBS.ACTIVE_PRESYMP)
        rate = np.zeros(len(uids))
        rate[self.on_treatment[uids]] = self.pars.rate_treatment_to_clear
        return 1 - np.exp(-rate)

    def p_presym_to_active(self, sim, uids):
        assert np.all(self.state[uids] == TBS.ACTIVE_PRESYMP)
        rate = np.full(len(uids), self.pars.rate_presym_to_active)
        return 1 - np.exp(-rate)

    def p_active_to_clear(self, sim, uids):
        assert np.all(np.isin(self.state[uids],
                                [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]))
        rate = np.full(len(uids), self.pars.rate_active_to_clear)
        rate[self.on_treatment[uids]] = self.pars.rate_treatment_to_clear
        rate *= self.rr_clearance[uids]
        return 1 - np.exp(-rate)

    def p_active_to_death(self, sim, uids):
        assert np.all(np.isin(self.state[uids],
                                [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]))
        rate = np.full(len(uids), self.pars.rate_exptb_to_dead)
        rate[self.state[uids] == TBS.ACTIVE_SMPOS] = self.pars.rate_smpos_to_dead
        rate[self.state[uids] == TBS.ACTIVE_SMNEG] = self.pars.rate_smneg_to_dead
        rate *= self.rr_death[uids]
        return 1 - np.exp(-rate)

    def process_transitions(self):
        """Encapsulate all state transitions for the TB infection."""
        ti = self.ti

        # Latent -> Pre-symptomatic
        latent_uids = ((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST)).uids
        new_presymp_uids = self._apply_transition(self.p_latent_to_presym, self.sim, latent_uids)
        if len(new_presymp_uids):
            self.state[new_presymp_uids] = TBS.ACTIVE_PRESYMP
            self.ti_cur[new_presymp_uids] = ti
            self.ti_presymp[new_presymp_uids] = ti
            self.susceptible[new_presymp_uids] = False
        self.results['new_active'][ti] = len(new_presymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(
            self.sim.people.age[new_presymp_uids] >= 15
        )

        # Pre-symptomatic transitions: clearance and progression to active
        presym_uids = (self.state == TBS.ACTIVE_PRESYMP).uids
        new_clear_presymp_uids = self._apply_transition(self.p_presym_to_clear, self.sim, presym_uids)
        new_active_uids = self._apply_transition(self.p_presym_to_active, self.sim, presym_uids)
        if len(new_active_uids):
            active_state = self.active_tb_state[new_active_uids]
            self.state[new_active_uids] = active_state
            self.ti_cur[new_active_uids] = ti
            self.ti_active[new_active_uids] = ti

        # Active -> Clear (via recovery or treatment)
        active_uids = ((self.state == TBS.ACTIVE_SMPOS) |
                       (self.state == TBS.ACTIVE_SMNEG) |
                       (self.state == TBS.ACTIVE_EXPTB)).uids
        new_clear_active_uids = self._apply_transition(self.p_active_to_clear, self.sim, active_uids)
        new_clear_uids = ss.uids.cat(new_clear_presymp_uids, new_clear_active_uids)
        if len(new_clear_uids):
            self.susceptible[new_clear_uids] = True
            self.infected[new_clear_uids] = False
            self.state[new_clear_uids] = TBS.NONE
            self.ti_cur[new_clear_uids] = ti
            self.active_tb_state[new_clear_uids] = TBS.NONE
            self.ti_presymp[new_clear_uids] = np.nan
            self.ti_active[new_clear_uids] = np.nan
            self.on_treatment[new_clear_uids] = False

        # Active -> Death
        # Recompute active_uids after clearances
        active_uids = ((self.state == TBS.ACTIVE_SMPOS) |
                       (self.state == TBS.ACTIVE_SMNEG) |
                       (self.state == TBS.ACTIVE_EXPTB)).uids
        new_death_uids = self._apply_transition(self.p_active_to_death, self.sim, active_uids)
        if len(new_death_uids):
            self.sim.people.request_death(new_death_uids)
            self.state[new_death_uids] = TBS.DEAD
            self.ti_cur[new_death_uids] = ti
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(
            self.sim.people.age[new_death_uids] >= 15
        )

        # Update transmissibility based on state and heterogeneity
        self.update_transmissibility()
        self.reset_relative_rates()

    def update_transmissibility(self):
        # Reset to baseline transmissibility
        self.rel_trans[:] = 1
        state_reltrans = [
            (TBS.ACTIVE_PRESYMP, self.pars.rel_trans_presymp),
            (TBS.ACTIVE_EXPTB, self.pars.rel_trans_exptb),
            (TBS.ACTIVE_SMPOS, self.pars.rel_trans_smpos),
            (TBS.ACTIVE_SMNEG, self.pars.rel_trans_smneg),
        ]
        for state, multiplier in state_reltrans:
            uids = (self.state == state)
            self.rel_trans[uids] *= multiplier

        # Apply individual-level heterogeneity
        uids = self.infectious
        self.rel_trans[uids] *= self.reltrans_het[uids]

        # Further reduce transmissibility for those on treatment
        uids = self.on_treatment
        self.rel_trans[uids] *= self.pars.rel_trans_treatment

    def reset_relative_rates(self):
        uids = self.sim.people.auids
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

# ---------------------------------------------------------------------------
# Treatment Handling Mixin
# ---------------------------------------------------------------------------
class TBTreatmentMixin:
    def start_treatment(self, uids):
        """Start treatment for active TB cases."""
        if len(uids) == 0:
            return 0
        rst = self.state[uids]
        is_active = np.isin(rst, [TBS.ACTIVE_PRESYMP,
                                   TBS.ACTIVE_SMPOS,
                                   TBS.ACTIVE_SMNEG,
                                   TBS.ACTIVE_EXPTB])
        tx_uids = uids[is_active]
        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(
            self.sim.people.age[tx_uids] >= 15
        )
        if len(tx_uids) == 0:
            return 0
        self.on_treatment[tx_uids] = True
        self.rr_death[tx_uids] = 0  # Zero death rate on treatment
        self.rel_trans[tx_uids] *= self.pars.rel_trans_treatment
        return len(tx_uids)

    def step_die(self, uids):
        if len(uids) == 0:
            return
        super().step_die(uids)
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.rel_trans[uids] = 0

# ---------------------------------------------------------------------------
# Results Handling Mixin
# ---------------------------------------------------------------------------
class TBResultsMixin:
    def init_tb_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_latent_slow',         dtype=int, label='Latent Slow'),
            ss.Result('n_latent_fast',         dtype=int, label='Latent Fast'),
            ss.Result('n_active',              dtype=int, label='Active (Combined)'),
            ss.Result('n_active_presymp',      dtype=int, label='Active Pre-Symptomatic'),
            ss.Result('n_active_presymp_15+',  dtype=int, label='Active Pre-Symptomatic, 15+'),
            ss.Result('n_active_smpos',        dtype=int, label='Active Smear Positive'),
            ss.Result('n_active_smpos_15+',    dtype=int, label='Active Smear Positive, 15+'),
            ss.Result('n_active_smneg',        dtype=int, label='Active Smear Negative'),
            ss.Result('n_active_smneg_15+',    dtype=int, label='Active Smear Negative, 15+'),
            ss.Result('n_active_exptb',        dtype=int, label='Active Extra-Pulmonary'),
            ss.Result('n_active_exptb_15+',    dtype=int, label='Active Extra-Pulmonary, 15+'),
            ss.Result('new_active',            dtype=int, label='New Active'),
            ss.Result('new_active_15+',        dtype=int, label='New Active, 15+'),
            ss.Result('cum_active',            dtype=int, label='Cumulative Active'),
            ss.Result('cum_active_15+',        dtype=int, label='Cumulative Active, 15+'),
            ss.Result('new_deaths',            dtype=int, label='New Deaths'),
            ss.Result('new_deaths_15+',        dtype=int, label='New Deaths, 15+'),
            ss.Result('cum_deaths',            dtype=int, label='Cumulative Deaths'),
            ss.Result('cum_deaths_15+',        dtype=int, label='Cumulative Deaths, 15+'),
            ss.Result('n_infectious',          dtype=int, label='Number Infectious'),
            ss.Result('n_infectious_15+',      dtype=int, label='Number Infectious, 15+'),
            ss.Result('prevalence_active',     dtype=float, scale=False, label='Prevalence (Active)'),
            ss.Result('incidence_kpy',         dtype=float, scale=False, label='Incidence per 1,000 person-years'),
            ss.Result('deaths_ppy',            dtype=float, label='Death per person-year'),
            ss.Result('n_reinfected',          dtype=int, label='Number reinfected'),
            ss.Result('new_notifications_15+', dtype=int, label='New TB notifications, 15+'),
            ss.Result('n_detectable_15+',      dtype=float, label='SmPos + SMNeg + cxr_asymp_sens * pre-symptomatic'),
        )

    def update_tb_results(self):
        super().update_results()
        res = self.results
        ti = self.ti
        ti_infctd = self.ti_infected
        dty = self.sim.t.dt_year
        n_alive = np.count_nonzero(self.sim.people.alive)

        res.n_latent_slow[ti]       = np.count_nonzero(self.state == TBS.LATENT_SLOW)
        res.n_latent_fast[ti]       = np.count_nonzero(self.state == TBS.LATENT_FAST)
        res.n_active_presymp[ti]    = np.count_nonzero(self.state == TBS.ACTIVE_PRESYMP)
        res['n_active_presymp_15+'][ti] = np.count_nonzero(
            (self.sim.people.age >= 15) & (self.state == TBS.ACTIVE_PRESYMP)
        )
        res.n_active_smpos[ti]      = np.count_nonzero(self.state == TBS.ACTIVE_SMPOS)
        res['n_active_smpos_15+'][ti] = np.count_nonzero(
            (self.sim.people.age >= 15) & (self.state == TBS.ACTIVE_SMPOS)
        )
        res.n_active_smneg[ti]      = np.count_nonzero(self.state == TBS.ACTIVE_SMNEG)
        res['n_active_smneg_15+'][ti] = np.count_nonzero(
            (self.sim.people.age >= 15) & (self.state == TBS.ACTIVE_SMNEG)
        )
        res.n_active_exptb[ti]      = np.count_nonzero(self.state == TBS.ACTIVE_EXPTB)
        res['n_active_exptb_15+'][ti] = np.count_nonzero(
            (self.sim.people.age >= 15) & (self.state == TBS.ACTIVE_EXPTB)
        )
        res.n_active[ti]            = np.count_nonzero(
            np.isin(self.state, [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS,
                                  TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
        )
        res.n_infectious[ti]        = np.count_nonzero(self.infectious)
        res['n_infectious_15+'][ti] = np.count_nonzero(
            self.infectious & (self.sim.people.age >= 15)
        )
        res['n_detectable_15+'][ti] = np.dot(
            self.sim.people.age >= 15,
            np.isin(self.state, [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG]) +
            self.pars.cxr_asymp_sens * (self.state == TBS.ACTIVE_PRESYMP)
        )

        if n_alive > 0:
            res.prevalence_active[ti] = res.n_active[ti] / n_alive
            res.incidence_kpy[ti]     = 1_000 * np.count_nonzero(ti_infctd == ti) / (n_alive * dty)
            res.deaths_ppy[ti]        = res.new_deaths[ti] / (n_alive * dty)

    def finalize_tb_results(self):
        super().finalize_results()
        res = self.results
        res['cum_deaths']     = np.cumsum(res['new_deaths'])
        res['cum_deaths_15+'] = np.cumsum(res['new_deaths_15+'])
        res['cum_active']     = np.cumsum(res['new_active'])
        res['cum_active_15+'] = np.cumsum(res['new_active_15+'])

# ---------------------------------------------------------------------------
# Plotting Mixin
# ---------------------------------------------------------------------------
class TBPlotMixin:
    def plot_tb_results(self):
        fig = plt.figure()
        for rkey in self.results.keys():
            if rkey == 'timevec':
                continue
            plt.plot(self.results['timevec'], self.results[rkey], label=rkey.title())
        plt.legend()
        return fig

# ---------------------------------------------------------------------------
# Main TB Class (Combining Mixins)
# ---------------------------------------------------------------------------
class TB(TBParameterMixin,
         TBTransitionMixin,
         TBTreatmentMixin,
         TBResultsMixin,
         TBPlotMixin,
         ss.Infection):
    
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_tb_parameters()
        self.define_tb_states()
        self.update_pars(pars, **kwargs)
        # Additional initialization can be added here if needed

    @property
    def infectious(self):
        """
        Return a boolean array indicating infectious individuals.
        Infectious if in any of the active states or on treatment.
        """
        return (self.on_treatment) | \
               (self.state == TBS.ACTIVE_PRESYMP) | \
               (self.state == TBS.ACTIVE_SMPOS) | \
               (self.state == TBS.ACTIVE_SMNEG) | \
               (self.state == TBS.ACTIVE_EXPTB)

    def set_prognoses(self, uids, from_uids=None):
        super().set_prognoses(uids, from_uids)
        p = self.pars

        # Determine latent fast vs slow for new infections
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.latent_tb_state[fast_uids] = TBS.LATENT_FAST
        self.latent_tb_state[slow_uids] = TBS.LATENT_SLOW
        self.state[slow_uids] = TBS.LATENT_SLOW
        self.state[fast_uids] = TBS.LATENT_FAST
        self.ti_cur[uids] = self.ti

        new_uids = uids[~self.infected[uids]]
        reinfected_uids = uids[(self.infected[uids]) & (self.state[uids] == TBS.LATENT_FAST)]
        self.results['n_reinfected'][self.ti] = len(reinfected_uids)

        # Update infection flags and parameters
        self.susceptible[fast_uids] = False
        self.infected[uids] = True
        self.rel_sus[slow_uids] = p.rel_sus_latentslow
        self.active_tb_state[uids] = self.pars.active_state.rvs(uids)
        self.reltrans_het[uids] = p.reltrans_het.rvs(uids)
        self.ti_infected[new_uids] = self.ti
        self.ti_infected[reinfected_uids] = self.ti
        self.ever_infected[uids] = True

    def step(self):
        # First, perform any generic infection model updates
        super().step()
        # Then process TB-specific transitions
        self.process_transitions()

    def init_results(self):
        self.init_tb_results()

    def update_results(self):
        self.update_tb_results()

    def finalize_results(self):
        self.finalize_tb_results()

    def plot(self):
        return self.plot_tb_results()

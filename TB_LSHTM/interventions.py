# tb_intervention.py

import starsim as ss
import sciris as sc
import numpy as np
from tbsim import TB as tbsim_TB, TBS
import datetime as dt

__all__ = ['CaseFinding', 'time_varying_parameter', 'sigmoidally_varying_parameter', 'set_beta']

class CaseFinding(ss.Intervention):
    """
    Intervention that simulates case finding in active TB cases. In more detail, this intervention --
    - Identifies the active TB individuals. This could be any one of the following states:
        ACTIVE_PRESYMP  (Active TB, pre-symptomatic)
        ACTIVE_SMPOS    (Active TB, smear positive)
        ACTIVE_SMNEG    (Active TB, smear negative)
        ACTIVE_EXPTB    (Active TB, extra-pulmonary)
    - Assign test sensitivity to accurately identify as Active TB
    - With some coverage rate, the intervention identifies the active TB cases and assigns them to treatment.
    - People who are found under active case finding are treated with a certain probability.
    """
    def __init__(self, pars=None, *args, **kwargs):
        """
        Initialize the intervention.
        """
        super().__init__()

        # Updated default parameters with time-aware p_found
        self.define_pars(
            p_treat = ss.bernoulli(p=1),

            date_cov = { # Numbers from Table 1, row "Persons who gave oral consent to participate — no. (%)"
                ss.date('2014-06-01'): 0.844,
                ss.date('2015-06-01'): 0.80,
                ss.date('2016-06-01'): 0.779,
                ss.date('2017-06-01'): 0.743,
            },
            interp = False,
            interp_year_max = None,

            age_min = 15,
            age_max = None,

            # Test sensitivity
            test_sens = {
                TBS.ACTIVE_SMPOS: 1,
                TBS.ACTIVE_PRESYMP: 0.9,
                TBS.ACTIVE_SMNEG: 0.8,
                TBS.ACTIVE_EXPTB: 0.1,
            },
        )
        self.update_pars(pars, **kwargs)

        # Convert datetime to float
        self.pars.date_cov = {
            t.to_year() if isinstance(t, dt.date) else t : v
            for t, v in self.pars.date_cov.items()
        }

        self.visit = ss.bernoulli(p=self.p_visit)
        self.test = ss.bernoulli(p=self.p_pos_test)
        self.dates = [] # Dates on which non-interpolated coverage was applied

        return

    @staticmethod
    def p_visit(self, sim, uids):
        # Determine which agents to visit based on coverage
        
        # NOTE: 
        # When date_cov inputs are provided as Calibpars, 
        # the __init__ does not convert them to float years. 
        # Explicitly making the conversion here
        if any(isinstance(t, dt.date) for t in self.pars.date_cov.keys()):
            # Convert datetime to float
            self.pars.date_cov = {
                t.to_year():v if isinstance(t, dt.date) else t 
                for t, v in self.pars.date_cov.items()
                }
        
        years = np.array(list(self.pars.date_cov.keys()))
        year = sim.t.now('year')
        if self.pars.interp:
            if self.pars.interp_year_max is not None:
                year = min(year, self.pars.interp_year_max)
            p_visit = np.interp(year, years, list(self.pars.date_cov.values()))
        else:
            in_year = (year >= years) & (year < years + self.t.dt_year)
            if in_year.any():
                dc_year = years[in_year][0]
                p_visit = self.pars.date_cov[dc_year]
                self.dates.append(year)
            else:
                p_visit = 0
        return p_visit

    @staticmethod
    def p_pos_test(self, sim, uids):
        p = np.zeros(len(uids))

        tb_state = sim.diseases.tb.state[uids]
        for tbs, sensitivity in self.pars.test_sens.items():
            p[tb_state == tbs] = sensitivity
        return p

    def init_results(self):
        self.define_results(
            ss.Result('n_elig', dtype=int, label='Number eligible', scale=True),
            ss.Result('n_tested', dtype=int, label='Number tested', scale=True),
            ss.Result('n_positive', dtype=int, label='Number positive', scale=True),
            ss.Result('expected_positive', dtype=float, label='Expected number positive', scale=True),
            ss.Result('n_treated', dtype=int, label='Number treated', scale=True),

            ss.Result('n_positive_presymp', dtype=int, label='Number positive: Presymp', scale=True),
            ss.Result('n_positive_smpos', dtype=int, label='Number positive: SmPos', scale=True),
            ss.Result('n_positive_smneg', dtype=int, label='Number positive: SmNeg', scale=True),
            ss.Result('n_positive_exp', dtype=int, label='Number positive: Exp', scale=True),

            ss.Result('n_positive_via_LF', dtype=int, label='Number positive: Fast', scale=True),
            ss.Result('n_positive_via_LS', dtype=int, label='Number positive: Slow', scale=True),
            ss.Result('n_positive_via_LF_dur', dtype=float, label='Mean dur from inf (via fast)', scale=False),
            ss.Result('n_positive_via_LS_dur', dtype=float, label='Mean dur from inf (via slow)', scale=False),
        )
        return

    def step(self):
        """ Apply the intervention """

        sim = self.sim
        tb = sim.diseases['tb']

        # Filter by treatment and age
        elig = ~tb.on_treatment
        if self.pars.age_min is not None:
            elig = elig & (sim.people.age >= self.pars.age_min)
        if self.pars.age_max is not None:
            elig = elig & (sim.people.age < self.pars.age_max)

        visit_uids = self.visit.filter(elig)

        expected_pos = 0
        if len(visit_uids) > 0:
            # Perform the test on the eligible individuals to find cases
            positive_uids = self.test.filter(visit_uids)
            expected_pos = np.sum(self.test._pars['p'])

            # Apply treatment to the positive cases
            treated_uids = self.pars.p_treat.filter(positive_uids)
            tb.start_treatment(treated_uids)
        else:
            positive_uids = ss.uids()
            treated_uids = ss.uids()

        # Update the results 
        ti = self.t.ti
        self.results.n_elig[ti] = np.count_nonzero(elig)
        self.results.n_tested[ti] = len(visit_uids)
        self.results.expected_positive[ti] = expected_pos
        self.results.n_positive[ti] = len(positive_uids)
        self.results.n_treated[ti] = len(treated_uids)

        if len(positive_uids) == 0:
            return

        # The following results are only relevant for the TBsim intrahost model
        if isinstance(tb, tbsim_TB):
            presym = tb.state[positive_uids] == TBS.ACTIVE_PRESYMP
            smpos = tb.state[positive_uids] == TBS.ACTIVE_SMPOS
            smneg = tb.state[positive_uids] == TBS.ACTIVE_SMNEG
            exp = tb.state[positive_uids] == TBS.ACTIVE_EXPTB

            self.results.n_positive_presymp[ti] = np.count_nonzero(presym)
            self.results.n_positive_smpos[ti] = np.count_nonzero(smpos)
            self.results.n_positive_smneg[ti] = np.count_nonzero(smneg)
            self.results.n_positive_exp[ti] = np.count_nonzero(exp)

            via_fast = tb.latent_tb_state[positive_uids] == TBS.LATENT_FAST
            self.results.n_positive_via_LF[ti] = np.count_nonzero(via_fast)
            if np.count_nonzero(via_fast) > 0:
                dur = (ti - tb.ti_infected[positive_uids[via_fast]]) * self.t.dt_year
                self.results.n_positive_via_LF_dur[ti] = np.mean(dur)

            via_slow = tb.latent_tb_state[positive_uids] == TBS.LATENT_SLOW
            self.results.n_positive_via_LS[ti] = np.count_nonzero(via_slow)
            if np.count_nonzero(via_slow) > 0:
                dur = (ti - tb.ti_infected[positive_uids[via_slow]]) * self.t.dt_year
                self.results.n_positive_via_LS_dur[ti] = np.mean(dur)

        return



#%% Intervention to reduce transmission and progression of the TB disease
class time_varying_parameter(ss.Intervention):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            tb_parameter = 'beta', # The parameter of the TB module to change
            rc_endpoint = 0.5,     # Will linearly interpolate from 1 at start to rc_endpoint at stop
            start = ss.date('1995-01-01'),
            stop = ss.date('2014-01-01'),
        )
        self.update_pars(pars, **kwargs)
        return
    
    def init_pre(self, sim, **kwargs):
        super().init_pre(sim, **kwargs)

        # Store the original value
        self.original_value = sim.diseases.tb.pars[self.pars.tb_parameter]

        # Make simulation and input time of the same type
        self.input_year = [t.to_year() for t in [self.pars.start, self.pars.stop]]
        return
 
    def step(self):
        # Interpolate the values and modify the parameter
        rc = np.interp(self.t.now('year'), self.input_year, [1, self.pars.rc_endpoint])
        self.sim.diseases.tb.pars[self.pars.tb_parameter] = self.original_value * rc
        if self.pars.tb_parameter == 'beta':
            for net in self.sim.networks: # Set MP too
                if isinstance(net, ss.MixingPool):
                    net.pars.beta = self.original_value * rc
        return

class sigmoidally_varying_parameter(ss.Intervention):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            tb_parameters = ['beta'], # The parameter of the TB module to change
            x_initial = 1,     # Will linearly interpolate from 1 at start to rc_endpoint at stop
            x_final = 1,
            dur_years = 15, # from 10% to 90% of 0 to 1 sigmoid value
            midpoint = ss.date('1995-01-01'),
            stop_year = None,
        )
        self.update_pars(pars, **kwargs)

        self.pars.tb_parameters = sc.promotetolist(self.pars.tb_parameters)

        # Calculate k (steepness parameter) from the slope at the midpoint
        #self.k = (4 * self.pars.slope_peryear_at_mid) / (self.pars.x_final - self.pars.x_initial)
        y1, y2 = 0.1, 0.9
        ln_term = np.log(y2 / (1 - y2)) - np.log(y1 / (1 - y1))
        self.k = ln_term / self.pars.dur_years
        return
    
    def init_pre(self, sim, **kwargs):
        super().init_pre(sim, **kwargs)

        # Store the original value
        self.original_value = {}
        for p in self.pars.tb_parameters:
            assert p in sim.diseases.tb.pars, f'Parameter {p} not found in TB module'
            val = sim.diseases.tb.pars[p]
            if isinstance(val, ss.Dist):
                if isinstance(val, ss.expon):
                    val = val.pars['scale']
                else:
                    raise NotImplementedError(f'Parameter {p} is a distribution of type {type(val)}')
            self.original_value[p] = val

        # HACK, see Starsim #840 (https://github.com/starsimhub/starsim/issues/840)
        if not isinstance(self.pars.midpoint, ss.date):
            self.pars.midpoint = ss.date._reset_class(self.pars.midpoint)

        # Make simulation and input time of the same type
        self.t_mid = self.pars.midpoint.to_year()
        return
 
    def step(self):

        if self.pars.stop_year is not None and self.t.now('year') > self.pars.stop_year:
            return # Don't do anything after the stop year

        # Interpolate the values and modify the parameter
        t = self.t.now('year')
        x = self.pars.x_initial + (self.pars.x_final - self.pars.x_initial) / (1 + np.exp(-self.k * (t - self.t_mid)))

        for p in self.pars.tb_parameters:
            val = self.sim.diseases.tb.pars[p]
            if isinstance(val, ss.Dist):
                if isinstance(val, ss.expon):
                    self.sim.diseases.tb.pars[p].pars['scale'] = self.original_value[p] * x
                else:
                    raise NotImplementedError(f'Parameter {p} is a distribution of type {type(val)}')
            else:
                self.sim.diseases.tb.pars[p] = self.original_value[p] * x
        return x


class set_beta(ss.Intervention):

    # Set beta to value in year
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            year = 2000,
            x_beta = 1,
        )
        self.update_pars(pars, **kwargs)

    def step(self):
        year = self.sim.t.now('year')
        if (year >= self.pars.year) & (year < self.pars.year + self.t.dt_year):
            self.sim.diseases.tb.pars['beta'] *= self.pars.x_beta

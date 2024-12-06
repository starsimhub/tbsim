# tb_intervention.py

import starsim as ss
import sciris as sc
import numpy as np
from tbsim import TBS
import datetime as dt

__all__ = ['ActiveCaseFinding', 'Product', 'TBVaccinationCampaign']

class ActiveCaseFinding(ss.Intervention):
    """
    Intervention that simulates active case finding in active Tb cases. In more detail, this intervention --
    - Identifies the active TB individuals. This could be any one of the following states:
        ACTIVE_PRESYMP  (Active TB, pre-symptomatic)
        ACTIVE_SMPOS    (Active TB, smear positive)
        ACTIVE_SMNEG    (Active TB, smear negative)
        ACTIVE_EXPTB    (Active TB, extra-pulmonary)
    - Assign test sensitivity to accurately identify as Active Tb
    - With some coverage rate, the intervention identifies the active TB cases and assigns them to treatment.
    - People who are found under active case finding are treated with a certain probability.
    """
    def __init__(self, pars=None, *args, **kwargs):
        """
        Initialize the intervention.
        """
        super().__init__(*args, **kwargs)

        # Updated default parameters with time-aware p_found
        self.define_pars(
            p_treat = ss.bernoulli(p=1),

            date_cov = {
                sc.date('2014-06-01'): 0.6,
                sc.date('2015-06-01'): 0.7,
                sc.date('2016-06-01'): 0.64,
                sc.date('2017-06-01'): 0.64, # setting this to the same as 2016 for now
            },

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
            sc.datetoyear(t):v if isinstance(t, dt.date) else t 
            for t, v in self.pars.date_cov.items()
        }

        self.test = ss.bernoulli(p=self.p_pos_test)

        return

    @staticmethod
    def p_pos_test(self, sim, uids):
        p = np.zeros(len(uids))

        tb_state = sim.diseases.tb.state[uids]
        for tbs, sensitivity in self.pars.test_sens.items():
            p[tb_state == tbs] = sensitivity
        return p

    def init_results(self):
        npts = len(self.pars.date_cov)
        self.define_results(
            ss.Result('time', dtype=float, shape=npts, label='Time', scale=True),
            ss.Result('n_elig', dtype=int, shape=npts, label='Number eligible', scale=True),
            ss.Result('n_found', dtype=int, shape=npts, label='Number found',scale=True),
            ss.Result('n_treated', dtype=int, shape=npts, label='Number treated', scale=True),
        )
        return

    def step(self):
        """ Apply the intervention """

        sim = self.sim

        # Determine when the intervention is active and return if nothing to do
        years = np.array(list(self.pars.date_cov.keys()))
        sim_year = self.t.now('year')
        is_active = (
            #(sim_year >= years) & (sim_year < years + self.t.dt_year)         # 50-age-specific-tb-reviewed
            (sim.t.now('year') >= years) & (sim.t.now('year') < years + self.sim.t.dt_year)
        )
        if not np.any(is_active):
            return

        tb = sim.diseases['tb']

        # Filter by treatment and age
        elig = ~tb.on_treatment
        if self.pars.age_min is not None:
            elig = elig & (sim.people.age >= self.pars.age_min)
        if self.pars.age_max is not None:
            elig = elig & (sim.people.age < self.pars.age_max)

        # Perform the test on the eligible individuals to find cases
        found_uids = self.test.filter(elig)

        # Apply treatment to the found cases
        treated_uids = self.pars.p_treat.filter(found_uids)
        tb.start_treatment(treated_uids)

        # Update the results 
        timepoint = np.where(is_active)[0][0]
        # self.results.time[timepoint] = sim_year                              # 50-age-specific-tb-reviewed
        self.results.time[timepoint] = sim.t.now('year')
        self.results.n_elig[timepoint] = np.sum(elig)
        self.results.n_found[timepoint] = len(found_uids)
        self.results.n_treated[timepoint] = len(treated_uids)

        return


class Product(ss.Module):
    """
    Class to define a vaccine product with specific attributes.
    """
    def __init__(self, name, efficacy, doses):
        self.name = name
        self.efficacy = efficacy
        self.doses = doses
        
    def init_pre(self, sim):
        if not self.initialized:
            super().init_pre(sim)
        else:
            return
    def __repr__(self):
        return f"Product(name={self.name}, efficacy={self.efficacy}, doses={self.doses})"
    
    def administer(self, people, inds):
        """ Adminster a Product - implemented by derived classes """
        print("vaccine administered to people")
        
        return

class TBVaccinationCampaign(ss.Intervention):
    """
    Base class for any intervention that uses campaign delivery; handles interpolation of input years.
    """

    def __init__(self, year=1900, product=None, rate =.015, target_gender='All', target_age=10, target_state=None, new_value_fraction=1, prob=None, *args, **kwargs):
        if product is None:
            raise NotImplementedError('No product specified')
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        self.target_gender = target_gender
        self.target_age = target_age
        self.prob = sc.promotetoarray(prob)
        self.product = product
        self.target_state = target_state
        self.new_value_fraction = new_value_fraction
        super().__init__(*args, **kwargs)
        self.p = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        return
    
    def update(self, sim):
        if sim.year < self.year[0]:
            return
        if self.product is None:
            raise NotImplementedError('No product specified')

        tb = sim.diseases['tb']
        ppl = sim.people
        
        # eligible = (tb.state == self.target_state) & ppl.alive & (ppl.age >= self.target_age) & (ppl.gender == self.target_gender)
        eligible = (tb.state == self.target_state) & ppl.alive
        
        eligible_uids = eligible.uids
        
        change_uids = self.p.filter(eligible_uids)
        
        # TODO: Add the actual value of the product's effectiveness here...
        tb.rel_LS_prog[change_uids] = tb.rel_LS_prog[change_uids]*0.9 # *self.product.efficacy   
        tb.rel_LF_prog[change_uids] = tb.rel_LF_prog[change_uids]*0.9  # *self.product.efficacy   
        
        return len(change_uids)

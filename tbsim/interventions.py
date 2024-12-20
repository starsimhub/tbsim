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
        super().__init__()

        # Updated default parameters with time-aware p_found
        self.define_pars(
            p_treat = ss.bernoulli(p=1),

            date_cov = { # Numbers from Table 1, row "Persons who gave oral consent to participate — no. (%)"
                sc.date('2014-06-01'): 0.844,
                sc.date('2015-06-01'): 0.80,
                sc.date('2016-06-01'): 0.779,
                sc.date('2017-06-01'): 0.743,
            },
            interp = False,

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

        self.visit = ss.bernoulli(p=self.p_visit)
        self.test = ss.bernoulli(p=self.p_pos_test)

        return

    @staticmethod
    def p_visit(self, sim, uids):
        # Determine which agents to visit based on coverage
        years = np.array(list(self.pars.date_cov.keys()))
        year = sim.t.now('year')
        if self.pars.interp:
            p_visit = np.interp(year, years, list(self.pars.date_cov.values()))
        else:
            in_year = (year >= years) & (year < years + self.t.dt_year)
            if in_year.any():
                year = years[in_year][0]
                p_visit = self.pars.date_cov[year]
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

        if len(visit_uids) > 0:
            # Perform the test on the eligible individuals to find cases
            positive_uids = self.test.filter(visit_uids)

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
        self.results.n_positive[ti] = len(positive_uids)
        self.results.n_treated[ti] = len(treated_uids)

        if len(positive_uids) == 0:
            return

        presym = tb.state[positive_uids] == TBS.ACTIVE_PRESYMP
        smpos = tb.state[positive_uids] == TBS.ACTIVE_SMPOS
        smneg = tb.state[positive_uids] == TBS.ACTIVE_SMNEG
        exp = tb.state[positive_uids] == TBS.ACTIVE_EXPTB

        self.results.n_positive_presymp[ti] = np.count_nonzero(presym)
        self.results.n_positive_smpos[ti] = np.count_nonzero(smpos)
        self.results.n_positive_smneg[ti] = np.count_nonzero(smneg)
        self.results.n_positive_exp[ti] = np.count_nonzero(exp)

        via_fast = tb.latent_tb_state[positive_uids] == TBS.LATENT_FAST
        via_slow = tb.latent_tb_state[positive_uids] == TBS.LATENT_SLOW
        self.results.n_positive_via_LF[ti] = np.count_nonzero(via_fast)
        self.results.n_positive_via_LS[ti] = np.count_nonzero(via_slow)

        if np.count_nonzero(via_fast) > 0:
            dur = (self.t.ti - tb.ti_infected[positive_uids[via_fast]])*self.t.dt_year
            self.results.n_positive_via_LF_dur[ti] = np.mean(dur)

        if np.count_nonzero(via_slow) > 0:
            dur = (self.t.ti - tb.ti_infected[positive_uids[via_slow]])*self.t.dt_year
            self.results.n_positive_via_LS_dur[ti] = np.mean(dur)

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

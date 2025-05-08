import starsim as ss
import sciris as sc
import numpy as np
from tbsim import TBS
import datetime as dt
import tbsim as mtb
import pandas as pd

__all__ = ['TbCascadeIntervention', 'TBTestScenario']


class TBTestScenario:
    BASE = 'base'
    SCENARIO = 'scenario'
    BENCHMARK = 'benchmark'


class TbCascadeIntervention(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.define_pars(
            mode=TBTestScenario.SCENARIO,
            scenario_id=1,
            screen_sens_age_hiv=None,
            test_scenarios=None,
            min_age=0,
            max_age=200,
            start=ss.date('2020-01-01'),
            stop=ss.date('2035-12-31'),
        )
        self.update_pars(pars, **kwargs)

        self.handlers = {
            TBTestScenario.BASE: self._apply_base_test,
            TBTestScenario.SCENARIO: self._apply_scenario_test,
            TBTestScenario.BENCHMARK: self._apply_benchmark_test,
        }

        self.pediatric_outcomes = []

    def step(self):
        t = self.sim.now
        if t < self.pars.start or t > self.pars.stop:
            return

        tb = self.sim.diseases.tb  # TB infection object
        people = self.sim.people

        # Get active symptomatic individuals
        active_tb_uids = (tb.state == TBS.ACTIVE_SMNEG) | (tb.state == TBS.ACTIVE_SMPOS) | (tb.state == TBS.ACTIVE_EXPTB)
        for uid in active_tb_uids.uids:
            person = people[uid]
            if not person.flags.get('sought_care'):
                self._seek_care(uid)

    def _seek_care(self, uid):
        person = self.sim.people[uid]
        person.flags['sought_care'] = True

        if self._screen(uid):
            handler = self.handlers.get(self.pars.mode)
            if handler is None:
                raise ValueError(f"Unsupported test mode: {self.pars.mode}")
            test_positive = handler(uid)
            self._handle_result(uid, test_positive)
        else:
            person.flags['returned_to_community'] = True

    def _screen(self, uid):
        person = self.sim.people[uid]
        age_group = 'under5' if person.age < 5 else '5plus'
        hiv_status = 'HIV+' if person.hiv_positive else 'HIV-'
        sens, spec = self.pars.screen_sens_age_hiv[age_group][hiv_status]
        has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
        return self._binary_test(has_tb, sens, spec)

    def _apply_scenario_test(self, uid):
        test_def = self.pars.test_scenarios[self.pars.scenario_id]
        has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
        return self._binary_test(has_tb, test_def['sensitivity'], test_def['specificity'])

    def _apply_base_test(self, uid):
        has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
        return self._binary_test(has_tb, 0.89, 0.98)  # GeneXpert default

    def _apply_benchmark_test(self, uid):
        test_def = self.pars.test_scenarios[self.pars.scenario_id]
        has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
        return self._binary_test(has_tb, test_def['sensitivity'], test_def['specificity'])

    def _handle_result(self, uid, test_positive):
        person = self.sim.people[uid]
        tb = self.sim.diseases.tb

        if test_positive:
            tb.start_treatment(ss.uids(uid))

            for hh_member in person.household:
                if not hh_member.flags.get('sought_care'):
                    self._seek_care(hh_member.uid)

                if hh_member.hiv_positive or hh_member.age < 5:
                    if tb.state[hh_member.uid] not in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]:
                        hh_member.initiate_tpt()
        else:
            person.flags['returned_to_community'] = True

        if person.age < 15:
            self._track_pediatric(person)

    def _track_pediatric(self, person):
        tb = self.sim.diseases.tb
        uid = person.uid
        self.pediatric_outcomes.append({
            'time': self.sim.t,
            'incidence': tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB],
            'mortality': tb.state[uid] == TBS.DEAD,
            'tpt': person.flags.get('received_tpt', False),
            'treatment_success': person.flags.get('tb_treatment_success', False),
        })

    def finalize(self):
        self._summarize_outcomes(self.sim)

    def _summarize_outcomes(self, sim):
        df = pd.DataFrame(self.pediatric_outcomes)
        if df.empty:
            sim.results['tb_pediatric_outcomes'] = pd.DataFrame()
            return
        df['year'] = df['time'] // 365
        sim.results['tb_pediatric_outcomes'] = df.groupby('year').sum()

    def _binary_test(self, condition, sens, spec):
        return np.random.rand() < sens if condition else np.random.rand() > spec


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

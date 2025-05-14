import starsim as ss
import sciris as sc
import numpy as np
from tbsim import TBS
import datetime as dt
import tbsim as mtb
import pandas as pd

__all__ = ['TPTInitiation', 'BCGProtection', 'Product', 'TBVaccinationCampaign', 'get_extrastates']


def get_extrastates():
    exs = [ss.State('sought_care', default=False),
        ss.State('returned_to_community', default=False),
        ss.State('received_tpt', default=False),
        ss.State('tb_treatment_success', default=False),
        ss.State('tested', default=False),
        ss.State('test_result', default=np.nan),
        ss.State('diagnosed', default=False),
        ss.State('on_tpt', default=True),
        ss.State('tb_smear', default=False),
        ss.State('hiv_positive', default=False),
        ss.State('eptb', default=False),
        ss.State('symptomatic', default=False),
        ss.State('presymptomatic', default=False),
        ss.State('non_symptomatic', default=True),
        ss.State('screen_negative', default=True),
        ss.State('household_contact', default=False),
        ss.FloatArr('vaccination_year', default=np.nan),]
    return exs


class TPTInitiation(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            p_tpt=ss.bernoulli(1.0),
            tpt_duration=2.0,
            max_age=5,
            hiv_status_threshold=False,
            p_3HP=0.3,
            start=ss.date('2000-01-01'),
        )
        self.update_pars(pars, **kwargs)

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb
        eligible = (~tb.on_treatment) & ppl['household_contact'] & (ppl['screen_negative'] | ppl['non_symptomatic'])
        #eligible &= (ppl.age < self.pars.max_age) | (ppl.hiv_positive == self.pars.hiv_status_threshold)

        tpt_candidates = self.pars.p_tpt.filter(eligible.uids)

        if len(tpt_candidates):
            use_3HP = sim.year >= self.pars.start.year
            assigned_3HP = np.random.rand(len(tpt_candidates)) < self.pars.p_3HP if use_3HP else np.zeros(len(tpt_candidates), dtype=bool)

            if not hasattr(tb, 'on_treatment_duration'):
                tb.define_states(ss.FloatArr('on_treatment_duration', default=0.0))

            tb.start_treatment(tpt_candidates)
            tb.on_treatment_duration[tpt_candidates] = self.pars.tpt_duration
            ppl['on_tpt'][tpt_candidates] = True

    def init_results(self):
        self.define_results(
            ss.Result('n_eligible', dtype=int),
            ss.Result('n_tpt_initiated', dtype=int),
            ss.Result('n_3HP_assigned', dtype=int),
        )
        
    def update_results(self):
        self.results['n_eligible'][self.ti] = np.count_nonzero(self.sim.people['household_contact'])
        self.results['n_tpt_initiated'][self.ti] = np.count_nonzero(self.sim.people['on_tpt'])
        self.results['n_3HP_assigned'][self.ti] = np.count_nonzero(self.sim.people['on_tpt'] & (self.sim.people.age < self.pars.max_age))


class BCGProtection(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            year=[1900],
            coverage=0.95,
            target_age=5,
        )
        self.update_pars(pars, **kwargs)
        self.vaccinated = None

    def initialize(self, sim):
        super().initialize(sim)
        ppl = sim.people
        eligible = (ppl.age < self.pars.target_age) & ppl.alive
        n_eligible = np.count_nonzero(eligible)
        vaccinated_mask = sim.rng.binomial(1, self.pars.coverage, n_eligible)
        self.vaccinated = eligible.uids[vaccinated_mask == 1]
        ppl['vaccination_year'][self.vaccinated] = sim.year

    def step(self):
        if self.vaccinated is None:
            return

        tb = self.sim.diseases.tb
        ppl = self.sim.people

        for uid in self.vaccinated:
            if not ppl.alive[uid]:
                continue

            age = ppl.age[uid]
            tb.rel_SL_prog[uid] *= 0.81

            if age < 3:
                tb.rel_LF_prog[uid] *= 0.58
            elif 3 <= age <= 15:
                tb.rel_LF_prog[uid] *= 0.81

            if age < 5:
                tb.rel_EP_prog[uid] *= 0.63

            if age < 5:
                tb.rel_ATB_death[uid] *= 0.20
                tb.rel_EP_death[uid] *= 0.20
            elif 5 <= age <= 14:
                tb.rel_ATB_death[uid] *= 0.13
                tb.rel_EP_death[uid] *= 0.13

    def init_results(self):
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )
        
    def update_results(self):
        self.results['n_vaccinated'][self.ti] = len(self.vaccinated) if self.vaccinated is not None else 0
        self.results['n_eligible'][self.ti] = np.count_nonzero(self.sim.people.age < self.pars.target_age)
        
        
# class TBTestScenario:
#     BASE = 'base'
#     SCENARIO = 'scenario'
#     BENCHMARK = 'benchmark'


# class TbCascadeIntervention(ss.Intervention):
#     def __init__(self, pars=None, **kwargs):
#         super().__init__(**kwargs)

#         self.define_pars(
#             mode=TBTestScenario.SCENARIO,
#             scenario_id=1,
#             screen_sens_age_hiv=None,
#             test_scenarios=None,
#             min_age=0,
#             max_age=200,
#             start=ss.date('2020-01-01'),
#             stop=ss.date('2035-12-31'),
#         )
#         self.update_pars(pars, **kwargs)

#         self.handlers = {
#             TBTestScenario.BASE: self._apply_base_test,
#             TBTestScenario.SCENARIO: self._apply_scenario_test,
#             TBTestScenario.BENCHMARK: self._apply_benchmark_test,
#         }

#         self.pediatric_outcomes = []

#     def step(self):
#         t = self.sim.now
#         if t < self.pars.start or t > self.pars.stop:
#             return

#         tb = self.sim.diseases.tb  # TB infection object
#         people = self.sim.people

#         # Get active symptomatic individuals
#         active_tb_uids = (tb.state == TBS.ACTIVE_SMNEG) | (tb.state == TBS.ACTIVE_SMPOS) | (tb.state == TBS.ACTIVE_EXPTB)
#         for uid in active_tb_uids.uids:
#             person = people[uid]
#             if not person.flags.get('sought_care'):
#                 self._seek_care(uid)

#     def _seek_care(self, uid):
#         person = self.sim.people[uid]
#         person.flags['sought_care'] = True

#         if self._screen(uid):
#             handler = self.handlers.get(self.pars.mode)
#             if handler is None:
#                 raise ValueError(f"Unsupported test mode: {self.pars.mode}")
#             test_positive = handler(uid)
#             self._handle_result(uid, test_positive)
#         else:
#             person.flags['returned_to_community'] = True

#     def _screen(self, uid):
#         person = self.sim.people[uid]
#         age_group = 'under5' if person.age < 5 else '5plus'
#         hiv_status = 'HIV+' if person.hiv_positive else 'HIV-'
#         sens, spec = self.pars.screen_sens_age_hiv[age_group][hiv_status]
#         has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
#         return self._binary_test(has_tb, sens, spec)

#     def _apply_scenario_test(self, uid):
#         test_def = self.pars.test_scenarios[self.pars.scenario_id]
#         has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
#         return self._binary_test(has_tb, test_def['sensitivity'], test_def['specificity'])

#     def _apply_base_test(self, uid):
#         has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
#         return self._binary_test(has_tb, 0.89, 0.98)  # GeneXpert default

#     def _apply_benchmark_test(self, uid):
#         test_def = self.pars.test_scenarios[self.pars.scenario_id]
#         has_tb = self.sim.diseases.tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]
#         return self._binary_test(has_tb, test_def['sensitivity'], test_def['specificity'])

#     def _handle_result(self, uid, test_positive):
#         person = self.sim.people[uid]
#         tb = self.sim.diseases.tb

#         if test_positive:
#             tb.start_treatment(ss.uids(uid))

#             for hh_member in person.household:
#                 if not hh_member.flags.get('sought_care'):
#                     self._seek_care(hh_member.uid)

#                 if hh_member.hiv_positive or hh_member.age < 5:
#                     if tb.state[hh_member.uid] not in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]:
#                         hh_member.initiate_tpt()
#         else:
#             person.flags['returned_to_community'] = True

#         if person.age < 15:
#             self._track_pediatric(person)

#     def _track_pediatric(self, person):
#         tb = self.sim.diseases.tb
#         uid = person.uid
#         self.pediatric_outcomes.append({
#             'time': self.sim.t,
#             'incidence': tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB],
#             'mortality': tb.state[uid] == TBS.DEAD,
#             'tpt': person.flags.get('received_tpt', False),
#             'treatment_success': person.flags.get('tb_treatment_success', False),
#         })

#     def finalize(self):
#         self._summarize_outcomes(self.sim)

#     def _summarize_outcomes(self, sim):
#         df = pd.DataFrame(self.pediatric_outcomes)
#         if df.empty:
#             sim.results['tb_pediatric_outcomes'] = pd.DataFrame()
#             return
#         df['year'] = df['time'] // 365
#         sim.results['tb_pediatric_outcomes'] = df.groupby('year').sum()

#     def _binary_test(self, condition, sens, spec):
#         return np.random.rand() < sens if condition else np.random.rand() > spec


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

import numpy as np
import pandas as pd
import starsim as ss
import tbsim as mtb

__all__ = ['TBTestScenario', 'TbCascadeIntervention']

TBS = mtb.TBS


class TBTestScenario:
    BASE = 'base'
    SCENARIO = 'scenario'
    BENCHMARK = 'benchmark'


class TbCascadeIntervention(ss.Intervention):
    """
    Vectorized intervention implementing a TB testing and treatment cascade based on care-seeking behavior,
    test sensitivity/specificity, and household contact tracing logic.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.define_pars(
            mode=TBTestScenario.SCENARIO,
            scenario_id=1,
            screen_sens_age_hiv={},
            test_scenarios={},
            min_age=0,
            max_age=200,
            start='2020-01-01',
            stop='2035-12-31',
        )
        self.update_pars(pars=pars, **kwargs)

        self.handlers = {
            TBTestScenario.BASE: self._apply_base_test_batch,
            TBTestScenario.SCENARIO: self._apply_scenario_test_batch,
            TBTestScenario.BENCHMARK: self._apply_benchmark_test_batch,
        }

        self.pediatric_outcomes = []
        self.sim = None

    def init_pre(self, sim):
        super().init_pre(sim)
        self.sim = sim
        self.pars.start = ss.date(self.pars.start)
        self.pars.stop = ss.date(self.pars.stop)

    def step(self):
        t = self.sim.now
        if t < self.pars.start or t > self.pars.stop:
            return

        tb = self.sim.diseases.tb
        people = self.sim.people

        active_tb_mask = (tb.state == TBS.ACTIVE_SMNEG) | (tb.state == TBS.ACTIVE_SMPOS) | (tb.state == TBS.ACTIVE_EXPTB)
        active_tb_uids = ss.uids(active_tb_mask)

        sought_care = people.sought_care[active_tb_uids]
        needs_care_uids = active_tb_uids[~sought_care]

        self._seek_care_batch(needs_care_uids)

    def _seek_care_batch(self, uids):
        if len(uids) == 0:
            return

        people = self.sim.people
        people.sought_care[uids] = True

        screened = self._screen_batch(uids)

        handler = self.handlers.get(self.pars.mode)
        if handler is None:
            raise ValueError(f"Unsupported test mode: {self.pars.mode}")

        test_positive = handler(uids)
        self._handle_results_batch(uids, test_positive, screened)

    def _screen_batch(self, uids):
        people = self.sim.people
        tb = self.sim.diseases.tb

        age = people.age[uids]
        hiv = people.hiv_positive[uids]
        tb_states = tb.state[uids]

        age_group = np.where(age < 5, 'under5', '5plus')
        hiv_status = np.where(hiv, 'HIV+', 'HIV-')
        has_tb = np.isin(tb_states, [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB])

        results = np.zeros(len(uids), dtype=bool)
        for group in ['under5', '5plus']:
            for status in ['HIV+', 'HIV-']:
                idx = (age_group == group) & (hiv_status == status)
                if not np.any(idx):
                    continue
                try:
                    sens, spec = self.pars.screen_sens_age_hiv[group][status]
                except KeyError:
                    continue
                results[idx] = self._binary_test_batch(has_tb[idx], sens, spec)
        return results

    def _apply_scenario_test_batch(self, uids):
        test_def = self.pars.test_scenarios.get(self.pars.scenario_id, {'sensitivity': 0.9, 'specificity': 0.95})
        tb_states = self.sim.diseases.tb.state[uids]
        has_tb = np.isin(tb_states, [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB])
        return self._binary_test_batch(has_tb, test_def['sensitivity'], test_def['specificity'])

    def _apply_base_test_batch(self, uids):
        tb_states = self.sim.diseases.tb.state[uids]
        has_tb = np.isin(tb_states, [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB])
        return self._binary_test_batch(has_tb, 0.89, 0.98)

    def _apply_benchmark_test_batch(self, uids):
        return self._apply_scenario_test_batch(uids)

    def _handle_results_batch(self, uids, test_positive, screened):
        people = self.sim.people
        tb = self.sim.diseases.tb
        positive_uids = uids[test_positive & screened]
        negative_uids = uids[~(test_positive & screened)]

        if len(positive_uids):
            tb.start_treatment(positive_uids)

            # Optional: restructure if household info is available as arrays
            for uid in positive_uids:
                person = people[uid]
                for hh_member in person.household:
                    if not hh_member.sought_care:
                        self._seek_care_batch(ss.uids(hh_member.uid))
                    if hh_member.hiv_positive or hh_member.age < 5:
                        if tb.state[hh_member.uid] not in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB]:
                            hh_member.initiate_tpt()

        people.returned_to_community[negative_uids] = True

        # Pediatric tracking
        under_15 = people.age[uids] < 15
        # self._track_pediatric_batch(uids[under_15])

        
    def _track_pediatric_batch(self, uids):
        if len(uids) == 0:
            return
        tb = self.sim.diseases.tb
        people = self.sim.people

        self.pediatric_outcomes.extend([
            {
                'time': self.sim.t,
                'incidence': tb.state[uid] in [TBS.ACTIVE_SMNEG, TBS.ACTIVE_SMPOS, TBS.ACTIVE_EXPTB],
                'mortality': tb.state[uid] == TBS.DEAD,
                'tpt': people[uid].received_tpt,
                'treatment_success': people[uid].tb_treatment_success,
            }
            for uid in uids
        ])

    def _binary_test_batch(self, condition, sens, spec):
        rand = np.random.rand(len(condition))
        return np.where(condition, rand < sens, rand > spec)

    def finalize(self):
        self._summarize_outcomes(self.sim)

    def _summarize_outcomes(self, sim):
        df = pd.DataFrame(self.pediatric_outcomes)
        if df.empty:
            sim.results['tb_pediatric_outcomes'] = pd.DataFrame()
            return
        df['year'] = df['time'] // 365
        sim.results['tb_pediatric_outcomes'] = df.groupby('year').sum()

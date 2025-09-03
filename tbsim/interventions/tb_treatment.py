import numpy as np
import starsim as ss
from tbsim import TBS
import matplotlib.ticker as mticker

__all__ = ['TBTreatment']


class TBTreatment(ss.Intervention):
    """
    Starts TB treatment for diagnosed individuals and applies treatment success/failure logic.

    Parameters:
        treatment_success_rate (float or Dist): Probability of cure if treated.
        reseek_multiplier (float): Care-seeking multiplier applied after failure.
        reset_flags (bool): Whether to reset tested/diagnosed flags after failure.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            treatment_success_rate=0.85,
            reseek_multiplier=2.0,
            reset_flags=True,
        )
        self.update_pars(pars=pars, **kwargs)

        # Storage for results
        self.new_treated = []
        self.successes = []
        self.failures = []

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Select individuals diagnosed with TB and alive
        diagnosed = ppl.diagnosed & ppl.alive
        active_tb = (((tb.state == TBS.ACTIVE_SMPOS) | (tb.state == TBS.ACTIVE_SMNEG) | (tb.state == TBS.ACTIVE_EXPTB)))
        uids = (diagnosed & active_tb).uids

        if len(uids) == 0:
            return

        # Start treatment
        started = tb.start_treatment(uids)

        # Treatment outcomes
        tx_uids = uids[tb.on_treatment[uids]]
        ppl.n_times_treated[tx_uids] += 1
        rand = np.random.rand(len(tx_uids))
        success_uids = tx_uids[rand < self.pars.treatment_success_rate]
        failure_uids = tx_uids[rand >= self.pars.treatment_success_rate]

        # Update success: instant clearance via TB logic
        tb.state[success_uids] = TBS.NONE
        tb.on_treatment[success_uids] = False
        tb.susceptible[success_uids] = True
        tb.infected[success_uids] = False
        tb.active_tb_state[success_uids] = TBS.NONE
        tb.ti_active[success_uids] = np.nan
        ppl.diagnosed[success_uids] = False
        ppl.tb_treatment_success[success_uids] = True

        # Update failure
        ppl.treatment_failure[failure_uids] = True

        if self.pars.reset_flags:
            ppl.diagnosed[failure_uids] = False
            ppl.tested[failure_uids] = False

        # Trigger renewed care-seeking for failures
        if len(failure_uids):
            ppl.sought_care[failure_uids] = False
            ppl.care_seeking_multiplier[failure_uids] *= self.pars.reseek_multiplier
            ppl.multiplier_applied[failure_uids] = True

        # Store
        self.new_treated = tx_uids
        self.successes = success_uids
        self.failures = failure_uids

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_treatment_success', dtype=int),
            ss.Result('n_treatment_failure', dtype=int),
            ss.Result('cum_treatment_success', dtype=int),
            ss.Result('cum_treatment_failure', dtype=int),
        )

    def update_results(self):
        n_treated = len(self.new_treated)
        n_success = len(self.successes)
        n_failure = len(self.failures)

        self.results['n_treated'][self.ti] = n_treated
        self.results['n_treatment_success'][self.ti] = n_success
        self.results['n_treatment_failure'][self.ti] = n_failure

        if self.ti > 0:
            self.results['cum_treatment_success'][self.ti] = self.results['cum_treatment_success'][self.ti - 1] + n_success
            self.results['cum_treatment_failure'][self.ti] = self.results['cum_treatment_failure'][self.ti - 1] + n_failure
        else:
            self.results['cum_treatment_success'][self.ti] = n_success
            self.results['cum_treatment_failure'][self.ti] = n_failure

        # Reset for next step
        self.new_treated = []
        self.successes = []
        self.failures = []



if __name__ == '__main__':
    # run scripts/interventions/run_tb_treatment.py 
    import scripts.interventions.run_tb_treatment as run_tb_treatment
    
    run_tb_treatment.run()
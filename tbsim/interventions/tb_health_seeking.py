import numpy as np
import starsim as ss
from tbsim import TBS

__all__ = ['HealthSeekingBehavior']


class HealthSeekingBehavior(ss.Intervention):
    """
    Trigger care-seeking behavior for individuals with active TB.

    Parameters:
        prob (float): Probability of seeking care per unit time.
        single_use (bool): Whether to expire the intervention after success.
        actual (Intervention): Optional downstream intervention (e.g. testing or treatment).
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            initial_care_seeking_rate = ss.perday(0.1),
            start = None,
            stop = None,
            single_use = True,
        )
        self.update_pars(pars=pars, **kwargs)

        self.dist_care_seeking = ss.bernoulli(p=self.p_care_seeking)

    @staticmethod
    def p_care_seeking(self, sim, uids):
        """ Calculate the probability of care-seeking for individuals."""
        # Get the base rate and unit from the TimePar object
        base_rate_val = self.pars.initial_care_seeking_rate.rate
        unit = self.pars.initial_care_seeking_rate.unit
        
        # Create rate array with the same rate for all individuals
        ratevals_arr = np.full(len(uids), base_rate_val)
        
        # Convert to Starsim rate object and apply to_prob()
        rate = ss.per(ratevals_arr, unit=unit)
        prob = rate.to_prob()  # Do not use sim.dt; module dt is used internally
        return prob

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb
        t = sim.now

        # Optional timing window
        if self.pars.start is not None and t < self.pars.start:
            return
        if self.pars.stop is not None and t > self.pars.stop:
            return


        not_yet_sought_uids = (((tb.state == TBS.ACTIVE_SMPOS) |
                                (tb.state == TBS.ACTIVE_SMNEG) |
                                (tb.state == TBS.ACTIVE_EXPTB)) &
                                ~ppl.sought_care & ppl.alive).uids
        sought_care_uids = self.dist_care_seeking.filter(not_yet_sought_uids)
        self.new_seekers_this_step = sought_care_uids

        if len(sought_care_uids) > 0:
            ppl.sought_care[sought_care_uids] = True
            tb.start_treatment(sought_care_uids)
            if self.pars.single_use:
                self.expired = True

    def init_results(self):
        """Define metrics to track over time."""
        super().init_results()
        self.define_results(
            ss.Result('new_sought_care', dtype=int),
            ss.Result('n_sought_care', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )
    
    def update_results(self):
        """Record who was eligible and who sought care at this timestep."""
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        active_tb = ((tb.state == TBS.ACTIVE_SMPOS) |
                     (tb.state == TBS.ACTIVE_SMNEG) |
                     (tb.state == TBS.ACTIVE_EXPTB))
        active_uids = active_tb.uids
        not_yet_sought = active_uids[~ppl.sought_care[active_uids]]

        self.results['new_sought_care'][self.ti] = len(self.new_seekers_this_step)
        self.results['n_sought_care'][self.ti] = np.count_nonzero(ppl.sought_care)
        self.results['n_eligible'][self.ti] = len(not_yet_sought)

        # Clear temporary storage
        self.new_seekers_this_step = []


# Sample usage: see tbsim_examples/run_tb_interventions.py
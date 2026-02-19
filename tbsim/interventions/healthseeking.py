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
            prob=0.1,              # Daily probability of seeking care if active
            single_use=True,       # Whether to expire after seeking care
            start=None,            # Optional start time
            stop=None,             # Optional stop time
        )
        self.update_pars(pars=pars, **kwargs)

        # Ensure prob is a Dist (created once, not every step)
        if not isinstance(self.pars.prob, ss.Dist):
            self.pars.prob = ss.bernoulli(p=self.pars.prob)

    def step(self):
        sim = self.sim
        t = sim.now
        ppl = sim.people
        tb = sim.diseases.tb

        # Optional timing window
        if self.pars.start is not None and t < self.pars.start:
            return
        if self.pars.stop is not None and t > self.pars.stop:
            return

        # Active TB and not yet sought care
        active_tb = (tb.state == TBS.ACTIVE_SMPOS) | (tb.state == TBS.ACTIVE_SMNEG) | (tb.state == TBS.ACTIVE_EXPTB)
        active_uids = active_tb.uids
        not_yet_sought = active_uids[~ppl.sought_care[active_uids]]

        seeking_uids = self.pars.prob.filter(not_yet_sought)

        if len(seeking_uids) == 0:
            return

        ppl.sought_care[seeking_uids] = True
        tb.start_treatment(seeking_uids)

        if self.pars.single_use:
            self.expired = True

    def init_results(self):
        """Define metrics to track over time."""
        super().init_results()
        self.define_results(
            ss.Result('n_sought_care', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )
    
    def update_results(self):
        """Record who was eligible and who sought care at this timestep."""
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        active_tb = (tb.state == TBS.ACTIVE_SMPOS) | (tb.state == TBS.ACTIVE_SMNEG) | (tb.state == TBS.ACTIVE_EXPTB)
        active_uids = active_tb.uids
        not_yet_sought = active_uids[~ppl.sought_care[active_uids]]

        self.results['n_eligible'][self.ti] = len(not_yet_sought)
        self.results['n_sought_care'][self.ti] = np.count_nonzero(ppl.sought_care)

# Sample calling function below
if __name__ == '__main__':

    import tbsim as mtb
    import starsim as ss
    import matplotlib.pyplot as plt
    sim = ss.Sim(
        people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),  # 👈 Add this!
        diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
        interventions=[mtb.HealthSeekingBehavior(pars={'prob': ss.bernoulli(p=0.2, strict=False)})],
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
        pars=dict(start=2000, stop=2020, dt=ss.days(1)),
    )
    sim.run()

    # Plot results
    sim.results.plot()
    hsb = sim.results['healthseekingbehavior']
    plt.plot(hsb['n_sought_care'].timevec, hsb['n_sought_care'].values, label='Sought care')
    plt.plot(hsb['n_eligible'].timevec, hsb['n_eligible'].values, label='Eligible')
    plt.xlabel('Time')
    plt.ylabel('People')
    plt.title('Health-Seeking Behavior Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
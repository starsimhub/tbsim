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
        active_tb = ((tb.state == TBS.ACTIVE_SMPOS) |
                     (tb.state == TBS.ACTIVE_SMNEG) |
                     (tb.state == TBS.ACTIVE_EXPTB))
        active_uids = active_tb.uids
        not_yet_sought = active_uids[~ppl.sought_care[active_uids]]

        # not_yet_sought = active_uids[
        #     (~ppl.sought_care[active_uids]) &
        #     (ppl.next_seek_time[active_uids] <= t)  # current_time)
        # ]

        # Initialize or extract a working distribution
        if isinstance(self.pars.prob, ss.Dist):
            dist = self.pars.prob
        else:
            # dist = ss.bernoulli(p=self.pars.prob)
            # Use per-person adjusted probability
            base_prob = self.pars.prob if isinstance(self.pars.prob, (int, float)) else self.pars.prob.p
            adjusted_prob = base_prob * ppl.care_seeking_multiplier[not_yet_sought]
            dist = ss.bernoulli(p=adjusted_prob)

        dist.init(sim)  # Explicitly initialize with the simulation context
        seeking_uids = dist.filter(not_yet_sought)

        # Store for use in update_results()
        self.new_seekers_this_step = seeking_uids
    
        if len(seeking_uids) == 0:
            return

        ppl.sought_care[seeking_uids] = True
        tb.start_treatment(seeking_uids)

        # ppl.next_seek_time[seeking_uids] = np.inf  # After someone seeks care

        if self.pars.single_use:
            self.expired = True

    def init_results(self):
        """Define metrics to track over time."""
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

        # print(f"[t={self.sim.ti}] new seekers: {len(self.new_seekers_this_step)}, total sought: {np.count_nonzero(ppl.sought_care)}")

        # Clear temporary storage
        self.new_seekers_this_step = []

# Sample calling function below
if __name__ == '__main__':

    # import tbsim as mtb
    # import starsim as ss
    # import matplotlib.pyplot as plt
    # sim = ss.Sim(
    #     people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),  # 👈 Add this!
    #     diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
    #     interventions=[mtb.HealthSeekingBehavior(pars={'prob': ss.bernoulli(p=0.1, strict=False)})],
    #     networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    #     pars=dict(start=2000, stop=2020, dt=1),
    # )
    # sim.run()

    # # Plot results
    # sim.results.plot()
    # hsb = sim.results['healthseekingbehavior']
    # plt.plot(hsb['n_sought_care'].timevec, hsb['n_sought_care'].values, label='Sought care')
    # plt.plot(hsb['n_eligible'].timevec, hsb['n_eligible'].values, label='Eligible')
    # plt.xlabel('Time')
    # plt.ylabel('People')
    # plt.title('Health-Seeking Behavior Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # import sys
    # import os
    # # Try standard import first
    # try:
    #     import plots as pl
    # except ModuleNotFoundError:
    #     # Try fallback paths
    #     candidate_paths = [
    #         os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')),
    #         os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')),
    #         os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')),
    #     ]
        
    #     for path in candidate_paths:
    #         if os.path.exists(os.path.join(path, 'plots.py')):
    #             sys.path.append(path)
    #             try:
    #                 # import plots as pl
    #                 from plots import plots_results
    #                 break  # Success
    #             except ModuleNotFoundError:
    #                 continue
    #     else:
    #         raise ImportError("Could not locate 'plots.py' in known fallback locations. Check project structure.")

    import tbsim as mtb
    import starsim as ss
    import matplotlib.pyplot as plt
    # import datetime
    import sciris as sc

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
    from plots import plot_results

    # Create and run the sim
    sim = ss.Sim(
        people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
        diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
        interventions=[
            mtb.HealthSeekingBehavior(pars={'prob': ss.bernoulli(p=0.1, strict=False)})
        ],
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
        pars=dict(start=2000, stop=2020, dt=1),
    )
    sim.run()

    print("Unique care_seeking_multiplier values:", np.unique(sim.people.care_seeking_multiplier))

    # Flatten results into format expected by plot_results()
    flat_results = {'TB + HSB': sim.results.flatten()}

    # Plot all matching metrics (you can adjust keywords below)
    plot_results(
        flat_results,
        keywords=['active', 'sought', 'eligible', 'incidence'],
        exclude=(),
        n_cols=2,
        dark=False,
        cmap='tab10',
        heightfold=3,
        style='default'
    )

    # Custom plot for new_sought_care + optional cumulative view
    hsb = sim.results['healthseekingbehavior']
    timevec = hsb['new_sought_care'].timevec
    new_sought = hsb['new_sought_care'].values
    cum_sought = np.cumsum(new_sought)

    plt.figure(figsize=(10, 5))
    plt.plot(timevec, new_sought, label='New sought care (this step)', linestyle='-', marker='o')
    plt.plot(timevec, cum_sought, label='Cumulative sought care', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Number of People')
    plt.title('New and Cumulative Health-Seeking Behavior Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # import tbsim as mtb
    # import starsim as ss
    # import matplotlib.pyplot as plt
    # import sys
    # import os
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
    # import plots as pl

    # results = {}
    # sim = ss.Sim(
    #     people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
    #     diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
    #     interventions=[
    #         mtb.HealthSeekingBehavior(pars={'prob': ss.bernoulli(p=0.1, strict=False)})
    #     ],
    #     networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    #     pars=dict(start=2000, stop=2020, dt=1),
    # )
    # sim.run()

    # # Flatten results for plotting
    # results[name] = sim.results.flatten()

    # pl.plot_results(
    #     results,
    #     keywords=['active', 'incidence', 'eligible', 'sought'],
    #     exclude=(),
    #     n_cols=2,
    #     dark=False,
    #     cmap='tab10',
    #     heightfold=3,
    #     style='default'
    # )
    # plt.show()

import numpy as np
import starsim as ss
from tbsim.interventions.base import TBIntervention

__all__ = ['HealthSeekingBehavior']


class HealthSeekingBehavior(TBIntervention):
    """
    Simulates initial care-seeking behaviour for individuals with active TB.

    At each time step the intervention identifies agents whose disease state
    is eligible for care-seeking (e.g. symptomatic, smear-positive) and who
    have not previously sought care.  A subset of those agents — determined
    by a probability or a rate — "seek care" and are handed to the TB
    disease module's ``start_treatment()`` method.

    The intervention works with every TB variant in tbsim (``TB``,
    ``TB_LSHTM``, ``TB_LSHTM_Acute``).  The base class automatically
    detects which variant is running and looks up the eligible states, so
    no manual configuration is needed.

    Parameters
    ----------
    prob : float or ss.Dist, default 0.1
        Per-step probability that an eligible agent seeks care.
        Ignored when ``initial_care_seeking_rate`` is provided.
    initial_care_seeking_rate : ss.Rate, optional
        A care-seeking rate expressed as an ``ss.Rate`` object
        (e.g. ``ss.perday(0.1)`` or ``ss.peryear(36.5)``).
        Internally converted to a per-step probability.
        When provided, ``prob`` is ignored.
    single_use : bool, default True
        If True, the intervention expires after the first time step in
        which at least one agent seeks care (one-shot trigger).
        Set to False for continuous care-seeking over time.
    start : date-like, optional
        Simulation date before which the intervention is inactive.
    stop : date-like, optional
        Simulation date after which the intervention is inactive.

    Results (recorded each time step)
    ----------------------------------
    new_sought_care : int
        Number of agents who sought care this step.
    n_sought_care : int
        Cumulative number of agents who have ever sought care.
    n_eligible : int
        Number of agents currently eligible but who have not yet sought care.

    Examples
    --------
    Rate-based, continuous care-seeking with the LSHTM model::

        sim = ss.Sim(
            diseases=TB_LSHTM(),
            interventions=HealthSeekingBehavior(pars={
                'initial_care_seeking_rate': ss.perday(0.1),
                'single_use': False,
            }),
            ...
        )
        sim.run()

    Probability-based, single-use with the legacy model::

        sim = ss.Sim(
            diseases=TB(),
            interventions=HealthSeekingBehavior(pars={'prob': 0.5}),
            ...
        )
    """

    _state_method = 'care_seeking_eligible'

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            prob=0.1,
            initial_care_seeking_rate=None,
            single_use=True,
            start=None,
            stop=None,
        )
        self.update_pars(pars=pars, **kwargs)
        self.define_states(ss.BoolArr('sought_care', default=False))
        self._new_seekers_count = 0
        self._care_seeking_dist = None

    def init_post(self):
        """Validate parameters and build the sampling distribution.

        Called automatically after the simulation is initialised.
        If ``initial_care_seeking_rate`` is set, it is validated here and
        converted to a probability each step.  Otherwise a Bernoulli
        distribution is created from ``prob`` and reused every step.
        """
        super().init_post()
        rate = self.pars.initial_care_seeking_rate
        if rate is not None:
            if not isinstance(rate, ss.Rate):
                raise TypeError(
                    f"initial_care_seeking_rate must be an ss.Rate (e.g. ss.perday(0.1), ss.peryear(0.1)), got {type(rate).__name__}"
                )
            return
        trace = self.name or 'HealthSeekingBehavior.care_seeking'
        if isinstance(self.pars.prob, ss.Dist):
            dist = self.pars.prob
        else:
            dist = ss.bernoulli(p=self.pars.prob)
        dist.init(trace=trace, sim=self.sim, module=self)
        self._care_seeking_dist = dist

    def step(self):
        """Identify eligible agents and sample who seeks care this step."""
        sim = self.sim
        t = sim.now
        if self.pars.start is not None and t < self.pars.start:
            return
        if self.pars.stop is not None and t > self.pars.stop:
            return

        active_uids = np.where(np.isin(self.tb.state, self.states))[0]
        not_yet_sought = active_uids[~self.sought_care[active_uids]]
        self._new_seekers_count = 0

        if len(not_yet_sought) == 0:
            return

        if self.pars.initial_care_seeking_rate is not None:
            rate = self.pars.initial_care_seeking_rate
            if not isinstance(rate, ss.Rate):
                raise TypeError(
                    f"initial_care_seeking_rate must be an ss.Rate (e.g. ss.perday(0.1)), got {type(rate).__name__}"
                )
            p = rate.to_prob()
            dist = ss.bernoulli(p=p)
            trace = self.name or 'HealthSeekingBehavior.care_seeking'
            dist.init(trace=trace, sim=sim, module=self)
            seeking_uids = dist.filter(not_yet_sought)
        else:
            seeking_uids = self._care_seeking_dist.filter(not_yet_sought)

        if len(seeking_uids) == 0:
            return
        self._new_seekers_count = len(seeking_uids)
        self.sought_care[seeking_uids] = True
        self.tb.start_treatment(seeking_uids)
        if self.pars.single_use:
            self.expired = True

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_sought_care', dtype=int),
            ss.Result('n_sought_care', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        """Record care-seeking counts and remaining eligible agents."""
        self.results['new_sought_care'][self.ti] = self._new_seekers_count
        self.results['n_sought_care'][self.ti] = np.count_nonzero(self.sought_care)
        active_uids = np.where(np.isin(self.tb.state, self.states))[0]
        not_yet_sought = active_uids[~self.sought_care[active_uids]]
        self.results['n_eligible'][self.ti] = len(not_yet_sought)

import warnings
import numpy as np
import starsim as ss
from tbsim import TBSL, TB_LSHTM, TB_LSHTM_Acute

__all__ = ['Immigration']


class Immigration(ss.Demographics):
    """
    Demographic module that adds new immigrants to the population each timestep.

    Arrivals are Poisson-distributed at the specified rate, assigned ages drawn
    from a configurable age distribution, and seeded into the TB state machine
    according to a user-supplied prevalence distribution. Requires either
    ``TB_LSHTM`` or ``TB_LSHTM_Acute`` to be present in the simulation.

    Parameters
    ----------
    pars : dict, optional
        Parameter overrides. Supported keys:

        immigration_rate : ss.freq
            Mean number of arrivals per year (default: 10). Must be an event-rate
            type (``ss.freqperyear``, etc.).
        rel_immigration : float
            Scalar multiplier applied to ``immigration_rate`` (default: 1.0).
            Useful for scenario scaling without changing the base rate.
        age_distribution : dict or None
            Mapping of ``{lower_age_bound: probability}`` defining the age-bin
            sampling distribution. Probabilities are normalized automatically.
            If ``None``, a default distribution spanning 0–85 years is used.
        tb_state_distribution : dict
            Mapping of ``{TBSL state name: probability}`` for the TB state
            assigned to each new arrival. Keys must be valid ``TBSL`` member
            names. Values are normalized automatically; a warning is raised if
            they sum to more than 1.

    States
    ------
    hhid : int
        Household ID assigned to the agent (-1 if unassigned).
    is_immigrant : bool
        ``True`` for agents that arrived via this module.
    immigration_time : float
        Timestep index at which the agent immigrated (NaN for non-immigrants).
    age_at_immigration : float
        Age in years at time of immigration (NaN for non-immigrants).
    immigration_tb_status : int
        TBSL state integer sampled at immigration (-1 for non-immigrants).

    Results
    -------
    n_immigrants : int
        Number of new arrivals recorded each timestep.

    Example
    -------
    ::

        import starsim as ss
        import tbsim

        sim = ss.Sim(
            diseases=tbsim.TB_LSHTM(),
            networks=tbsim.HouseholdNet(),
            demographics=tbsim.Immigration(pars=dict(immigration_rate=500)),
        )
        sim.run()
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__()
        
        self.define_pars(
            immigration_rate=ss.freqperyear(10),
            rel_immigration=1.0,
            age_distribution=None,
            tb_state_distribution=dict(
                SUSCEPTIBLE=0.6517,
                INFECTION=0.33,
                CLEARED=0.007,
                RECOVERED=0.005,
                NON_INFECTIOUS=0.0008,
                ASYMPTOMATIC=0.0015,
                SYMPTOMATIC=0.0010,
                TREATED=0.003,
                TREATMENT=0.0,
            ),
        )
        self.update_pars(pars, **kwargs)

        self.pars.tb_state_distribution = self._validate_tb_state_distribution(self.pars.tb_state_distribution)
        
        # Random draws (CRN-safe): keep one call per distribution per timestep
        self._dist_n = ss.poisson(name='imm_n', lam=self._lam_per_timestep)
        self._dist_agebin = ss.choice(name='imm_agebin', a=[0], p=[1.0])  # configured in init_pre()
        self._dist_ageu = ss.random(name='imm_ageu')  # uniform(0,1) for age within bin
        self._dist_tbstate = ss.choice(name='imm_tbstate', a=[-1], p=[1.0])  # configured in init_pre()
        self._dist_hhu = ss.random(name='imm_hhu')  # uniform(0,1) for household assignment

        # Cached bin edges for age sampling (configured in init_pre)
        self._age_lows = None
        self._age_highs = None
        
        # Tracking per person (set when someone immigrates)
        self.define_states(
            ss.IntArr('hhid', default=-1),
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),
            ss.FloatArr('age_at_immigration', default=np.nan),
            ss.IntArr('immigration_tb_status', default=-1),
        )
        
        # Tracking per timestep
        self.n_immigrants = 0
        
        return
    
    def init_post(self):
        """Resolve the TB disease module name after the simulation is assembled."""
        super().init_post()
        self._tb_name = next(
            (k for k, d in self.sim.diseases.items() if isinstance(d, (TB_LSHTM, TB_LSHTM_Acute))),
            None,
        )
        if self._tb_name is None:
            raise RuntimeError('Expected TB_LSHTM(_Acute) disease module for immigration initialization')

    def init_pre(self, sim):
        """Initialize with simulation information."""
        super().init_pre(sim)
        
        if self.pars.age_distribution is None:
            # Default age bins (years) if none provided
            self.pars.age_distribution = {
                0: 0.15,
                5: 0.20,
                15: 0.25,
                30: 0.20,
                50: 0.15,
                65: 0.05,
            }

        # Configure age-bin sampler
        ad = self.pars.age_distribution
        if isinstance(ad, dict) and len(ad):
            keys = np.array(sorted(ad.keys()), dtype=float)
            probs = np.array([ad[k] for k in keys], dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(keys)) / len(keys)

            self._age_lows = keys
            self._age_highs = np.r_[keys[1:], 85.0]
            self._dist_agebin.pars.a = np.arange(len(keys), dtype=int)
            self._dist_agebin.pars.p = probs

        # Configure TB state sampler (already validated and normalized in __init__)
        dist = self.pars.tb_state_distribution
        self._dist_tbstate.pars.a = np.array([int(getattr(TBSL, k)) for k in dist], dtype=int)
        self._dist_tbstate.pars.p = np.array(list(dist.values()), dtype=float)
        
        return
    
    def init_results(self):
        """Initialize results tracking."""
        super().init_results()
        self.define_results(
            ss.Result('n_immigrants', dtype=int, label='Number of immigrants'),
        )
        return

    def expected_immigrants_per_timestep(self):
        """Return expected number of immigrants in the current timestep."""
        r = self.pars.immigration_rate
        if r is None:
            return 0.0

        if isinstance(r, ss.Rate):
            if not isinstance(r, ss.freq):
                raise ValueError('immigration_rate must be an event rate (ss.freq...), e.g. ss.freqperyear(1000)')
            rate = r
        else:
            rate = ss.freqperyear(float(r))

        rate = rate * float(self.pars.rel_immigration)

        dt = getattr(self.sim, 'dt', None)
        dt_dur = dt if isinstance(dt, ss.dur) else ss.dur(dt)
        lam = float(rate.to_events(dt_dur))
        if not np.isfinite(lam) or lam < 0:
            lam = 0.0
        return lam

    def _lam_per_timestep(self, module):
        """Callable function for ss.poisson(lam=...)."""
        return module.expected_immigrants_per_timestep()

    def _sample_ages(self, n):
        """Sample ages (years) for n immigrants."""
        if n <= 0:
            return np.empty(0, dtype=float)

        if self._age_lows is None or self._age_highs is None:
            u = self._dist_ageu.rvs(n)
            return u * 85.0

        bin_idx = self._dist_agebin.rvs(n).astype(int)
        u = self._dist_ageu.rvs(n)
        lo = self._age_lows[bin_idx]
        hi = self._age_highs[bin_idx]
        return lo + u * (hi - lo)

    def _get_immigrant_characteristics(self, n_immigrants):
        """Generate characteristics for new immigrants."""
        if n_immigrants == 0:
            return {}
        return {'ages': self._sample_ages(n_immigrants)}


    def _init_tb_lshtm(self, new_uids):
        """
        Seed TB states for newly arrived agents.

        Samples a TB state for each UID from ``tb_state_distribution`` and
        sets all derived flags (``infected``, ``susceptible``, ``ever_infected``,
        ``on_treatment``, ``rel_sus``, ``rel_trans``) to be consistent with the
        sampled state.

        Parameters
        ----------
        new_uids : array-like
            UIDs of the agents just added to the population.

        Returns
        -------
        sampled : np.ndarray of int
            TBSL integer states assigned to each UID.
        """
        tb = self.sim.diseases[self._tb_name]
        # Only allow ACUTE if the TB module is the acute variant
        state_vals = np.asarray(self._dist_tbstate.pars.a).astype(int)
        has_acute = int(TBSL.ACUTE) in set(map(int, state_vals))
        if has_acute and not isinstance(tb, TB_LSHTM_Acute):
            raise ValueError('tb_state_distribution includes ACUTE but TB module is not TB_LSHTM_Acute')

        n = len(new_uids)
        sampled = self._dist_tbstate.rvs(n).astype(int)
        tb.state[new_uids] = sampled

        # Keep flags consistent with the TB state machine
        infected_states = np.array([int(TBSL.INFECTION), int(TBSL.NON_INFECTIOUS), int(TBSL.ASYMPTOMATIC),
                                    int(TBSL.SYMPTOMATIC), int(TBSL.TREATMENT)])
        if isinstance(tb, TB_LSHTM_Acute):
            infected_states = np.append(infected_states, int(TBSL.ACUTE))

        reinfectable_states = np.array([int(TBSL.SUSCEPTIBLE), int(TBSL.CLEARED), int(TBSL.RECOVERED), int(TBSL.TREATED)])

        is_inf = np.isin(sampled, infected_states)
        tb.infected[new_uids] = is_inf
        tb.susceptible[new_uids] = np.isin(sampled, reinfectable_states)
        tb.ever_infected[new_uids] = sampled != int(TBSL.SUSCEPTIBLE)

        tb.ti_infected[new_uids] = np.where(is_inf, self.ti, -np.inf)
        tb.on_treatment[new_uids] = sampled == int(TBSL.TREATMENT)

        # rel_sus: RECOVERED/TREATED have modifiers
        tb.rel_sus[new_uids] = 1.0
        tb.rel_sus[new_uids[sampled == int(TBSL.RECOVERED)]] = float(tb.pars.rr_rec)
        tb.rel_sus[new_uids[sampled == int(TBSL.TREATED)]] = float(tb.pars.rr_treat)

        # rel_trans: ASYMPTOMATIC uses kappa; ACUTE uses alpha; DEAD = 0
        tb.rel_trans[new_uids] = 1.0
        tb.rel_trans[new_uids[sampled == int(TBSL.ASYMPTOMATIC)]] = float(tb.pars.trans_asymp)
        if isinstance(tb, TB_LSHTM_Acute):
            tb.rel_trans[new_uids[sampled == int(TBSL.ACUTE)]] = float(tb.pars.trans_acute)
        tb.rel_trans[new_uids[sampled == int(TBSL.DEAD)]] = 0.0

        return sampled
    
    def step(self):
        """Add immigrants to the population."""
        n_immigrants = int(self._dist_n.rvs(1)[0])
        
        if n_immigrants == 0:
            self.n_immigrants = 0
            return []
        
        characteristics = self._get_immigrant_characteristics(n_immigrants)
        
        new_uids = self.sim.people.grow(n_immigrants)
        
        self.sim.people.age[new_uids] = characteristics['ages']
        
        
        sampled_states = self._init_tb_lshtm( new_uids)
        self.immigration_tb_status[new_uids] = sampled_states
        
        self.assign_immigrants_to_households(new_uids)

        self.is_immigrant[new_uids] = True
        self.immigration_time[new_uids] = float(self.ti)
        self.age_at_immigration[new_uids] = self.sim.people.age[new_uids]
        
        self.n_immigrants = n_immigrants
        return new_uids
    
    def assign_immigrants_to_households(self, new_uids):
        """
        Assign new arrivals to existing households.

        Searches the simulation's networks for the first one that exposes an
        ``hhid`` array, then randomly assigns each immigrant to one of the
        existing household IDs. If no households exist yet, each immigrant is
        placed in its own singleton household.

        Parameters
        ----------
        new_uids : array-like
            UIDs of the agents to assign.
        """
        # Find the first network with an hhid array
        hh_net = None
        for net in self.sim.networks.values():
            hhid = getattr(net, 'hhid', None)
            if isinstance(hhid, (np.ndarray, ss.BaseArr)):
                hh_net = net
                break
        
        if hh_net is None:
            return
        
        # NOTE: simple assignment to existing households
        existing_hhids = np.unique(hh_net.hhid[hh_net.hhid >= 0])
        
        if len(existing_hhids) > 0:
            u = self._dist_hhu.rvs(len(new_uids))
            idx = np.floor(u * len(existing_hhids)).astype(int)
            idx = np.clip(idx, 0, len(existing_hhids) - 1)
            assigned_hhids = existing_hhids[idx]
            hh_net.hhid[new_uids] = assigned_hhids
            self.hhid[new_uids] = assigned_hhids
        else:
            for i, uid in enumerate(new_uids):
                hh_net.hhid[uid] = i
                self.hhid[uid] = i
    
    def update_results(self):
        """Update results tracking."""
        super().update_results()
        if isinstance(getattr(self, 'results', None), ss.Results):
            self.results['n_immigrants'][self.ti] = int(self.n_immigrants)
        return

    @staticmethod
    def _validate_tb_state_distribution(tb_state_distribution):
        """Validate and normalize tb_state_distribution."""
        if not tb_state_distribution:
            raise ValueError('tb_state_distribution must be provided')
        dist = {k: float(v) for k, v in dict(tb_state_distribution).items() if v}
        for k, v in dist.items():
            if k not in TBSL.__members__:
                raise KeyError(f'Unknown LSHTM state "{k}" in tb_state_distribution')
            if v < 0:
                raise ValueError(f'tb_state_distribution["{k}"] is negative ({v})')
        total = sum(dist.values())
        if total > 1.0:
            warnings.warn(f'tb_state_distribution sums to {total:.6g} (>1); normalizing automatically')
        return {k: v / total for k, v in dist.items()}


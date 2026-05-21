import warnings
import numpy as np
import starsim as ss
from .tb import TBS, TB, TBAcute

__all__ = ['Immigration']


class Immigration(ss.Demographics):
    """
    Stochastic exogenous-entry operator for TB state-space simulations.
    Demographic module that adds new immigrants to the population each timestep.

    Arrivals are Poisson-distributed at the specified rate, assigned ages drawn
    from a configurable age distribution, and seeded into the TB state machine
    according to a user-supplied prevalence distribution. Requires either
    ``TB`` or ``TBAcute`` to be present in the simulation.

    This module defines a marked point process over simulation time and applies
    it as a demographic inflow to the agent system. At each model step ``t``:

    1. Arrival count is sampled as
       ``N_t ~ Poisson(lambda_t)``, with
       ``lambda_t = rel_immigration * immigration_rate.to_events(dt_t)``.
    2. ``N_t`` agents are appended to the population.
    3. Each new agent receives:
       - an age sampled from a piecewise-uniform mixture induced by
         ``age_distribution``,
       - a TB state sampled from ``tb_state_distribution`` on the ``TBS`` state set,
       - a household assignment (if a compatible household-ID array is found).
    4. TB-derived flags/states needed for imported-agent consistency are made
       internally consistent with the sampled TB state. The full set updated in
       ``_init_tb_states`` is:
       ``state``, ``infected``, ``susceptible``, ``ever_infected``,
       ``on_treatment``, ``ti_infected``, ``rr_reinfection``,
       ``ti_rr_reinfection_wane``, ``rel_sus``, and ``rel_trans``.

    The inflow is explicitly *exogenous* with respect to local infection hazard:
    every imported agent is stamped with ``ti_infected = -np.inf`` so imported
    infected cases are not counted as model-generated incident infections
    (susceptible imports are unaffected by this assignment in practice).

    Mathematical objects:
        - Arrival intensity: ``lambda_t in R_{>=0}``
        - Age-bin index: ``B_i ~ Categorical(p_age)``
        - Age within bin: ``U_i ~ Uniform(0,1)``,
          ``A_i = L_{B_i} + U_i * (H_{B_i} - L_{B_i})``
        - TB state: ``S_i ~ Categorical(p_tb)`` over ``TBS`` codes

    Compatibility constraints:
        - Requires disease module instance of ``TB`` or ``TBAcute``.
        - If ``ACUTE`` has non-zero mass in ``tb_state_distribution``, the active
          disease module must be ``TBAcute``.

    Parameters
    ----------
    pars : dict, optional
        Parameter overrides. Supported keys:

        immigration_rate : ss.freq or float
            Mean number of arrivals per year (default: ``ss.freqperyear(10)``).
            An event-rate type (``ss.freqperyear``, etc.) is preferred; bare
            scalars are coerced to ``ss.freqperyear``. Passing a non-event
            ``ss.Rate`` subclass raises ``ValueError``.
        rel_immigration : float
            Scalar multiplier applied to ``immigration_rate`` (default: 1.0).
            Useful for scenario scaling without changing the base rate.
        age_distribution : dict or None
            Mapping of ``{lower_age_bound: probability}`` defining the age-bin
            sampling distribution. Probabilities are normalized automatically.
            Weights must be finite and non-negative, with strictly positive
            total mass. If ``None``, a default distribution spanning 0–85
            years is used.
        tb_state_distribution : dict
            Mapping of ``{TBS state name: probability}`` for the TB state
            assigned to each new arrival. Keys must be valid ``TBS`` member
            names. Values must be finite and non-negative, and are normalized
            automatically; a warning is raised if they sum to more than 1.

    Agent-level states introduced by the module:
        hhid (ss.IntArr): Assigned household ID (module copy), ``-1`` if unset.
        is_immigrant (ss.BoolState): Indicator for module-origin agents.
        immigration_time (ss.FloatArr): Immigration step index.
        age_at_immigration (ss.FloatArr): Entry age (years).
        immigration_tb_status (ss.IntArr): TB state code at entry.

    Result channels:
        n_immigrants (ss.Result): Per-step arrival count ``N_t``.

    Implementation invariants:
        - ``n_immigrants == len(new_uids)`` whenever ``step()`` returns non-empty.
        - For immigrants ``u``, ``immigration_tb_status[u] == tb.state[u]`` at entry.
        - For every imported agent ``u``, ``tb.ti_infected[u] = -np.inf``
          (regardless of whether the sampled state is infected or susceptible).
        - For every imported agent ``u``, ``is_immigrant[u] = True``,
          ``immigration_time[u] = self.ti``, and ``age_at_immigration[u] = age[u]``.
        - If a household network with writable edge arrays is present, each
          immigrant is connected by undirected edges to all other members of
          their assigned household after assignment (including other newcomers
          assigned to the same household); only pairs not already present are
          appended.
        - Household assignment is uniform over the set of existing household IDs;
          if no households exist yet (no finite, non-negative IDs in the
          network's household array), the ``N_t`` immigrants are assigned the
          batch-local IDs ``0..N_t-1`` (one singleton household each).

    Example:
        sim = tbsim.Sim(
            demographics=[tbsim.Immigration(pars=dict(immigration_rate=ss.freqperyear(500)))],
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
                NON_INFECTIOUS=0.0008,
                ASYMPTOMATIC=0.0015,
                SYMPTOMATIC=0.0010,
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
        """
        Resolve the TB disease module after the simulation is assembled.

        Scans ``self.sim.diseases`` and caches the key of the first module that
        is an instance of ``TB`` or ``TBAcute``. Raises ``RuntimeError`` if no
        compatible disease module is present, because subsequent state writes
        in ``_init_tb_states`` depend on this handle.
        """
        super().init_post()
        self._tb_name = next(
            (k for k, d in self.sim.diseases.items() if isinstance(d, (TB, TBAcute))),
            None,
        )
        if self._tb_name is None:
            raise RuntimeError('Expected TB or TBAcute disease module for immigration initialization')

    def init_pre(self, sim):
        """
        Configure stochastic samplers using simulation-level information.

        - If ``age_distribution`` is ``None``, installs a default 0–85 year
          age profile with 6 bins.
        - Builds ``_age_lows`` and ``_age_highs`` from the sorted keys of
          ``age_distribution``; the top bin's upper bound is fixed at 85 years.
        - Normalizes the age bin weights to a probability simplex and writes
          them into ``_dist_agebin``.
        - Configures ``_dist_tbstate`` over the integer ``TBS`` codes named in
          ``tb_state_distribution`` (already validated/normalized in ``__init__``).
        """
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
            if np.any(~np.isfinite(keys)):
                raise ValueError('age_distribution keys must be finite age-bin lower bounds')
            if np.any(~np.isfinite(probs)):
                raise ValueError('age_distribution weights must be finite')
            if np.any(probs < 0):
                raise ValueError('age_distribution weights must be non-negative')
            total = probs.sum()
            if total <= 0:
                raise ValueError('age_distribution must include at least one positive weight')
            probs = probs / total

            self._age_lows = keys
            self._age_highs = np.r_[keys[1:], 85.0]
            self._dist_agebin.pars.a = np.arange(len(keys), dtype=int)
            self._dist_agebin.pars.p = probs

        # Configure TB state sampler (already validated and normalized in __init__)
        dist = self.pars.tb_state_distribution
        self._dist_tbstate.pars.a = np.array([int(getattr(TBS, k)) for k in dist], dtype=int)
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
        """
        Compute the Poisson intensity for the current timestep.

        Resolves ``immigration_rate`` to an ``ss.freq`` (coercing bare scalars
        via ``ss.freqperyear``), applies ``rel_immigration`` as a multiplicative
        scaler, then converts to expected events over the current step duration
        ``self.sim.t.dt`` using ``rate.to_events(dt)``. Non-finite or negative
        values are clamped to zero so the downstream ``ss.poisson`` draw is
        always well-defined.

        Returns:
            lam (float): Expected arrivals during the current timestep.
        """
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

        dt = getattr(self.sim.t, 'dt', getattr(self.sim, 'dt', None))
        dt_dur = dt if isinstance(dt, ss.dur) else ss.dur(dt)
        lam = float(rate.to_events(dt_dur))
        if not np.isfinite(lam) or lam < 0:
            lam = 0.0
        return lam

    def _lam_per_timestep(self, module):
        """
        Callback bound to ``ss.poisson(lam=...)``.

        Starsim invokes this each timestep with the owning module instance,
        allowing the Poisson rate to track dynamic changes to
        ``immigration_rate``, ``rel_immigration``, or ``sim.t.dt``.
        """
        return module.expected_immigrants_per_timestep()

    def _sample_ages(self, n):
        """
        Sample ages (in years) for ``n`` immigrants.

        Two-stage sampling: a bin index ``B_i`` is drawn from ``_dist_agebin``
        with probabilities given by the configured ``age_distribution``, then
        an in-bin position ``U_i ~ Uniform(0,1)`` is drawn from ``_dist_ageu``
        to yield ``A_i = L_{B_i} + U_i * (H_{B_i} - L_{B_i})``.

        If no bin structure has been configured (``_age_lows`` is ``None``),
        falls back to ``Uniform(0, 85)`` years.

        Returns:
            ages (np.ndarray): Float ages of length ``n`` (empty if ``n <= 0``).
        """
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


    def _init_tb_states(self, new_uids):
        """
        Seed TB state and imported-agent consistency fields for newly arrived agents.

        For each UID in ``new_uids``, samples a ``TBS`` state from
        ``_dist_tbstate`` and writes the TB fields needed for a consistent
        imported state:

        - ``tb.state`` is set to the sampled code.
        - ``tb.infected`` is ``True`` for any state outside
          ``{SUSCEPTIBLE, CLEARED, DEAD}`` (note: ``DEAD`` imports therefore
          have both ``infected = False`` and ``susceptible = False``).
        - ``tb.susceptible`` is ``True`` for ``SUSCEPTIBLE`` and ``CLEARED``.
        - ``tb.ever_infected`` is ``True`` for any state other than
          ``SUSCEPTIBLE``.
        - ``tb.on_treatment`` is ``True`` for ``TREATMENT``.
        - ``tb.ti_infected`` is set to ``-np.inf`` for all imports, so imported
          cases are not classified as model-generated incident infections.
        - ``tb.rr_reinfection`` is reset to 1.0, then overridden with
          ``tb.pars.rr_reinfection_cleared`` for ``CLEARED`` imports.
        - ``tb.ti_rr_reinfection_wane`` is set to ``np.inf``.
        - ``tb.rel_sus`` is 1.0 for non-cleared imports, mirroring
          ``rr_reinfection`` for ``CLEARED`` imports.
        - ``tb.rel_trans`` is 1.0 by default, ``tb.pars.trans_asymp`` for
          ``ASYMPTOMATIC``, ``tb.pars.trans_acute`` for ``ACUTE`` (only when
          the disease module is ``TBAcute``), and 0.0 for ``DEAD``.

        Raises:
            ValueError: If ``tb_state_distribution`` assigns positive mass to
                ``ACUTE`` (i.e., ``ACUTE`` survives zero-weight pruning in the
                validator) while the active disease module is a base ``TB``,
                not ``TBAcute``.

        Args:
            new_uids (array-like): UIDs of agents just appended to the population.

        Returns:
            sampled (np.ndarray): Integer ``TBS`` codes assigned to each UID,
                aligned with ``new_uids``.
        """
        tb = self.sim.diseases[self._tb_name]

        # Only allow ACUTE if the TB module is the acute variant
        state_vals = np.asarray(self._dist_tbstate.pars.a).astype(int)
        has_acute = int(TBS.ACUTE) in set(map(int, state_vals))
        if has_acute and not isinstance(tb, TBAcute):
            raise ValueError('tb_state_distribution includes ACUTE but TB module is not TBAcute')

        n = len(new_uids)
        sampled = self._dist_tbstate.rvs(n).astype(int)
        tb.state[new_uids] = sampled

        # Keep flags consistent with the TB state machine.
        # Imported TB is not counted as "new infection" for model incidence metrics.
        non_infected_states = np.array([int(TBS.SUSCEPTIBLE), int(TBS.CLEARED), int(TBS.DEAD)])
        is_inf = ~np.isin(sampled, non_infected_states)
        tb.infected[new_uids] = is_inf
        tb.susceptible[new_uids] = np.isin(sampled, [int(TBS.SUSCEPTIBLE), int(TBS.CLEARED)])
        tb.ever_infected[new_uids] = sampled != int(TBS.SUSCEPTIBLE)
        tb.on_treatment[new_uids] = sampled == int(TBS.TREATMENT)

        # Imported cases are exogenous; avoid recording them as incident infections.
        tb.ti_infected[new_uids] = -np.inf

        # Reinfection state for CLEARED imports.
        tb.rr_reinfection[new_uids] = 1.0
        tb.ti_rr_reinfection_wane[new_uids] = np.inf
        cleared = sampled == int(TBS.CLEARED)
        if np.any(cleared):
            tb.rr_reinfection[new_uids[cleared]] = float(tb.pars.rr_reinfection_cleared)

        # rel_sus: CLEARED carries reinfection susceptibility modifier.
        tb.rel_sus[new_uids] = 1.0
        tb.rel_sus[new_uids[cleared]] = tb.rr_reinfection[new_uids[cleared]]

        # rel_trans: ASYMPTOMATIC uses kappa; ACUTE uses alpha; DEAD = 0.
        tb.rel_trans[new_uids] = 1.0
        tb.rel_trans[new_uids[sampled == int(TBS.ASYMPTOMATIC)]] = float(tb.pars.trans_asymp)
        if isinstance(tb, TBAcute):
            tb.rel_trans[new_uids[sampled == int(TBS.ACUTE)]] = float(tb.pars.trans_acute)
        tb.rel_trans[new_uids[sampled == int(TBS.DEAD)]] = 0.0

        return sampled
    
    def step(self):
        """
        Execute one immigration event cycle for the current timestep.

        Sequence:

        1. Draw arrival count ``N_t = self._dist_n.rvs(1)[0]``.
        2. If ``N_t == 0``, reset ``self.n_immigrants`` and return ``[]``.
        3. Sample arrival characteristics (ages) via
           ``_get_immigrant_characteristics``.
        4. Grow the population (``self.sim.people.grow(N_t)``) and write ages.
        5. Seed TB states for the new UIDs via ``_init_tb_states`` and record
           the sampled codes in ``immigration_tb_status``.
        6. Assign new agents to households (and household edges, if available)
           via ``assign_immigrants_to_households``.
        7. Mark ``is_immigrant``, ``immigration_time`` (current step index),
           and ``age_at_immigration``.
        8. Cache ``self.n_immigrants`` for ``update_results``.

        Returns:
            new_uids: UIDs of newly added agents, or ``[]`` if ``N_t == 0``.
        """
        n_immigrants = int(self._dist_n.rvs(1)[0])
        
        if n_immigrants == 0:
            self.n_immigrants = 0
            return []
        
        characteristics = self._get_immigrant_characteristics(n_immigrants)
        
        new_uids = self.sim.people.grow(n_immigrants)
        
        self.sim.people.age[new_uids] = characteristics['ages']
        
        
        sampled_states = self._init_tb_states(new_uids)
        self.immigration_tb_status[new_uids] = sampled_states
        
        self.assign_immigrants_to_households(new_uids)

        self.is_immigrant[new_uids] = True
        self.immigration_time[new_uids] = float(self.ti)
        self.age_at_immigration[new_uids] = self.sim.people.age[new_uids]
        
        self.n_immigrants = n_immigrants
        return new_uids
    
    def assign_immigrants_to_households(self, new_uids):
        """
        Assign new arrivals to households and insert household edges.

        Searches ``self.sim.networks`` for the first network exposing a
        household-ID attribute. Attribute lookup is order-sensitive: ``hhid``
        (legacy convention) is checked before ``household_ids`` (current
        Starsim ``HouseholdNet``); the first match wins.

        Then:

        - If at least one valid (finite and ``>= 0``) household ID exists in
          that array, each immigrant is assigned uniformly at random from the
          set of distinct existing IDs using ``_dist_hhu`` (a single
          ``Uniform(0,1)`` draw per immigrant, mapped to a household index via
          ``floor(u * n_hh)``).
        - If no valid IDs exist, the ``N_t`` immigrants are assigned the
          batch-local IDs ``0..N_t-1`` (one singleton household each); no
          attempt is made to avoid collisions with IDs that may appear in
          subsequent timesteps.

        The assigned IDs are written both to the network's household array and
        to the module-local ``self.hhid``. Finally, ``_connect_immigrants_to_households``
        is called to append undirected household-network edges between each
        newcomer and all other members of the same household after assignment
        (including other newcomers assigned to that household), only for pairs
        not already present in ``hh_net.edges``.

        If no compatible household network is found, this method is a no-op.

        Args:
            new_uids (array-like): UIDs of the agents to assign.
        """
        # Find the first network with a household id array
        hh_net = None
        hh_attr = None
        for net in self.sim.networks.values():
            for attr in ['hhid', 'household_ids']:
                hh_arr = getattr(net, attr, None)
                if isinstance(hh_arr, (np.ndarray, ss.BaseArr)):
                    hh_net = net
                    hh_attr = attr
                    break
            if hh_net is not None:
                hh_net = net
                break

        if hh_net is None:
            return

        hh_arr = getattr(hh_net, hh_attr)
        hh_vals = np.asarray(hh_arr, dtype=float)
        valid = np.isfinite(hh_vals) & (hh_vals >= 0)
        existing_hhids = np.unique(hh_vals[valid]).astype(int)

        if len(existing_hhids):
            u = self._dist_hhu.rvs(len(new_uids))
            idx = np.floor(u * len(existing_hhids)).astype(int)
            idx = np.clip(idx, 0, len(existing_hhids) - 1)
            assigned_hhids = existing_hhids[idx]
            hh_arr[new_uids] = assigned_hhids
            self.hhid[new_uids] = assigned_hhids
        else:
            assigned_hhids = np.arange(len(new_uids), dtype=int)
            hh_arr[new_uids] = assigned_hhids
            self.hhid[new_uids] = assigned_hhids
        self._connect_immigrants_to_households(hh_net=hh_net, hh_arr=hh_arr, new_uids=new_uids, assigned_hhids=assigned_hhids)
        return

    @staticmethod
    def _has_edge_struct(net):
        """Return True if ``net`` exposes ``edges.p1`` and ``edges.p2`` arrays."""
        edges = getattr(net, 'edges', None)
        has_edges = edges is not None and hasattr(edges, 'p1') and hasattr(edges, 'p2')
        return bool(has_edges)

    def _connect_immigrants_to_households(self, hh_net, hh_arr, new_uids, assigned_hhids):
        """
        Append household-network edges for newly assigned immigrants.

        For each household ID assigned to any newcomer, retrieves the household
        membership after assignment (UID-safe via ``(hh_arr == hhid).uids``
        when ``hh_arr`` is an ``ss.BaseArr``, otherwise positional
        ``np.where``), and generates undirected ``(min(u,v), max(u,v))`` pairs
        between each newcomer and every other member of the same household.
        This includes newcomer-newcomer pairs when multiple immigrants are
        assigned to the same household. Pairs that are already present in
        ``hh_net.edges`` are skipped, and within-batch deduplication is done
        via a ``set``. New edges are appended with ``beta=1.0``.

        No-op if ``hh_net`` does not expose ``edges.p1``/``edges.p2`` (as
        checked by ``_has_edge_struct``); callers must also ensure the network
        supports ``hh_net.append(p1=..., p2=..., beta=...)`` for the new edges
        to be persisted.

        Complexity per household with ``k`` members and ``n_new`` newcomers is
        ``O(k * n_new)`` before deduplication.
        """
        if not self._has_edge_struct(hh_net):
            return

        hh_vals = np.asarray(hh_arr, dtype=float)
        p1_existing = np.asarray(hh_net.edges.p1, dtype=int)
        p2_existing = np.asarray(hh_net.edges.p2, dtype=int)
        existing_pairs = {
            (int(min(a, b)), int(max(a, b)))
            for a, b in zip(p1_existing, p2_existing)
            if a != b
        }

        assigned_hhids = np.asarray(assigned_hhids, dtype=int)
        pairs = set()
        for hhid in np.unique(assigned_hhids):
            if isinstance(hh_arr, ss.BaseArr):
                members = np.asarray((hh_arr == hhid).uids, dtype=int)
            else:
                members = np.where(hh_vals == hhid)[0].astype(int)
            if len(members) < 2:
                continue

            newcomers = np.asarray(new_uids[assigned_hhids == hhid], dtype=int)
            for u in newcomers:
                for v in members:
                    if u == v:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) not in existing_pairs:
                        pairs.add((a, b))

        if len(pairs):
            p1 = np.fromiter((a for a, _ in sorted(pairs)), dtype=int)
            p2 = np.fromiter((b for _, b in sorted(pairs)), dtype=int)
            hh_net.append(p1=ss.uids(p1), p2=ss.uids(p2), beta=np.ones(len(p1), dtype=float))
        return

    def update_results(self):
        """
        Write the current-timestep arrival count to ``results.n_immigrants``.

        Silent no-op if ``self.results`` has not yet been initialized as an
        ``ss.Results`` instance (e.g., when the module is constructed but the
        sim hasn't been run).
        """
        super().update_results()
        if isinstance(getattr(self, 'results', None), ss.Results):
            self.results['n_immigrants'][self.ti] = int(self.n_immigrants)
        return

    @staticmethod
    def _validate_tb_state_distribution(tb_state_distribution):
        """
        Validate ``tb_state_distribution`` and normalize it to a probability simplex.

        Drops zero-weight entries, then enforces:

        - non-empty mapping after dropping zeros,
        - every key is a valid ``TBS`` member name,
        - finite, non-negative weights,
        - strictly positive total mass.

        If the total exceeds 1, a warning is emitted but normalization
        proceeds. Returns a new dict ``{name: w / total}`` with weights
        summing to 1.

        Raises:
            ValueError: empty, all-zero, non-finite, or negative weights.
            KeyError: unknown ``TBS`` state name.
        """
        if not tb_state_distribution:
            raise ValueError('tb_state_distribution must be provided')
        dist = {k: float(v) for k, v in dict(tb_state_distribution).items() if v}
        for k, v in dist.items():
            if k not in TBS.__members__:
                raise KeyError(f'Unknown TB state "{k}" in tb_state_distribution')
            if not np.isfinite(v):
                raise ValueError(f'tb_state_distribution["{k}"] must be finite ({v})')
            if v < 0:
                raise ValueError(f'tb_state_distribution["{k}"] is negative ({v})')
        total = sum(dist.values())
        if total <= 0:
            raise ValueError('tb_state_distribution must include at least one positive probability')
        if total > 1.0:
            warnings.warn(f'tb_state_distribution sums to {total:.6g} (>1); normalizing automatically')
        return {k: v / total for k, v in dist.items()}

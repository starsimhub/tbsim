"""Migration demographics for TBsim (bidirectional immigration and emigration)."""

import warnings
import numpy as np
import starsim as ss

from .tb import TBS, TBAcute, get_tb

__all__ = ['Migration']


class Migration(ss.Demographics):
    """Bidirectional migration with household-aware turnover.

    Simulates both immigration (new agents entering the population) and
    emigration (existing agents leaving). Each timestep, arrivals are drawn
    from a Poisson process and departures are sampled from the active
    population. Immigrants are assigned an age, a TB disease state, and
    (when a ``HouseholdNet`` is present) a household. Emigrants are removed
    from the disease model and from household networks. Setting
    ``emigration_rate=0`` gives an immigration-only module.

    When ``maintain_population`` is enabled, the number of immigrants per
    step is adjusted so that the active (non-terminal) population stays
    close to the size recorded at the start of the simulation.

    If ``tb_state_distribution`` is not provided, a default entry-state mix
    is derived from the TB module's ``init_prev``, ``inf_non``, and
    ``inf_asy`` parameters so that immigrants roughly mirror the initial
    burden.

    Args (pars):
        immigration_rate (ss.freq/float): Expected immigrant arrivals per
            year, specified as an event rate (e.g. ``ss.freqperyear(100)``).
            Using ``ss.peryear`` (a probability rate) will trigger a warning.
        emigration_rate (ss.freq/float): Expected emigrant departures per
            year, same format as ``immigration_rate``.
        maintain_population (bool): If True, adjusts immigration each step to
            keep the active (non-terminal) population near the baseline
            recorded at ``init_post``.
        immigration_age_distribution (dict/None): Immigration age profile,
            ``{age_lower_bound: weight}`` for piecewise-uniform sampling of
            arriving immigrant ages. Does not affect emigration.
        emigration_age_distribution (dict/None): Optional age weights
            ``{age_lower_bound: weight}`` used to bias emigrant selection
            toward certain age groups. If None, emigrants are chosen
            uniformly at random from the active population.
        age_data (DataFrame/Series/array/str/None): Immigration age histogram
            in Starsim ``People`` format; overrides
            ``immigration_age_distribution`` when both are provided.
        max_age (float): Upper bound on sampled immigrant ages (default 85).
        tb_state_distribution (dict/None): ``{TBS state name: weight}`` mix
            for immigrants at entry. Terminal states (``DEAD``, ``REMOVED``)
            are stripped. If None, defaults are derived from TB module
            parameters.

    Attributes:
        hhid (IntArr): Household index on the network, or -1 if unassigned.
        is_immigrant (BoolState): True for agents created by immigration.
        immigration_time (FloatArr): Timestep index when the agent arrived.
        age_at_immigration (FloatArr): Age (years) at arrival.
        immigration_tb_status (IntArr): ``TBS`` value assigned at entry.
        is_emigrant (BoolState): True for agents removed by emigration.
        emigration_time (FloatArr): Timestep index when the agent departed.

    Results:
        n_immigrants: Number of arrivals in the last step.
        n_emigrants: Number of departures in the last step.
        net_migration: ``n_immigrants - n_emigrants`` for the last step.

    **Household integration**: When a ``HouseholdNet`` is present in the sim,
    immigrants are added to existing households (weighted by household size)
    and connected via complete-graph edges. Emigrants are removed from their
    household membership and edges.

    Example::

        import starsim as ss
        import tbsim

        mig = tbsim.Migration(pars=dict(
            immigration_rate=ss.freqperyear(100),
            emigration_rate=ss.freqperyear(80),
        ))
        sim = tbsim.Sim(demographics=[ss.Births(), ss.Deaths(), mig])
        sim.run()
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        # Track whether the user gave an explicit TB-state mix, so we know whether to derive a default at init.
        specified = ('tb_state_distribution' in (pars or {})) or ('tb_state_distribution' in kwargs)
        self.define_pars(
            immigration_rate=ss.freqperyear(10),
            emigration_rate=ss.freqperyear(0),
            maintain_population=False,
            immigration_age_distribution=None,
            emigration_age_distribution=None,
            age_data=None,
            max_age=85.0,
            tb_state_distribution=None,
        )
        self.update_pars(pars, **kwargs)
        self.tb_state_distribution_specified = bool(specified and self.pars.tb_state_distribution is not None)

        # Samplers. dist_age_data is the optional histogram sampler built from `age_data`; when it is None,
        # ages come either from piecewise-uniform bins (age_lows/age_highs + dist_age_bin) or a uniform draw.
        self.dist_n_immigrants = ss.poisson(lam=self._immigrants_per_timestep)
        self.dist_n_emigrants = ss.poisson(lam=self._emigrants_per_timestep)
        self.dist_age_bin = ss.choice(a=[0], p=[1.0])     # Which immigration age bin
        self.dist_age_uniform = ss.random()               # Position within an age bin, or scaled to max_age
        self.dist_tb_state = ss.choice(a=[-1], p=[1.0])    # Immigrant entry TB state
        self.dist_household = ss.random()                  # Household assignment for immigrants
        self.dist_emigrant = ss.random()                   # Emigrant selection scores
        self.dist_age_data = None
        self.age_lows = None
        self.age_highs = None
        self.emig_age_lows = None
        self.emig_age_weights = None

        self.define_states(
            ss.IntArr('hhid', default=-1),
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),
            ss.FloatArr('age_at_immigration', default=np.nan),
            ss.IntArr('immigration_tb_status', default=-1),
            ss.BoolState('is_emigrant', default=False),
            ss.FloatArr('emigration_time', default=np.nan),
        )
        self.baseline_population = None
        self.tb_name = None
        self.n_immigrants = 0
        self.n_emigrants = 0
        return

    def init_pre(self, sim):
        """Resolve the TB module and configure the age and TB-state samplers."""
        super().init_pre(sim)
        try:
            tb = get_tb(sim)
        except ValueError as exc:
            raise RuntimeError('Expected TB or TBAcute disease module for migration initialization') from exc
        self.tb_name = tb.name

        if not self.tb_state_distribution_specified:
            self.pars.tb_state_distribution = self._derive_default_tb_state_distribution(tb)
        self.pars.tb_state_distribution = self._validate_tb_state_distribution(self.pars.tb_state_distribution)

        self._configure_age_sampling()
        self._validate_max_age()
        self._configure_immigration_age_bins()
        self._configure_tb_state_sampler()
        self._configure_emig_age_weights()
        return

    def init_post(self):
        """Record the starting active population as the maintenance target."""
        super().init_post()
        self.baseline_population = self._count_active_pop()
        return

    def init_results(self):
        """Define per-step immigration, emigration, and net-migration counts."""
        super().init_results()
        self.define_results(
            ss.Result('n_immigrants', dtype=int, label='Number of immigrants'),
            ss.Result('n_emigrants', dtype=int, label='Number of emigrants'),
            ss.Result('net_migration', dtype=int, label='Net migration'),
        )
        return

    @property
    def tb(self):
        """The resolved TB (or TBAcute) disease module."""
        return self.sim.diseases[self.tb_name]

    # --- Configuration helpers (init-time) ---------------------------------

    def _validate_max_age(self):
        """Clamp ``max_age`` to a positive finite value, warning if it was invalid."""
        max_age = float(self.pars.max_age)
        if not np.isfinite(max_age) or max_age <= 0:
            warnings.warn(f'max_age invalid ({self.pars.max_age!r}); using 85.0', stacklevel=2)
            self.pars.max_age = 85.0
        return

    def _configure_age_sampling(self):
        """Build the immigrant-age histogram sampler from ``age_data``, if provided."""
        if self.pars.age_data is None:
            return
        if self.pars.immigration_age_distribution is not None:
            warnings.warn('age_data is set; ignoring immigration_age_distribution', stacklevel=2)
        self.dist_age_data = ss.People.get_age_dist(self.pars.age_data)
        self.age_lows = None
        self.age_highs = None
        return

    def _configure_immigration_age_bins(self):
        """Build piecewise-uniform age bins for sampling immigrant ages.

        Uses ``immigration_age_distribution`` (a ``{lower_bound: weight}``
        dict), installing a default profile when none is supplied. Skipped
        when ``age_data`` has already configured a histogram sampler.
        """
        if self.dist_age_data is not None:
            return
        max_age = float(self.pars.max_age)
        if self.pars.immigration_age_distribution is None:
            keys, weights = [0, 5, 15, 30, 50, 65], [0.15, 0.20, 0.25, 0.20, 0.15, 0.05]
            self.pars.immigration_age_distribution = {k: w for k, w in zip(keys, weights) if k < max_age}

        age_bins = self.pars.immigration_age_distribution
        if not (isinstance(age_bins, dict) and len(age_bins)):
            return

        edges = np.array(sorted(age_bins.keys()), dtype=float)
        bin_weights = np.array([age_bins[k] for k in edges], dtype=float)
        valid = np.isfinite(edges) & np.isfinite(bin_weights)
        edges, bin_weights = edges[valid], np.clip(bin_weights[valid], 0, None)
        edges = edges[edges < max_age]
        bin_weights = bin_weights[:len(edges)]
        if len(edges) == 0 or bin_weights.sum() <= 0:
            warnings.warn('immigration_age_distribution has no usable bins; using uniform [0, max_age)', stacklevel=2)
            return

        self.age_lows = edges
        self.age_highs = np.r_[edges[1:], max_age]
        self.dist_age_bin.pars.a = np.arange(len(edges), dtype=int)
        self.dist_age_bin.pars.p = bin_weights / bin_weights.sum()
        return

    def _configure_tb_state_sampler(self):
        """Point the entry-state sampler at the validated TB-state distribution."""
        weights = self.pars.tb_state_distribution
        self.dist_tb_state.pars.a = np.array([int(TBS[name]) for name in weights], dtype=int)
        self.dist_tb_state.pars.p = np.array(list(weights.values()), dtype=float)
        return

    def _configure_emig_age_weights(self):
        """Parse ``emigration_age_distribution`` into normalized per-bin selection weights."""
        spec = self.pars.emigration_age_distribution
        self.emig_age_lows = None
        self.emig_age_weights = None
        if spec is None:
            return
        if not isinstance(spec, dict) or not len(spec):
            warnings.warn('emigration_age_distribution is empty; using uniform emigration', stacklevel=2)
            return

        lows = np.array(sorted(spec.keys()), dtype=float)
        weights = np.array([spec[k] for k in lows], dtype=float)
        valid = np.isfinite(lows) & np.isfinite(weights)
        lows, weights = lows[valid], np.clip(weights[valid], 0, None)
        if len(lows) == 0 or weights.sum() <= 0:
            warnings.warn('emigration_age_distribution has no usable bins; using uniform emigration', stacklevel=2)
            return
        self.emig_age_lows = lows
        self.emig_age_weights = weights / weights.sum()
        return

    @staticmethod
    def _extract_scalar_from_spec(spec, candidates=('p', 'v', 'value')):
        """Extract a representative scalar from a Starsim spec or plain numeric input."""
        if spec is None:
            return None
        if np.isscalar(spec):
            return float(spec)

        def first_scalar(values):
            """Return the first usable scalar (or mean of an array) from ``values``."""
            for val in values:
                if val is None:
                    continue
                if np.isscalar(val):
                    return float(val)
                arr = np.asarray(val, dtype=float)
                if arr.size:
                    return float(np.nanmean(arr))
            return None

        pars = getattr(spec, 'pars', None)
        if pars is not None:
            result = first_scalar(pars[key] for key in candidates if key in pars)
            if result is not None:
                return result
        return first_scalar(getattr(spec, key, None) for key in candidates)

    @classmethod
    def _derive_default_tb_state_distribution(cls, tb):
        """Derive a default immigrant TB-state mix from the TB module's parameters.

        Splits ``init_prev`` between INFECTION and ASYMPTOMATIC in proportion
        to the relative ``inf_non``/``inf_asy`` progression rates, with the
        remainder SUSCEPTIBLE.
        """
        init_prev = cls._extract_scalar_from_spec(tb.pars.init_prev, candidates=('p', 'v', 'value')) or 0.0
        init_prev = float(np.clip(init_prev, 0.0, 1.0))

        inf_non = max(float(cls._extract_scalar_from_spec(tb.pars.inf_non, candidates=('value', 'v')) or 0.0), 0.0)
        inf_asy = max(float(cls._extract_scalar_from_spec(tb.pars.inf_asy, candidates=('value', 'v')) or 0.0), 0.0)
        total = inf_non + inf_asy
        asymp_fraction = (inf_asy / total) if total > 0 else 0.0

        asymptomatic = init_prev * asymp_fraction
        return {
            TBS.SUSCEPTIBLE.name: 1.0 - init_prev,
            TBS.INFECTION.name: init_prev - asymptomatic,
            TBS.ASYMPTOMATIC.name: asymptomatic,
        }

    @staticmethod
    def _validate_tb_state_distribution(tb_state_distribution):
        """Drop unknown/terminal states, coerce weights, and return a normalized mix."""
        if not tb_state_distribution:
            raise ValueError('tb_state_distribution must be provided')
        valid = {}
        for name, weight in dict(tb_state_distribution).items():
            if name not in TBS._member_names_:
                warnings.warn(f'Ignoring unknown TB state "{name}" in tb_state_distribution', stacklevel=2)
                continue
            w = float(weight)
            valid[name] = w if (np.isfinite(w) and w >= 0) else 0.0
        for term in [TBS.DEAD.name, TBS.REMOVED.name]:
            if term in valid:
                warnings.warn(f'Removing terminal state {term} from tb_state_distribution', stacklevel=2)
                valid.pop(term)
        weight_sum = sum(valid.values())
        if weight_sum <= 0:
            raise ValueError('tb_state_distribution must include at least one positive probability')
        return {k: v / weight_sum for k, v in valid.items() if v > 0}

    # --- Rates -------------------------------------------------------------

    @staticmethod
    def _expected_events_per_timestep(rate_spec, dt):
        """Convert an annual ``ss.freq`` (or scalar) migration rate into expected events per timestep."""
        if rate_spec is None:
            return 0.0
        if isinstance(rate_spec, ss.Rate):
            if not isinstance(rate_spec, ss.freq):
                warnings.warn('Migration rates should be ss.freq (event rate), not ss.peryear; treating as event rate', stacklevel=2)
            annual_rate = rate_spec
        else:
            annual_rate = ss.freqperyear(float(rate_spec))
        expected = float(annual_rate.to_events(dt))
        return expected if (np.isfinite(expected) and expected >= 0) else 0.0

    def expected_immigrants_per_timestep(self):
        """Expected number of immigrant arrivals this timestep."""
        return self._expected_events_per_timestep(self.pars.immigration_rate, self.t.dt)

    def expected_emigrants_per_timestep(self):
        """Expected number of emigrant departures this timestep."""
        return self._expected_events_per_timestep(self.pars.emigration_rate, self.t.dt)

    def _immigrants_per_timestep(self, module):
        """Poisson rate callback for immigrant arrivals."""
        return module.expected_immigrants_per_timestep()

    def _emigrants_per_timestep(self, module):
        """Poisson rate callback for emigrant departures."""
        return module.expected_emigrants_per_timestep()

    # --- Population bookkeeping --------------------------------------------

    def _active_pop_uids(self):
        """UIDs of alive agents that are not in a terminal TB state."""
        uid = np.asarray(self.sim.people.uid, dtype=int)
        alive = np.asarray(self.sim.people.alive, dtype=bool)
        active = ~np.isin(np.asarray(self.tb.state), [*TBS.terminal_states()])
        return ss.uids(uid[alive & active])

    def _count_active_pop(self):
        """Number of alive, non-terminal agents."""
        return int(len(self._active_pop_uids()))

    def _adjust_arrivals_for_pop_target(self, n_immigrants, n_emigrants):
        """Under ``maintain_population``, top up arrivals to hold the active population at baseline."""
        if not self.pars.maintain_population:
            return n_immigrants
        n_active = self._count_active_pop()
        n_target = self.baseline_population if self.baseline_population is not None else n_active
        n_projected = n_active + n_immigrants - n_emigrants
        return max(int(n_immigrants + (n_target - n_projected)), 0)

    # --- Age sampling ------------------------------------------------------

    def _sample_ages(self, n):
        """Sample ``n`` immigrant ages from the configured age histogram, bins, or a uniform fallback."""
        if n <= 0:
            return np.empty(0, dtype=float)
        if self.dist_age_data is not None:
            return np.asarray(self.dist_age_data.rvs(n), dtype=float)
        if self.age_lows is None or self.age_highs is None:
            return self.dist_age_uniform.rvs(n) * float(self.pars.max_age)
        age_bin = self.dist_age_bin.rvs(n).astype(int)
        within_bin = self.dist_age_uniform.rvs(n)
        return self.age_lows[age_bin] + within_bin * (self.age_highs[age_bin] - self.age_lows[age_bin])

    # --- Household integration ---------------------------------------------

    def _find_household_network(self):
        """Return the household network if one is present in the sim, else None."""
        for net in self.sim.networks.values():
            if hasattr(net, 'household_ids') and hasattr(net, 'remove_uids'):
                return net
        return None

    def _household_sizes(self, household_net):
        """Return member counts indexed by household ID, computed from live agents."""
        alive = self.sim.people.alive.uids
        ids = np.asarray(household_net.household_ids[alive], dtype=float)
        valid = ~np.isnan(ids)
        if not np.any(valid):
            return np.empty(0, dtype=float)
        return np.bincount(ids[valid].astype(int)).astype(float)

    def _weighted_household_indices(self, household_net, sample_uids):
        """Pick household IDs for new members with probability proportional to household size."""
        household_sizes = self._household_sizes(household_net)
        if len(household_sizes) == 0 or household_sizes.sum() <= 0:
            return np.empty(0, dtype=int)
        draws = np.asarray(self.dist_household.rvs(sample_uids), dtype=float)
        cdf = np.cumsum(household_sizes / household_sizes.sum())
        return np.searchsorted(cdf, draws, side='right').astype(int)

    def _append_household_edges(self, household_net, uid, member_uids):
        """Connect a new member ``uid`` to every existing member of its household."""
        member_uids = member_uids[member_uids != uid]
        if len(member_uids) == 0:
            return
        p1 = ss.uids(member_uids)
        p2 = ss.uids(np.full(len(member_uids), int(uid), dtype=int))
        beta = np.ones(len(p1), dtype=ss.dtypes.float)
        household_net.append(p1=p1, p2=p2, beta=beta)
        return

    def _create_household_singletons(self, household_net, new_uids):
        """Place each new agent in its own fresh household (used when no households exist yet)."""
        new_uids = ss.uids(new_uids)
        assigned = household_net.n_households + np.arange(len(new_uids))
        household_net.n_households += len(new_uids)
        household_net.household_ids[new_uids] = assigned
        self.hhid[new_uids] = assigned
        return assigned

    def assign_immigrants_to_households(self, new_uids):
        """Assign immigrants to existing households (size-weighted) and wire up their edges."""
        household_net = self._find_household_network()
        if household_net is None:
            return None

        household_indices = self._weighted_household_indices(household_net, new_uids)
        if len(household_indices) == 0:
            return self._create_household_singletons(household_net, new_uids)

        assigned = np.empty(len(new_uids), dtype=int)
        for i, uid in enumerate(np.asarray(new_uids, dtype=int)):
            household_index = int(household_indices[i])
            members = ss.uids(household_net.household_ids == household_index)
            household_net.household_ids[ss.uids(uid)] = household_index
            self._append_household_edges(household_net, uid, members)
            assigned[i] = household_index
        self.hhid[new_uids] = assigned
        return assigned

    # --- Immigration -------------------------------------------------------

    def _init_tb_states(self, new_uids):
        """Assign immigrant entry TB states and the dependent TB flags/timers, returning the states."""
        tb = self.tb
        if TBS.ACUTE in np.asarray(self.dist_tb_state.pars.a, dtype=int) and not isinstance(tb, TBAcute):
            raise ValueError(f'tb_state_distribution includes {TBS.ACUTE.name} but TB module is not TBAcute')

        entry_states = self.dist_tb_state.rvs(len(new_uids)).astype(int)
        susceptible_like = [TBS.SUSCEPTIBLE, TBS.CLEARED]
        infected_mask = ~np.isin(entry_states, [*susceptible_like, *TBS.terminal_states()])

        tb.state[new_uids] = entry_states
        tb.infected[new_uids] = infected_mask
        tb.susceptible[new_uids] = np.isin(entry_states, susceptible_like)
        tb.ever_infected[new_uids] = entry_states != TBS.SUSCEPTIBLE
        tb.on_treatment[new_uids] = entry_states == TBS.TREATMENT

        # Imported infections are not counted as model-incident, so leave ti_infected at -inf except for the
        # current step on agents that arrive already infected.
        tb.ti_infected[new_uids] = -np.inf
        tb.ti_infected[new_uids[infected_mask]] = self.ti
        tb.ti_asymp[new_uids] = -np.inf
        asymptomatic = entry_states == TBS.ASYMPTOMATIC
        if np.any(asymptomatic):
            tb.ti_asymp[new_uids[asymptomatic]] = self.ti

        # Reinfection susceptibility: cleared arrivals carry the cleared-relative-risk.
        is_cleared = entry_states == TBS.CLEARED
        tb.rr_reinfection[new_uids] = 1.0
        tb.ti_rr_reinfection_wane[new_uids] = np.inf
        if np.any(is_cleared):
            tb.rr_reinfection[new_uids[is_cleared]] = float(tb.pars.rr_reinfection_cleared)
        tb.rel_sus[new_uids] = 1.0
        tb.rel_sus[new_uids[is_cleared]] = tb.rr_reinfection[new_uids[is_cleared]]

        # Relative transmissibility by state.
        tb.rel_trans[new_uids] = 1.0
        tb.rel_trans[new_uids[asymptomatic]] = float(tb.pars.trans_asymp)
        if isinstance(tb, TBAcute):
            tb.rel_trans[new_uids[entry_states == TBS.ACUTE]] = float(tb.pars.trans_acute)
        return entry_states

    def _perform_immigration(self, n_arrivals):
        """Create ``n_arrivals`` immigrants with ages, TB states, and household membership."""
        self.n_immigrants = max(int(n_arrivals), 0)
        if self.n_immigrants <= 0:
            return ss.uids()
        new_uids = self.sim.people.grow(self.n_immigrants)
        self.sim.people.age[new_uids] = self._sample_ages(self.n_immigrants)
        self.immigration_tb_status[new_uids] = self._init_tb_states(new_uids)
        self.assign_immigrants_to_households(new_uids)
        self.is_immigrant[new_uids] = True
        self.immigration_time[new_uids] = float(self.ti)
        self.age_at_immigration[new_uids] = self.sim.people.age[new_uids]
        return new_uids

    # --- Emigration --------------------------------------------------------

    def _draw_emigrants(self):
        """Draw the number of emigrant departures this timestep (0 if emigration is off)."""
        if self.expected_emigrants_per_timestep() <= 0:
            return 0
        return int(self.dist_n_emigrants.rvs(1)[0])

    def _emig_weights_for_uids(self, uids):
        """Per-agent emigration weights from the age profile, or None for uniform selection."""
        if self.emig_age_lows is None or self.emig_age_weights is None or len(uids) == 0:
            return None
        ages = np.asarray(self.sim.people.age[uids], dtype=float)
        age_bins = np.searchsorted(self.emig_age_lows, ages, side='right') - 1
        weights = np.zeros(len(uids), dtype=float)
        valid = age_bins >= 0
        if np.any(valid):
            weights[valid] = self.emig_age_weights[age_bins[valid]]
        return weights

    def _sample_emigrants(self, n_requested):
        """Select up to ``n_requested`` emigrants, age-weighted if configured, else uniformly at random."""
        eligible = self._active_pop_uids()
        if len(eligible) == 0 or n_requested <= 0:
            return ss.uids()
        n_select = min(int(n_requested), len(eligible))

        weights = self._emig_weights_for_uids(eligible)
        if weights is None:
            scores = np.asarray(self.dist_emigrant.rvs(eligible), dtype=float)
            return eligible[np.argsort(scores)[:n_select]]

        # Weighted sampling without replacement via the exponential (Efraimidis-Spirakis) key trick.
        selected = np.array([], dtype=int)
        pos_mask = weights > 0
        weighted_uids = np.asarray(eligible[pos_mask], dtype=int)
        if len(weighted_uids):
            draws = np.clip(np.asarray(self.dist_emigrant.rvs(weighted_uids), dtype=float), 1e-12, 1.0)
            keys = -np.log(draws) / weights[pos_mask]
            n_weighted = min(n_select, len(weighted_uids))
            selected = weighted_uids[np.argsort(keys)[:n_weighted]]

        # If the weighted pool was too small, fill the remainder uniformly from the rest.
        if len(selected) < n_select:
            sel_set = set(selected.tolist())
            fallback = np.array([u for u in np.asarray(eligible, dtype=int) if u not in sel_set], dtype=int)
            if len(fallback):
                n_fill = n_select - len(selected)
                fallback_scores = np.asarray(self.dist_emigrant.rvs(fallback), dtype=float)
                selected = np.concatenate([selected, fallback[np.argsort(fallback_scores)[:n_fill]]])

        return ss.uids(selected)

    def _apply_emigration(self, emigrant_uids):
        """Remove emigrants from the TB model, their households, and the active population."""
        self.n_emigrants = len(emigrant_uids)
        if self.n_emigrants == 0:
            return emigrant_uids

        tb = self.tb
        tb.state[emigrant_uids] = TBS.REMOVED
        tb.infected[emigrant_uids] = False
        tb.susceptible[emigrant_uids] = False
        tb.on_treatment[emigrant_uids] = False
        tb.rel_sus[emigrant_uids] = 0
        tb.rel_trans[emigrant_uids] = 0

        household_net = self._find_household_network()
        if household_net is not None:
            household_net.remove_uids(emigrant_uids)
        self.hhid[emigrant_uids] = -1
        self.is_emigrant[emigrant_uids] = True
        self.emigration_time[emigrant_uids] = float(self.ti)
        self.sim.people.request_removal(emigrant_uids)
        return emigrant_uids

    # --- Step / results ----------------------------------------------------

    def step(self):
        """Sample and apply this timestep's emigration, then immigration."""
        n_immigrants = int(self.dist_n_immigrants.rvs(1)[0])
        emig_uids = self._sample_emigrants(self._draw_emigrants())
        n_adjusted = self._adjust_arrivals_for_pop_target(n_immigrants, len(emig_uids))
        self._apply_emigration(emig_uids)
        return self._perform_immigration(n_adjusted)

    def update_results(self):
        """Record this step's immigration, emigration, and net-migration counts."""
        super().update_results()
        if isinstance(self.results, ss.Results):
            self.results['n_immigrants'][self.ti] = int(self.n_immigrants)
            self.results['n_emigrants'][self.ti] = int(self.n_emigrants)
            self.results['net_migration'][self.ti] = int(self.n_immigrants - self.n_emigrants)
        return

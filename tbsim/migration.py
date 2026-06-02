"""Migration and immigration demographics for TBsim."""

import warnings
import numpy as np
import starsim as ss

from .tb import TBS, TB, TBAcute, get_tb

__all__ = ['Migration']


class Migration(ss.Demographics):
    """Bidirectional migration with household-aware turnover.

    Simulates both immigration (new agents entering the population) and
    emigration (existing agents leaving). Each timestep, arrivals are drawn
    from a Poisson process and departures are sampled from the active
    population. Immigrants are assigned an age, a TB disease state, and
    (when a ``HouseholdNet`` is present) a household. Emigrants are removed
    from the disease model and from household networks.

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
            toward certain age groups. Bins are half-open intervals
            ``[lower, next_lower)``, with the last bin ending at ``max_age``.
            Ages below the smallest key or at/above ``max_age`` are not
            age-weighted (they enter the uniform fallback pool). If None,
            emigrants are chosen uniformly at random from the active population.
        age_data (DataFrame/Series/array/str/None): Immigration age histogram
            in Starsim ``People`` format; overrides
            ``immigration_age_distribution`` when both are provided.
        max_age (float): Upper age bound for immigration sampling and for the
            top end of emigration age bins (default 85).
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
        input_pars = {} if pars is None else dict(pars)
        tb_state_dist_specified = ('tb_state_distribution' in input_pars) or ('tb_state_distribution' in kwargs)
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
        self._tb_state_distribution_specified = bool(tb_state_dist_specified and self.pars.tb_state_distribution is not None)

        self._dist_n_immigrants = ss.poisson(lam=self._lam_immigrants_per_timestep)
        self._dist_n_emigrants = ss.poisson(lam=self._lam_emigrants_per_timestep)
        self._dist_agebin = ss.choice(a=[0], p=[1.0])
        self._dist_ageu = ss.random()
        self._dist_tbstate = ss.choice(a=[-1], p=[1.0])
        self._dist_hhu = ss.random()
        self._dist_emig_u = ss.random()
        self._dist_age = None
        self._age_lows = None
        self._age_highs = None
        self._emig_age_lows = None
        self._emig_age_highs = None
        self._emig_age_weights = None

        self.define_states(
            ss.IntArr('hhid', default=-1),
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),
            ss.FloatArr('age_at_immigration', default=np.nan),
            ss.IntArr('immigration_tb_status', default=-1),
            ss.BoolState('is_emigrant', default=False),
            ss.FloatArr('emigration_time', default=np.nan),
        )
        self._baseline_population = None
        self._fresh_import_uids = None
        self._tb_name = None
        self.n_immigrants = 0
        self.n_emigrants = 0
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        try:
            tb = get_tb(sim)
            self._tb_name = tb.name
        except ValueError as exc:
            raise RuntimeError('Expected TB or TBAcute disease module for migration initialization') from exc
        if not self._tb_state_distribution_specified:
            self.pars.tb_state_distribution = self._derive_default_tb_state_distribution(tb)
        self.pars.tb_state_distribution = self._validate_tb_state_distribution(self.pars.tb_state_distribution)
        self._configure_age_sampling()
        max_age = float(self.pars.max_age)
        if not np.isfinite(max_age) or max_age <= 0:
            max_age = 85.0
            warnings.warn(f'max_age invalid ({self.pars.max_age!r}); using {max_age}', stacklevel=2)
            self.pars.max_age = max_age
        if self._dist_age is None and self.pars.immigration_age_distribution is None:
            default_keys = [0, 5, 15, 30, 50, 65]
            self.pars.immigration_age_distribution = {k: w for k, w in zip(default_keys, [0.15, 0.20, 0.25, 0.20, 0.15, 0.05]) if k < max_age}

        immigration_age_bins = self.pars.immigration_age_distribution
        if isinstance(immigration_age_bins, dict) and len(immigration_age_bins) and self._dist_age is None:
            bin_edges = np.array(sorted(immigration_age_bins.keys()), dtype=float)
            bin_weights = np.array([immigration_age_bins[k] for k in bin_edges], dtype=float)
            valid = np.isfinite(bin_edges) & np.isfinite(bin_weights)
            bin_edges, bin_weights = bin_edges[valid], bin_weights[valid]
            bin_weights = np.clip(bin_weights, 0, None)
            bin_edges = bin_edges[bin_edges < max_age]
            bin_weights = bin_weights[:len(bin_edges)]
            weight_sum = bin_weights.sum()
            if len(bin_edges) == 0 or weight_sum <= 0:
                warnings.warn('immigration_age_distribution has no usable bins; using uniform [0, max_age)', stacklevel=2)
                self._age_lows = None
                self._age_highs = None
            else:
                bin_weights = bin_weights / weight_sum
                self._age_lows = bin_edges
                self._age_highs = np.r_[bin_edges[1:], max_age]
                self._dist_agebin.pars.a = np.arange(len(bin_edges), dtype=int)
                self._dist_agebin.pars.p = bin_weights

        tb_entry_weights = self.pars.tb_state_distribution
        self._dist_tbstate.pars.a = np.array([int(TBS[state_name]) for state_name in tb_entry_weights], dtype=int)
        self._dist_tbstate.pars.p = np.array(list(tb_entry_weights.values()), dtype=float)
        self._configure_emig_age_weights()
        return

    def init_post(self):
        super().init_post()
        self._baseline_population = self._count_active_pop()
        return

    def _configure_age_sampling(self):
        age_data = self.pars.age_data
        if age_data is None:
            return
        if self.pars.immigration_age_distribution is not None:
            warnings.warn('age_data is set; ignoring immigration_age_distribution', stacklevel=2)
        self._dist_age = ss.People.get_age_dist(age_data)
        self._age_lows = None
        self._age_highs = None
        return

    def _configure_emig_age_weights(self):
        spec = self.pars.emigration_age_distribution
        if spec is None:
            self._emig_age_lows = None
            self._emig_age_highs = None
            self._emig_age_weights = None
            return
        if not isinstance(spec, dict) or not len(spec):
            warnings.warn('emigration_age_distribution is empty; using uniform emigration', stacklevel=2)
            self._emig_age_lows = None
            self._emig_age_highs = None
            self._emig_age_weights = None
            return

        max_age = float(self.pars.max_age)
        age_lows = np.array(sorted(spec.keys()), dtype=float)
        age_weights = np.array([spec[k] for k in age_lows], dtype=float)
        valid = np.isfinite(age_lows) & np.isfinite(age_weights)
        age_lows, age_weights = age_lows[valid], age_weights[valid]
        age_weights = np.clip(age_weights, 0, None)
        age_lows = age_lows[age_lows < max_age]
        age_weights = age_weights[:len(age_lows)]
        weight_sum = age_weights.sum()
        if len(age_lows) == 0 or weight_sum <= 0:
            warnings.warn('emigration_age_distribution has no usable bins; using uniform emigration', stacklevel=2)
            self._emig_age_lows = None
            self._emig_age_highs = None
            self._emig_age_weights = None
            return
        self._emig_age_lows = age_lows
        self._emig_age_highs = np.r_[age_lows[1:], max_age]
        self._emig_age_weights = age_weights / weight_sum
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_immigrants', dtype=int, label='Number of immigrants'),
            ss.Result('n_emigrants', dtype=int, label='Number of emigrants'),
            ss.Result('net_migration', dtype=int, label='Net migration'),
        )
        return

    @staticmethod
    def _expected_events_per_timestep(rate_spec, dt):
        if rate_spec is None:
            return 0.0
        if isinstance(rate_spec, ss.Rate):
            if not isinstance(rate_spec, ss.freq):
                warnings.warn('Migration rates should be ss.freq (event rate), not ss.peryear; treating as event rate', stacklevel=2)
            annual_rate = rate_spec
        else:
            annual_rate = ss.freqperyear(float(rate_spec))
        expected_events = float(annual_rate.to_events(dt))
        if not np.isfinite(expected_events) or expected_events < 0:
            expected_events = 0.0
        return expected_events

    @staticmethod
    def _extract_scalar_from_spec(spec, candidates=('p', 'v', 'value')):
        """Extract a scalar value from a Starsim spec or plain numeric input."""
        if spec is None:
            return None
        if np.isscalar(spec):
            return float(spec)
        pars = getattr(spec, 'pars', None)
        if pars is not None:
            for key in candidates:
                if key in pars:
                    val = pars[key]
                    if np.isscalar(val):
                        return float(val)
                    arr = np.asarray(val, dtype=float)
                    if arr.size:
                        return float(np.nanmean(arr))
        for key in candidates:
            if hasattr(spec, key):
                val = getattr(spec, key)
                if np.isscalar(val):
                    return float(val)
                arr = np.asarray(val, dtype=float)
                if arr.size:
                    return float(np.nanmean(arr))
        return None

    @classmethod
    def _derive_default_tb_state_distribution(cls, tb):
        """Derive default migrant TB-state mix from TB module parameters."""
        init_prev = cls._extract_scalar_from_spec(tb.pars.init_prev, candidates=('p', 'v', 'value'))
        if init_prev is None:
            init_prev = 0.0
        init_prev = float(np.clip(init_prev, 0.0, 1.0))

        inf_non = cls._extract_scalar_from_spec(tb.pars.inf_non, candidates=('value', 'v'))
        inf_asy = cls._extract_scalar_from_spec(tb.pars.inf_asy, candidates=('value', 'v'))
        inf_non = max(float(inf_non or 0.0), 0.0)
        inf_asy = max(float(inf_asy or 0.0), 0.0)
        progression_total = inf_non + inf_asy
        asymp_fraction = (inf_asy / progression_total) if progression_total > 0 else 0.0

        asymptomatic = init_prev * asymp_fraction
        infection = init_prev - asymptomatic
        susceptible = 1.0 - init_prev

        return {
            TBS.SUSCEPTIBLE.name: susceptible,
            TBS.INFECTION.name: infection,
            TBS.ASYMPTOMATIC.name: asymptomatic,
        }

    def expected_immigrants_per_timestep(self):
        return self._expected_events_per_timestep(self.pars.immigration_rate, self.t.dt)

    def expected_emigrants_per_timestep(self):
        return self._expected_events_per_timestep(self.pars.emigration_rate, self.t.dt)

    def _lam_immigrants_per_timestep(self, module):
        return module.expected_immigrants_per_timestep()

    def _lam_emigrants_per_timestep(self, module):
        return module.expected_emigrants_per_timestep()

    def _bound_ages(self, ages):
        """Coerce sampled ages into the valid simulation range [0, max_age)."""
        ages = np.asarray(ages, dtype=float)
        max_age = float(self.pars.max_age)
        if not np.isfinite(max_age) or max_age <= 0:
            return np.clip(ages, 0.0, None)
        return np.clip(ages, 0.0, np.nextafter(max_age, 0.0))

    def _sample_ages(self, n):
        if n <= 0:
            return np.empty(0, dtype=float)
        if self._dist_age is not None:
            return self._bound_ages(self._dist_age.rvs(n))
        if self._age_lows is None or self._age_highs is None:
            return self._bound_ages(self._dist_ageu.rvs(n) * float(self.pars.max_age))
        age_bin = self._dist_agebin.rvs(n).astype(int)
        within_bin = self._dist_ageu.rvs(n)
        bin_lower = self._age_lows[age_bin]
        bin_upper = self._age_highs[age_bin]
        return self._bound_ages(bin_lower + within_bin * (bin_upper - bin_lower))

    def _draw_emigrants(self):
        if self.expected_emigrants_per_timestep() <= 0:
            return 0
        return int(self._dist_n_emigrants.rvs(1)[0])

    def _adjust_arrivals_for_pop_target(self, n_immigrants, n_emigrants):
        if not self.pars.maintain_population:
            return n_immigrants

        n_active    = self._count_active_pop()
        n_target    = self._baseline_population if self._baseline_population is not None else n_active
        n_projected = n_active + n_immigrants - n_emigrants
        delta       = n_target - n_projected
        return max(int(n_immigrants + delta), 0)

    def _active_pop_uids(self):
        tb = self.sim.diseases[self._tb_name]
        uid   = np.asarray(self.sim.people.uid, dtype=int)
        alive = np.asarray(self.sim.people.alive, dtype=bool)
        ok    = ~np.isin(np.asarray(tb.state), [*TBS.terminal_states()])
        return ss.uids(uid[alive & ok])

    def _count_active_pop(self):
        return int(len(self._active_pop_uids()))

    def _find_household_network(self):
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
        household_sizes = self._household_sizes(household_net)
        if len(household_sizes) == 0 or household_sizes.sum() <= 0:
            return np.empty(0, dtype=int)
        draws = np.asarray(self._dist_hhu.rvs(sample_uids), dtype=float)
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
        new_uids = ss.uids(new_uids)
        assigned = household_net.n_households + np.arange(len(new_uids))
        household_net.n_households += len(new_uids)
        household_net.household_ids[new_uids] = assigned
        self.hhid[new_uids] = assigned
        return assigned

    def assign_immigrants_to_households(self, new_uids):
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

    def _init_tb_states(self, new_uids):
        tb = self.sim.diseases[self._tb_name]
        entry_state_codes = np.asarray(self._dist_tbstate.pars.a, dtype=int)
        if TBS.ACUTE in entry_state_codes and not isinstance(tb, TBAcute):
            raise ValueError(f'tb_state_distribution includes {TBS.ACUTE.name} but TB module is not TBAcute')

        entry_states = self._dist_tbstate.rvs(len(new_uids)).astype(int)
        tb.state[new_uids] = entry_states
        tb.infected[new_uids] = ~np.isin(entry_states, [TBS.SUSCEPTIBLE, TBS.CLEARED, *TBS.terminal_states()])
        tb.susceptible[new_uids] = np.isin(entry_states, [TBS.SUSCEPTIBLE, TBS.CLEARED])
        tb.ever_infected[new_uids] = entry_states != TBS.SUSCEPTIBLE
        tb.on_treatment[new_uids] = entry_states == TBS.TREATMENT
        tb.ti_infected[new_uids] = -np.inf
        infected_states = ~np.isin(entry_states, [TBS.SUSCEPTIBLE, TBS.CLEARED, *TBS.terminal_states()])
        tb.ti_infected[new_uids[infected_states]] = self.ti
        tb.ti_asymp[new_uids] = -np.inf
        asymptomatic = entry_states == TBS.ASYMPTOMATIC
        if np.any(asymptomatic):
            tb.ti_asymp[new_uids[asymptomatic]] = self.ti
        tb.rr_reinfection[new_uids] = 1.0
        tb.ti_rr_reinfection_wane[new_uids] = np.inf
        is_cleared = entry_states == TBS.CLEARED
        if np.any(is_cleared):
            tb.rr_reinfection[new_uids[is_cleared]] = float(tb.pars.rr_reinfection_cleared)
        tb.rel_sus[new_uids] = 1.0
        tb.rel_sus[new_uids[is_cleared]] = tb.rr_reinfection[new_uids[is_cleared]]
        tb.rel_trans[new_uids] = 1.0
        tb.rel_trans[new_uids[asymptomatic]] = float(tb.pars.trans_asymp)
        if isinstance(tb, TBAcute):
            acute = entry_states == TBS.ACUTE
            tb.rel_trans[new_uids[acute]] = float(tb.pars.trans_acute)
        return entry_states

    def _perform_immigration(self, n_arrivals):
        if n_arrivals <= 0:
            self.n_immigrants = 0
            self._fresh_import_uids = None
            return ss.uids()
        arrival_ages = self._sample_ages(n_arrivals)
        new_uids = self.sim.people.grow(n_arrivals)
        self.sim.people.age[new_uids] = arrival_ages
        self.immigration_tb_status[new_uids] = self._init_tb_states(new_uids)
        self.assign_immigrants_to_households(new_uids)
        self.is_immigrant[new_uids] = True
        self.immigration_time[new_uids] = float(self.ti)
        self.age_at_immigration[new_uids] = self.sim.people.age[new_uids]
        self.n_immigrants = n_arrivals
        self._fresh_import_uids = new_uids
        return new_uids

    def _eligible_emigrants(self):
        return self._active_pop_uids()

    def _emig_weights_for_uids(self, uids):
        if self._emig_age_lows is None or self._emig_age_weights is None or len(uids) == 0:
            return None
        ages = np.asarray(self.sim.people.age[uids], dtype=float)
        weights = np.zeros(len(uids), dtype=float)
        for lo, hi, w in zip(self._emig_age_lows, self._emig_age_highs, self._emig_age_weights):
            in_bin = (ages >= lo) & (ages < hi)
            weights[in_bin] = w
        return weights

    def _sample_emigrants(self, n_requested):
        eligible = self._eligible_emigrants()
        if len(eligible) == 0 or n_requested <= 0:
            return ss.uids()
        n_select = min(int(n_requested), len(eligible))

        weights = self._emig_weights_for_uids(eligible)
        if weights is None:
            scores = np.asarray(self._dist_emig_u.rvs(eligible), dtype=float)
            order  = np.argsort(scores)
            return eligible[order[:n_select]]

        selected     = np.array([], dtype=int)
        pos_mask     = weights > 0
        weighted_uids = np.asarray(eligible[pos_mask], dtype=int)
        if len(weighted_uids):
            pos_weights = weights[pos_mask]
            draws       = np.asarray(self._dist_emig_u.rvs(weighted_uids), dtype=float)
            draws       = np.clip(draws, 1e-12, 1.0)
            keys        = -np.log(draws) / pos_weights
            order       = np.argsort(keys)
            n_weighted  = min(n_select, len(weighted_uids))
            selected    = weighted_uids[order[:n_weighted]]

        if len(selected) < n_select:
            sel_set       = set(selected.tolist())
            fallback_uids = np.array([u for u in np.asarray(eligible, dtype=int) if u not in sel_set], dtype=int)
            if len(fallback_uids):
                n_fill         = n_select - len(selected)
                fallback_scores = np.asarray(self._dist_emig_u.rvs(fallback_uids), dtype=float)
                fallback_order  = np.argsort(fallback_scores)
                selected        = np.concatenate([selected, fallback_uids[fallback_order[:n_fill]]])

        return ss.uids(selected)

    def _apply_emigration(self, emigrant_uids):
        if len(emigrant_uids) == 0:
            self.n_emigrants = 0
            return emigrant_uids

        tb = self.sim.diseases[self._tb_name]
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
        self.n_emigrants = len(emigrant_uids)
        return emigrant_uids

    def step(self):
        self._fresh_import_uids = None
        n_immigrants = int(self._dist_n_immigrants.rvs(1)[0])
        n_emigrants  = self._draw_emigrants()
        emig_uids    = self._sample_emigrants(n_emigrants)
        n_adjusted   = self._adjust_arrivals_for_pop_target(n_immigrants, len(emig_uids))
        self._apply_emigration(emig_uids)
        return self._perform_immigration(n_adjusted)

    def update_results(self):
        super().update_results()
        if isinstance(self.results, ss.Results):
            self.results['n_immigrants'][self.ti] = int(self.n_immigrants)
            self.results['n_emigrants'][self.ti] = int(self.n_emigrants)
            self.results['net_migration'][self.ti] = int(self.n_immigrants - self.n_emigrants)
        self._fresh_import_uids = None
        return

    @staticmethod
    def _validate_tb_state_distribution(tb_state_distribution):
        if not tb_state_distribution:
            raise ValueError('tb_state_distribution must be provided')
        raw = dict(tb_state_distribution)
        valid = {}
        for name, weight in raw.items():
            if name not in TBS._member_names_:
                warnings.warn(f'Ignoring unknown TB state "{name}" in tb_state_distribution', stacklevel=2)
                continue
            w = float(weight)
            if not np.isfinite(w) or w < 0:
                w = 0.0
            valid[name] = w
        for term in [TBS.DEAD.name, TBS.REMOVED.name]:
            if term in valid:
                warnings.warn(f'Removing terminal state {term} from tb_state_distribution', stacklevel=2)
                valid.pop(term)
        weight_sum = sum(valid.values())
        if weight_sum <= 0:
            raise ValueError('tb_state_distribution must include at least one positive probability')
        return {k: v / weight_sum for k, v in valid.items() if v > 0}

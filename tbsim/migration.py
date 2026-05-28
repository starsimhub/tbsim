"""Immigration demographics for TBsim. """

import warnings
import numpy as np
import starsim as ss

from .tb import TBS, TB, TBAcute

__all__ = ['Immigration']


class Immigration(ss.Demographics):
    """
    Add immigrants on a Poisson schedule with preset age and TB states.

    Args (pars):
        immigration_rate (ss.freq/float): Arrivals per year (default
            ``ss.freqperyear(10)``). Must be ``ss.freq...``, not ``ss.peryear``.
        age_distribution (dict/None): ``{age_lower_bound: weight}`` for
            piecewise-uniform ages; top bin ends at ``max_age``. Used when
            ``age_data`` is None; if both are None, a built-in 0--max_age profile
            is installed in ``init_pre``. All keys must be strictly less than
            ``max_age``.
        age_data (DataFrame/Series/array/str/None): Age histogram in Starsim
            ``People`` format; overrides ``age_distribution``.
        max_age (float): Upper bound on sampled ages for ``age_distribution`` and
            the no-distribution fallback. Default 85.0.
        tb_state_distribution (dict): ``{TBS name: weight}`` at entry. Weights
            are normalized. ``DEAD`` is rejected. ``ACUTE`` requires ``TBAcute``.

    Attributes:
        hhid (IntArr): Household ID on the network, or -1.
        is_immigrant (BoolState): Set for agents created by this module.
        immigration_time (FloatArr): Step index when the agent arrived.
        age_at_immigration (FloatArr): Age at arrival (years).
        immigration_tb_status (IntArr): ``TBS`` value assigned at entry.
        results['n_immigrants'] (Result): Count of arrivals in the last step.

    **Example**::

        import starsim as ss
        import tbsim

        sim = ss.Sim(
            diseases=tbsim.TB(),
            demographics=[tbsim.Immigration(pars=dict(immigration_rate=ss.freqperyear(200)))],
        )
        sim.run()
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            immigration_rate=ss.freqperyear(10),   # annual arrival intensity
            age_distribution=None,                 # age distribution for immigrants
            age_data=None,                         # age histogram for immigrants
            max_age=85.0,                          # upper age bound for age_distribution sampling
            # TODO: Which values should we use as defauls?
            # Values below are randomly chosen for demo purposes
            tb_state_distribution={                
                TBS.SUSCEPTIBLE.name: 0.6517,
                TBS.INFECTION.name: 0.33,
                TBS.CLEARED.name: 0.007,
                TBS.NON_INFECTIOUS.name: 0.0008,
                TBS.ASYMPTOMATIC.name: 0.0015,
                TBS.SYMPTOMATIC.name: 0.0010,
                TBS.TREATMENT.name: 0.0,
            },
        )
        self.update_pars(pars, **kwargs)
        self.pars.tb_state_distribution = self._validate_tb_state_distribution(self.pars.tb_state_distribution)

        self._dist_n = ss.poisson(lam=self._lam_per_timestep)   # Poisson distribution for number of immigrants per timestep
        self._dist_agebin = ss.choice(a=[0], p=[1.0])           # Choice distribution for age bin
        self._dist_ageu = ss.random()                           # Random distribution for age within bin
        self._dist_tbstate = ss.choice(a=[-1], p=[1.0])         # Choice distribution for TB state
        self._dist_hhu = ss.random()                            # Random distribution for household ID
        self._dist_age = None                                   # Age distribution
        self._age_lows = None                                   # Age bin lower bounds
        self._age_highs = None                                  # Age bin upper bounds

        self.define_states(
            ss.IntArr('hhid', default=-1),                                   # Household ID
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),                 # Time of immigration
            ss.FloatArr('age_at_immigration', default=np.nan),               # Age at immigration
            ss.IntArr('immigration_tb_status', default=-1),                  # TB state at immigration
        )
        self.n_immigrants = 0
        self._fresh_import_uids = None  # set by step(); cleared by update_results()
        return

    def init_post(self):
        """Find ``TB`` or ``TBAcute`` in ``sim.diseases`` (required)."""
        super().init_post()
        self._tb_name = next((k for k, d in self.sim.diseases.items() if isinstance(d, (TB, TBAcute))), None)
        if self._tb_name is None:
            raise RuntimeError('Expected TB or TBAcute disease module for immigration initialization')
        return

    def init_pre(self, sim):
        """Set up age and TB entry distributions from ``pars``."""
        super().init_pre(sim)
        self._configure_age_sampling()
        max_age = float(self.pars.max_age)
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError(f'max_age must be a positive finite number, got {self.pars.max_age!r}')
        if self._dist_age is None and self.pars.age_distribution is None:
            default_keys = [0, 5, 15, 30, 50, 65]
            self.pars.age_distribution = {k: w for k, w in zip(default_keys, [0.15, 0.20, 0.25, 0.20, 0.15, 0.05]) if k < max_age}

        age_bins = self.pars.age_distribution
        if isinstance(age_bins, dict) and len(age_bins) and self._dist_age is None:
            bin_edges = np.array(sorted(age_bins.keys()), dtype=float)
            bin_weights = np.array([age_bins[k] for k in bin_edges], dtype=float)
            if np.any(~np.isfinite(bin_edges)):
                raise ValueError('age_distribution keys must be finite age-bin lower bounds')
            if np.any(~np.isfinite(bin_weights)):
                raise ValueError('age_distribution weights must be finite')
            if np.any(bin_weights < 0):
                raise ValueError('age_distribution weights must be non-negative')
            if bin_edges[-1] >= max_age:
                raise ValueError(f'age_distribution keys must be strictly less than max_age={max_age}; got max key {bin_edges[-1]}')
            weight_sum = bin_weights.sum()
            if weight_sum <= 0:
                raise ValueError('age_distribution must include at least one positive weight')
            bin_weights = bin_weights / weight_sum
            self._age_lows = bin_edges
            self._age_highs = np.r_[bin_edges[1:], max_age]
            self._dist_agebin.pars.a = np.arange(len(bin_edges), dtype=int)
            self._dist_agebin.pars.p = bin_weights

        tb_entry_weights = self.pars.tb_state_distribution
        self._dist_tbstate.pars.a = np.array([int(TBS[state_name]) for state_name in tb_entry_weights], dtype=int)
        self._dist_tbstate.pars.p = np.array(list(tb_entry_weights.values()), dtype=float)
        return

    def _configure_age_sampling(self):
        """Use ``age_data`` for ``_dist_age`` when supplied."""
        age_data = self.pars.age_data
        if age_data is None:
            return
        if self.pars.age_distribution is not None:
            warnings.warn('age_data is set; ignoring age_distribution', stacklevel=2)
        self._dist_age = ss.People.get_age_dist(age_data)
        self._age_lows = None
        self._age_highs = None
        return

    def init_results(self):
        """Define ``n_immigrants`` result."""
        super().init_results()
        self.define_results(ss.Result('n_immigrants', dtype=int, label='Number of immigrants'))
        return

    def expected_immigrants_per_timestep(self):
        """Poisson mean ``immigration_rate.to_events(self.t.dt)`` for this step."""
        rate_spec = self.pars.immigration_rate
        if rate_spec is None:
            return 0.0
        if isinstance(rate_spec, ss.Rate):
            if not isinstance(rate_spec, ss.freq):
                raise ValueError('immigration_rate must be an event rate (ss.freq...), e.g. ss.freqperyear(1000)')
            annual_rate = rate_spec
        else:
            annual_rate = ss.freqperyear(float(rate_spec))
        expected_arrivals = float(annual_rate.to_events(self.t.dt))
        if not np.isfinite(expected_arrivals) or expected_arrivals < 0:
            expected_arrivals = 0.0
        return expected_arrivals

    def _lam_per_timestep(self, module):
        """Callback for ``_dist_n`` so λ tracks the current step."""
        return module.expected_immigrants_per_timestep()

    def _sample_ages(self, n):
        """Draw ``n`` entry ages in years."""
        if n <= 0:
            return np.empty(0, dtype=float)
        if self._dist_age is not None:
            return np.asarray(self._dist_age.rvs(n), dtype=float)
        if self._age_lows is None or self._age_highs is None:
            return self._dist_ageu.rvs(n) * float(self.pars.max_age)
        age_bin = self._dist_agebin.rvs(n).astype(int)
        within_bin = self._dist_ageu.rvs(n)
        bin_lower = self._age_lows[age_bin]
        bin_upper = self._age_highs[age_bin]
        return bin_lower + within_bin * (bin_upper - bin_lower)

    def _init_tb_states(self, new_uids):
        """Write TB flags for new arrivals; return sampled ``TBS`` codes."""
        tb = self.sim.diseases[self._tb_name]
        entry_state_codes = np.asarray(self._dist_tbstate.pars.a, dtype=int)
        if TBS.ACUTE in entry_state_codes and not isinstance(tb, TBAcute):
            raise ValueError(f'tb_state_distribution includes {TBS.ACUTE.name} but TB module is not TBAcute')

        entry_states = self._dist_tbstate.rvs(len(new_uids)).astype(int)
        tb.state[new_uids] = entry_states
        tb.infected[new_uids] = ~np.isin(entry_states, [TBS.SUSCEPTIBLE, TBS.CLEARED, TBS.DEAD])
        tb.susceptible[new_uids] = np.isin(entry_states, [TBS.SUSCEPTIBLE, TBS.CLEARED])
        tb.ever_infected[new_uids] = entry_states != TBS.SUSCEPTIBLE
        tb.on_treatment[new_uids] = entry_states == TBS.TREATMENT
        tb.ti_infected[new_uids] = -np.inf
        
        # TODO: Question -should we consider setting ti_asymp to the arrival time?
        # ti_asymp == ti is how TB counts new_active; -inf never matches a real ti.
        # so incidence_kpy and new_active are not spiked on the arrival step.
        # tb.ti_asymp[new_uids] = -np.inf
        
        tb.ti_asymp[new_uids] = self.ti
        tb.rr_reinfection[new_uids] = 1.0
        tb.ti_rr_reinfection_wane[new_uids] = np.inf
        is_cleared = entry_states == TBS.CLEARED
        if np.any(is_cleared):
            tb.rr_reinfection[new_uids[is_cleared]] = float(tb.pars.rr_reinfection_cleared)
        tb.rel_sus[new_uids] = 1.0
        tb.rel_sus[new_uids[is_cleared]] = tb.rr_reinfection[new_uids[is_cleared]]
        tb.rel_trans[new_uids] = 1.0
        tb.rel_trans[new_uids[entry_states == TBS.ASYMPTOMATIC]] = float(tb.pars.trans_asymp)
        if isinstance(tb, TBAcute):
            tb.rel_trans[new_uids[entry_states == TBS.ACUTE]] = float(tb.pars.trans_acute)
        return entry_states

    def step(self):
        """Draw arrivals, grow ``people``, seed TB, and assign households.

        Returns the new UIDs (same pattern as ``ss.Births.step``); the sim loop
        does not use the return value.
        """
        n_arrivals = int(self._dist_n.rvs(1)[0])
        if n_arrivals == 0:
            self.n_immigrants = 0
            self._fresh_import_uids = None
            return []
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

    def assign_immigrants_to_households(self, new_uids):
        """Place ``new_uids`` in a household and add edges if the network supports it.

        Uses the first network with ``hhid`` or ``household_ids``. Picks an
        existing household at random (alive, not ``TBS.DEAD``). If none qualify,
        assigns singleton IDs ``0 .. n-1`` for this batch.
        """
        household_net = None
        household_id_attr = None
        for net in self.sim.networks.values():
            for attr in ['hhid', 'household_ids']:  # hhid is used in tbsim household classes. household_ids is for starsim.
                household_ids = getattr(net, attr, None)
                if isinstance(household_ids, (np.ndarray, ss.BaseArr)):
                    household_net = net
                    household_id_attr = attr
                    break
            if household_net is not None:
                break
        if household_net is None:
            return

        household_ids = getattr(household_net, household_id_attr)
        tb = self.sim.diseases[self._tb_name]
        household_ids_arr = np.asarray(household_ids, dtype=float)
        has_household_id = np.isfinite(household_ids_arr) & (household_ids_arr >= 0)
        is_active = np.asarray(self.sim.people.alive) & (np.asarray(tb.state) != TBS.DEAD)
        occupied_household_ids = np.unique(household_ids_arr[has_household_id & is_active]).astype(int)

        if len(occupied_household_ids):
            household_draw = self._dist_hhu.rvs(len(new_uids))
            household_idx = np.floor(household_draw * len(occupied_household_ids)).astype(int)
            household_idx = np.clip(household_idx, 0, len(occupied_household_ids) - 1)
            assigned_household_ids = occupied_household_ids[household_idx]
        else:
            assigned_household_ids = np.arange(len(new_uids), dtype=int)

        household_ids[new_uids] = assigned_household_ids
        self.hhid[new_uids] = assigned_household_ids
        self._connect_immigrants_to_households(
            household_net=household_net,
            household_ids=household_ids,
            new_uids=new_uids,
            assigned_household_ids=assigned_household_ids,
        )
        return

    @staticmethod
    def _has_edge_struct(net):
        """Return whether ``net`` exposes ``edges.p1`` and ``edges.p2``."""
        edges = getattr(net, 'edges', None)
        return bool(edges is not None and hasattr(edges, 'p1') and hasattr(edges, 'p2'))

    def _connect_immigrants_to_households(self, household_net, household_ids, new_uids, assigned_household_ids):
        """Append undirected edges between newcomers and active household members."""
        if not self._has_edge_struct(household_net):
            return
        tb = self.sim.diseases[self._tb_name]
        household_ids_arr = np.asarray(household_ids, dtype=float)
        edge_p1 = np.asarray(household_net.edges.p1, dtype=int)
        edge_p2 = np.asarray(household_net.edges.p2, dtype=int)
        existing_edge_pairs = {(int(min(a, b)), int(max(a, b))) for a, b in zip(edge_p1, edge_p2) if a != b}
        assigned_household_ids = np.asarray(assigned_household_ids, dtype=int)
        new_edge_pairs = set()
        for household_id in np.unique(assigned_household_ids):
            if isinstance(household_ids, ss.BaseArr):
                member_uids = np.asarray((household_ids == household_id).uids, dtype=int)
            else:
                member_uids = np.where(household_ids_arr == household_id)[0].astype(int)
            is_active_member = np.asarray(self.sim.people.alive[member_uids]) & (tb.state[member_uids] != TBS.DEAD)
            member_uids = member_uids[is_active_member]
            if len(member_uids) < 2:
                continue
            immigrant_uids = np.asarray(new_uids[assigned_household_ids == household_id], dtype=int)
            for immigrant_uid in immigrant_uids:
                for member_uid in member_uids:
                    if immigrant_uid == member_uid:
                        continue
                    uid_lo, uid_hi = (immigrant_uid, member_uid) if immigrant_uid < member_uid else (member_uid, immigrant_uid)
                    if (uid_lo, uid_hi) not in existing_edge_pairs:
                        new_edge_pairs.add((uid_lo, uid_hi))
        if len(new_edge_pairs):
            new_p1 = np.fromiter((lo for lo, _ in sorted(new_edge_pairs)), dtype=int)
            new_p2 = np.fromiter((hi for _, hi in sorted(new_edge_pairs)), dtype=int)
            household_net.append(p1=ss.uids(new_p1), p2=ss.uids(new_p2), beta=np.ones(len(new_p1), dtype=float))
        
        # TODO: Question -should we consider add a portion of the immigrants without household? 
        # While this may not impact the household dynamics, it can seed prevalence into the population.
        return

    def update_results(self):
        """Store ``n_immigrants`` for this step in ``results``.

        Also re-applies the ``ti_asymp = -inf`` sentinel on fresh imports to undo
        any within-step ASYMP rebound writes from ``TB.transition`` (FR12: imports
        are exogenous, not endogenous incident cases).
        """
        super().update_results()
        if isinstance(self.results, ss.Results):
            self.results['n_immigrants'][self.ti] = int(self.n_immigrants)
        if self._fresh_import_uids is not None and len(self._fresh_import_uids):
            tb = self.sim.diseases[self._tb_name]
            tb.ti_asymp[self._fresh_import_uids] = -np.inf
            self._fresh_import_uids = None
        return

    @staticmethod
    def _validate_tb_state_distribution(tb_state_distribution):
        """Check ``tb_state_distribution`` keys and normalize weights to sum to 1."""
        if not tb_state_distribution:
            raise ValueError('tb_state_distribution must be provided')
        weights_by_state = {k: float(v) for k, v in dict(tb_state_distribution).items() if v}
        if TBS.DEAD.name in weights_by_state:
            raise ValueError(f'tb_state_distribution cannot include {TBS.DEAD.name}; immigrants must enter alive')
        for state_name, weight in weights_by_state.items():
            if state_name not in TBS._member_names_:
                raise KeyError(f'Unknown TB state "{state_name}" in tb_state_distribution (expected a {TBS.__name__} name)')
            if not np.isfinite(weight):
                raise ValueError(f'tb_state_distribution["{state_name}"] must be finite ({weight})')
            if weight < 0:
                raise ValueError(f'tb_state_distribution["{state_name}"] is negative ({weight})')
        weight_sum = sum(weights_by_state.values())
        if weight_sum <= 0:
            raise ValueError('tb_state_distribution must include at least one positive probability')
        if weight_sum > 1.0:
            warnings.warn(f'tb_state_distribution sums to {weight_sum:.6g} (>1); normalizing automatically')
        return {k: v / weight_sum for k, v in weights_by_state.items()}

"""Exogenous immigration for TBsim (Term 1). See ``docs/immigration.md`` for the full guide."""

import warnings
import numpy as np
import starsim as ss

from .tb import TBS, TB, TBAcute

__all__ = ['Immigration']


class Immigration(ss.Demographics):
    """
    Exogenous immigration as a Poisson arrival process with TB entry states.

    Each step draws ``N ~ Poisson(λ)`` newcomer agents, where
    ``λ = immigration_rate.to_events(self.t.dt)``. New agents
    receive sampled ages, a ``TBS`` entry state from ``tb_state_distribution``,
    and optional placement in a household network. Imported infection is
    exogenous: ``ti_infected`` is set to ``-inf`` so arrivals do not count as
    model-generated incidence.

    Requires ``TB`` or ``TBAcute`` in the simulation. ``TBS.DEAD`` cannot be
    used in ``tb_state_distribution``. Household assignment skips agents who are
    not ``people.alive`` or are in ``TBS.DEAD``.

    Args (pars):
        immigration_rate (ss.freq/float): Mean arrivals per year (default
            ``ss.freqperyear(10)``). Scalars are coerced to ``ss.freqperyear``.
            Must be an event frequency (``ss.freq...``), not e.g. ``ss.peryear``.
        age_distribution (dict/None): ``{lower_age_bound: weight}`` for
            piecewise-uniform entry ages. Ignored if ``age_data`` is set.
        age_data (DataFrame/Series/array/str/None): Starsim age histogram (same
            formats as ``ss.People(age_data=...)``).
        tb_state_distribution (dict): ``{TBS name: weight}`` for entry TB state.
            Weights are normalized; ``TBS.DEAD`` is not allowed.

    Attributes:
        hhid (IntArr): Assigned household ID, or -1 if unset.
        is_immigrant (BoolState): True for agents added by this module.
        immigration_time (FloatArr): Step index at entry.
        age_at_immigration (FloatArr): Age at entry (years).
        immigration_tb_status (IntArr): ``TBS`` code at entry.
        results['n_immigrants'] (Result): Arrival count each step.

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
        """Define parameters, RNGs, and per-agent immigration states."""
        super().__init__()
        self.define_pars(
            immigration_rate=ss.freqperyear(10),    # Default annual arrival rate of 10 immigrants
            age_distribution=None,                  # Default age distribution (see below)
            age_data=None,                          # Default age data (see below)  
            tb_state_distribution={                 # Default TB state distribution (see below)
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

        self._dist_n = ss.poisson(name='imm_n', lam=self._lam_per_timestep)
        self._dist_agebin = ss.choice(name='imm_agebin', a=[0], p=[1.0])
        self._dist_ageu = ss.random(name='imm_ageu')
        self._dist_tbstate = ss.choice(name='imm_tbstate', a=[-1], p=[1.0])
        self._dist_hhu = ss.random(name='imm_hhu')
        self._dist_age = None
        self._age_lows = None
        self._age_highs = None

        self.define_states(
            ss.IntArr('hhid', default=-1),
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),
            ss.FloatArr('age_at_immigration', default=np.nan),
            ss.IntArr('immigration_tb_status', default=-1),
        )
        self.n_immigrants = 0
        return

    def init_post(self):
        """Resolve the ``TB`` or ``TBAcute`` module after the sim is assembled."""
        super().init_post()
        self._tb_name = next((k for k, d in self.sim.diseases.items() if isinstance(d, (TB, TBAcute))), None)
        if self._tb_name is None:
            raise RuntimeError('Expected TB or TBAcute disease module for immigration initialization')
        return

    def init_pre(self, sim):
        """Configure age-bin and ``TBS`` entry samplers from ``pars``.

        Args:
            sim (ss.Sim): Simulation being initialized.
        """
        super().init_pre(sim)
        self._configure_age_sampling()
        if self._dist_age is None and self.pars.age_distribution is None:
            self.pars.age_distribution = {0: 0.15, 5: 0.20, 15: 0.25, 30: 0.20, 50: 0.15, 65: 0.05}

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
            weight_sum = bin_weights.sum()
            if weight_sum <= 0:
                raise ValueError('age_distribution must include at least one positive weight')
            bin_weights = bin_weights / weight_sum
            self._age_lows = bin_edges
            self._age_highs = np.r_[bin_edges[1:], 85.0]
            self._dist_agebin.pars.a = np.arange(len(bin_edges), dtype=int)
            self._dist_agebin.pars.p = bin_weights

        tb_entry_weights = self.pars.tb_state_distribution
        self._dist_tbstate.pars.a = np.array([int(TBS[state_name]) for state_name in tb_entry_weights], dtype=int)
        self._dist_tbstate.pars.p = np.array(list(tb_entry_weights.values()), dtype=float)
        return

    def _configure_age_sampling(self):
        """Build ``_dist_age`` from ``pars.age_data`` when provided."""
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
        """Register the per-step ``n_immigrants`` result channel."""
        super().init_results()
        self.define_results(ss.Result('n_immigrants', dtype=int, label='Number of immigrants'))
        return

    def expected_immigrants_per_timestep(self):
        """Return the Poisson mean for the current step.
        Non-finite or negative values are clamped to zero.

        Returns:
            expected_arrivals (float): Expected number of immigrants this step.
        """
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
        """Poisson rate callback for ``_dist_n`` (called each step)."""
        return module.expected_immigrants_per_timestep()

    def _sample_ages(self, n):
        """Sample entry ages for ``n`` immigrants (years).

        Args:
            n (int): Number of ages to draw.

        Returns:
            ages (ndarray): Length-``n`` float array of ages.
        """
        if n <= 0:
            return np.empty(0, dtype=float)
        if self._dist_age is not None:
            return np.asarray(self._dist_age.rvs(n), dtype=float)
        if self._age_lows is None or self._age_highs is None:
            return self._dist_ageu.rvs(n) * 85.0
        age_bin = self._dist_agebin.rvs(n).astype(int)
        within_bin = self._dist_ageu.rvs(n)
        bin_lower = self._age_lows[age_bin]
        bin_upper = self._age_highs[age_bin]
        return bin_lower + within_bin * (bin_upper - bin_lower)

    def _init_tb_states(self, new_uids):
        """Seed TB state and consistency fields for new arrivals.

        Uses ``TBS.non_infected_states()`` and ``TBS.susceptible_states()`` like
        ``TB``. Sets ``ti_infected`` to ``-inf`` for all imports.

        Args:
            new_uids (array-like): UIDs of agents just added to the population.

        Returns:
            entry_states (ndarray): Sampled ``TBS`` integer codes.
        """
        tb = self.sim.diseases[self._tb_name]
        entry_state_codes = np.asarray(self._dist_tbstate.pars.a, dtype=int)
        if TBS.ACUTE in entry_state_codes and not isinstance(tb, TBAcute):
            raise ValueError(f'tb_state_distribution includes {TBS.ACUTE.name} but TB module is not TBAcute')

        entry_states = self._dist_tbstate.rvs(len(new_uids)).astype(int)
        tb.state[new_uids] = entry_states
        tb.infected[new_uids] = ~np.isin(entry_states, TBS.non_infected_states())
        tb.susceptible[new_uids] = np.isin(entry_states, TBS.susceptible_states())
        tb.ever_infected[new_uids] = entry_states != TBS.SUSCEPTIBLE
        tb.on_treatment[new_uids] = entry_states == TBS.TREATMENT
        tb.ti_infected[new_uids] = -np.inf
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
        """Add immigrants for the current timestep.

        Draws ``N ~ Poisson(λ)``, grows the population, seeds TB and household
        state, and flags audit fields on new agents.

        Returns:
            new_uids (list): UIDs of agents added this step, or ``[]`` if ``N=0``.
        """
        n_arrivals = int(self._dist_n.rvs(1)[0])
        if n_arrivals == 0:
            self.n_immigrants = 0
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
        return new_uids

    def assign_immigrants_to_households(self, new_uids):
        """Assign household IDs and append missing household-network edges.

        Looks for ``hhid`` or ``household_ids`` on the first compatible network.
        Assignment pools only households with at least one alive, non-``TBS.DEAD``
        member. If none exist, each arrival gets a singleton household ID.

        Args:
            new_uids (array-like): UIDs of agents just added to the population.
        """
        household_net = None
        household_id_attr = None
        for net in self.sim.networks.values():
            for attr in ['hhid', 'household_ids']:
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
        return

    def update_results(self):
        """Write ``results.n_immigrants[self.ti]`` from the last ``step()``."""
        super().update_results()
        if isinstance(getattr(self, 'results', None), ss.Results):
            self.results['n_immigrants'][self.ti] = int(self.n_immigrants)
        return

    @staticmethod
    def _validate_tb_state_distribution(tb_state_distribution):
        """Validate and normalize ``tb_state_distribution`` to ``TBS`` weights.

        Args:
            tb_state_distribution (dict): ``{TBS name: weight}``; zero weights dropped.

        Returns:
            weights_by_state (dict): Normalized weights summing to 1.
        """
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

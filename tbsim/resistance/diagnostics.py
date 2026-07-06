"""Drug susceptibility testing (DST) for the TB resistance overlay.

The product/delivery split here mirrors ``tbsim/interventions/diagnostics.py``
(``Dx`` + ``DxDelivery``):

- :class:`DSTDx` is the **product** — owns per-drug sensitivity / specificity
  as proper ``ss.bernoulli`` parameters under ``self.pars`` so they share the
  Starsim RNG plumbing and are reproducible.
- :class:`DSTDelivery` is the **delivery** — owns eligibility, coverage,
  per-agent state arrays, and a ``step()`` orchestrator that defers to small
  ``step_*`` submethods (`step_select_eligible`, `step_administer`,
  `step_update_states`) so subclasses can override one phase at a time.
"""

import numpy as np
import starsim as ss
import tbsim

__all__ = ['DSTDx', 'DSTDelivery', 'RegimenRouter',
           'treatment_monitoring_eligibility']


class DSTDx(ss.Product):
    """
    Drug susceptibility test (DST) product.

    DST observes an agent-level phenotype: for each drug in the panel, the
    test reports "resistant" or "susceptible". Internal strain state is not
    exposed; the test sees the union of carried strains' resistance bits.

    Per-drug sensitivity/specificity are applied at the strain level and then
    aggregated to an agent-level observed phenotype (any observed strain with
    observed resistance to a drug yields agent-level observed resistance for
    that drug). A per-strain observability term ``p_strain_obs`` models
    strain-level drop-out in mixed infections.

    Args:
        catalog (StrainCatalog): Strain catalog.
        drugs (list[str]): Subset of drugs to report on (default: all in
            catalog).
        sensitivity (dict|float): Per-drug sensitivity. Default 0.95.
        specificity (dict|float): Per-drug specificity. Default 0.99.
        p_strain_obs (float|dict|None): Probability a carried strain is
            observed at all by DST before applying sens/spec. If ``None``
            (default), uses strain fitness per catalog entry. If float,
            applies one strain-agnostic value to all strains.

    Example::

        dst = DSTDx(catalog, drugs=['INH', 'RIF'], sensitivity=0.95,
                    specificity=0.99)
        result = dst.administer(sim, agent_uids)
        # result['INH'] is a boolean array of observed INH-resistance
    """

    def __init__(self, catalog, drugs=None, sensitivity=0.95,
                 specificity=0.99, p_strain_obs=None, **kwargs):
        super().__init__()
        self.catalog = catalog
        if drugs is None:
            drugs = list(catalog.drugs)
        for d in drugs:
            if d not in catalog.drugs:
                raise ValueError(
                    f'DSTDx drug {d!r} not in catalog drugs {catalog.drugs}'
                )
        self.drugs = list(drugs)

        # Normalize scalar inputs to per-drug dicts.
        if isinstance(sensitivity, (int, float)):
            sensitivity = {d: float(sensitivity) for d in self.drugs}
        if isinstance(specificity, (int, float)):
            specificity = {d: float(specificity) for d in self.drugs}

        # Normalize strain observability inputs.
        if p_strain_obs is None:
            strain_obs = np.asarray(catalog.fitness, dtype=float)
        elif isinstance(p_strain_obs, (int, float)):
            strain_obs = np.full(catalog.n, float(p_strain_obs), dtype=float)
        elif isinstance(p_strain_obs, dict):
            if 'all' in p_strain_obs:
                strain_obs = np.full(catalog.n, float(p_strain_obs['all']), dtype=float)
            else:
                strain_obs = np.array([
                    float(p_strain_obs.get(uid, catalog.fitness[i]))
                    for i, uid in enumerate(catalog.uids)
                ], dtype=float)
        else:
            raise TypeError(
                f'p_strain_obs must be float, dict, or None; got {type(p_strain_obs).__name__}'
            )
        if np.any((strain_obs < 0.0) | (strain_obs > 1.0)):
            raise ValueError(f'p_strain_obs values must be in [0, 1]; got {strain_obs}')

        # Define per-drug Bernoullis as proper Starsim pars. ``strict=False``
        # allows DSTDx to be invoked standalone (e.g. from tests that build
        # a DSTDx outside a DSTDelivery), while keeping the standard RNG
        # plumbing intact when the product is wrapped in a DSTDelivery and
        # the sim's intervention init flow runs.
        bernoulli_pars = {}
        for d in self.drugs:
            bernoulli_pars[f'p_sens_{d}'] = ss.bernoulli(
                p=float(sensitivity[d]), strict=False,
            )
            bernoulli_pars[f'p_spec_{d}'] = ss.bernoulli(
                p=float(specificity[d]), strict=False,
            )
        for s_idx in range(catalog.n):
            bernoulli_pars[f'p_obs_strain_{s_idx}'] = ss.bernoulli(
                p=float(strain_obs[s_idx]), strict=False,
            )
        self.define_pars(**bernoulli_pars)
        self.update_pars(**kwargs)

        # Eagerly init so the distributions are sampleable even when the
        # product is not yet owned by an intervention.
        for name in bernoulli_pars:
            dist = self.pars[name]
            if not dist.initialized:
                dist.init()
        return

    def true_phenotype(self, tb, uids, drug):
        """Return per-uid boolean array of true resistance to *drug*.

        A UID is "truly resistant" to a drug if it carries at least one
        strain whose catalog phenotype is resistant to that drug. The
        carrier check is expressed via :class:`ss.BoolArr.uids` set
        intersection with *uids* — the canonical Starsim filtering idiom.
        """
        if not len(uids):
            return np.zeros(0, dtype=bool)
        profile = tb.agent_strains
        if profile is None:
            return np.zeros(len(uids), dtype=bool)
        drug_col = self.catalog.drugs.index(drug)
        resistant_strain_idx = np.where(
            self.catalog.resistance[:, drug_col] == 1
        )[0]
        true_pos = np.zeros(len(uids), dtype=bool)
        for s_idx in resistant_strain_idx:
            strain_arr = getattr(profile._tb, profile.names[int(s_idx)])
            carrier_uids = strain_arr.uids.intersect(uids)
            if len(carrier_uids) == 0:
                continue
            # Mark positions in `uids` where the carrier_uids intersect.
            true_pos |= np.isin(uids, carrier_uids)
        return true_pos

    def administer(self, sim, uids):
        """Run DST on *uids* and return ``{drug: bool array}``.

        We sample by integer size rather than by UID because the test- and
        documentation-friendly standalone use-case (DSTDx constructed
        outside a DSTDelivery) does not have agent slots wired into the
        Bernoulli — and per-drug DST does not depend on UID-keyed CRN here.
        """
        if len(uids) == 0:
            return {}
        tb = tbsim.get_tb(sim)
        n = len(uids)
        results = {drug: np.zeros(n, dtype=bool) for drug in self.drugs}
        profile = tb.agent_strains

        # Fallback for non-overlay contexts: all are effectively susceptible,
        # with per-drug false positives controlled by specificity.
        if profile is None:
            for drug in self.drugs:
                spec_neg = np.asarray(self.pars[f'p_spec_{drug}'].rvs(n), dtype=bool)
                results[drug] = ~spec_neg
            return results

        uid_to_pos = {int(uid): i for i, uid in enumerate(uids)}
        drug_cols = {drug: self.catalog.drugs.index(drug) for drug in self.drugs}

        # Apply DST at strain level, then aggregate to agent-level phenotype.
        for s_idx in range(self.catalog.n):
            strain_arr = getattr(profile._tb, profile.names[s_idx])
            carrier_uids = strain_arr.uids.intersect(uids)
            if len(carrier_uids) == 0:
                continue

            obs_draw = np.asarray(self.pars[f'p_obs_strain_{s_idx}'].rvs(len(carrier_uids)), dtype=bool)
            observed_uids = carrier_uids[obs_draw]
            if len(observed_uids) == 0:
                continue

            for drug in self.drugs:
                m = len(observed_uids)
                sens_pos = np.asarray(self.pars[f'p_sens_{drug}'].rvs(m), dtype=bool)
                spec_neg = np.asarray(self.pars[f'p_spec_{drug}'].rvs(m), dtype=bool)
                is_resistant = bool(self.catalog.resistance[s_idx, drug_cols[drug]])
                observed_resistant = sens_pos if is_resistant else ~spec_neg
                if not observed_resistant.any():
                    continue
                hit_uids = observed_uids[observed_resistant]
                hit_idx = [uid_to_pos[int(uid)] for uid in hit_uids]
                results[drug][hit_idx] = True
        return results


class DSTDelivery(ss.Intervention):
    """
    Delivers a :class:`DSTDx` product to diagnosed agents and stashes the
    observed phenotype on per-drug state arrays.

    Mirrors the :class:`tbsim.DxDelivery` shape: ``__init__`` declares pars
    (``p_coverage``) and per-agent states; ``step()`` is split into
    ``step_select_eligible`` / ``step_administer`` / ``step_update_states``;
    ``init_results`` / ``update_results`` / ``finalize_results`` populate
    per-timestep and cumulative counters; ``shrink()`` drops transients.

    Args:
        product (DSTDx): The DST product.
        coverage (float): Probability of receiving DST among eligible agents.
            Default 1.0.
        eligibility (callable): Optional ``(sim) -> uids`` override.
            Default: diagnosed & alive & not-yet-DST-tested.
    """

    def __init__(self, product, coverage=1.0, eligibility=None, **kwargs):
        super().__init__()
        self.product = product
        self.eligibility = eligibility

        self.define_pars(
            p_coverage=ss.bernoulli(p=float(coverage)),
        )

        states = [
            ss.BoolArr('tested_dst', default=False),
            ss.IntArr('n_times_dst_tested', default=0),
            ss.FloatArr('ti_dst_tested', default=np.nan),
        ]
        for drug in product.drugs:
            states.append(ss.BoolArr(f'observed_{drug}_resistant', default=False))
        self.define_states(*states)

        self.update_pars(**kwargs)
        product.name = f'{self.name}_product'
        return

    def init_post(self):
        """Resolve a reference to the diagnostic delivery whose ``diagnosed``
        flag we gate on. Falls back to ``None`` for plain ``ss.Sim`` parents
        that don't expose ``get_dx`` — matches ``TxDelivery.init_post``.
        """
        super().init_post()
        self._dx = None
        if hasattr(self.sim, 'get_dx'):
            try:
                self._dx = self.sim.get_dx(result_state='diagnosed')
            except Exception:
                self._dx = None
        return

    def _get_eligible(self, sim):
        """Custom-or-default eligibility, in canonical Starsim UID idiom."""
        if self.eligibility is not None:
            return ss.uids(self.eligibility(sim))
        if self._dx is None:
            return ss.uids()
        return (self._dx.diagnosed & sim.people.alive & ~self.tested_dst).uids

    def init_results(self):
        super().init_results()
        results = [
            ss.Result('n_tested_dst', dtype=int),
            ss.Result('cum_tested_dst', dtype=int),
        ]
        for drug in self.product.drugs:
            results.append(ss.Result(f'n_obs_{drug}_resistant', dtype=int))
            results.append(ss.Result(f'cum_obs_{drug}_resistant', dtype=int))
        self.define_results(*results)
        return

    def step(self):
        """Orchestrator — see ``step_*`` submethods for details."""
        self.step_select_eligible()
        self.step_administer()
        self.step_update_states()
        return

    def step_select_eligible(self):
        """Pick eligible agents and apply the coverage filter."""
        eligible = self._get_eligible(self.sim)
        if len(eligible):
            self._selected = self.pars.p_coverage.filter(eligible)
        else:
            self._selected = ss.uids()
        return self._selected

    def step_administer(self):
        """Run the DST product on the selected agents."""
        if len(self._selected) == 0:
            self._results = {}
            return self._results
        self._results = self.product.administer(self.sim, self._selected)
        return self._results

    def step_update_states(self):
        """Write per-drug observed-resistance and bookkeeping flags."""
        selected = self._selected
        if len(selected) == 0:
            return
        for drug, obs in self._results.items():
            arr = getattr(self, f'observed_{drug}_resistant')
            arr[selected[obs]] = True
        self.tested_dst[selected] = True
        self.n_times_dst_tested[selected] += 1
        self.ti_dst_tested[selected] = self.ti
        return

    def update_results(self):
        ti = self.ti
        self.results.n_tested_dst[ti] = len(self._selected)
        for drug in self.product.drugs:
            obs = self._results.get(drug, np.zeros(0, dtype=bool))
            self.results[f'n_obs_{drug}_resistant'][ti] = int(np.count_nonzero(obs))
        return

    def finalize_results(self):
        super().finalize_results()
        self.results.cum_tested_dst[:] = np.cumsum(self.results.n_tested_dst)
        for drug in self.product.drugs:
            self.results[f'cum_obs_{drug}_resistant'][:] = np.cumsum(
                self.results[f'n_obs_{drug}_resistant']
            )
        return

    def shrink(self):
        """Drop per-step transient references so multisim pickling is small."""
        self._selected = None
        self._results = None
        self._dx = None
        return


class RegimenRouter:
    """
    Build DST-aware eligibility lambdas that route agents to regimens.

    Spec §"Diagnostics & Treatment Modification": treatment provision can be
    dependent on the observed DST phenotype. This helper turns a
    :class:`DSTDelivery` into composable eligibility functions for use by
    multiple :class:`StrainAwareTxDelivery` instances.

    The router does not own any state itself; it produces lambdas of the
    form ``lambda sim -> ss.uids`` that the user passes as
    ``eligibility=`` to the relevant ``StrainAwareTxDelivery``. Routing is
    idempotent: agents already on treatment (``tb.on_treatment == True``)
    are filtered out.

    Args:
        dst (DSTDelivery): The DST delivery whose observed-resistance state
            arrays we read.
        diagnosed_state (str): Optional state name to require for routing
            (default ``'diagnosed'`` — reads ``sim.get_dx(result_state=...)``).
            Pass ``None`` to skip the diagnosed gate (DST-tested only).
        require_dst_tested (bool): If True (default), require
            ``dst.tested_dst == True`` (agent has actually been DST-tested).

    Example:
        ::

            dst = DSTDelivery(product=DSTDx(catalog, drugs=['INH','RIF','BDQ']))
            router = RegimenRouter(dst)

            first_line  = StrainAwareTxDelivery(
                product=first_line_tx,
                eligibility=router.matches(INH=False, RIF=False),
                name='first_line',
            )
            second_line = StrainAwareTxDelivery(
                product=second_line_tx,
                eligibility=router.matches(INH=True, RIF=True),  # MDR
                name='second_line',
            )
            sim = tbsim.Sim(..., interventions=[
                hsb, screen, dst, first_line, second_line,
            ])
    """

    def __init__(self, dst, diagnosed_state='diagnosed', require_dst_tested=True):
        self.dst = dst
        self.diagnosed_state = diagnosed_state
        self.require_dst_tested = require_dst_tested
        return

    def _base_eligible(self, sim):
        """UIDs that have passed the diagnosis + DST-tested gates.

        Implemented entirely in the Starsim UID API: every filter is
        expressed as ``ss.uids`` set operations rather than numpy boolean
        manipulation.
        """
        elig = sim.people.alive.uids
        if self.diagnosed_state is not None:
            dx = sim.get_dx(result_state=self.diagnosed_state)
            if dx is None:
                return ss.uids()
            elig = elig.intersect(getattr(dx, self.diagnosed_state).uids)
        if self.require_dst_tested:
            elig = elig.intersect(self.dst.tested_dst.uids)
        # Exclude agents already on treatment
        tb = tbsim.get_tb(sim)
        elig = elig.intersect(tb.on_treatment.false())
        return elig

    def matches(self, **per_drug_resistance):
        """Return an eligibility lambda matching agents whose observed phenotype
        matches the given drug→bool dict.

        Args:
            **per_drug_resistance: e.g. ``INH=True, RIF=True`` (MDR) or
                ``RIF=False, BDQ=False`` (BDQ-susceptible RIF-susceptible).

        Returns:
            callable: ``(sim) -> ss.uids`` selecting matching agents.
        """
        dst = self.dst
        spec = dict(per_drug_resistance)
        # Validate drugs exist on DST
        for d in spec:
            attr = f'observed_{d}_resistant'
            if not hasattr(dst, attr):
                raise ValueError(
                    f'RegimenRouter.matches: drug {d!r} has no observed_{d}_resistant '
                    f'state on DSTDelivery. Available: {dst.product.drugs}'
                )

        def _elig(sim):
            base = self._base_eligible(sim)
            if len(base) == 0:
                return base
            out = base
            for drug, want_resistant in spec.items():
                arr = getattr(dst, f'observed_{drug}_resistant')
                match_uids = arr.uids if want_resistant else arr.false()
                out = out.intersect(match_uids)
            return out
        _elig.__name__ = 'router_matches_' + '_'.join(
            f'{d}{"+" if v else "-"}' for d, v in spec.items()
        )
        return _elig

    def default(self):
        """Eligibility lambda for "everything that didn't match a specific phenotype"
        — useful as the fallback first-line tier. Note: this returns *all*
        base-eligible agents; place the router-specific tiers BEFORE the
        default tier so the specific tiers consume their agents first."""
        return lambda sim: self._base_eligible(sim)


def treatment_monitoring_eligibility(tx_delivery_name, after_steps=4, every_steps=None):
    """
    Build a treatment-monitoring eligibility lambda (spec §"Treatment monitoring").

    Returns a function ``(sim) -> ss.uids`` selecting agents who are currently
    on a given treatment course AND have been on treatment for at least
    ``after_steps`` simulation steps. Use as ``eligibility=`` on a standard
    ``DxDelivery`` to gate a "still bacteriologically positive" check. The
    output of that Dx can in turn gate a regimen-switch
    :class:`StrainAwareTxDelivery`.

    Args:
        tx_delivery_name (str): The ``name`` of the TxDelivery to monitor.
            Used to look up its state via ``sim.interventions[name]``.
        after_steps (int): Minimum number of timesteps since
            ``ti_treatment_start`` before an agent is eligible for monitoring.
            Default 4 steps.
        every_steps (int|None): If given, only retest every N steps after
            the first eligibility. ``None`` means a single test at exactly
            ``after_steps``.

    Returns:
        callable: ``(sim) -> ss.uids`` selecting eligible agents.

    Example:
        ::

            tx = StrainAwareTxDelivery(product=first_line, name='first_line')
            monitor = tbsim.DxDelivery(
                name='monitor', product=tbsim.Xpert(), coverage=0.9,
                eligibility=treatment_monitoring_eligibility('first_line',
                                                              after_steps=8),
                result_state='still_positive',
            )
            switch = StrainAwareTxDelivery(
                product=second_line,
                name='second_line',
                eligibility=lambda sim: sim.interventions['monitor'].still_positive.uids,
            )
    """
    def _elig(sim):
        tx = sim.interventions.get(tx_delivery_name)
        if tx is None:
            return ss.uids()
        tb = tbsim.get_tb(sim)
        # On-treatment agents (TB-state, not just intervention bookkeeping)
        on_tx_uids = tb.on_treatment.uids
        if len(on_tx_uids) == 0:
            return on_tx_uids
        # Time on treatment, in sim steps. ti_treatment_start is a FloatArr
        # indexed by UID; np.asarray here is only used to do arithmetic on
        # the resulting positional values.
        ti_start = np.asarray(tx.ti_treatment_start[on_tx_uids], dtype=float)
        elapsed = sim.ti - ti_start
        if every_steps:
            ready_mask = elapsed >= float(after_steps)
            ready_mask &= (
                (elapsed - float(after_steps)) % float(every_steps) == 0
            )
        else:
            # One-shot semantics: eligible only on the first due step.
            ready_mask = np.isclose(elapsed, float(after_steps))
        return on_tx_uids[ready_mask]
    _elig.__name__ = f'monitoring_after_{after_steps}_steps'
    return _elig

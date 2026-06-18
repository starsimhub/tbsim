"""Drug susceptibility testing (DST) for the TB resistance overlay."""

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

    Per-drug sensitivity (probability of correctly detecting resistance when
    any carried strain is resistant) and specificity (probability of
    correctly reporting susceptibility when no carried strain is resistant)
    are configurable.

    Args:
        registry (StrainRegistry): Strain registry.
        drugs (list[str]): Subset of drugs to report on (default: all in
            registry).
        sensitivity (dict|float): Per-drug sensitivity. Default 0.95 for all drugs.
        specificity (dict|float): Per-drug specificity. Default 0.99 for all drugs.

    Example:
        ::

            dst = DSTDx(registry, drugs=['INH', 'RIF'], sensitivity=0.95,
                        specificity=0.99)
            result = dst.administer(sim, agent_uids)
            # result['INH'] is a boolean array of observed INH-resistance
    """

    def __init__(self, registry, drugs=None, sensitivity=0.95, specificity=0.99):
        super().__init__()
        self.registry = registry
        if drugs is None:
            drugs = list(registry.drugs)
        for d in drugs:
            if d not in registry.drugs:
                raise ValueError(f'DSTDx drug {d!r} not in registry drugs {registry.drugs}')
        self.drugs = list(drugs)

        if isinstance(sensitivity, (int, float)):
            sensitivity = {d: float(sensitivity) for d in self.drugs}
        if isinstance(specificity, (int, float)):
            specificity = {d: float(specificity) for d in self.drugs}
        self.sensitivity = {d: float(sensitivity[d]) for d in self.drugs}
        self.specificity = {d: float(specificity[d]) for d in self.drugs}

        self._rng_sens = {
            d: ss.bernoulli(p=self.sensitivity[d], name=f'dst_sens_{d}', strict=False)
            for d in self.drugs
        }
        self._rng_spec = {
            d: ss.bernoulli(p=self.specificity[d], name=f'dst_spec_{d}', strict=False)
            for d in self.drugs
        }
        for dist in list(self._rng_sens.values()) + list(self._rng_spec.values()):
            if not dist.initialized:
                dist.init()
        return

    def true_phenotype(self, tb, uids, drug):
        """Return per-uid boolean array of true resistance to *drug*."""
        if not len(uids):
            return np.zeros(0, dtype=bool)
        profile = tb.strain_profile
        if profile is None:
            return np.zeros(len(uids), dtype=bool)
        drug_col = self.registry.drugs.index(drug)
        resistant_strain_idx = np.where(self.registry.resistance[:, drug_col] == 1)[0]
        true_pos = np.zeros(len(uids), dtype=bool)
        for s_idx in resistant_strain_idx:
            carrier_mask = np.asarray(
                getattr(profile._tb, profile.names[int(s_idx)])[uids], dtype=bool,
            )
            true_pos |= carrier_mask
        return true_pos

    def administer(self, sim, uids):
        """Run DST on *uids* and return per-drug observed-resistance dict."""
        tb = tbsim.get_tb(sim)
        results = {}
        n = len(uids)
        for drug in self.drugs:
            true_res = self.true_phenotype(tb, uids, drug)
            sens_pos = np.asarray(self._rng_sens[drug].rvs(n), dtype=bool)
            spec_neg = np.asarray(self._rng_spec[drug].rvs(n), dtype=bool)
            observed = np.where(true_res, sens_pos, ~spec_neg)
            results[drug] = observed
        return results


class DSTDelivery(ss.Intervention):
    """
    Run DST on diagnosed agents and stash observed phenotype on a state array.

    Each per-drug result is stored on a ``BoolArr`` named
    ``observed_<drug>_resistant``. Downstream treatment routing can read these
    arrays to choose a regimen.

    Args:
        product (DSTDx): The DST product.
        eligibility (callable): Optional function ``(sim) -> uids``.
            Default: agents who have been diagnosed by any DxDelivery.
        coverage (float): Probability of receiving DST among eligible agents.
            Default 1.0.
    """

    def __init__(self, product, eligibility=None, coverage=1.0, **kwargs):
        super().__init__(**kwargs)
        self.product = product
        self.eligibility = eligibility
        self.define_pars(
            coverage=ss.bernoulli(p=float(coverage)),
        )
        states = [ss.BoolArr('tested_dst', default=False)]
        for drug in product.drugs:
            states.append(ss.BoolArr(f'observed_{drug}_resistant', default=False))
        self.define_states(*states)
        return

    def _default_eligibility(self, sim):
        dx = sim.get_dx(result_state='diagnosed')
        if dx is None:
            return ss.uids()
        return (dx.diagnosed & sim.people.alive & ~self.tested_dst).uids

    def step(self):
        if self.eligibility is not None:
            uids = ss.uids(self.eligibility(self.sim))
        else:
            uids = self._default_eligibility(self.sim)
        if len(uids) == 0:
            return
        uids = self.pars.coverage.filter(uids)
        if len(uids) == 0:
            return
        results = self.product.administer(self.sim, uids)
        for drug, obs in results.items():
            arr = getattr(self, f'observed_{drug}_resistant')
            arr[uids[obs]] = True
        self.tested_dst[uids] = True
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

            dst = DSTDelivery(product=DSTDx(registry, drugs=['INH','RIF','BDQ']))
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
            the first eligibility. ``None`` means a single test once-after.

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
        ready_mask = elapsed >= float(after_steps)
        if every_steps:
            ready_mask &= (
                (elapsed - float(after_steps)) % float(every_steps) == 0
            )
        return on_tx_uids[ready_mask]
    _elig.__name__ = f'monitoring_after_{after_steps}_steps'
    return _elig

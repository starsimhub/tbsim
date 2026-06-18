"""Drug susceptibility testing (DST) for the TB resistance overlay."""

import numpy as np
import starsim as ss
import tbsim

__all__ = ['DSTDx', 'DSTDelivery']


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

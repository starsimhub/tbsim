"""
Drug-susceptibility testing (DST) for multi-strain TB.

``DST`` (product) produces an *observed* ``n``-bit resistance profile from an agent's
carried strains, applying per-drug sensitivity/specificity **at the strain level** and
an optional per-strain observation probability ``p_strain_obs`` (a within-host/culture
bottleneck; default = strain fitness). ``DSTDelivery`` (intervention) administers it to
eligible agents and stores the observed profile, which downstream treatment eligibility
can read — either a single-drug callable (``observed_resistant``) or a composable
multi-drug router (``matches``) for DST-dependent regimen selection.
"""

import numpy as np
import starsim as ss

from ..tb import TBS, get_tb
from .tb_resistant import TBResistant

__all__ = ['DST', 'DSTDelivery', 'reset_dst_on_cure']


class DST(ss.Product):
    """
    Drug-susceptibility test.

    Args:
        strains (Strains): the strain registry (e.g. ``tb.strains``).
        sens (float/dict): per-drug sensitivity (P(observe resistant | strain is resistant)). Default 1.
        spec (float/dict): per-drug specificity (P(observe susceptible | strain is susceptible)). Default 1.
        p_strain_obs (None/float/dict): probability each carried strain is observed at all (culture
            bottleneck). ``None`` (default) uses the strain's transmission fitness as a bacillary-load proxy.
    """

    def __init__(self, strains, sens=1.0, spec=1.0, p_strain_obs=None, **kwargs):
        super().__init__(**kwargs)
        self.strains = strains
        self.sens = np.array([_perdrug(sens, d, 1.0) for d in strains.drugs])
        self.spec = np.array([_perdrug(spec, d, 1.0) for d in strains.drugs])
        if p_strain_obs is None:
            self.p_obs_by_id = strains.fitness.copy()
        elif np.isscalar(p_strain_obs):
            self.p_obs_by_id = np.full(strains.m, float(p_strain_obs))
        else:
            self.p_obs_by_id = np.array([p_strain_obs.get(i, 1.0) for i in range(strains.m)])

        # Independent CRN stream per strain, so a phenotype carried by several strains has a higher
        # detection probability (spec §DST) and per-strain observation drop-out is independent across strains.
        self._obs_rngs = [ss.random(name=f'dst_obs_{j}') for j in range(strains.m)]
        self._call_rngs = [ss.random(name=f'dst_call_{j}') for j in range(strains.m)]
        return

    def administer(self, tb, uids):
        """Return the observed ``n``-bit resistance profile (as an integer per agent) for ``uids``.

        DST is applied at the strain level then aggregated to the agent phenotype: each carried strain
        is independently observed (culture bottleneck ``p_strain_obs``), and each observed strain
        independently passes sensitivity (if truly resistant) or fails specificity (if susceptible). A
        drug is called resistant for an agent if *any* of its observed strains reads resistant — so a
        phenotype carried by several strains is more likely detected (spec §DST). Sensitivity and
        specificity for the drugs within one strain share that strain's call draw (a minor, deliberate
        within-strain correlation); independence across strains is what drives the multi-strain boost.
        """
        m = self.strains
        n_u = len(uids)
        if n_u == 0:
            return np.zeros(0, dtype=int)
        carried = m.carried(tb.strain_mask[uids])  # (n_u, m)
        profile = np.zeros(n_u, dtype=int)
        for j in range(m.m):
            cj = carried[:, j]
            if not cj.any():
                continue
            seen = cj & (np.asarray(self._obs_rngs[j].rvs(uids), dtype=float) < self.p_obs_by_id[j])
            if not seen.any():
                continue
            call = np.asarray(self._call_rngs[j].rvs(uids), dtype=float)
            for di in range(m.n):
                if m.profile[j, di]:
                    hit = seen & (call < self.sens[di])    # truly resistant → read resistant w.p. sens
                else:
                    hit = seen & (call >= self.spec[di])   # truly susceptible → false positive w.p. 1-spec
                profile[hit] |= (1 << di)
        return profile


class DSTDelivery(ss.Intervention):
    """
    Administers a ``DST`` to eligible agents and stores the observed resistance profile.

    Args:
        product (DST): the DST product.
        eligibility (callable): ``sim -> uids`` (default: active TB, alive, not yet tested).
    """

    def __init__(self, product, eligibility=None, **kwargs):
        super().__init__()
        self.product = product
        self.eligibility = eligibility
        self._n_tested = 0
        self.define_states(
            ss.IntArr('dst_profile', default=0),   # observed n-bit resistance profile
            ss.BoolArr('dst_tested', default=False),
            ss.FloatArr('ti_dst', default=np.nan),
        )
        self.update_pars(**kwargs)
        product.name = f'{self.name}_product'
        return

    def _get_eligible(self, sim):
        if self.eligibility is not None:
            return ss.uids(self.eligibility(sim))
        tb = get_tb(sim, which=TBResistant)
        return (tb.active_tb & sim.people.alive & ~self.dst_tested).uids

    def reset(self, uids):
        """Clear sticky DST state for *uids* (e.g. after cure) so stale profiles cannot re-route treatment."""
        uids = ss.uids(uids)
        if len(uids) == 0:
            return
        self.dst_tested[uids] = False
        self.dst_profile[uids] = 0
        self.ti_dst[uids] = np.nan
        return

    def observed_resistant(self, drug, require_active=True, exclude_on_treatment=True):
        """Return a callable ``sim -> uids`` selecting agents observed resistant to ``drug`` (for treatment eligibility).

        By default restricts to alive agents with current active TB and not on treatment, so cured agents
        with a stale positive profile are not re-selected.
        """
        di = self.product.strains.drug_idx[drug]
        name = self.name  # resolve the sim's own (copied) DST instance at call time
        def _elig(sim):
            dst = sim.interventions[name]
            tb = get_tb(sim, which=TBResistant)
            sel = (dst.dst_tested & sim.people.alive).uids
            if len(sel) == 0:
                return sel
            obs = ((np.asarray(dst.dst_profile[sel]) >> di) & 1).astype(bool)
            out = sel[obs]
            if require_active and len(out):
                out = out[tb.active_tb[out]]
            if exclude_on_treatment and len(out):
                out = out[tb.state[out] != TBS.TREATMENT]
            return out
        return _elig

    def matches(self, require_tested=True, exclude_on_treatment=True, require_active=True, **per_drug):
        """Return an eligibility callable selecting agents whose observed DST profile matches ``per_drug``.

        E.g. ``matches(RIF=True, BDQ=False)`` selects observed-RIF-resistant, observed-BDQ-susceptible
        agents. The returned ``sim -> uids`` callable restricts to DST-tested (unless
        ``require_tested=False``), to current active TB (unless ``require_active=False``), and, unless
        ``exclude_on_treatment=False``, not-currently-on-treatment agents. Compose with
        ``TxDeliveryR(eligibility=..., supersedes=[...])`` to route or switch regimens.
        """
        strains = self.product.strains
        spec = {strains.drug_idx[d]: bool(v) for d, v in per_drug.items()}
        name = self.name
        def _elig(sim):
            dst = sim.interventions[name]
            tb = get_tb(sim, which=TBResistant)
            sel = dst.dst_tested.uids if require_tested else sim.people.alive.uids
            if len(sel) == 0:
                return sel
            prof = np.asarray(dst.dst_profile[sel])
            mask = np.ones(len(sel), dtype=bool)
            for di, want in spec.items():
                bit = ((prof >> di) & 1).astype(bool)
                mask &= bit if want else ~bit
            out = sel[mask]
            if require_active and len(out):
                out = out[tb.active_tb[out]]
            if exclude_on_treatment and len(out):
                out = out[tb.state[out] != TBS.TREATMENT]
            return out
        _elig.__name__ = 'dst_matches_' + '_'.join(f'{d}{"+" if v else "-"}' for d, v in per_drug.items())
        return _elig

    def step(self):
        tb = get_tb(self.sim, which=TBResistant)
        elig = self._get_eligible(self.sim)
        self._n_tested = len(elig)
        if len(elig) == 0:
            return
        self.dst_profile[elig] = self.product.administer(tb, elig)
        self.dst_tested[elig] = True
        self.ti_dst[elig] = self.ti
        return

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('n_tested', dtype=int))
        return

    def update_results(self):
        self.results.n_tested[self.ti] = self._n_tested
        return


def reset_dst_on_cure(sim, uids):
    """Reset every ``DSTDelivery`` sticky profile for cured agents (if any DST modules are present)."""
    uids = ss.uids(uids)
    if len(uids) == 0:
        return
    for iv in sim.interventions.values():
        if isinstance(iv, DSTDelivery):
            iv.reset(uids)
    return


def _perdrug(val, drug, default):
    """Resolve a scalar-or-dict per-drug parameter to a value for ``drug``."""
    if np.isscalar(val):
        return float(val)
    return float(val.get(drug, default))

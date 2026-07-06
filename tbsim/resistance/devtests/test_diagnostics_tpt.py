"""
Diagnostics (DST) and strain-aware TPT — spec §"Diagnostics & Treatment Modification" and §TPT.

DST: sensitivity/specificity recovery, the mixed-infection detection boost, and the
``p_strain_obs`` culture bottleneck. TPT: per-strain sterilization / resistance unmasking (the
Mills–Cohen "IPT drives resistance" dynamic) and the state-dependent TPT-failure acquisition gradient.
"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS


def _init(tb, interventions, n=20000, stop='2001-12-31'):
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))
    sim = ss.Sim(n_agents=n, networks=net, diseases=tb, interventions=interventions, dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date(stop), rand_seed=0, verbose=0)
    sim.init()
    return sim


# --------------------------------------------------------------------------- DST sens/spec
def test_dst_recovers_sens_spec_mono():
    """In the mono-infection limit the observed calls recover the configured sens/spec."""
    sens, spec = 0.9, 0.95
    tb = tbsim.TBResistant(pars=dict(init_prev=ss.bernoulli(0.0)))
    sim = _init(tb, tbsim.DSTDelivery(product=tbsim.DST(strains=tb.strains, sens=sens, spec=spec, p_strain_obs=1.0)))
    tb = tbsim.get_tb(sim, which=tbsim.TBResistant)
    prod = next(iv for iv in sim.interventions.values() if isinstance(iv, tbsim.DSTDelivery)).product
    res, sus = ss.uids(np.arange(10000)), ss.uids(np.arange(10000, 20000))
    tb.strain_mask[res] = 2  # mono-B (resistant)
    tb.strain_mask[sus] = 1  # mono-A (susceptible)
    assert np.isclose((prod.administer(tb, res) & 1).astype(bool).mean(), sens, atol=0.02)
    assert np.isclose(1 - (prod.administer(tb, sus) & 1).astype(bool).mean(), spec, atol=0.02)


def test_dst_multiple_strains_raise_detection():
    """Spec §DST: at p_strain_obs=1, a phenotype carried by *two* strains is detected with higher
    probability than by one — P(detect) = 1-(1-sens)^2 vs sens (requires independent per-strain calls)."""
    sens = 0.6
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], pars=dict(init_prev=ss.bernoulli(0.0)))
    dst = tbsim.DSTDelivery(product=tbsim.DST(strains=tb.strains, sens={'RIF': sens}, spec=1.0, p_strain_obs=1.0))
    sim = _init(tb, dst)
    tb = tbsim.get_tb(sim, which=tbsim.TBResistant)
    prod = next(iv for iv in sim.interventions.values() if isinstance(iv, tbsim.DSTDelivery)).product
    one = ss.uids(np.arange(10000))
    two = ss.uids(np.arange(10000, 20000))
    tb.strain_mask[one] = (1 << 1)              # carries only strain 1 = {RIF}
    tb.strain_mask[two] = (1 << 1) | (1 << 3)   # carries strain 1={RIF} and strain 3={RIF,BDQ} — both RIF-R
    det_one = (prod.administer(tb, one) & 1).astype(bool).mean()   # bit 0 = RIF
    det_two = (prod.administer(tb, two) & 1).astype(bool).mean()
    assert np.isclose(det_one, sens, atol=0.02)
    assert np.isclose(det_two, 1 - (1 - sens) ** 2, atol=0.02)     # ≈ 0.84 > 0.6
    assert det_two > det_one


def test_dst_p_strain_obs_bottleneck_lowers_detection():
    """Spec §DST: p_strain_obs < 1 (strain drop-out at sampling/culture) reduces overall DST sensitivity."""
    sens = 0.9
    tb = tbsim.TBResistant(pars=dict(init_prev=ss.bernoulli(0.0)))
    def detect(p_obs):
        dst = tbsim.DSTDelivery(product=tbsim.DST(strains=tb.strains, sens=sens, spec=1.0, p_strain_obs=p_obs))
        sim = _init(tb, dst, n=10000)
        tbx = tbsim.get_tb(sim, which=tbsim.TBResistant)
        prod = next(iv for iv in sim.interventions.values() if isinstance(iv, tbsim.DSTDelivery)).product
        u = ss.uids(np.arange(10000)); tbx.strain_mask[u] = 2  # mono-B (resistant)
        return (prod.administer(tbx, u) & 1).astype(bool).mean()
    full, bottleneck = detect(1.0), detect(0.5)
    assert np.isclose(full, sens, atol=0.02)                 # all strains observed → recovers sens
    assert np.isclose(bottleneck, 0.5 * sens, atol=0.03)     # half observed → ~half the detections
    assert bottleneck < full


# --------------------------------------------------------------------------- TPT: unmasking / selection
def test_tpt_unmasks_and_selects_resistance():
    """Spec §TPT (Mills–Cohen): a susceptible-strain-clearing TPT applied to a mixed epidemic tilts
    competition toward the resistant strain — the resistant fraction of active TB is *higher* with
    TPT than without, even though a less-fit resistant strain would otherwise be out-competed."""
    def run(with_tpt):
        tb = tbsim.TBResistant(drugs=['INH'], rel_fitness={'INH': 0.9},
                               pars=dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.15),
                                         init_strains=[0.8, 0.2], rr_reinfection_inf=0.0, rr_reinfection_non=0.0))
        ivs = None
        if with_tpt:
            ivs = tbsim.TPTSimple(product=tbsim.TPTRx(strains=tb.strains, regimen_drugs=['INH'],
                                  pars=dict(efficacy=ss.bernoulli(0.9), p_sterilize=ss.bernoulli(1.0))),
                                  pars=dict(coverage=ss.bernoulli(0.5)))
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=10), dur=0))
        sim = ss.Sim(n_agents=4000, networks=net, diseases=tb, interventions=ivs, dt=ss.days(30),
                     start=ss.date('2000-01-01'), stop=ss.date('2035-12-31'), rand_seed=0, verbose=0)
        sim.run()
        r = sim.results.tb
        return float(np.mean(r['frac_resist'][-6:]))  # late-window resistant fraction of active TB
    no_tpt = run(False)
    tpt = run(True)
    assert tpt > no_tpt, f'TPT should raise the resistant fraction (unmasking): TPT {tpt:.3f} vs none {no_tpt:.3f}'


def test_tpt_failure_acquisition_state_gradient():
    """Spec §TPT: TPT-failure resistance acquisition risk varies by TB state — very low for INFECTION,
    high for SYMPTOMATIC (default RR gradient 0.05 / 0.5 / 1 / 1)."""
    tb = tbsim.TBResistant(drugs=['INH'], pars=dict(init_prev=ss.bernoulli(0.0)))
    tpt = tbsim.TPTSimple(product=tbsim.TPTRx(strains=tb.strains, regimen_drugs=['INH'],
                          p_tpt_acq={'INH': 1.0}))  # p=1 → acquisition fraction = the per-state RR
    sim = _init(tb, tpt, n=20000)
    tb = tbsim.get_tb(sim, which=tbsim.TBResistant)
    prod = next(iv for iv in sim.interventions.values() if isinstance(iv, tbsim.TPTSimple)).product
    latent = ss.uids(np.arange(10000))
    sympt = ss.uids(np.arange(10000, 20000))
    for u, st in [(latent, TBS.INFECTION), (sympt, TBS.SYMPTOMATIC)]:
        tb.state[u] = st
        tb.strain_mask[u] = 1  # mono INH-susceptible (can acquire INH resistance)
    prod._acquire(ss.uids(np.arange(20000)))
    acq_latent = np.mean(tb.strain_mask[latent] == 2)   # became INH-resistant
    acq_sympt = np.mean(tb.strain_mask[sympt] == 2)
    assert np.isclose(acq_latent, 0.05, atol=0.02)      # INFECTION RR = 0.05
    assert np.isclose(acq_sympt, 1.0, atol=0.02)        # SYMPTOMATIC RR = 1.0
    assert acq_sympt > acq_latent


if __name__ == '__main__':
    import sys, pytest
    sys.exit(pytest.main([__file__, '-v']))

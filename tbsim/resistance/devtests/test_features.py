"""
Usability features from the update plan: eligibility combinators / monitoring sugar / failure oracle
(L5), the strain-agnostic convenience factory (L7), and drug-name validation (L8).
"""

import numpy as np
import pytest
import starsim as ss
import tbsim
from tbsim import TBS


# --------------------------------------------------------------------------- L5: eligibility combinators
def test_eligibility_combinators_intersect_and_union():
    """eligibility_all / eligibility_any intersect / union the uids returned by their callables."""
    a = lambda sim: ss.uids([1, 2, 3, 4])
    b = lambda sim: ss.uids([3, 4, 5, 6])
    assert set(tbsim.eligibility_all(a, b)(None).tolist()) == {3, 4}
    assert set(tbsim.eligibility_any(a, b)(None).tolist()) == {1, 2, 3, 4, 5, 6}
    assert len(tbsim.eligibility_all()(None)) == 0


def test_monitoring_require_and_will_fail():
    """L5: treatment_monitoring_eligibility(require=...) narrows monitoring to agents also matching a
    DST profile (via eligibility_all), and will_fail(tx) selects on-treatment agents whose pre-rolled
    course outcome is a failure."""
    tb = tbsim.TBResistant(drugs=['RIF'], init_prev=ss.bernoulli(0.0))
    first = tbsim.TxDeliveryR(name='first', product=tbsim.TxR(strains=tb.strains, base_efficacy=0.5))
    dst = tbsim.DSTDelivery(name='dst', product=tbsim.DST(strains=tb.strains, sens=1.0, spec=1.0))
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=1), dur=0))
    sim = tbsim.Sim(n_agents=400, networks=net, diseases=tb, demographics=[], interventions=[dst, first], dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date('2001-12-31'), rand_seed=0, verbose=0)
    sim.init()
    tb = sim.get_tb()
    first, dst = sim.interventions['first'], sim.interventions['dst']
    coh = ss.uids(np.arange(200))
    tb.state[coh] = TBS.TREATMENT
    first.on_course[coh] = True
    first.ti_treatment_start[coh] = -3           # started 3 steps ago (≥ after_steps=2)
    rif_r = ss.uids(np.arange(100))
    dst.dst_tested[coh] = True
    dst.ti_dst[coh] = 0
    dst.dst_profile[rif_r] = 1                    # observed RIF-resistant
    failing = ss.uids(np.arange(50, 150))
    first.pending_surv[coh] = 0                  # default: pre-rolled as cured
    first.pending_surv[failing] = 1              # these will fail

    mon = tbsim.treatment_monitoring_eligibility('first', after_steps=2)
    assert set(mon(sim).tolist()) == set(coh.tolist())          # everyone ≥2 steps into first-line

    matches_rif = dst.matches(RIF=True, exclude_on_treatment=False)  # monitored agents are on treatment
    mon_req = tbsim.treatment_monitoring_eligibility('first', after_steps=2, require=matches_rif)
    assert set(mon_req(sim).tolist()) == set(rif_r.tolist())    # narrowed to observed RIF-resistant
    assert set(tbsim.eligibility_all(mon, matches_rif)(sim).tolist()) == set(rif_r.tolist())  # same via combinator

    assert set(tbsim.will_fail('first')(sim).tolist()) == set(failing.tolist())


# --------------------------------------------------------------------------- L7: agnostic convenience mode
def test_agnostic_reproduces_single_strain_tb():
    """L7: TBResistant.agnostic() reproduces single-strain tbsim.TB (no resistance ever arises, and the
    endemic active prevalence tracks base TB within stochastic tolerance on a matched scenario)."""
    pars = dict(beta=ss.permonth(0.2), init_prev=ss.bernoulli(0.05))
    def build(tb):
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
        return tbsim.Sim(n_agents=5000, networks=net, diseases=tb, demographics=[], dt=ss.days(30),
                      start=ss.date('2000-01-01'), stop=ss.date('2035-12-31'), rand_seed=1, verbose=0)
    s_tb = build(tbsim.TB(name='tb', pars=pars)); s_tb.run()
    s_ag = build(tbsim.TBResistant.agnostic(pars=pars)); s_ag.run()
    assert float(np.max(s_ag.results.tb['frac_resist'])) == 0.0            # no resistance ever
    p_tb = np.array(s_tb.results.tb['prevalence_active'])[-24:].mean()
    p_ag = np.array(s_ag.results.tb['prevalence_active'])[-24:].mean()
    assert abs(p_tb - p_ag) < 0.02                                         # tracks single-strain TB


# --------------------------------------------------------------------------- L8: drug-name validation
def test_validate_drugs_raises_on_typos():
    """L8: mistyped drug names fail fast (instead of resolving silently via .get()) in TxR, TPTRx, DST."""
    tb = tbsim.TBResistant(drugs=['RIF'], init_prev=ss.bernoulli(0.0))
    s = tb.strains
    with pytest.raises(ValueError, match='RIFF'):
        tbsim.TxR(strains=s, regimen_drugs=['RIFF'])            # typo in regimen_drugs
    with pytest.raises(ValueError, match='RIFF'):
        tbsim.TxR(strains=s, q_acq={'RIFF': 0.1})               # typo in q_acq
    with pytest.raises(ValueError, match='RIFF'):
        tbsim.TxR(strains=s, resist_penalty={'RIFF': 0.1})      # typo in resist_penalty
    with pytest.raises(ValueError, match='RIFF'):
        tbsim.TPTRx(strains=s, p_tpt_acq={'RIFF': 0.1})         # typo in TPT acquisition
    with pytest.raises(ValueError, match='RIFF'):
        tbsim.DST(strains=s, sens={'RIFF': 0.9})                # typo in DST sensitivity
    # A correct drug name still works.
    tbsim.TxR(strains=s, regimen_drugs=['RIF'], q_acq={'RIF': 0.1})


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))

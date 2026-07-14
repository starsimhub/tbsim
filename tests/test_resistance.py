"""
Tests for the multi-strain (drug-resistance) TB extension (tbsim.resistance).

Covers: the Strains registry, the transmission/superinfection mechanics, the
strain-aware natural history (progression bottleneck, per-drug de-novo acquisition),
the strain-resolved treatment operator, treatment monitoring / regimen switching,
strain-aware TPT, DST + DST-routing, and validation against the two-strain reference
ODE (tbsim.compartmental.TwoStrainODE, a port of ode.r) — including the model-tests.md
§8 conservation / single-strain-reduction / strain-symmetry checks.
"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS
import tbsim.compartmental as tc
import pytest


# --------------------------------------------------------------------------- helpers
def make_sim(tb, seed=1, n=3000, start=ss.date('2000-01-01'), stop=ss.date('2035-12-31'),
             dt=ss.days(30), interventions=None, analyzers=None):
    # NB: pass time objects as function defaults (not freshly built inline) so sim.init() sizes
    # the state arrays — matches the tests/test_tb.py convention.
    if isinstance(stop, str):
        stop = ss.date(stop)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=10), dur=0))
    sim = tbsim.Sim(tb_model=tb, n_agents=n, networks=net, demographics=[], interventions=interventions,
                    analyzers=analyzers, dt=dt, start=start, stop=stop, rand_seed=seed, verbose=0)
    return sim


NOSUPER = dict(rr_reinfection_inf=0.0, rr_reinfection_non=0.0)  # σ = 0


def _product(sim, cls):
    """The (initialized) product/intervention of type ``cls`` from the run/inited sim."""
    return next(iv for iv in sim.interventions.values() if isinstance(iv, cls))


# --------------------------------------------------------------------------- Strains registry
def test_strains_registry():
    s = tbsim.Strains(['RIF', 'BDQ'], rel_fitness={'RIF': 0.5, 'BDQ': 0.8})
    assert s.n == 2 and s.m == 4
    # fitness = product of costs over resistant drugs; id 0 pan-susceptible = 1
    assert np.allclose(s.fitness, [1.0, 0.5, 0.8, 0.4])
    # human-readable labels
    assert s.labels == ['pan', 'RIF', 'BDQ', 'RIF+BDQ']
    # strain id 3 (mask over strains) — carried decode of an agent carrying strains 0 and 1
    assert list(s.carried(np.array([3]))[0]) == [True, True, False, False]
    # transmit ∝ fitness among carried strains (agent carrying strains 1 and 2: 0.5 vs 0.8)
    tp = s.transmit_probs(np.array([6]))[0]  # mask bits 1,2 set
    assert np.isclose(tp[1], 0.5 / 1.3) and np.isclose(tp[2], 0.8 / 1.3)
    # aggregate phenotype: agent carrying strains 1 (RIF) and 2 (BDQ) is resistant to both
    assert list(s.phenotype_any(np.array([6]))[0]) == [True, True]


def test_strains_validation():
    with pytest.raises(ValueError):
        tbsim.Strains(['RIF', 'RIF'])                       # duplicate drug
    with pytest.raises(ValueError):
        tbsim.Strains(['RIF'], rel_fitness={'BDQ': 0.5})    # unknown drug
    with pytest.raises(ValueError):
        tbsim.Strains(['RIF'], rel_fitness={'RIF': 1.5})    # fitness out of range


# --------------------------------------------------------------------------- reduction to single-strain TB
def test_single_strain_reduction_matches_tb():
    """A single pan-susceptible strain, σ=0, reproduces the single-strain TB trajectory closely."""
    pars = dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12))
    sim_tb = make_sim(tbsim.TB(pars=pars)); sim_tb.run()
    sim_r = make_sim(tbsim.TBResistant(**pars, **NOSUPER)); sim_r.run()
    a = np.array([sim_tb.results.tb[f'n_{s.name}'][-1] for s in TBS])
    b = np.array([sim_r.results.tb[f'n_{s.name}'][-1] for s in TBS])
    # RNG streams differ (extra strain draws), so require close, not identical.
    assert np.allclose(a, b, rtol=0.15, atol=25)
    assert sim_r.results.tb['frac_resist'][-1] == 0.0  # no resistance ever created


# --------------------------------------------------------------------------- symmetry (neutral drift)
def test_strain_symmetry_neutral():
    """Two equally-fit strains, σ=0, symmetric seeding → resistant fraction averages ~0.5 over seeds."""
    fr = []
    for seed in range(6):
        tb = tbsim.TBResistant(rel_fitness={'TX': 1.0},
                               beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12),
                               init_strains=[0.5, 0.5], **NOSUPER)
        sim = make_sim(tb, seed=seed); sim.run()
        assert sim.results.tb['frac_super'][-1] == 0.0  # σ=0 → no superinfection
        fr.append(sim.results.tb['frac_resist'][-1])
    assert 0.30 < np.mean(fr) < 0.70  # symmetric in expectation


# --------------------------------------------------------------------------- competitive exclusion
def test_competitive_exclusion_no_treatment():
    """Without treatment a less-fit resistant strain is out-competed (frac_resist declines toward 0)."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 0.55},
                           beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12),
                           init_strains=[0.7, 0.3], **NOSUPER)
    sim = make_sim(tb, stop='2060-12-31'); sim.run()
    fr = sim.results.tb['frac_resist']
    assert fr[3] > 0.15 and fr[-1] < fr[3] / 2  # started substantial, then declined


# --------------------------------------------------------------------------- superinfection gating
def test_superinfection_requires_sigma():
    """AB co-infections appear only when the superinfection susceptibility σ>0."""
    base = dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12), init_strains=[0.5, 0.5])
    off = make_sim(tbsim.TBResistant(rel_fitness={'TX': 1.0}, **base, **NOSUPER)); off.run()
    on = make_sim(tbsim.TBResistant(rel_fitness={'TX': 1.0}, **base,
                  rr_reinfection_inf=1.0, rr_reinfection_non=1.0)); on.run()
    assert off.results.tb['frac_super'][-1] == 0.0
    assert on.results.tb['frac_super'][-1] > 0.05
    assert np.sum(on.results.tb['new_identical_superinf']) > 0  # identical-strain re-exposures now counted (spec §1)


# --------------------------------------------------------------------------- de-novo acquisition
def test_denovo_mixed_vs_replacement():
    """De-novo resistance from a pure-A epidemic: mixed makes AB, replacement does not; no p_rand makes none."""
    base = dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12), init_strains=[1.0, 0.0])
    none = make_sim(tbsim.TBResistant(rel_fitness={'TX': 0.9}, **base, **NOSUPER),
                    stop='2050-12-31'); none.run()
    assert none.results.tb['frac_resist'][-1] == 0.0  # no mechanism to create B

    mixed = make_sim(tbsim.TBResistant(rel_fitness={'TX': 0.9},
                     **base, p_rand={'TX': 0.02}, prog_resist_mode='mixed', **NOSUPER),
                     stop='2050-12-31'); mixed.run()
    rep = make_sim(tbsim.TBResistant(rel_fitness={'TX': 0.9},
                   **base, p_rand={'TX': 0.02}, prog_resist_mode='replacement', **NOSUPER),
                   stop='2050-12-31'); rep.run()
    assert np.sum(mixed.results.tb['new_denovo_resistance']) > 0
    assert mixed.results.tb['frac_resist'][-1] > 0
    assert mixed.results.tb['frac_super'][-1] > 0    # mixed creates AB
    assert rep.results.tb['frac_super'][-1] == 0.0   # replacement never creates AB


def test_denovo_per_drug_specificity():
    """p_rand is per drug: with p_rand on BDQ only, active TB acquires BDQ resistance but never RIF."""
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], rel_fitness={'BDQ': 0.9},
                           beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12),
                           init_strains=[1.0, 0.0, 0.0, 0.0], p_rand={'BDQ': 0.05}, **NOSUPER)
    sim = make_sim(tb, stop='2050-12-31'); sim.run()
    r = sim.results.tb
    assert np.sum(r['new_denovo_resistance']) > 0
    assert r['frac_resist_BDQ'][-1] > 0.01   # BDQ resistance emerges de-novo
    assert r['frac_resist_RIF'][-1] == 0.0   # RIF never mutates (p_rand['RIF'] == 0)


# --------------------------------------------------------------------------- treatment
def test_treatment_selects_for_resistance():
    """Treatment that cures A well but B poorly drives the resistant fraction up."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 0.575},
                           beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12),
                           init_strains=[0.9, 0.1], rr_reinfection_inf=1.0, rr_reinfection_non=1.0)
    tx = tbsim.TxDeliveryR(product=tbsim.TxR(strains=tb.strains, base_efficacy=0.75,
                           resist_penalty={'TX': 1/3.}, adherence=0.9, q_acq={'TX': 0.05}),
                           rate_sym=ss.peryear(1.2), rate_asym=ss.peryear(0.05))
    sim = make_sim(tb, stop='2050-12-31', interventions=tx); sim.run()
    r = sim.results.tb
    assert r['frac_resist'][-1] > 0.5                          # resistance selected for
    assert np.sum(sim.results[tx.name].n_acquired) > 0         # some acquisition-on-failure


def test_treatment_outcome_operator_matches_ode():
    """Per-strain cure of an AB cohort reproduces the ODE outcome probabilities (adherence=1, q=0)."""
    ea, eb = 0.75, 0.25
    tb = tbsim.TBResistant(rel_fitness={'TX': 0.575}, init_prev=ss.bernoulli(0.0))
    prod = tbsim.TxR(strains=tb.strains, base_efficacy=ea, resist_penalty={'TX': eb/ea}, adherence=1.0, q_acq=None)
    sim = make_sim(tb, n=20000, interventions=tbsim.TxDeliveryR(product=prod)); sim.init()
    # ss.Sim copies its modules, so operate on the sim's initialized copies.
    tb = sim.get_tb()
    prod = _product(sim, tbsim.TxDeliveryR).product
    uids = ss.uids(np.arange(20000))
    tb.strain_mask[uids] = 3  # everyone AB
    surv = prod.roll_survivors(tb, uids)
    frac = lambda v: np.mean(surv == v)
    assert np.isclose(frac(0), ea * eb, atol=0.02)             # both cured
    assert np.isclose(frac(1), (1 - ea) * eb, atol=0.02)       # A survives, B cured
    assert np.isclose(frac(2), ea * (1 - eb), atol=0.02)       # B survives, A cured
    assert np.isclose(frac(3), (1 - ea) * (1 - eb), atol=0.02) # both survive


def test_treatment_interrupt_reverts_and_preserves_strains():
    """interrupt() prematurely stops this delivery's course, reverting agents to their prior
    active state with strains intact (the mechanism behind treatment-monitoring regimen switching)."""
    tb = tbsim.TBResistant(init_prev=ss.bernoulli(0.0))
    tx = tbsim.TxDeliveryR(product=tbsim.TxR(strains=tb.strains), rate_sym=ss.peryear(0.0))
    sim = make_sim(tb, n=1000, interventions=tx); sim.init()
    tb = sim.get_tb()
    txd = _product(sim, tbsim.TxDeliveryR)
    uids = ss.uids(np.arange(500))
    # Manually put a cohort on this delivery's course.
    txd.prior_state[uids] = int(TBS.SYMPTOMATIC)
    tb.strain_mask[uids] = 1
    tb.state[uids] = TBS.TREATMENT
    txd.ti_treatment_start[uids] = sim.ti
    txd.ti_treatment_end[uids] = sim.ti + 10
    reverted = txd.interrupt(uids)
    assert len(reverted) == 500
    assert (tb.state[uids] == TBS.SYMPTOMATIC).all()          # reverted to prior state
    assert (tb.strain_mask[uids] == 1).all()                  # strains preserved for re-treatment
    assert np.isnan(txd.ti_treatment_end[uids]).all()         # course cleared


def test_treatment_monitoring_switches_regimen():
    """Monitoring eligibility + supersedes moves agents from a first-line to a second-line course."""
    tb = tbsim.TBResistant(drugs=['INH', 'RIF'], rel_fitness={'INH': 0.95},
                           beta=ss.permonth(0.3), init_prev=ss.bernoulli(0.10),
                           init_strains=[0.5, 0.5, 0.0, 0.0])  # pan + INH-resistant
    first = tbsim.TxDeliveryR(name='first', rate_sym=ss.peryear(2.0),
                              product=tbsim.TxR(strains=tb.strains, regimen_drugs=['INH'],
                                                base_efficacy=0.8, resist_penalty={'INH': 0.1}))
    second = tbsim.TxDeliveryR(name='second', supersedes=['first'],
                               eligibility=tbsim.treatment_monitoring_eligibility('first', after_steps=1),
                               product=tbsim.TxR(strains=tb.strains, regimen_drugs=['RIF'], base_efficacy=0.8))
    sim = make_sim(tb, interventions=[first, second], stop='2012-12-31'); sim.run()
    # The monitoring→interrupt→switch chain ran and moved at least some agents onto second-line.
    assert np.sum(sim.results['second'].n_treated) > 0


# --------------------------------------------------------------------------- TPT
def test_tpt_strain_aware_sterilization():
    """Strain-aware TPT clears regimen-susceptible strains but leaves resistant strains latent."""
    tb = tbsim.TBResistant(drugs=['INH'], init_prev=ss.bernoulli(0.0))
    tpt = tbsim.TPTSimple(product=tbsim.TPTRx(strains=tb.strains, regimen_drugs=['INH']))
    sim = make_sim(tb, n=2000, interventions=tpt); sim.init()
    tb = sim.get_tb()
    prod = _product(sim, tbsim.TPTSimple).product
    pan = ss.uids(np.arange(1000))
    res = ss.uids(np.arange(1000, 2000))
    tb.state[pan] = TBS.INFECTION; tb.strain_mask[pan] = 1  # strain 0 = INH-susceptible → bit 0
    tb.state[res] = TBS.INFECTION; tb.strain_mask[res] = 2  # strain 1 = INH-resistant → bit 1
    prod._apply_sterilization(ss.uids(np.arange(2000)))
    assert (tb.state[pan] == TBS.CLEARED).all()              # susceptible strain cleared
    assert (tb.strain_mask[pan] == 0).all()
    assert (tb.state[res] == TBS.INFECTION).all()            # resistant strain persists latently
    assert (tb.strain_mask[res] == 2).all()


# --------------------------------------------------------------------------- DST
def test_dst_recovers_sens_spec():
    """DST observed calls recover the configured sensitivity/specificity in the mono-infection limit."""
    sens, spec = 0.9, 0.95
    tb = tbsim.TBResistant(init_prev=ss.bernoulli(0.0))
    dst = tbsim.DSTDelivery(product=tbsim.DST(strains=tb.strains, sens=sens, spec=spec, p_strain_obs=1.0))
    sim = make_sim(tb, n=20000, interventions=dst); sim.init()
    # ss.Sim copies its modules, so operate on the sim's initialized copies.
    tb = sim.get_tb()
    prod = _product(sim, tbsim.DSTDelivery).product
    res_uids = ss.uids(np.arange(10000))
    sus_uids = ss.uids(np.arange(10000, 20000))
    tb.strain_mask[res_uids] = 2  # mono-B (resistant)
    tb.strain_mask[sus_uids] = 1  # mono-A (susceptible)
    obs_res = (prod.administer(tb, res_uids) & 1).astype(bool)
    obs_sus = (prod.administer(tb, sus_uids) & 1).astype(bool)
    assert np.isclose(obs_res.mean(), sens, atol=0.02)         # sensitivity
    assert np.isclose(1 - obs_sus.mean(), spec, atol=0.02)     # specificity


def test_dst_router_matches_observed_profile():
    """DSTDelivery.matches builds composable eligibility from the observed n-bit profile."""
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], init_prev=ss.bernoulli(0.0))
    dst = tbsim.DSTDelivery(product=tbsim.DST(strains=tb.strains))
    sim = make_sim(tb, n=300, interventions=dst); sim.init()
    dstd = _product(sim, tbsim.DSTDelivery)
    rif_only = ss.uids(np.arange(100))
    both = ss.uids(np.arange(100, 200))
    sus = ss.uids(np.arange(200, 300))
    dstd.dst_tested[ss.uids(np.arange(300))] = True
    dstd.dst_profile[rif_only] = 0b01   # observed RIF-resistant (bit 0 = RIF)
    dstd.dst_profile[both] = 0b11       # RIF + BDQ
    dstd.dst_profile[sus] = 0b00
    any_rif = set(dstd.matches(RIF=True)(sim))
    rif_not_bdq = set(dstd.matches(RIF=True, BDQ=False)(sim))
    assert any_rif == set(rif_only) | set(both)   # all observed RIF-resistant
    assert rif_not_bdq == set(rif_only)           # RIF-resistant, BDQ-susceptible only


# --------------------------------------------------------------------------- explicit efficacy vector / adherence distribution / retreatment classifier
def test_treatment_explicit_efficacy_vector():
    """TxR(efficacy_by_strain=...) uses the given per-strain vector T_l verbatim as the cure probabilities
    (spec §Treatment), overriding the base_efficacy × penalty parameterization."""
    tb = tbsim.TBResistant(drugs=['RIF'], init_prev=ss.bernoulli(0.0))
    eff = np.array([0.9, 0.2])  # pan, RIF-resistant
    prod = tbsim.TxR(strains=tb.strains, efficacy_by_strain=eff, adherence=1.0)
    sim = make_sim(tb, n=20000, interventions=tbsim.TxDeliveryR(product=prod)); sim.init()
    tb = sim.get_tb(); prod = _product(sim, tbsim.TxDeliveryR).product
    assert np.allclose(prod.eff_by_id, eff)
    u = ss.uids(np.arange(20000))
    tb.strain_mask[ss.uids(np.arange(10000))] = 1        # pan
    tb.strain_mask[ss.uids(np.arange(10000, 20000))] = 2  # RIF-resistant
    surv = prod.roll_survivors(tb, u)
    assert np.isclose(np.mean(surv[:10000] == 0), 0.9, atol=0.02)   # pan cured at t_pan
    assert np.isclose(np.mean(surv[10000:] == 0), 0.2, atol=0.02)   # resistant cured at t_res


def test_treatment_adherence_distribution():
    """adherence as a callable makes completion a per-agent distribution applied across all the agent's
    strains (spec §Treatment): non-completers clear nothing that course."""
    tb = tbsim.TBResistant(drugs=['RIF'], init_prev=ss.bernoulli(0.0))
    adh = lambda uids: np.where(np.asarray(uids) < 5000, 1.0, 0.0)  # first half fully adherent, rest never
    prod = tbsim.TxR(strains=tb.strains, base_efficacy=1.0, adherence=adh)
    sim = make_sim(tb, n=10000, interventions=tbsim.TxDeliveryR(product=prod)); sim.init()
    tb = sim.get_tb(); prod = _product(sim, tbsim.TxDeliveryR).product
    assert callable(prod.adherence_distribution)
    u = ss.uids(np.arange(10000)); tb.strain_mask[u] = 1
    surv = prod.roll_survivors(tb, u)
    assert np.mean(surv[:5000] == 0) == 1.0   # adherent + efficacy 1 → all cured
    assert np.mean(surv[5000:] == 0) == 0.0   # non-adherent → none cured


def test_failure_vs_new_case_classification():
    """TxDeliveryR.failure_case_eligibility partitions active TB by time since last treatment initiation
    (durable tb.ti_last_treatment): recent → treatment failure/retreatment, old or never-treated → new
    case (spec §Diagnostics)."""
    tb = tbsim.TBResistant(init_prev=ss.bernoulli(0.0))
    sim = make_sim(tb, n=600); sim.init()
    tb = sim.get_tb(); ti = sim.ti
    tb.state[ss.uids(np.arange(300))] = TBS.SYMPTOMATIC  # candidate pool = active TB (uids 0-299)
    recent, old = ss.uids(np.arange(100)), ss.uids(np.arange(100, 200))
    tb.ti_last_treatment[recent] = ti          # treated now → failure/retreatment
    tb.ti_last_treatment[old] = ti - 100       # treated long ago → new case; uids 200-299 never treated (nan)
    failed = tbsim.TxDeliveryR.failure_case_eligibility(within=ss.years(2))
    newcase = tbsim.TxDeliveryR.failure_case_eligibility(within=ss.years(2), new_case=True)
    assert set(failed(sim).tolist()) == set(range(100))
    assert set(newcase(sim).tolist()) == set(range(100, 300))


# --------------------------------------------------------------------------- per-strain counter
def test_strain_counter_and_count_weighting():
    """Identical-strain re-exposure increments the per-strain count (previously blocked) and the
    transmission multinomial is weighted by count × fitness (spec updates §1)."""
    s = tbsim.Strains(['TX'], rel_fitness=None)  # neutral fitness
    tp = s.transmit_probs(np.array([0b11]), counts=np.array([[2, 1]]))[0]
    assert np.allclose(tp, [2/3, 1/3]) and s.max_fitness(np.array([0b11]))[0] == 1.0  # split by count, infectiousness count-free
    tb = tbsim.TBResistant(init_prev=ss.bernoulli(0.0))
    sim = make_sim(tb, n=200); sim.init(); tb = sim.get_tb()
    tgt, src = ss.uids(np.arange(50)), ss.uids(np.arange(50, 100))
    for u in (tgt, src):
        tb.strain_mask[u] = 1; tb.strain_counts[0][u] = 1; tb.state[u] = TBS.INFECTION
    tb.set_prognoses(tgt, sources=src)
    assert (tb.strain_mask[tgt] == 1).all()                    # carried set unchanged
    assert (np.asarray(tb.strain_counts[0][tgt]) == 2).all()   # count incremented 1 → 2


# --------------------------------------------------------------------------- ODE reference self-checks (model-tests.md §8)
def test_ode_conservation():
    """Summed over all 22 compartments the population is conserved at N."""
    ode = tc.TwoStrainODE(); df = ode.run()
    dyn = df[[c for c in tc.two_strain_ode.STATES if c != 'DTH']].sum(axis=1)
    assert np.allclose(dyn, ode.pars.N, rtol=1e-6)


def test_ode_single_strain_reduction():
    """Seed only A with q=q_p=0, σ=0, treatment off → strain-summed ODE equals the single-strain TB_ODE."""
    ode = tc.TwoStrainODE(fit_b=1.0, q_treat=0, q_prog=0, rr_reinfection_inf=0, rr_reinfection_non=0,
                          r_treat_asym=0, r_treat_sym=0)
    ode.run(SY_A=1e3)  # no B seed
    coll = ode.collapse().iloc[-1]
    single = tc.TB_ODE(beta=16.45, theta=0, phi=0).run()  # existing single-strain reference (objdict of arrays)
    for state in ['INFECTION', 'NON_INFECTIOUS', 'ASYMPTOMATIC', 'SYMPTOMATIC', 'CLEARED']:
        assert np.isclose(coll[state], single[state][-1], rtol=1e-3, atol=1.0), state


def test_ode_strain_symmetry():
    """r_a=r_b, e_a=e_b, q=q_p=0, symmetric seed → A- and B-compartments identical for all t."""
    ode = tc.TwoStrainODE(fit_a=1.0, fit_b=1.0, eff_a=0.5, eff_b=0.5, q_treat=0, q_prog=0)
    df = ode.run(SY_A=5e2, SY_B=5e2)
    for a, b in [('L_A', 'L_B'), ('N_A', 'N_B'), ('AS_A', 'AS_B'), ('SY_A', 'SY_B')]:
        assert np.allclose(df[a], df[b], atol=1e-6)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))

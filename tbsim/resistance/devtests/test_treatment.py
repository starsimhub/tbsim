"""
Strain-resolved treatment: the outcome operator π(m→s) (model-tests.md §6), acquisition-on-failure
with the per-state RR, and treatment-driven selection for resistance (ABM vs ODE).
"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS
from tbsim.resistance.devtests import ode_utils as ou

BETA_ODE = 45.0
YEARS = 80
NSEEDS = 2


def _init_product(ea, eb, q):
    """Build an initialized TxR (n=1: strain 0 = A susceptible, strain 1 = B resistant)."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 0.6}, init_prev=ss.bernoulli(0.0))
    prod = tbsim.TxR(strains=tb.strains, base_efficacy=ea, resist_penalty={'TX': eb / ea},
                     adherence=1.0, q_acq={'TX': q})
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))
    sim = tbsim.Sim(n_agents=30000, networks=net, diseases=tb, demographics=[],
                 interventions=tbsim.TxDeliveryR(product=prod), dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date('2001-12-31'), rand_seed=0, verbose=0)
    sim.init()
    tb = sim.get_tb()
    prod = next(iv for iv in sim.interventions.values() if isinstance(iv, tbsim.TxDeliveryR)).product
    return tb, prod


def _set_mask_counts(tb, uids, mask):
    """Set an agent cohort to carry ``mask``, one copy per carried strain (invariant count>0 ⟺ bit set)."""
    tb.strain_mask[uids] = mask
    for j in range(tb.strains.m):
        tb.strain_counts[j][uids] = 1 if (mask >> j) & 1 else 0
    return


def _outcome_dist(tb, prod, mask, n=30000):
    """Apply the full operator (roll_survivors → acquire_counts, at SYMPTOMATIC so state-RR=1) to a cohort."""
    uids = ss.uids(np.arange(n))
    _set_mask_counts(tb, uids, mask)
    surv = prod.roll_survivors(tb, uids)                                   # surviving strain mask (bits ⊆ mask)
    counts_surv = tb._counts(uids) * tb.strains.carried(surv)              # cured strains → 0
    _, mask_out, _ = prod.acquire_counts(tb, uids, counts_surv, states=np.full(n, int(TBS.SYMPTOMATIC)))
    return {v: float(np.mean(mask_out == v)) for v in (0, 1, 2, 3)}


# --------------------------------------------------------------------------- §6: outcome operator
def test_treatment_operator_matches_ode_pi_table():
    """The ABM's per-strain cure + replacement-acquisition reproduces the ODE π(m→s) table exactly
    (two_strain_ode.py `pi_*`), including the AB→B collapse when a failed AB acquires resistance."""
    ea, eb, q = 0.75, 0.3, 0.1
    tb, prod = _init_product(ea, eb, q)

    # A cohort (mask 1): cure ea; else A→A w.p. (1-ea)(1-q), A→B w.p. (1-ea)q.
    dA = _outcome_dist(tb, prod, 1)
    assert np.isclose(dA[0], ea, atol=0.02)
    assert np.isclose(dA[1], (1 - ea) * (1 - q), atol=0.02)   # pi_A_to_A
    assert np.isclose(dA[2], (1 - ea) * q, atol=0.02)         # pi_A_to_B

    # B cohort (mask 2): cure eb; else stays B (already resistant → no acquisition).
    dB = _outcome_dist(tb, prod, 2)
    assert np.isclose(dB[0], eb, atol=0.02)
    assert np.isclose(dB[2], (1 - eb), atol=0.02)             # pi_B_to_B

    # AB cohort (mask 3): matches the ODE's pi_AB_to_{A,B,AB} + cure.
    dAB = _outcome_dist(tb, prod, 3)
    assert np.isclose(dAB[0], ea * eb, atol=0.02)                                  # both cured
    assert np.isclose(dAB[1], (1 - ea) * eb * (1 - q), atol=0.02)                  # pi_AB_to_A
    pi_ab_to_b = ea * (1 - eb) + (1 - ea) * eb * q + (1 - ea) * (1 - eb) * q
    assert np.isclose(dAB[2], pi_ab_to_b, atol=0.02)                              # pi_AB_to_B (incl. AB→B collapse)
    assert np.isclose(dAB[3], (1 - ea) * (1 - eb) * (1 - q), atol=0.02)            # pi_AB_to_AB


# --------------------------------------------------------------------------- acquisition state-RR
def test_acquisition_only_in_active_states():
    """Spec: the acquisition-on-failure RR is 0 outside ASYMPTOMATIC/SYMPTOMATIC. Treating (by an
    eligibility override) a NON_INFECTIOUS cohort must yield no acquired resistance; SYMPTOMATIC must."""
    tb, prod = _init_product(ea=0.5, eb=0.5, q=1.0)  # q=1 → deterministic acquisition where RR>0
    uids = ss.uids(np.arange(20000))
    _set_mask_counts(tb, uids, 1)  # all mono-A (susceptible; a failure can acquire)
    surv = prod.roll_survivors(tb, uids)          # ~half survive (eff 0.5)
    failed = surv != 0
    counts_surv = tb._counts(uids) * tb.strains.carried(surv)
    # NON_INFECTIOUS → RR 0 → no acquisition (no A→B), so no surviving strain becomes B (mask 2).
    _, mask_ni, _ = prod.acquire_counts(tb, uids, counts_surv.copy(), states=np.full(len(uids), int(TBS.NON_INFECTIOUS)))
    assert np.count_nonzero(mask_ni == 2) == 0
    # SYMPTOMATIC → RR 1 → every surviving A acquires B (mask 1 → 2).
    _, mask_sy, _ = prod.acquire_counts(tb, uids, counts_surv.copy(), states=np.full(len(uids), int(TBS.SYMPTOMATIC)))
    assert np.count_nonzero(mask_sy == 2) == np.count_nonzero(failed)
    assert np.count_nonzero(mask_sy == 1) == 0


# --------------------------------------------------------------------------- ABM ↔ ODE: selection
def test_treatment_selects_for_resistance_matches_ode():
    """Treatment that cures A well but B poorly drives the resistant fraction up in both models."""
    be = ou.calibrate_beta(BETA_ODE)
    fit_b, initB = 0.575, 0.1
    treat = dict(eff_a=0.75, eff_b=0.25, q_treat=0.03, r_treat_sym=1.0, r_treat_asym=0.05)
    seeds = dict(L_A=(1 - initB) * 0.05 * 1e5, L_B=initB * 0.05 * 1e5)
    o = ou.run_ode(BETA_ODE, years=YEARS, fit_b=fit_b, seeds=seeds, treat=treat,
                   rr_reinfection_inf=1.0, rr_reinfection_non=1.0)
    a = [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': fit_b}, init_strains=[1 - initB, initB],
                    treat=treat, rr_reinfection_inf=1.0, rr_reinfection_non=1.0) for s in range(NSEEDS)]
    assert ou.late_mean(o, 'frac_resist') > initB              # ODE: treatment selects resistance up
    assert ou.final_mean(a, 'frac_resist') > initB             # ABM: same
    assert ou.final_mean(a, 'frac_resist') > 0.5               # strongly selected under this pressure


# --------------------------------------------------------------------------- §5: counter coupling / reset
def test_treatment_efficacy_independent_of_count():
    """Spec §5 coupling: treatment effectiveness does not depend on strain count (identical strains are
    cured together). Two cohorts with the same strain but different counts are cured at the same rate."""
    tb, prod = _init_product(ea=0.6, eb=0.6, q=0.0)
    a, b = ss.uids(np.arange(15000)), ss.uids(np.arange(15000, 30000))
    tb.strain_mask[a] = 1; tb.strain_counts[0][a] = 1
    tb.strain_mask[b] = 1; tb.strain_counts[0][b] = 3
    cure_a = float(np.mean(prod.roll_survivors(tb, a) == 0))
    cure_b = float(np.mean(prod.roll_survivors(tb, b) == 0))
    assert np.isclose(cure_a, cure_b, atol=0.03) and np.isclose(cure_a, 0.6, atol=0.03)


def test_successful_cure_resets_count_survivor_unchanged():
    """Spec §5: a successful cure of a strain resets its count to 0 (treatment clears all copies),
    while a surviving (uncured) strain keeps its count. Set an agent carrying strain 0 (pan) ×2 and
    strain 1 (resistant) ×1; a regimen that cures pan but not the resistant strain zeroes count[0] and
    leaves count[1]."""
    # base_efficacy=1 on the regimen drug, resist_penalty 0 → strain 0 always cured, strain 1 never.
    tb = tbsim.TBResistant(rel_fitness={'TX': 1.0}, init_prev=ss.bernoulli(0.0))
    prod = tbsim.TxR(strains=tb.strains, base_efficacy=1.0, resist_penalty={'TX': 0.0}, adherence=1.0)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))
    sim = tbsim.Sim(n_agents=2000, networks=net, diseases=tb, demographics=[], interventions=tbsim.TxDeliveryR(product=prod),
                 dt=ss.days(30), start=ss.date('2000-01-01'), stop=ss.date('2001-12-31'), rand_seed=0, verbose=0)
    sim.init()
    tb = sim.get_tb()
    tx = next(iv for iv in sim.interventions.values() if isinstance(iv, tbsim.TxDeliveryR))
    prod = tx.product
    u = ss.uids(np.arange(2000))
    tb.state[u] = TBS.SYMPTOMATIC
    tb.strain_mask[u] = (1 << 0) | (1 << 1)     # carries pan + resistant
    tb.strain_counts[0][u] = 2                  # pan ×2
    tb.strain_counts[1][u] = 1                  # resistant ×1
    # Put them on a course and force it to complete this step, then resolve.
    tx.prior_state[u] = tb.state[u]
    tx.pending_surv[u] = prod.roll_survivors(tb, u)
    tb.state[u] = TBS.TREATMENT
    tx.ti_treatment_end[u] = -1                 # already ended → resolve now
    tx._resolve()
    assert (tb.strain_mask[u] == (1 << 1)).all()          # pan cured, resistant survives
    assert (np.asarray(tb.strain_counts[0][u]) == 0).all()  # cured strain's count reset to 0
    assert (np.asarray(tb.strain_counts[1][u]) == 1).all()  # surviving strain's count unchanged


# --------------------------------------------------------------------------- L3: latent-treatment divergence
def test_latent_treatment_divergence():
    """L3 Option B: by default (treat_latent=False) latent agents selected for treatment are cleared
    immediately (→CLEARED, no course), so they never enter TREATMENT and never acquire resistance —
    matching base tbsim. With treat_latent=True the same agents run a course that here fails and
    acquires resistance."""
    def run(treat_latent):
        tb = tbsim.TBResistant(init_prev=ss.bernoulli(0.3), beta=ss.permonth(0.0),  # seed latent, no transmission
                               init_strains=[1.0, 0.0])
        tx = tbsim.TxDeliveryR(name='tx', treat_latent=treat_latent, dur_treatment=ss.months(3),
                               eligibility=lambda sim: sim.get_tb().latent.uids,
                               product=tbsim.TxR(strains=tb.strains, base_efficacy=0.0, adherence=1.0,
                                                 q_acq={'TX': 1.0}, acq_state_rr={int(TBS.INFECTION): 1.0}))
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=1), dur=0))
        sim = tbsim.Sim(n_agents=3000, networks=net, diseases=tb, demographics=[], interventions=tx, dt=ss.days(30),
                     start=ss.date('2000-01-01'), stop=ss.date('2003-12-31'), rand_seed=0, verbose=0)
        sim.run()
        return sim
    default, coursed = run(False), run(True)
    r_def, r_crs = default.results['tx'], coursed.results['tx']
    # Default: latent agents cleared immediately — no course, no acquisition, and they reach CLEARED.
    assert int(np.sum(r_def.n_treated)) == 0
    assert int(np.sum(r_def.n_acquired)) == 0
    assert int(default.results.tb['n_CLEARED'][-1]) > 0
    # treat_latent=True: latent agents run a course (n_treated>0) that fails and acquires resistance.
    assert int(np.sum(r_crs.n_treated)) > 0
    assert int(np.sum(r_crs.n_acquired)) > 0


if __name__ == '__main__':
    import sys, pytest
    sys.exit(pytest.main([__file__, '-v']))

"""
Natural history: superinfection eligibility, progression bottleneck, and the superinfection
rate modifiers — the model-tests.md §10 questions Q1, Q2a, Q2b, Q3a, Q3b.

Each question is checked *directionally against the ODE*: does the ABM shift the observable the
same way the deterministic reference does when the knob moves off its null? Directional checks are
robust to the ABM's stochasticity and to the network-vs-mass-action β mismatch.
"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS
from tbsim.resistance.devtests import ode_utils as ou

BETA_ODE = 45.0
YEARS = 70
NSEEDS = 2
SEEDS_5050 = dict(L_A=0.5 * 0.05 * 1e5, L_B=0.5 * 0.05 * 1e5)  # symmetric latent seed for the ODE


def _abm(be, psi=1.0, pmulti=1.0, fit_b=1.0, select=0, sigmaN=1.0, sigmaA=0.0, sigmaY=0.0):
    """Two-strain ABM (σ_L=1), averaged observables over NSEEDS."""
    return [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': fit_b}, init_strains=[0.5, 0.5],
                       rr_reinfection_inf=1.0, rr_reinfection_non=sigmaN,
                       rr_reinfection_asy=sigmaA, rr_reinfection_sym=sigmaY,
                       rr_prog_super=psi, p_multi=pmulti, prog_select_fitness=select)
            for s in range(NSEEDS)]


def _ode(psi=1.0, pmulti=1.0, fit_b=1.0, select=0, sigmaN=1.0, sigmaA=0.0, sigmaY=0.0):
    return ou.run_ode(BETA_ODE, years=YEARS, fit_b=fit_b, seeds=SEEDS_5050,
                      rr_reinfection_inf=1.0, rr_reinfection_non=sigmaN,
                      rr_reinfection_asy=sigmaA, rr_reinfection_sym=sigmaY,
                      rr_prog_super=psi, p_multi=pmulti, prog_select_fitness=select)


# --------------------------------------------------------------------------- direct state invariants
def test_natural_clearance_removes_all_strains():
    """Spec §Clearance: natural clearance/resolution is immune-mediated → clears *all* strains
    (contrast with treatment). No CLEARED agent may carry a strain."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 1.0},
                           beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.15),
                           init_strains=[0.5, 0.5], rr_reinfection_inf=1.0, rr_reinfection_non=1.0)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=10), dur=0))
    sim = tbsim.Sim(n_agents=3000, networks=net, diseases=tb, demographics=[], dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date('2020-12-31'), rand_seed=0, verbose=0)
    sim.run()
    tb = sim.get_tb()
    cleared = ss.uids(tb.state == TBS.CLEARED)
    assert len(cleared) > 0
    assert int(tb.strain_mask[cleared].sum()) == 0  # every cleared agent carries no strain


def test_identical_strain_superinfection_allowed_and_counted():
    """Spec §1 (reversal): superinfection with an identical strain is now *allowed* — it leaves the
    carried set unchanged but increments that strain's per-strain count, and the events are tallied in
    ``new_identical_superinf``. Such events occur once σ>0 and a strain is common."""
    on = _abm(BETA_ODE and ou.calibrate_beta(BETA_ODE))
    total = sum(int(np.sum(r.sim.results.tb['new_identical_superinf'])) for r in on)
    assert total > 0
    # And the counter really recorded multiplicity >1 for some agent (a superinfected identical strain).
    max_count = max(int(np.max(np.stack([c.values for c in r.sim.diseases.tb.strain_counts], axis=1)))
                    for r in on)
    assert max_count >= 2


# --------------------------------------------------------------------------- Q1: ψ (rr_prog_super)
def test_q1_superinfection_faster_progression():
    """Q1: ψ>1 speeds AB progression, raising the superinfected fraction of active TB — ODE and ABM
    agree in direction."""
    be = ou.calibrate_beta(BETA_ODE)
    o_null, o_psi = _ode(psi=1.0), _ode(psi=3.0)
    assert ou.late_mean(o_psi, 'frac_super') > ou.late_mean(o_null, 'frac_super')
    a_null, a_psi = _abm(be, psi=1.0), _abm(be, psi=3.0)
    assert ou.final_mean(a_psi, 'frac_super') > ou.final_mean(a_null, 'frac_super')


# --------------------------------------------------------------------------- Q2a: p_multi bottleneck
def test_q2a_bottleneck_reduces_superinfection():
    """Q2a: lowering p_multi funnels multi-strain agents to a single strain at progression, so the
    superinfected fraction of active TB falls — ODE and ABM agree."""
    be = ou.calibrate_beta(BETA_ODE)
    o_full, o_bott = _ode(pmulti=1.0), _ode(pmulti=0.0)
    assert ou.late_mean(o_bott, 'frac_super') < ou.late_mean(o_full, 'frac_super')
    a_full, a_bott = _abm(be, pmulti=1.0), _abm(be, pmulti=0.0)
    assert ou.final_mean(a_bott, 'frac_super') < ou.final_mean(a_full, 'frac_super')


# --------------------------------------------------------------------------- Q2b: selection rule
def test_q2b_fitness_selection_lowers_resistance():
    """Q2b: when the bottleneck picks one strain, fitness-weighted selection favors the fitter
    (susceptible) strain, lowering the resistant fraction vs. count-only ("random") selection — ODE
    and ABM agree in direction.

    NB: with the per-strain counter (spec update §2) the bottleneck survivor is count-weighted even in
    "random" mode, and count-weighted transmission + bottleneck amplify competitive exclusion, so a
    less-fit strain is driven fully extinct by end-of-run in *both* modes (the ABM endpoint ties at 0,
    unlike the count-less ODE). The fitness effect is therefore compared on the **time-averaged**
    resistant fraction — fitness selection drives it down faster — which stays above the extinction floor."""
    be = ou.calibrate_beta(BETA_ODE)
    o_rand = _ode(pmulti=0.2, fit_b=0.7, select=0)
    o_fit = _ode(pmulti=0.2, fit_b=0.7, select=1)
    assert ou.late_mean(o_fit, 'frac_resist') < ou.late_mean(o_rand, 'frac_resist')
    a_rand = _abm(be, pmulti=0.2, fit_b=0.7, select=0)
    a_fit = _abm(be, pmulti=0.2, fit_b=0.7, select=1)
    timeavg = lambda runs: float(np.mean([np.mean(r.frac_resist) for r in runs]))
    assert timeavg(a_fit) < timeavg(a_rand)


# --------------------------------------------------------------------------- Q3a: σ_N eligibility
def test_q3a_noninfectious_eligibility_raises_superinfection():
    """Q3a: allowing superinfection in NON_INFECTIOUS (σ_N=1) rather than INFECTED-only (σ_N=0)
    raises the superinfected fraction — ODE and ABM agree."""
    be = ou.calibrate_beta(BETA_ODE)
    o_inf_only, o_both = _ode(sigmaN=0.0), _ode(sigmaN=1.0)
    assert ou.late_mean(o_both, 'frac_super') > ou.late_mean(o_inf_only, 'frac_super')
    a_inf_only, a_both = _abm(be, sigmaN=0.0), _abm(be, sigmaN=1.0)
    assert ou.final_mean(a_both, 'frac_super') > ou.final_mean(a_inf_only, 'frac_super')


# --------------------------------------------------------------------------- Q3b: active-disease superinfection
def test_q3b_active_superinfection_adds_coinfection():
    """Q3b: opening superinfection during active disease (σ_A, σ_Y > 0) adds coinfection beyond the
    latent-only default — ODE and ABM agree that frac_super rises."""
    be = ou.calibrate_beta(BETA_ODE)
    o_off, o_on = _ode(sigmaA=0.0, sigmaY=0.0), _ode(sigmaA=1.0, sigmaY=1.0)
    assert ou.late_mean(o_on, 'frac_super') > ou.late_mean(o_off, 'frac_super')
    a_off, a_on = _abm(be, sigmaA=0.0, sigmaY=0.0), _abm(be, sigmaA=1.0, sigmaY=1.0)
    assert ou.final_mean(a_on, 'frac_super') > ou.final_mean(a_off, 'frac_super')


if __name__ == '__main__':
    import sys, pytest
    sys.exit(pytest.main([__file__, '-v']))

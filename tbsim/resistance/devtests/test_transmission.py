"""
Transmission: fitness bottleneck, ∝-fitness strain selection, and ABM↔ODE agreement.

Covers spec §"Transmission" (the worked 56/44 example), the reduction check (model-tests.md
§8.2), competitive exclusion under a fitness cost, and Q5 (transmission bottleneck vs
independent — the ABM implements only the bottleneck; independence is ODE-only).
"""

import numpy as np
import pytest
import tbsim
from tbsim.resistance.devtests import ode_utils as ou

BETA_ODE = 45.0
YEARS = 80
NSEEDS = 3


# --------------------------------------------------------------------------- operator: fitness split
def test_transmission_split_matches_spec_table():
    """Spec Table (agent C): a source carrying {RIF} and {RIF,BDQ} transmits at the fittest strain's
    rate (max fitness) and splits which strain is passed ∝ fitness → 56% / 44% at r_BDQ=0.8."""
    s = tbsim.Strains(['RIF', 'BDQ'], rel_fitness={'RIF': 0.5, 'BDQ': 0.8})
    mask = np.array([(1 << 1) | (1 << 3)])         # carries strain 1={RIF} and strain 3={RIF,BDQ}
    # Relative effective contact rate = max fitness over carried strains.
    assert np.isclose(s.max_fitness(mask)[0], 0.5)  # r_RIF, not r_RIF*r_BDQ
    tp = s.transmit_probs(mask)[0]
    assert np.isclose(tp[1], 1 / 1.8, atol=1e-9)    # 56% — strain {RIF}
    assert np.isclose(tp[3], 0.8 / 1.8, atol=1e-9)  # 44% — strain {RIF,BDQ}
    assert np.isclose(tp.sum(), 1.0)
    # Per-strain transmission rate = split × max-fitness × β → 0.28β / 0.22β (spec rounds 0.2778/0.2222).
    assert np.isclose(tp[1] * s.max_fitness(mask)[0], 0.28, atol=0.01)
    assert np.isclose(tp[3] * s.max_fitness(mask)[0], 0.22, atol=0.01)


def test_superinfection_does_not_reduce_infectiousness():
    """Spec: superinfection does not lower a source's overall transmission — rel_trans stays at the
    fittest strain, identical to a mono-infection with that strain."""
    s = tbsim.Strains(['RIF'], rel_fitness={'RIF': 0.6})
    mono_pan = np.array([1 << 0])           # {pan} only
    mono_rif = np.array([1 << 1])           # {RIF} only
    both = np.array([(1 << 0) | (1 << 1)])  # {pan, RIF}
    assert np.isclose(s.max_fitness(mono_pan)[0], 1.0)
    assert np.isclose(s.max_fitness(mono_rif)[0], 0.6)
    assert np.isclose(s.max_fitness(both)[0], 1.0)  # = the fitter (pan) mono-infection, not reduced


# --------------------------------------------------------------------------- ABM ↔ ODE: reduction
def test_single_strain_reduction_matches_ode_compartments():
    """model-tests.md §8.2: one strain, σ=0, no treatment/acquisition → the ABM's endemic
    compartment fractions match the (strain-summed) ODE within stochastic tolerance."""
    be = ou.calibrate_beta(BETA_ODE)
    o = ou.run_ode(BETA_ODE, years=YEARS, rr_reinfection_inf=0, rr_reinfection_non=0)
    a = [ou.run_abm(be, years=YEARS, seed=s, rr_reinfection_inf=0.0, rr_reinfection_non=0.0)
         for s in range(NSEEDS)]
    for state in ou.STATES:
        ode_f = o.fracs[state][-1]
        abm_f = np.mean([r.fracs[state][-1] for r in a])
        # Coarse agreement: network-ABM vs mass-action ODE differ structurally in the latent pool;
        # β is calibrated on active prevalence, so ~5pp is the fair bar for compartment fractions.
        assert abs(abm_f - ode_f) < 0.05, f'{state}: ABM {abm_f:.3f} vs ODE {ode_f:.3f}'
    assert ou.final_mean(a, 'frac_resist') == 0.0  # never any resistance created


# --------------------------------------------------------------------------- ABM ↔ ODE: competitive exclusion
def test_competitive_exclusion_matches_ode():
    """A less-fit resistant strain with no treatment is out-competed in *both* models, and the ABM's
    residual resistant fraction tracks the ODE's."""
    be = ou.calibrate_beta(BETA_ODE)
    fit_b, initB = 0.7, 0.25
    seeds = dict(L_A=(1 - initB) * 0.05 * 1e5, L_B=initB * 0.05 * 1e5)
    o = ou.run_ode(BETA_ODE, years=YEARS, fit_b=fit_b, seeds=seeds,
                   rr_reinfection_inf=0, rr_reinfection_non=0)
    a = [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': fit_b}, init_strains=[1 - initB, initB],
                    rr_reinfection_inf=0.0, rr_reinfection_non=0.0) for s in range(NSEEDS)]
    assert o.frac_resist[-1] < initB                       # ODE: resistance declines
    assert ou.final_mean(a, 'frac_resist') < initB         # ABM: resistance declines
    assert abs(ou.final_mean(a, 'frac_resist') - o.frac_resist[-1]) < 0.10  # and they agree


# --------------------------------------------------------------------------- Q5: bottleneck only
def test_transmission_bottleneck_only_no_independent_mode():
    """Q5: the ABM implements only the transmission *bottleneck* (rel_trans = max fitness, then split
    ∝ fitness). Independent per-strain transmission is ODE-only (`transmission_independent`) and is
    deliberately not exposed on the ABM — it would require per-strain FOI channels."""
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], rel_fitness={'RIF': 0.5, 'BDQ': 0.8})
    assert 'transmission_independent' not in tb.pars  # no independent-mode knob on the ABM
    # Confirm bottleneck semantics are what the ABM uses.
    mask = np.array([(1 << 1) | (1 << 3)])
    assert np.isclose(tb.strains.max_fitness(mask)[0], 0.5)


if __name__ == '__main__':
    import sys, pytest
    sys.exit(pytest.main([__file__, '-v']))

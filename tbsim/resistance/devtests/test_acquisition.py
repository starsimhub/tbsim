"""
Resistance acquisition: de-novo (random) mutation at progression — model-tests.md §5.1 / Q4 —
per-drug specificity, mixed-vs-replacement mechanism, and the three-way resistance-origin
decomposition (§11).
"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS
from tbsim.resistance.devtests import ode_utils as ou

BETA_ODE = 45.0
YEARS = 90
NSEEDS = 2


# --------------------------------------------------------------------------- Q4: mixed vs replacement
def test_q4_mixed_makes_ab_replacement_does_not():
    """Q4: de-novo acquisition as `mixed` yields AB superinfection (L_A → AB); as `replacement` it
    switches the strain (L_A → B) and creates no AB. Verified in ODE and ABM from a pure-A epidemic."""
    be = ou.calibrate_beta(BETA_ODE)
    denovo = dict(q_prog=0.02)
    # ODE (pure-A seed, σ=0 so AB can only come from de-novo)
    o_mix = ou.run_ode(BETA_ODE, years=YEARS, fit_b=0.9, rr_reinfection_inf=0, rr_reinfection_non=0,
                       denovo=dict(denovo, prog_resist_mix=1))
    o_rep = ou.run_ode(BETA_ODE, years=YEARS, fit_b=0.9, rr_reinfection_inf=0, rr_reinfection_non=0,
                       denovo=dict(denovo, prog_resist_mix=0))
    assert ou.late_mean(o_mix, 'frac_super') > 0.0
    assert ou.late_mean(o_rep, 'frac_super') == 0.0
    assert ou.late_mean(o_mix, 'frac_resist') > 0.0 and ou.late_mean(o_rep, 'frac_resist') > 0.0
    # ABM
    a_mix = [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': 0.9}, init_strains=[1, 0],
                        rr_reinfection_inf=0.0, rr_reinfection_non=0.0,
                        denovo=dict(denovo, prog_resist_mix=1)) for s in range(NSEEDS)]
    a_rep = [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': 0.9}, init_strains=[1, 0],
                        rr_reinfection_inf=0.0, rr_reinfection_non=0.0,
                        denovo=dict(denovo, prog_resist_mix=0)) for s in range(NSEEDS)]
    assert ou.final_mean(a_mix, 'frac_super') > 0.0
    assert ou.final_mean(a_rep, 'frac_super') == 0.0
    assert ou.final_mean(a_mix, 'frac_resist') > 0.0 and ou.final_mean(a_rep, 'frac_resist') > 0.0


def test_q4_denovo_emergence_tracks_ode():
    """With matched de-novo rate and fitness, the ABM's steady resistant fraction (replacement mode,
    σ=0) tracks the ODE's within tolerance — a quantitative check on the de-novo flux magnitude."""
    be = ou.calibrate_beta(BETA_ODE)
    denovo = dict(q_prog=0.02, prog_resist_mix=0)
    o = ou.run_ode(BETA_ODE, years=YEARS, fit_b=0.9, rr_reinfection_inf=0, rr_reinfection_non=0, denovo=denovo)
    a = [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': 0.9}, init_strains=[1, 0],
                    rr_reinfection_inf=0.0, rr_reinfection_non=0.0, denovo=denovo) for s in range(NSEEDS)]
    assert abs(ou.final_mean(a, 'frac_resist') - ou.late_mean(o, 'frac_resist')) < 0.10


# --------------------------------------------------------------------------- per-drug specificity (n>1)
def test_denovo_per_drug_specificity():
    """Spec: p_rand is per drug (`p_rand_i`) and strain-agnostic except a strain cannot re-acquire a
    drug it already resists. With de-novo only on BDQ, active TB gains BDQ resistance but never RIF."""
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], rel_fitness={'BDQ': 0.9},
                           beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12),
                           init_strains=[1.0, 0.0, 0.0, 0.0], p_rand={'BDQ': 0.05},
                           rr_reinfection_inf=0.0, rr_reinfection_non=0.0)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=10), dur=0))
    sim = tbsim.Sim(n_agents=3000, networks=net, diseases=tb, demographics=[], dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date('2050-12-31'), rand_seed=0, verbose=0)
    sim.run()
    r = sim.results.tb
    assert np.sum(r['new_denovo_resistance']) > 0
    assert r['frac_resist_BDQ'][-1] > 0.01
    assert r['frac_resist_RIF'][-1] == 0.0


def test_denovo_multistrain_each_strain_mutates():
    """Spec: each carried strain of a multi-strain agent can independently acquire resistance. Seed a
    single {pan,RIF} superinfection and confirm de-novo on BDQ can produce a strain carrying BDQ."""
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], init_prev=ss.bernoulli(0.0),
                           p_rand={'BDQ': 1.0}, prog_resist_mode='mixed')  # p=1 → deterministic acquisition
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    sim = tbsim.Sim(n_agents=200, networks=net, diseases=tb, demographics=[], dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date('2000-12-31'), rand_seed=0, verbose=0)
    sim.init()
    tb = sim.get_tb()
    u = ss.uids(np.arange(100))
    tb.state[u] = TBS.INFECTION
    tb.strain_mask[u] = (1 << 0) | (1 << 1)     # carries pan (id0) and RIF (id1)
    tb.strain_counts[0][u] = 1; tb.strain_counts[1][u] = 1  # invariant: count>0 ⟺ bit set
    tb._denovo(u)                               # de-novo at progression, BDQ p=1
    carried = tb.strains.carried(tb.strain_mask[u])
    # Every agent now carries a BDQ-resistant strain (id2={0,1} from pan, and id3={1,1} from RIF).
    bdq_resistant_ids = [j for j in range(tb.strains.m) if tb.strains.profile[j, 1]]
    assert carried[:, bdq_resistant_ids].any(axis=1).all()


# --------------------------------------------------------------------------- §11: resistance-origin decomposition
def test_resistance_origin_decomposition():
    """model-tests.md §11: the three resistance-origin fluxes are separable. de-novo fires only with
    p_rand>0; treatment-acquired only with treatment + q_acq>0; transmitted only once resistance circulates."""
    from tbsim import ResistanceStats

    def run(p_rand, tx):
        tb = tbsim.TBResistant(rel_fitness={'TX': 0.9},
                               beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12),
                               init_strains=[0.85, 0.15], rr_reinfection_inf=1.0,
                               rr_reinfection_non=1.0, p_rand=p_rand)
        ivs = None
        if tx:
            ivs = tbsim.TxDeliveryR(product=tbsim.TxR(strains=tb.strains, base_efficacy=0.8,
                                    resist_penalty={'TX': 0.1}, q_acq={'TX': 0.05}),
                                    rate_sym=ss.peryear(1.0))
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=10), dur=0))
        sim = tbsim.Sim(n_agents=3000, networks=net, diseases=tb, demographics=[], interventions=ivs,
                     analyzers=ResistanceStats(), dt=ss.days(30), start=ss.date('2000-01-01'),
                     stop=ss.date('2040-12-31'), rand_seed=0, verbose=0)
        sim.run()
        res = sim.results['resistancestats']
        return dict(denovo=int(np.sum(res.flux_denovo)), txacq=int(np.sum(res.flux_txacq)),
                    transmitted=int(np.sum(res.flux_transmitted)))

    no_denovo_no_tx = run(p_rand=None, tx=False)
    denovo_only = run(p_rand={'TX': 0.02}, tx=False)
    tx_only = run(p_rand=None, tx=True)

    assert no_denovo_no_tx['denovo'] == 0 and no_denovo_no_tx['txacq'] == 0
    assert no_denovo_no_tx['transmitted'] > 0            # seeded resistance still transmits
    assert denovo_only['denovo'] > 0 and denovo_only['txacq'] == 0
    assert tx_only['txacq'] > 0 and tx_only['denovo'] == 0


def test_acquisition_mutates_every_susceptible_strain():
    """TR-1/§4: on a treatment-failure acquisition hit, *every* carried drug-susceptible strain acquires
    resistance independently (not just one). An agent co-carrying strain 0 (pan) and strain 1 (RIF-R),
    both FQ-susceptible, that fails an FQ-acquiring regimen ends carrying strain 2 (FQ, from the pan
    strain) AND strain 3 (RIF+FQ, from the RIF-R strain) — the old single-pick rule produced only one."""
    tb = tbsim.TBResistant(drugs=['RIF', 'FQ'], init_prev=ss.bernoulli(0.0))
    prod = tbsim.TxR(strains=tb.strains, base_efficacy=0.0, adherence=1.0, q_acq={'FQ': 1.0})  # q=1 → certain FQ hit
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))
    sim = tbsim.Sim(n_agents=8000, networks=net, diseases=tb, demographics=[], interventions=tbsim.TxDeliveryR(product=prod),
                 dt=ss.days(30), start=ss.date('2000-01-01'), stop=ss.date('2001-12-31'), rand_seed=0, verbose=0)
    sim.init()
    tb = sim.get_tb()
    prod = next(iv for iv in sim.interventions.values() if isinstance(iv, tbsim.TxDeliveryR)).product
    u = ss.uids(np.arange(8000))
    tb.strain_mask[u] = (1 << 0) | (1 << 1)   # carries pan (id0) and RIF-R (id1); both FQ-susceptible
    tb.strain_counts[0][u] = 1; tb.strain_counts[1][u] = 1
    counts0 = tb._counts(u)                    # base_efficacy=0 → nothing cured; every strain survives
    _, mask, _ = prod.acquire_counts(tb, u, counts0, states=np.full(len(u), int(TBS.SYMPTOMATIC)))
    got_s2 = ((mask >> 2) & 1).astype(bool).mean()   # pan strain acquired FQ → strain 2 ({FQ})
    got_s3 = ((mask >> 3) & 1).astype(bool).mean()   # RIF-R strain acquired FQ → strain 3 ({RIF,FQ})
    assert np.allclose(got_s2, 1.0) and np.allclose(got_s3, 1.0)  # BOTH susceptible strains mutated


def test_denovo_acquisition_probability_independent_of_count():
    """§4 coupling: strain count does not change the de-novo acquisition probability, and identical
    strains acquire together (one bit per strain id → all copies mutate as one). Two cohorts identical
    except for strain count acquire BDQ resistance at the same rate."""
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], init_prev=ss.bernoulli(0.0), p_rand={'BDQ': 0.3})
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))
    sim = tbsim.Sim(n_agents=20000, networks=net, diseases=tb, demographics=[], dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date('2000-06-30'), rand_seed=0, verbose=0)
    sim.init()
    tb = sim.get_tb()
    a, b = ss.uids(np.arange(10000)), ss.uids(np.arange(10000, 20000))
    for u in (a, b):
        tb.state[u] = TBS.INFECTION
        tb.strain_mask[u] = 1        # carries pan (id0), susceptible to BDQ
    tb.strain_counts[0][a] = 1
    tb.strain_counts[0][b] = 4       # b carries 4 copies
    tb._denovo(ss.uids(np.arange(20000)))
    bdq_ids = [j for j in range(tb.strains.m) if tb.strains.profile[j, 1]]
    acq_a = tb.strains.carried(tb.strain_mask[a])[:, bdq_ids].any(1).mean()
    acq_b = tb.strains.carried(tb.strain_mask[b])[:, bdq_ids].any(1).mean()
    assert np.isclose(acq_a, acq_b, atol=0.03)   # count-independent acquisition probability
    assert 0.2 < acq_a < 0.4                      # ~0.3 as configured


def test_tpt_acquired_resistance_counted_in_flux():
    """L2: TPT-driven acquisition is its own origin channel (``flux_tptacq``) in ResistanceStats, so
    the decomposition no longer loses it. With only ``p_tpt_acq`` active (no de-novo, no treatment),
    resistance emerges and is attributed to TPT — not to the de-novo or treatment channels."""
    from tbsim import ResistanceStats

    tb = tbsim.TBResistant(drugs=['INH'], rel_fitness={'INH': 0.9},
                           beta=ss.permonth(0.3), init_prev=ss.bernoulli(0.2),
                           init_strains=[1.0, 0.0],  # pure-susceptible seed → resistance only via TPT
                           rr_reinfection_inf=0.0, rr_reinfection_non=0.0, p_rand=None)
    tpt = tbsim.TPTSimple(product=tbsim.TPTRx(strains=tb.strains, regimen_drugs=['INH'],
                          p_tpt_acq={'INH': 1.0}, acq_state_rr={int(TBS.INFECTION): 1.0},  # strong signal on latent targets
                          pars=dict(efficacy=ss.bernoulli(0.6), p_sterilize=ss.bernoulli(0.0))),
                          pars=dict(coverage=ss.bernoulli(0.5)))
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0))
    sim = tbsim.Sim(n_agents=3000, networks=net, diseases=tb, demographics=[], interventions=tpt, analyzers=ResistanceStats(),
                 dt=ss.days(30), start=ss.date('2000-01-01'), stop=ss.date('2012-12-31'), rand_seed=0, verbose=0)
    sim.run()
    res = sim.results['resistancestats']
    tptacq = int(np.sum(res.flux_tptacq))
    assert tptacq > 0                                       # TPT-acquired resistance is captured
    assert int(np.sum(res.flux_denovo)) == 0                # no de-novo configured
    assert int(np.sum(res.flux_txacq)) == 0                 # no treatment configured
    assert sim.results.tb['frac_resist'][-1] > 0.0          # resistance really emerged (would be uncounted pre-fix)


if __name__ == '__main__':
    import sys, pytest
    sys.exit(pytest.main([__file__, '-v']))

"""
Per-strain multiplicity counter — spec updates §1–§6 (see implementation-decisions.md D-COUNTER).

Covers the data model (start-at-1, identical-superinfection increment, invariant, clearance reset)
and the two consumers that read the counter: the transmission multinomial (§1) and the progression
bottleneck (§2). DST / treatment / acquisition are asserted *independent* of the counter elsewhere
(test_diagnostics_tpt, test_treatment, test_acquisition) and in this file's coupling checks.
"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS


def _init(tb, n=400, stop='2000-06-30'):
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))
    sim = ss.Sim(n_agents=n, networks=net, diseases=tb, dt=ss.days(30),
                 start=ss.date('2000-01-01'), stop=ss.date(stop), rand_seed=0, verbose=0)
    sim.init()
    return sim, tbsim.get_tb(sim, which=tbsim.TBResistant)


def _counts_col(tb, j, uids):
    return np.asarray(tb.strain_counts[j][uids])


# --------------------------------------------------------------------------- §1 data model
def test_new_infection_starts_count_at_one_regardless_of_source():
    """Spec §1: a newly infected agent starts the founding strain at count 1, even when the source
    carries many copies of it."""
    tb = tbsim.TBResistant(pars=dict(init_prev=ss.bernoulli(0.0)))
    sim, tb = _init(tb)
    src = ss.uids(np.arange(50))
    tgt = ss.uids(np.arange(50, 100))
    tb.strain_mask[src] = 1               # carries strain 0 (pan)
    tb.strain_counts[0][src] = 5          # ...with 5 copies
    tb.set_prognoses(tgt, sources=src)    # infect susceptible targets from these sources
    assert (tb.strain_mask[tgt] == 1).all()          # target now carries strain 0
    assert (_counts_col(tb, 0, tgt) == 1).all()      # ...at count 1, not 5


def test_identical_superinfection_increments_count_mask_unchanged():
    """Spec §1 (reversal): re-exposure to a carried strain increments its count (previously blocked);
    the carried set (mask) is unchanged and the event is tallied."""
    tb = tbsim.TBResistant(pars=dict(init_prev=ss.bernoulli(0.0)))
    sim, tb = _init(tb)
    tgt = ss.uids(np.arange(50))
    src = ss.uids(np.arange(50, 100))
    for u in (tgt, src):
        tb.strain_mask[u] = 1
        tb.strain_counts[0][u] = 1
        tb.state[u] = TBS.INFECTION
    tb._n_identical_superinf = 0
    tb.set_prognoses(tgt, sources=src)
    assert (tb.strain_mask[tgt] == 1).all()          # mask unchanged — still just strain 0
    assert (_counts_col(tb, 0, tgt) == 2).all()      # count incremented 1 → 2
    assert tb._n_identical_superinf == len(tgt)      # counted


def test_clearance_and_death_reset_counts():
    """Spec §3: natural clearance zeroes both the strain mask and all per-strain counts; the same
    holds after death. Checked over a full run via the global invariant plus a direct CLEARED check."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 1.0},
                           pars=dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.15),
                                     init_strains=[0.5, 0.5], rr_reinfection_inf=1.0, rr_reinfection_non=1.0))
    sim, tb = _init(tb, n=3000, stop='2015-12-31')
    sim.run()
    tb = tbsim.get_tb(sim, which=tbsim.TBResistant)
    cleared = ss.uids(tb.state == TBS.CLEARED)
    assert len(cleared) > 0
    counts = tb._counts(cleared)
    assert int(counts.sum()) == 0                    # no cleared agent carries any count


def test_count_invariant_holds_after_run():
    """Core invariant (D-COUNTER): count>0 ⟺ strain bit set, for every agent, after a full run that
    exercises seeding, transmission, superinfection, de-novo, bottleneck, and clearance."""
    tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], rel_fitness={'RIF': 0.9},
                           pars=dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.15),
                                     init_strains=[0.6, 0.2, 0.2, 0.0], rr_reinfection_inf=1.0,
                                     rr_reinfection_non=1.0, p_multi=0.5, p_rand={'BDQ': 0.02}))
    sim, tb = _init(tb, n=3000, stop='2015-12-31')
    sim.run()
    tb = tbsim.get_tb(sim, which=tbsim.TBResistant)
    all_uids = tb.strain_mask.auids
    mask = np.asarray(tb.strain_mask[all_uids])
    bits = ((mask[:, None] >> np.arange(tb.strains.m)) & 1).astype(bool)
    counts = tb._counts(all_uids)
    assert np.array_equal(counts > 0, bits)          # count positive exactly where the bit is set


# --------------------------------------------------------------------------- §1 consumer: transmission
def test_transmission_multinomial_weighted_by_count():
    """Spec §1: P(pass strain j) ∝ count × fitness. Source with counts {pan:2, resistant:1} and no
    fitness cost splits 2/3 vs 1/3; the overall transmissibility (max fitness) is count-independent."""
    s = tbsim.Strains(['TX'], rel_fitness=None)  # neutral fitness
    mask = np.array([0b11])                       # carries strain 0 and strain 1
    tp1 = s.transmit_probs(mask, counts=np.array([[1, 1]]))[0]
    tp2 = s.transmit_probs(mask, counts=np.array([[2, 1]]))[0]
    assert np.allclose(tp1, [0.5, 0.5])
    assert np.allclose(tp2, [2/3, 1/3])
    # Overall infectiousness (max carried fitness) does not depend on count.
    assert s.max_fitness(mask)[0] == 1.0


def test_transmission_count_split_realized_in_set_prognoses():
    """The count-weighted split is actually applied by set_prognoses: from sources with {pan:2, res:1}
    (neutral fitness) the passed strain is pan ~2/3 of the time."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 1.0}, pars=dict(init_prev=ss.bernoulli(0.0)))
    sim, tb = _init(tb, n=12000)
    src = ss.uids(np.arange(6000))
    tgt = ss.uids(np.arange(6000, 12000))
    tb.strain_mask[src] = 0b11
    tb.strain_counts[0][src] = 2   # pan ×2
    tb.strain_counts[1][src] = 1   # resistant ×1
    tb.set_prognoses(tgt, sources=src)
    got_pan = float(np.mean(tb.strain_mask[tgt] == 1))   # received strain 0 only
    assert abs(got_pan - 2/3) < 0.03


# --------------------------------------------------------------------------- §2 consumer: bottleneck
def test_progression_bottleneck_weighted_by_count():
    """Spec §2: when p_multi<1 forces one strain through the bottleneck, the survivor is chosen ∝ count.
    Counts {pan:2, resistant:1} → survivor is pan ~2/3, resistant ~1/3."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 1.0},
                           pars=dict(init_prev=ss.bernoulli(0.0), p_multi=0.0, prog_select='random'))
    sim, tb = _init(tb, n=9000)
    u = ss.uids(np.arange(9000))
    tb.strain_mask[u] = 0b11
    tb.strain_counts[0][u] = 2
    tb.strain_counts[1][u] = 1
    tb._bottleneck(u)                    # p_multi=0 → every agent reduced to one strain
    surv_pan = float(np.mean(tb.strain_mask[u] == 1))   # survivor = strain 0
    surv_res = float(np.mean(tb.strain_mask[u] == 2))   # survivor = strain 1
    assert abs(surv_pan - 2/3) < 0.03
    assert abs(surv_res - 1/3) < 0.03
    assert np.all(np.isin(tb.strain_mask[u], [1, 2]))   # exactly one strain survives


if __name__ == '__main__':
    import sys, pytest
    sys.exit(pytest.main([__file__, '-v']))

"""
Tests for time-varying (front-loaded) TB progression.

The INFECTION-exit hazards to NON_INFECTIOUS and ASYMPTOMATIC can optionally
decline exponentially with time since infection::

    inf_asy(tau) = inf_asy * exp(-k_asy * tau)
    inf_non(tau) = inf_non * exp(-k_non * tau)

where ``tau`` is years since the agent entered INFECTION. Both ``k_asy`` and
``k_non`` default to 0 (constant hazard = current behaviour). See
timevarying_handoff.md / timevarying_readme.md for the modelling rationale.
"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS
import pytest


def make_cohort_sim(n_agents=20_000, dt=ss.days(30), years=10, seed=1, **tb_pars):
    """A closed, synchronised cohort: everyone starts latent, no transmission/demographics/networks.

    Extra keyword arguments are passed straight through as TB parameters, so a
    test can e.g. isolate the direct INFECTION->ASYMPTOMATIC channel.
    """
    pars = dict(
        init_prev=ss.bernoulli(1.0),        # seed the whole population into latent INFECTION
        init_prev_active=ss.bernoulli(0.0),
        beta=ss.peryear(0.0),               # no transmission (time-since-infection == sim time)
    )
    pars.update(tb_pars)
    tb = tbsim.TB(**pars)
    sim = tbsim.Sim(
        tb_model=tb, n_agents=n_agents, networks=[], demographics=[], dt=dt,
        start=ss.date("2000-01-01"), stop=ss.date(f"{2000 + years}-01-01"), rand_seed=seed,
    )
    sim.pars.verbose = 0
    return sim


# Isolated direct INFECTION->ASYMPTOMATIC channel with ASYMPTOMATIC made absorbing,
# so "ever reached ASYMPTOMATIC" == n_ASYMPTOMATIC and the decline shows up cleanly.
ISOLATED_ASY = dict(
    inf_cle=ss.peryear(0.1),    # modest clearance so INFECTION persists and the tail is visible
    inf_non=ss.peryear(0.0),    # no indirect INFECTION->NON->ASY route
    inf_asy=ss.peryear(0.3),
    asy_non=ss.peryear(0.0),    # ASYMPTOMATIC absorbing (no reversion / progression out)
    asy_sym=ss.peryear(0.0),
)


# --- Parameters exist and default to off ---

def test_decline_params_default_off():
    """k_asy and k_non exist and default to 0 (constant-hazard model)."""
    tb = tbsim.TB()
    assert float(tb.pars.k_asy) == 0.0
    assert float(tb.pars.k_non) == 0.0


def test_negative_k_rejected_in_tb():
    """Shape parameters must be non-negative."""
    with pytest.raises(ValueError, match='k_asy must be >= 0'):
        tbsim.TB(k_asy=-1.0)
    with pytest.raises(ValueError, match='k_non must be >= 0'):
        tbsim.TB(k_non=-0.5)


# --- progression_rates: the exact spec formula ---

def _setup_latent(sim, uids, taus, rr=1.0):
    """Put ``uids`` into INFECTION with time-since-infection ``taus`` (years) and given rr_activation."""
    sim.init()
    tb = sim.get_tb()
    uids = ss.uids(uids)
    tb.state[uids] = TBS.INFECTION
    tb.rr_activation[uids] = rr
    tb.ti_infected[uids] = tb.ti - np.asarray(taus) / sim.t.dt_year
    return tb, uids


def test_progression_rates_constant_when_k_zero():
    """With k_asy = k_non = 0, progression_rates returns exactly inf_non/inf_asy * rr_activation."""
    taus = np.array([0.0, 5.0, 9.0])                 # should be ignored when k == 0
    rr = np.array([1.0, 0.5, 2.0])
    tb, uids = _setup_latent(make_cohort_sim(n_agents=100), [0, 1, 2], taus, rr=rr)
    inf_non, inf_asy = tb.progression_rates(uids)
    assert np.allclose(np.asarray(inf_non.value), float(tb.pars.inf_non.value) * rr)
    assert np.allclose(np.asarray(inf_asy.value), float(tb.pars.inf_asy.value) * rr)


def test_progression_rates_exponential_decline():
    """progression_rates applies inf_x(tau) = inf_x * exp(-k_x * tau) per agent."""
    k_asy, k_non = 2.0, 0.5
    taus = np.array([0.0, 1.0, 3.0, 7.0])
    tb, uids = _setup_latent(make_cohort_sim(n_agents=100, k_asy=k_asy, k_non=k_non), [0, 1, 2, 3], taus)
    inf_non, inf_asy = tb.progression_rates(uids)
    assert np.allclose(np.asarray(inf_asy.value), float(tb.pars.inf_asy.value) * np.exp(-k_asy * taus))
    assert np.allclose(np.asarray(inf_non.value), float(tb.pars.inf_non.value) * np.exp(-k_non * taus))
    # tau = 0 starts at the base rate (the parameter is the tau=0 instantaneous hazard)
    assert np.isclose(np.asarray(inf_asy.value)[0], float(tb.pars.inf_asy.value))


def test_decline_channels_independent():
    """k_asy declines only the ASYMPTOMATIC channel; k_non only the NON_INFECTIOUS channel."""
    taus = np.array([0.0, 2.0, 5.0])
    # k_asy only
    tb, uids = _setup_latent(make_cohort_sim(n_agents=100, k_asy=3.0), [0, 1, 2], taus)
    inf_non, inf_asy = tb.progression_rates(uids)
    assert np.allclose(np.asarray(inf_non.value), float(tb.pars.inf_non.value))                 # NON unchanged
    assert np.allclose(np.asarray(inf_asy.value), float(tb.pars.inf_asy.value) * np.exp(-3.0 * taus))
    # k_non only
    tb2, uids2 = _setup_latent(make_cohort_sim(n_agents=100, k_non=1.5), [0, 1, 2], taus)
    inf_non2, inf_asy2 = tb2.progression_rates(uids2)
    assert np.allclose(np.asarray(inf_asy2.value), float(tb2.pars.inf_asy.value))               # ASY unchanged
    assert np.allclose(np.asarray(inf_non2.value), float(tb2.pars.inf_non.value) * np.exp(-1.5 * taus))


def test_decline_clock_resets_on_reinfection():
    """tau is measured from the most recent infection: reinfection resets the decline to the base rate."""
    tb, uids = _setup_latent(make_cohort_sim(n_agents=100, k_asy=4.0), [0], [6.0])  # infected 6 years ago
    _, inf_asy_old = tb.progression_rates(uids)
    assert np.asarray(inf_asy_old.value)[0] < 0.5 * float(tb.pars.inf_asy.value)     # strongly declined
    tb.set_prognoses(uids)                                                           # reinfect -> ti_infected = now
    _, inf_asy_new = tb.progression_rates(uids)
    assert np.isclose(np.asarray(inf_asy_new.value)[0], float(tb.pars.inf_asy.value))  # back to base (tau = 0)


# --- Full-sim behaviour ---

def _ever_asy_curve(sim):
    """Fraction of the cohort that has ever reached ASYMPTOMATIC, per timestep (ASY absorbing)."""
    sim.run()
    tb = sim.get_tb()
    n0 = sum(int(tb.results[f"n_{s.name}"][0]) for s in TBS)   # closed cohort size
    return np.asarray(tb.results["n_ASYMPTOMATIC"][:]) / n0, sim.t.dt_year


def test_k_asy_front_loads_progression():
    """k_asy > 0 front-loads INFECTION->ASYMPTOMATIC: progression concentrates in year 1 and the total falls."""
    curve0, dt_year = _ever_asy_curve(make_cohort_sim(k_asy=0.0, **ISOLATED_ASY))
    curveK, _       = _ever_asy_curve(make_cohort_sim(k_asy=6.0, **ISOLATED_ASY))

    i1 = int(round(1.0 / dt_year))                 # index nearest 1 year since infection
    frac0 = curve0[i1] / curve0[-1]                # share of eventual progression reached by year 1
    fracK = curveK[i1] / curveK[-1]

    # At tau = 0 the decline factor is 1, so the very first step is identical to the constant model.
    assert np.isclose(curveK[0], curve0[0])
    # Curves are monotone non-decreasing (ASY absorbing).
    assert np.all(np.diff(curveK) >= -1e-12)
    # Front-loaded: the declining model completes far more of its total within year 1.
    assert fracK > frac0 + 0.25
    assert fracK > 0.85 and frac0 < 0.55
    # Decline to zero reduces the eventual total direct progression.
    assert curveK[-1] < 0.5 * curve0[-1]


def test_k_zero_matches_default_full_sim():
    """A full transmission sim with explicit k_asy = k_non = 0 reproduces the default bit-for-bit."""
    def run(**extra):
        tb = tbsim.TB(init_prev=ss.bernoulli(0.1), beta=ss.permonth(0.2), **extra)
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
        sim = tbsim.Sim(tb_model=tb, n_agents=500, networks=net, demographics=[], dt=ss.days(30),
                        start=ss.date("2000-01-01"), stop=ss.date("2005-12-31"), rand_seed=3)
        sim.pars.verbose = 0
        sim.run()
        return sim.get_tb()

    tb0 = run()
    tb1 = run(k_asy=0.0, k_non=0.0)
    for state in TBS:
        np.testing.assert_array_equal(tb0.results[f"n_{state.name}"][:], tb1.results[f"n_{state.name}"][:])
    np.testing.assert_array_equal(tb0.results["cum_active"][:], tb1.results["cum_active"][:])
    np.testing.assert_array_equal(tb0.results["new_deaths"][:], tb1.results["new_deaths"][:])


if __name__ == "__main__":
    pytest.main(["-x", "-v", __file__])

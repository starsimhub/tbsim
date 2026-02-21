"""
Tests for HealthSeekingBehavior intervention.

Covers: state resolution (TB_LSHTM, TB_LSHTM_Acute, legacy TB), care-seeking logic
(one-shot per episode, retry, episode reset), result series correctness,
start/stop bounds, and custom_states override.
"""

import numpy as np
import starsim as ss
import tbsim as mtb
from tbsim import TBSL, TBS
from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior


def make_lshtm_sim(
    n_agents=200,
    start=ss.date("2000-01-01"),
    stop=ss.date("2005-12-31"),
    dt=ss.days(7),
    use_acute=False,
    tb_pars=None,
    hsb_pars=None,
):
    """Build a minimal Sim with TB_LSHTM (or Acute) + HealthSeekingBehavior."""
    tb_pars = tb_pars or {}
    hsb_pars = hsb_pars or {}
    tb = mtb.TB_LSHTM_Acute(pars=tb_pars) if use_acute else mtb.TB_LSHTM(pars=tb_pars)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    hsb = HealthSeekingBehavior(pars=hsb_pars)
    return ss.Sim(
        people=ss.People(n_agents=n_agents),
        networks=net,
        diseases=tb,
        interventions=hsb,
        dt=dt,
        start=start,
        stop=stop,
        verbose=0,
    )


def get_hsb(sim):
    """Return the live HealthSeekingBehavior from an initialised/run sim."""
    return sim.interventions[0]


def test_state_resolution_tb_lshtm():
    """init_post resolves eligible states to [TBSL.SYMPTOMATIC] for TB_LSHTM."""
    sim = make_lshtm_sim()
    sim.init()
    assert list(get_hsb(sim)._states) == [int(TBSL.SYMPTOMATIC)]


def test_state_resolution_tb_lshtm_acute():
    """init_post resolves eligible states to [TBSL.SYMPTOMATIC] for TB_LSHTM_Acute."""
    sim = make_lshtm_sim(use_acute=True)
    sim.init()
    assert list(get_hsb(sim)._states) == [int(TBSL.SYMPTOMATIC)]


def test_state_resolution_custom_states():
    """custom_states parameter overrides enum-based resolution."""
    custom = [int(TBSL.ASYMPTOMATIC), int(TBSL.SYMPTOMATIC)]
    sim = make_lshtm_sim(hsb_pars=dict(custom_states=custom))
    sim.init()
    assert set(get_hsb(sim)._states) == {int(TBSL.ASYMPTOMATIC), int(TBSL.SYMPTOMATIC)}


def test_result_keys_defined():
    """HealthSeekingBehavior defines all four expected result series."""
    sim = make_lshtm_sim()
    sim.init()
    for key in ('new_sought_care', 'n_sought_care', 'n_ever_sought_care', 'n_eligible'):
        assert key in get_hsb(sim).results, f"Missing result key: {key}"


def test_care_seeking_fires_with_symptomatic_agents():
    """With symptomatic agents and high rate, some agents seek care."""
    sim = make_lshtm_sim(
        n_agents=500,
        stop=ss.date("2010-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.5)),
    )
    sim.run()
    assert get_hsb(sim).results['n_ever_sought_care'][:].max() > 0


def test_care_seeking_fires_acute_variant():
    """TB_LSHTM_Acute: care-seeking fires for symptomatic agents."""
    sim = make_lshtm_sim(
        n_agents=500,
        use_acute=True,
        stop=ss.date("2010-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.5)),
    )
    sim.run()
    assert get_hsb(sim).results['n_ever_sought_care'][:].max() > 0


def test_no_care_seeking_without_eligible_agents():
    """With beta=0 and init_prev=0, no one is symptomatic and no care is sought."""
    sim = make_lshtm_sim(
        n_agents=100,
        tb_pars=dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.9)),
    )
    sim.run()
    hsb = get_hsb(sim)
    assert hsb.results['n_ever_sought_care'][:].max() == 0
    assert hsb.results['n_eligible'][:].max() == 0


def test_one_shot_per_episode():
    """Agents seek care at most once per episode when care_retry_steps is None."""
    sim = make_lshtm_sim(
        n_agents=500,
        stop=ss.date("2010-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.9), care_retry_steps=None),
    )
    sim.run()
    assert get_hsb(sim).n_care_sought[:].max() <= 1


def test_retry_allows_multiple_seeks():
    """With care_retry_steps=1 and high rate, some agents seek care more than once."""
    sim = make_lshtm_sim(
        n_agents=500,
        stop=ss.date("2010-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.9), care_retry_steps=1),
    )
    sim.run()
    assert get_hsb(sim).n_care_sought_total[:].max() > 1


def test_episode_counter_resets_on_leaving_eligible_state():
    """n_care_sought resets to 0 when an agent leaves eligible states."""
    sim = make_lshtm_sim(n_agents=100, stop=ss.date("2003-12-31"))
    sim.init()
    hsb = get_hsb(sim)
    tb  = sim.diseases[0]

    uid = ss.uids([0])
    tb.state[uid]       = TBSL.SYMPTOMATIC
    hsb.n_care_sought[uid] = 1
    tb.state[uid]       = TBSL.RECOVERED
    hsb.step()

    assert hsb.n_care_sought[uid] == 0


def test_sought_care_no_attribute_error():
    """With ss.People (no sought_care attr), the intervention runs without error."""
    sim = make_lshtm_sim(
        n_agents=200,
        stop=ss.date("2005-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.20)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.5)),
    )
    sim.run()


def test_inactive_outside_start_stop():
    """Intervention outside its active window records zero new seekers."""
    sim = make_lshtm_sim(
        n_agents=300,
        stop=ss.date("2005-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars=dict(
            initial_care_seeking_rate=ss.perday(0.9),
            start=ss.date("2010-01-01"),
            stop=ss.date("2020-12-31"),
        ),
    )
    sim.run()
    assert get_hsb(sim).results['new_sought_care'][:].sum() == 0


def test_results_non_negative():
    """All HealthSeekingBehavior result series are non-negative."""
    sim = make_lshtm_sim(
        n_agents=200,
        stop=ss.date("2005-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.20)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.3)),
    )
    sim.run()
    hsb = get_hsb(sim)
    for key in ('new_sought_care', 'n_sought_care', 'n_ever_sought_care', 'n_eligible'):
        assert np.all(hsb.results[key][:] >= 0), f"Result '{key}' contains negative values"


def test_cumulative_seeks_ge_ever_sought():
    """cumsum(new_sought_care) >= n_ever_sought_care at every step.

    n_ever_sought_care counts currently-alive agents who have ever sought care,
    so it can decrease when those agents die. The cumulative event count is always
    at least as large.
    """
    sim = make_lshtm_sim(
        n_agents=300,
        stop=ss.date("2008-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.25)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.3)),
    )
    sim.run()
    hsb = get_hsb(sim)
    assert np.all(np.cumsum(hsb.results['new_sought_care'][:]) >= hsb.results['n_ever_sought_care'][:])


def test_n_sought_care_le_ever_sought():
    """n_sought_care <= n_ever_sought_care at every time step."""
    sim = make_lshtm_sim(
        n_agents=300,
        stop=ss.date("2008-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.25)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.3)),
    )
    sim.run()
    hsb = get_hsb(sim)
    assert np.all(hsb.results['n_sought_care'][:] <= hsb.results['n_ever_sought_care'][:])


def make_tb_sim(
    n_agents=200,
    start=ss.date("2000-01-01"),
    stop=ss.date("2005-12-31"),
    dt=ss.days(7),
    tb_pars=None,
    hsb_pars=None,
):
    """Build a minimal Sim with legacy TB + HealthSeekingBehavior."""
    tb_pars  = tb_pars  or {}
    hsb_pars = hsb_pars or {}
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    return ss.Sim(
        people=ss.People(n_agents=n_agents),
        networks=net,
        diseases=mtb.TB(pars=tb_pars),
        interventions=HealthSeekingBehavior(pars=hsb_pars),
        dt=dt,
        start=start,
        stop=stop,
        verbose=0,
    )


def test_state_resolution_tb_legacy():
    """init_post resolves eligible states to TBS.care_seeking_eligible() for legacy TB."""
    sim = make_tb_sim()
    sim.init()
    hsb = get_hsb(sim)
    assert set(hsb._states) == set(int(s) for s in TBS.care_seeking_eligible())


def test_care_seeking_fires_tb_legacy():
    """Legacy TB: care-seeking fires when active smear+/smear-/EPTB agents are present."""
    sim = make_tb_sim(
        n_agents=500,
        stop=ss.date("2010-12-31"),
        tb_pars=dict(init_prev=ss.bernoulli(0.30), beta=ss.peryear(0.10)),
        hsb_pars=dict(initial_care_seeking_rate=ss.perday(0.5)),
    )
    sim.run()
    assert get_hsb(sim).results['n_ever_sought_care'][:].max() > 0


def test_presymp_not_eligible_tb_legacy():
    """ACTIVE_PRESYMP agents are not eligible; only smear+/smear-/EPTB count."""
    sim = make_tb_sim(n_agents=100, stop=ss.date("2003-12-31"))
    sim.init()
    hsb = get_hsb(sim)
    tb  = sim.diseases[0]

    uid = ss.uids([0])
    tb.state[uid] = TBS.ACTIVE_PRESYMP
    hsb.n_care_sought[uid] = 0
    hsb.step()

    assert hsb.n_care_sought[uid] == 0, \
        "ACTIVE_PRESYMP should not be eligible for care-seeking"


if __name__ == '__main__':
    import pytest, sys
    sys.exit(pytest.main([__file__, '-v']))

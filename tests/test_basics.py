import pytest
import numpy as np
import starsim as ss
import tbsim as mtb

def make_tb_simplified(agents=20, start=2000, end=2020, dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(pars={'beta': 0.01, 'init_prev': 0.25})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=dict(dt=dt, start=start, end=end), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim

def test_initial_states():
    tb = mtb.TB()
    print(tb.states)
    assert isinstance(tb.susceptible, ss.BoolArr)
    assert isinstance(tb.infected, ss.BoolArr)
    assert isinstance(tb.rel_sus, ss.FloatArr)
    assert isinstance(tb.rel_trans, ss.FloatArr)
    assert isinstance(tb.ti_infected, ss.FloatArr)

    assert isinstance(tb.state, ss.FloatArr)
    assert isinstance(tb.active_tb_state, ss.FloatArr)
    assert isinstance(tb.rr_activation, ss.FloatArr)
    assert isinstance(tb.rr_clearance, ss.FloatArr)
    assert isinstance(tb.rr_death, ss.FloatArr)
    assert isinstance(tb.on_treatment, ss.BoolArr)
    assert isinstance(tb.ti_active, ss.FloatArr)
    assert isinstance(tb.ti_active, ss.FloatArr)
    assert isinstance(tb.ti_active, ss.FloatArr)

def test_default_parameters():
    tb = mtb.TB()
    print(tb)
    assert tb.pars['init_prev'] is not None
    assert isinstance(tb.pars['rate_LS_to_presym'], float)
    assert isinstance(tb.pars['rate_LF_to_presym'], float)
    assert isinstance(tb.pars['rate_active_to_cure'], float)
    assert isinstance(tb.pars['rate_exptb_to_dead'], float)
    assert isinstance(tb.pars['rate_smpos_to_dead'], float)
    assert isinstance(tb.pars['rate_smneg_to_dead'], float)
    assert isinstance(tb.pars['rel_trans_smpos'], float)

def test_infectious():
    tb = mtb.TB()
    assert not tb.infectious

    tb.state = mtb.TBS.ACTIVE_PRESYMP
    assert tb.infectious

    tb.state = mtb.TBS.ACTIVE_SMPOS
    assert tb.infectious

    tb.state = mtb.TBS.ACTIVE_SMNEG
    assert tb.infectious

    tb.state = mtb.TBS.ACTIVE_EXPTB
    assert tb.infectious

def test_set_prognoses():
    sim = make_tb_simplified()
    sim.run()
    tb = sim.diseases['tb']
    before = tb.state.copy()
    uids = ss.uids([1, 2, 3, 7, 9])
    tb.set_prognoses(uids)
    after = tb.state
    assert not np.array_equal(before, after)
    print("Before: ", before)
    print("After: ", after)

def test_update_pre():
    sim = make_tb_simplified(agents=300)
    sim.run()
    tb = sim.diseases['tb']
    assert len(tb.state[tb.state == mtb.TBS.NONE]) == 0
    assert len(tb.state[tb.state == mtb.TBS.LATENT_SLOW]) > 0
    assert len(tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG]) > 0

    print("none", tb.state[tb.state == mtb.TBS.NONE])
    print("Slow:", tb.state[tb.state == mtb.TBS.LATENT_SLOW])
    print("Fast:", tb.state[tb.state == mtb.TBS.LATENT_FAST])
    print("Active Presymp:", tb.state[tb.state == mtb.TBS.ACTIVE_PRESYMP])
    print("Active ExpTB:", tb.state[tb.state == mtb.TBS.ACTIVE_EXPTB])
    print("Active Smear Negative: ", tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG])
    print("Active Smear Positive: ", tb.state[tb.state == mtb.TBS.ACTIVE_SMPOS])

if __name__ == '__main__':
    pytest.main()

import pytest
import numpy as np
from unittest.mock import MagicMock
from tbsim.comorbidities.hiv import HIV, HIVStage
import tbsim as mtb
import starsim as ss

def make_tb_simplified(agents=20, start=2000, stop=2020, dt=7/365):
    pop = ss.People(n_agents=agents)
    hiv = mtb.HIV()
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=hiv, pars=dict(dt=dt, start=start, stop=stop), demographics=dems)
    return sim


def test_initialization():
    hiv_model = mtb.HIV()
    assert hiv_model.pars['init_prev'] == 0.01
    assert hiv_model.pars['transmission_rate'] == 0.0001
    assert hiv_model.pars['cd4_decline_rate'] == 5.0
    assert hiv_model.pars['cd4_recovery_rate'] == 10.0
    assert hiv_model.pars['vl_rise_rate'] == 1000.0
    assert hiv_model.pars['vl_fall_rate'] == 3000.0


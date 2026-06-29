"""Smoke-test runnable code blocks from docs/examples.md."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl
import tbsim
from tbsim import TB
from tbsim.analyzers import DwellTime
from tbsim.comorbidities.hiv import HIV


def test_example_basic_tb_simulation():
    sim = ss.Sim(diseases=TB())
    sim.run()
    sim.plot()


def test_example_tb_with_interventions():
    tb = TB()
    bcg = tbsim.BCGRoutine(pars=dict(
        coverage=ss.bernoulli(p=0.8),
        start=ss.date('1980-01-01'),
        stop=ss.date('2030-12-31'),
        age_range=[0, 5],
    ))
    tpt = tbsim.TPTSimple(pars=dict(
        start=ss.date('1990-01-01'),
        stop=ss.date('2030-12-31'),
    ))
    sim = ss.Sim(
        diseases=tb,
        interventions=[bcg, tpt],
        pars=dict(start=ss.date('1975-01-01'), stop=ss.date('2030-12-31')),
    )
    sim.run()


def test_example_tb_hiv_comorbidity():
    sim = ss.Sim(diseases=[TB(), HIV()])
    sim.run()


def test_example_household_networks():
    dhs_data = sc.dataframe(
        hh_id=[0, 1, 2],
        ages=['72, 17, 30', '37', '13, 55, 36'],
    )
    sim = ss.Sim(
        networks=ssl.networks.HouseholdNet(dhs_data=dhs_data, dynamic=False),
        diseases=TB(),
    )
    sim.run()


def test_example_advanced_analysis():
    sim = ss.Sim(diseases=[TB()], analyzers=DwellTime(scenario_name="Baseline"))
    sim.run()
    az = sim.analyzers[0]
    az.plot('histogram')
    az.plot('kaplan_meier')
    az.plot('network')


def test_example_parameter_sweeps():
    results = []
    for rate in np.linspace(0.1, 0.5, 5):
        sim = ss.Sim(diseases=TB(pars={'beta': ss.peryear(rate)}))
        sim.run()
        results.append(sim.results)
    assert len(results) == 5

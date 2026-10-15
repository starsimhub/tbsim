"""
Test migration: people arriving (`immigration`, into the states given by `immigrants=`) and leaving (`emigration`)
"""

import numpy as np
import sciris as sc
import tbsim

pars = dict(start=2000, stop=2010, total_pop=10_000, verbose=0)


def run(solver='ode', **kwargs):
    """ Run TB with no transmission, so that changes are from migration """
    tb = tbsim.TB(beta=0, init=dict(infection=0.02), **kwargs)
    return tbsim.Sim(tb=tb, solver=solver, n_agents=10_000, **pars).run()


def n_people(sim):
    """ The number of people at the end, from the TB states """
    tb = sim.diseases.tb
    return sum(tb.results[f'n_{s.name}'][-1] for s in tb.graph.people_states)


@sc.timer()
def test_population():
    sc.heading('Migration grows or shrinks the population, as an ODE and as agents')
    base = n_people(run())
    grow = n_people(run(immigration=500, emigration=0.01))
    shrink = n_people(run(immigration=100, emigration=0.05))
    balanced = n_people(run(immigration=500, emigration=0.05))
    print(f'People after 10 years: {base:n} without migration, {grow:n} growing, {shrink:n} shrinking, {balanced:n} balanced')
    assert shrink < base < grow
    assert np.isclose(balanced, base, rtol=0.01)

    abm = run('abm', immigration=500, emigration=0.01)
    assert np.isclose(n_people(abm), grow, rtol=0.03)
    res = abm.results
    assert res.new_emigrants.sum() == res.tb.emigrants.sum() > 0 # Emigrants leave, rather than die
    return abm


@sc.timer()
def test_immigrants():
    sc.heading('People arriving with TB raise its prevalence')
    base = run()
    sim = run(immigration=500, immigrants=dict(susceptible=0.5, symptomatic=0.5))
    before, after = base.results.tb.prevalence_active[-1], sim.results.tb.prevalence_active[-1]
    print(f'Active TB prevalence: {before:0.4f} without immigration, {after:0.4f} with')
    assert after > 2*before
    return sim


if __name__ == '__main__':
    test_population()
    test_immigrants()

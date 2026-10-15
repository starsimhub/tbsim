"""
Test drug resistance: tbsim.TB with drugs, whose strains are a dimension of the model (see tbsim.resistance)

The model is checked against the reference ODE (two_strain_ode.R) in test_models.py, and as agents
against the ODE in test_compartmental.py; these tests check its behavior.
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim

pars = dict(start=2000, stop=2030, unit='year', total_pop=1e5, verbose=0) # As an ODE, with a yearly step
init = dict(infection=0.05, symptomatic=0.005)


def run(solver='ode', **kwargs):
    """ Run TB with resistance to one drug (unless other drugs are given) """
    kwargs = sc.mergedicts(dict(drugs=['RIF'], init=init), kwargs)
    return tbsim.Sim(tb=tbsim.TB(**kwargs), solver=solver, **pars).run()


def by_strain(sim, state='symptomatic'):
    """ The number in a state for each set of strains at the end, by label """
    labels = sim.diseases.tb.graph.strats[0].labels
    return dict(zip(labels, sim.results.tb[f'n_{state}'].by('resistance').values[-1]))


@sc.timer()
def test_strains():
    sc.heading('The strains and the sets of them that people can carry')
    st = tbsim.Strains(['RIF'])
    assert st.labels == ['DS', 'RIF', 'DS+RIF']
    st = tbsim.Strains(['RIF', 'INH'])
    assert len(st.strains) == 4 and len(st.labels) == 4 + 6 # Every strain, and every pair
    assert 'RIF/INH' in st.labels and 'DS+RIF/INH' in st.labels
    assert len(tbsim.Strains(['RIF', 'INH'], max_strains=1).labels) == 4
    tb = tbsim.TB(drugs=['RIF', 'INH'])
    assert tb.graph.strats[0].labels == st.labels
    assert 'susceptible' not in tb.graph.strats[0].states # Only infection and TB have strains
    return st


@sc.timer()
def test_single_strain():
    sc.heading('With no resistance acquired and nobody resistant at the start, the model is the one without drugs')
    plain = tbsim.Sim(tb=tbsim.TB(theta=0.5, init=init), **pars).run()
    sim = run(theta=0.5, q_prog=0, q_treat=0)
    assert np.allclose(sim.results.tb.active.values, plain.results.tb.active.values, rtol=1e-8)
    assert tbsim.resistant_share(sim).max() == 0
    return sim


@sc.timer()
def test_symmetry():
    sc.heading('Two strains that differ only in name stay equal')
    sim = run(theta=0.5, fitness=1, rr_cure=1, q_prog=0, q_treat=0, strains=dict(DS=0.5, RIF=0.5))
    counts = by_strain(sim)
    assert np.isclose(counts['DS'], counts['RIF'], rtol=1e-8)
    return sim


@sc.timer()
def test_selection():
    sc.heading('A less-fit resistant strain is out-competed without treatment, and selected for by treatment')
    kw = dict(fitness=0.7, rr_cure=0.3, strains=dict(DS=0.9, RIF=0.1))
    untreated = tbsim.resistant_share(run(**kw))
    treated = tbsim.resistant_share(run(theta=1.0, **kw))
    print(f'Resistant share after 30 years: {untreated[-1]:0.3f} without treatment, {treated[-1]:0.3f} with')
    assert untreated[-1] < untreated[0]
    assert treated[-1] > 2*untreated[-1]
    return treated


@sc.timer()
def test_superinfection():
    sc.heading('People carry two strains only through superinfection')
    kw = dict(q_prog=0, strains=dict(DS=0.5, RIF=0.5))
    none = by_strain(run(rr_superinfection_latent=0, rr_superinfection_non=0, **kw), 'infection')
    some = by_strain(run(**kw), 'infection')
    assert none['DS+RIF'] == 0
    assert some['DS+RIF'] > 0
    return some


@sc.timer()
def test_denovo():
    sc.heading('Resistance acquired on progression is added to the strain, or replaces it')
    kw = dict(beta=1, q_prog=0.01, rr_superinfection_latent=0, rr_superinfection_non=0)
    mixed = by_strain(run(**kw), 'non_infectious')
    replaced = by_strain(run(prog_resist_mix=False, **kw), 'non_infectious')
    print(f'Mixed: {mixed}\nReplaced: {replaced}')
    assert mixed['DS+RIF'] > 2*mixed['RIF'] # With few new infections, most RIF is carried with DS
    assert replaced['DS+RIF'] == 0 and replaced['RIF'] > 0
    return mixed, replaced


@sc.timer()
def test_two_drugs():
    sc.heading('Two drugs: resistance to each, and to both, is acquired during treatment and transmitted; as agents too')
    kw = dict(drugs=['RIF', 'INH'], theta=0.3, q_treat=dict(RIF=0.02, INH=0.05), fitness=dict(RIF=0.8, INH=0.95))
    ode = run(**kw)
    for drug in ['RIF', 'INH']:
        assert tbsim.resistant_share(ode, drug)[-1] > 0
    assert by_strain(ode)['RIF/INH'] > 0

    weekly = sc.mergedicts(pars, dict(stop=2010, dt=ss.weeks(1)))
    ode = tbsim.Sim(tb=tbsim.TB(init=init, **kw), **weekly).run()
    abm = tbsim.Sim(tb=tbsim.TB(init=init, **kw), solver='abm', n_agents=20_000, **weekly).run()
    ratio = abm.results.tb.active.values.mean()/ode.results.tb.active.values.mean() # Over the whole run, since TB is declining to small numbers by the end
    print(f'Active TB as agents is {ratio:0.2f} times that of the ODE')
    assert 0.8 < ratio < 1.25
    return ode


@sc.timer()
def test_interventions():
    sc.heading('Interventions work as they do without drugs: care, and a test for resistance')
    make = lambda: tbsim.TB(drugs=['RIF'], strains=dict(DS=0.8, RIF=0.2), init=init)
    weekly = sc.mergedicts(pars, dict(dt=ss.weeks(1)))
    tb = make()
    dst = ss.Test(detects=tb.resistance != 'DS', name='dst')
    screen = ss.screen(dst, eligible=tb.symptomatic, coverage=0.5, interval=ss.years(1), name='screen')
    sim = tbsim.Sim(tb=tb, interventions=[tbsim.care(tb, test=tbsim.Xpert(by_age=False)), screen], **weekly).run()
    base = tbsim.Sim(tb=make(), **weekly).run()
    assert sim.results.tb.active[-1] < 0.8*base.results.tb.active[-1]
    res = sim.results.screen
    share = res.diagnosed.values.sum()/res.delivered.values.sum()
    resistant = tbsim.resistant_share(sim, states=['symptomatic'])
    print(f'The test for resistance found {share:0.3f} of those tested resistant; {resistant.mean():0.3f} of people with symptomatic TB were')
    assert 0.5*resistant.mean() < share < 2*resistant.mean()
    return sim


if __name__ == '__main__':
    test_strains()
    test_single_strain()
    test_symmetry()
    test_selection()
    test_superinfection()
    test_denovo()
    test_two_drugs()
    test_interventions()

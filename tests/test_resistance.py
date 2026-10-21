"""
Test drug resistance: tbsim.TB with drugs, whose strains are a dimension of the model (see tbsim.resistance)

The model is checked against the reference ODE (two_strain_ode.R) in test_models.py, and as agents
against the ODE in test_compartmental.py; these tests check its behavior.
"""

import pytest
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


@sc.timer()
def test_second_line():
    sc.heading('A test for resistance sends people to a second-line regimen, which cures the resistant strain; as agents too')
    def make(solver, second_line):
        tb = tbsim.TB(drugs=['RIF'], strains=dict(DS=0.7, RIF=0.3), init=dict(infection=0.3), rr_cure=dict(RIF=0.1), second_line=second_line, **tbsim.regimen('dots'))
        care = tbsim.care(tb, seeking=ss.peryear(1.0), dst=tbsim.DST(tb) if second_line else None)
        agents = dict(n_agents=50_000) if solver == 'abm' else {}
        return tbsim.Sim(tb=tb, interventions=care, solver=solver, **sc.mergedicts(pars, dict(stop=2020, dt=ss.weeks(1)), agents)).run()
    first = make('ode', None)
    sims = sc.objdict(ode=make('ode', True), abm=make('abm', True))
    shares = {key:tbsim.resistant_share(sim)[-1] for key,sim in dict(first_line=first, **sims).items()}
    print(f'Share of active TB resistant to RIF: {sc.strjoin(f"{k} {v:0.3f}" for k,v in shares.items())}')
    assert shares['ode'] < 0.6*shares['first_line']
    res = sims.ode.results
    assert np.isclose(res.care_dst_positive.delivered.sum(), res.tb.diagnosis_second_line.sum())
    assert res.tb.n_second_line.values.max() > 0
    ratio = sims.abm.results.tb.n_second_line.values.mean()/res.tb.n_second_line.values.mean()
    print(f'On second-line treatment as agents: {ratio:0.2f} times the ODE')
    assert 0.8 < ratio < 1.25
    return sims


@sc.timer()
def test_tpt_strains():
    sc.heading('TPT protects only against the strains its drugs act on')
    def progression(drugs):
        tb = tbsim.TB(beta=0, drugs=['RIF', 'INH'], init=dict(infection=0.5), strains={'DS':0.5, 'INH':0.25, 'DS+INH':0.25})
        tpt = ss.treat(tbsim.TPT(efficacy=1.0, activation=0.0, dur=None, tb=tb, drugs=drugs), eligible=tb.infection, years=[2000], name='tpt_delivery')
        sim = tbsim.Sim(tb=tb, interventions=tpt, **sc.mergedicts(pars, dict(stop=2003, dt=ss.weeks(1)))).run()
        return {key:round(val) for key,val in by_strain(sim, 'non_infectious').items() if key in ['DS', 'INH', 'DS+INH']}
    none = progression(['RIF'])
    inh = progression(['INH'])
    print(f'Non-infectious TB with RIF-based TPT: {none}\n... with INH-based TPT: {inh}')
    assert none['DS'] == none['INH'] == none['DS+INH'] == 0
    assert inh['DS'] == 0 and inh['INH'] > 0 and 0 < inh['DS+INH'] < inh['INH']
    return none, inh


@sc.timer()
def test_monitoring():
    sc.heading('People on first-line treatment found resistant switch to second line; as agents too')
    def make(solver, monitor):
        tb = tbsim.TB(drugs=['RIF'], strains=dict(DS=0.7, RIF=0.3), init=dict(infection=0.3), rr_cure=dict(RIF=0.1), second_line=True, **tbsim.regimen('dots'))
        intvs = [tbsim.care(tb, seeking=ss.peryear(1.0))] + [tbsim.monitoring(tb)]*monitor
        agents = dict(n_agents=50_000) if solver == 'abm' else {}
        return tbsim.Sim(tb=tb, interventions=intvs, solver=solver, **sc.mergedicts(pars, dict(stop=2020, dt=ss.weeks(1)), agents)).run()
    base = make('ode', False)
    sims = sc.objdict(ode=make('ode', True), abm=make('abm', True))
    switched = {key:sim.results.tb.switch_second_line.sum() for key,sim in sims.items()}
    print(f'Switched to second line: {switched}; resistant share {tbsim.resistant_share(base)[-1]:0.3f} without monitoring, {tbsim.resistant_share(sims.ode)[-1]:0.3f} with')
    assert tbsim.resistant_share(sims.ode)[-1] < 0.6*tbsim.resistant_share(base)[-1]
    assert np.isclose(switched['abm'], switched['ode'], rtol=0.2)
    return sims


@sc.timer()
def test_tpt_infection():
    sc.heading('TPT clears the strains it acts on, and others can become resistant under it; as agents too')
    out = sc.objdict()
    for solver in ['ode', 'abm']:
        tb = tbsim.TB(beta=0, drugs=['RIF', 'INH'], init=dict(infection=0.4, symptomatic=0.05), strains={'DS':0.5, 'INH':0.25, 'DS+INH':0.25}, tpt=dict(drugs=['INH'], clearance=4.0, acquisition=0.5))
        tpt = ss.treat(tbsim.TPT(tb=tb, dur=ss.years(100)), eligible=tb.infection | tb.symptomatic, years=[2000], name='tpt_delivery') # Held for the whole run
        agents = dict(n_agents=40_000) if solver == 'abm' else {}
        sim = tbsim.Sim(tb=tb, interventions=tpt, solver=solver, **sc.mergedicts(pars, dict(stop=2002, dt=ss.weeks(1)), agents)).run()
        res = sim.results.tb
        latent = by_strain(sim, 'infection')
        out[solver] = sc.objdict(DS=latent['DS'], INH=latent['INH'], mixed=latent['DS+INH'], partial=res.tpt_partial_DS_and_INH_to_INH.sum(),
                                 acquired=sum(res[key].sum() for key in res.keys() if key.startswith('tpt_acquisition_') and '_to_' in key))
        print(f'{solver}: {sc.strjoin(f"{k} {v:.0f}" for k,v in out[solver].items())}')
        assert latent['DS'] < 0.05*latent['INH'] and latent['DS+INH'] < 0.05*latent['INH'] # Cleared, or left with the strain it does not act on
        assert out[solver].partial > 0 and out[solver].acquired > 0
    assert np.isclose(out.abm.acquired, out.ode.acquired, rtol=0.2)

    # Without TPT, none of this happens
    tb = tbsim.TB(drugs=['RIF', 'INH'], init=init, tpt=dict(drugs=['INH'], clearance=4.0, acquisition=0.5))
    res = tbsim.Sim(tb=tb, **pars).run().results.tb
    assert res.tpt_clearance.sum() == 0
    return out


@sc.timer()
def test_retreatment():
    sc.heading('Only people treated before are tested for resistance (as agents)')
    tb = tbsim.TB(drugs=['RIF'], strains=dict(DS=0.7, RIF=0.3), init=dict(infection=0.3), rr_cure=dict(RIF=0.1), second_line=True, **tbsim.regimen('dots'))
    care = tbsim.care(tb, seeking=ss.peryear(1.0), dst=tbsim.DST(tb), retreatment=ss.years(2))
    sim = tbsim.Sim(tb=tb, interventions=care, solver='abm', n_agents=20_000, **sc.mergedicts(pars, dict(stop=2015, dt=ss.weeks(1)))).run()
    res = sim.results
    print(f'{res.care_history.delivered.sum():.0f} tested positive; {res.care_history.diagnosed.sum():.0f} had been treated within 2 years, and were tested for resistance')
    assert 0 < res.care_dst.delivered.sum() == res.care_history.diagnosed.sum() < res.care_history.delivered.sum()
    assert np.isclose(res.tb.diagnosis.sum() + res.tb.diagnosis_second_line.sum(), res.care_history.delivered.sum())
    tb = tbsim.TB(drugs=['RIF'], second_line=True)
    with pytest.raises(ss.CodedError, match='E804'): # Only agents keep when people were treated
        tbsim.Sim(tb=tb, interventions=tbsim.care(tb, dst=tbsim.DST(tb), retreatment=ss.years(2)), solver='ode', verbose=0).init()
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
    test_second_line()
    test_tpt_strains()
    test_monitoring()
    test_tpt_infection()
    test_retreatment()

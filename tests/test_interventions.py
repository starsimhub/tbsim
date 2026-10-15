"""
Test the TB interventions: the care cascade (tests and treatment), TPT, BCG, and changes to transmission
"""
import numpy as np
import sciris as sc
import starsim as ss
import tbsim

kw = dict(start=2000, stop=2015, dt=ss.weeks(1), unit='year', total_pop=1e5, verbose=0)


def make_tb(**pars):
    tb = tbsim.TB(init=dict(infection=0.05, symptomatic=0.003), **pars)
    return tb.stratify(age=[0, 15])


def run(tb, interventions=None, solver='ode', **kwargs):
    agents = dict(n_agents=50_000) if solver == 'abm' else {}
    return ss.Sim(tb, interventions=interventions, solver=solver, **sc.mergedicts(kw, agents, kwargs)).run()


@sc.timer()
def test_tests():
    """ A TB test has a sensitivity for each state, by age """
    xpert = tbsim.Xpert()
    assert set(xpert.sensitivity) == {'symptomatic', 'asymptomatic', 'non_infectious'}
    assert xpert.sensitivity['symptomatic'].values == [0.73, 0.909]
    assert tbsim.CAD(by_age=False).sensitivity['symptomatic'] == 0.66
    custom = tbsim.test(dict(symptomatic=[0.5, 0.9]))
    assert list(custom.sensitivity) == ['symptomatic']
    return xpert


@sc.timer()
def test_care():
    """ The care cascade finds and treats TB, under every solver alike """
    base = run(make_tb(**tbsim.regimen('dots'))).results.tb
    sims = sc.objdict()
    for solver in ['ode', 'abm']:
        tb = make_tb(**tbsim.regimen('dots'))
        sims[solver] = run(tb, tbsim.care(tb), solver=solver)
    res = sims.ode.results
    assert res.tb.infectious[-1] < 0.5*base.infectious[-1], 'Care should lower TB'
    assert res.tb.diagnosis.sum() > 0 and np.isclose(res.tb.diagnosis.sum(), res.care_positive.delivered.sum())
    found = res.care.diagnosed.sum()
    for solver,sim in sims.items():
        print(f'{solver}: {sim.results.care.diagnosed.sum():.0f} diagnosed, {sim.results.tb.completion.sum():.0f} treated')
    assert np.isclose(sims.abm.results.care.diagnosed.sum(), found, rtol=0.15)
    return sims


@sc.timer()
def test_prevention():
    """ TPT and BCG slow progression from latent infection for the people who hold them """
    def latent_progression(intv):
        tb = make_tb(beta=0)
        res = run(tb, intv).results.tb
        return (res.latent_asy + res.latent_non).sum()
    base = latent_progression(None)
    tpt = latent_progression(ss.treat(tbsim.TPT(efficacy=1.0, activation=0.0), coverage=1.0, eligible=None, years=[2000]))
    bcg = latent_progression(ss.vaccinate(tbsim.BCG(take=1.0, activation=0.5), coverage=1.0, years=[2000]))
    print(f'Progression from latent infection: {base:.0f} without, {tpt:.0f} with TPT, {bcg:.0f} with BCG')
    assert tpt < 0.7*base and bcg < 0.9*base
    return base, tpt, bcg


@sc.timer()
def test_change_beta():
    """ Transmission can change over time with ss.change (what BetaByYear did) """
    tb = make_tb()
    res = run(tb, ss.change(tb.pars.beta, at=[2005], by=[0.0])).results.tb
    assert res.transmission[-10:].sum() == 0 and res.transmission[:100].sum() > 0
    return res


if __name__ == '__main__':
    test_tests()
    test_care()
    test_prevention()
    test_change_beta()

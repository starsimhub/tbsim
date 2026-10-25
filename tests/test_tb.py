"""
Test the TB model (tbsim.TB): its natural history, and that it runs the same as an ODE and as agents
"""
import numpy as np
import sciris as sc
import starsim as ss
import tbsim

kw = dict(start=2000, stop=2020, dt=ss.weeks(1), unit='year', total_pop=1e5, verbose=0)
states = ['susceptible', 'infection', 'cleared', 'recovered', 'non_infectious', 'asymptomatic', 'symptomatic', 'treatment', 'treated']


def run(tb, solver='ode', **kwargs):
    agents = dict(n_agents=50_000) if solver == 'abm' else {}
    return ss.Sim(tb, solver=solver, **sc.mergedicts(kw, agents, kwargs)).run().results.tb


@sc.timer()
def test_natural_history():
    """ People stay in the model (births replace deaths), the results add up, and transmission keeps TB going """
    for solver in ['ode', 'abm']:
        res = run(tbsim.TB(init=dict(infection=0.05, symptomatic=0.002)), solver=solver)
        total = sum(res[s].values for s in states)
        assert np.allclose(total, 1e5, rtol=0.02 if solver == 'abm' else 1e-9), f'The population should stay the same size with {solver}'
        assert np.allclose(res.infectious, res.asymptomatic + res.symptomatic)
        assert np.allclose(res.active, res.infectious + res.non_infectious)
        assert np.allclose(res.prevalence_active, res.infectious/total, rtol=0.02)
        assert res.transmission.sum() > 0 and res.reinfect_rec.sum() > 0
        assert res.infectious[-1] > 50, f'TB should persist with {solver}'
    return res


@sc.timer()
def test_solvers_agree():
    """ As agents, the model is close to the ODE """
    tb = lambda: tbsim.TB(init=dict(infection=0.03, symptomatic=0.005))
    ode = run(tb()).infectious[-104:].mean()
    abm = run(tb(), solver='abm', n_agents=100_000).infectious[-104:].mean()
    print(f'Infectious TB over the last two years: {ode:.0f} as an ODE, {abm:.0f} as agents')
    assert 0.8 < abm/ode < 1.25
    return ode, abm


@sc.timer()
def test_front_loading():
    """ With k_latent > 0, latent infections become remote, after which they rarely progress: progression is front-loaded """
    def cohort(**pars):
        tb = tbsim.TB(beta=0, mu=0, init=dict(infection=1.0), **pars)
        return run(tb, stop=2010)
    flat = cohort()
    front = cohort(k_latent=1.0, inf_asy_remote=0.0, inf_non_remote=0.0, inf_asy=0.12, inf_non=0.32)
    early = lambda r: (r.latent_asy + r.latent_non)[:52].sum()/(r.latent_asy + r.latent_non).sum()
    assert early(front) > early(flat), 'Progression should happen earlier with front-loading'
    assert 'remote' in front.keys() and 'infection_remote' in front.keys()
    return flat, front


@sc.timer()
def test_treatment_and_relapse():
    """ Treatment ends in success or failure as the regimen says, and some successes relapse """
    pars = tbsim.regimen('dots')
    assert np.isclose(pars['delta']/(pars['delta'] + pars['phi']), 0.85*0.85) # The chance of success
    tb = tbsim.TB(beta=0, mu=0, theta=10, init=dict(symptomatic=0.1), **pars) # Everyone symptomatic is soon treated
    res = run(tb, stop=2008)
    success = res.completion.sum()/(res.completion.sum() + res.failure.sum())
    assert np.isclose(success, 0.85*0.85, atol=0.01)
    assert np.isclose(res.relapse.sum()/res.completion.sum(), 0.05, atol=0.01)
    return res


@sc.timer()
def test_waning():
    """ With dur_reinfection_protection, people who cleared TB go back to being susceptible """
    res = run(tbsim.TB(dur_reinfection_protection=ss.years(5)))
    assert res.waning_cleared.sum() > 0 and res.waning_recovered.sum() > 0
    assert 'waning_cleared' not in run(tbsim.TB()).keys()
    return res


@sc.timer()
def test_sim():
    """ tbsim.Sim takes TB parameters directly, runs as an ODE without a network, and as agents with one """
    sim = tbsim.Sim(beta=12, verbose=0).run()
    assert sim.solver == 'ode' and sim.get_tb().pars.beta.raw == ss.peryear(12)
    sim = tbsim.Sim(networks=ss.RandomNet(n_contacts=5), beta=2, n_agents=2000, verbose=0).run()
    assert sim.solver == 'abm'
    sim = tbsim.demo(plot=False, verbose=0)
    assert isinstance(sim, tbsim.Sim)
    return sim


if __name__ == '__main__':
    test_natural_history()
    test_solvers_agree()
    test_front_loading()
    test_treatment_and_relapse()
    test_waning()
    test_sim()

"""
Compare the reference ODEs with the declared models, run as ODEs and as agents
"""

import pytest
import numpy as np
import sciris as sc
import starsim as ss
import tbsim
import tbsim.compartmental as tbc

@sc.timer()
def test_ode(do_plot=False):
    """ Run and compare the reference ODE and the declared model (tbsim.TB), run as an ODE """
    start_time = 1920
    end_time = 2020

    # Run pure Python version
    tbr = tbc.TB_ODE()
    tbr.run(start_time=start_time, end_time=end_time) # No dt since exact solver
    if do_plot:
        tbr.plot()

    # Run the declared model
    sim = ss.Sim(tbsim.TB(init_n=dict(symptomatic=1000)), start=start_time, stop=end_time, unit='year', total_pop=1e5, verbose=0)
    sim.run()
    assert np.allclose(sim.results.tb.symptomatic.values, tbr.results.SYMPTOMATIC, rtol=1e-4) # Both at the start of each year
    if do_plot:
        sim.plot()

    return tbr, sim


@pytest.mark.parametrize('treat', [0, 1.0]) # Separately, so they run in parallel
@sc.timer()
def test_two_strain_abm_vs_ode(treat):
    """ The model with resistance, as agents, is close to the same model as an ODE: a less-fit resistant strain is out-competed, and treatment selects for it """
    pars = dict(start=2000, stop=2030, dt=ss.months(2), unit='year', total_pop=1e5, verbose=0) # Agents differ from the ODE by order dt, but at this dt that is small next to the noise between seeds
    last = 6 # The steps in the last year
    tb_kw = dict(beta=40, fitness=0.85, strains=dict(DS=0.8, R=0.2), init=dict(infection=0.05), rr_superinfection_latent=0, rr_superinfection_non=0)
    sims = sc.objdict()
    for solver in ['ode', 'abm']:
        kw = dict(n_agents=50_000, rand_seed=1) if solver == 'abm' else {}
        sims[solver] = ss.Sim(tbsim.LSHTMResistant(r_treat_sym=treat, **tb_kw), solver=solver, **pars, **kw).run()
    share = sc.objdict({solver:tbsim.resistant_share(sim)[-last:].mean() for solver,sim in sims.items()}) # Over the last year
    print(f'treat={treat}: {share}')
    if treat == 0:
        for solver in ['ode', 'abm']:
            assert share[solver] < 0.2, f'{solver}: without treatment, the less-fit resistant strain should be out-competed' # As agents, 0.05 to 0.10 over 10 seeds
        active = sc.objdict({solver:sim.results.tb.active.values[-last:].mean() for solver,sim in sims.items()})
        assert 0.8 < active.abm/active.ode < 1.25 # 0.98 to 1.00 over 10 seeds
    else:
        untreated = tbsim.resistant_share(ss.Sim(tbsim.LSHTMResistant(r_treat_sym=0, **tb_kw), **pars).run())[-last:].mean() # Without treatment, as an ODE (quick); the agents without treatment are in the other case
        for solver in ['ode', 'abm']:
            assert share[solver] > untreated, f'{solver}: treatment should select for resistance' # 0.07 untreated; as agents, 0.15 to 0.32 with treatment over 10 seeds (with fewer agents, a seed can fall below)
        assert abs(share.abm - share.ode) < 0.25 # With treatment, agents differ from the ODE by -0.08 to +0.08 over 10 seeds (at 20k agents, weekly: -0.15 to +0.06)
    return share


if __name__ == '__main__':
    tbr, sim = test_ode(do_plot=True)
    for treat in [0, 1.0]:
        test_two_strain_abm_vs_ode(treat)

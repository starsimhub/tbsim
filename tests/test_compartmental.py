"""
Compare the reference ODEs with the declared models, run as ODEs and as agents
"""

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
    sim = ss.Sim(tbsim.TB(init=dict(symptomatic=1000)), start=start_time, stop=end_time, unit='year', total_pop=1e5, verbose=0)
    sim.run()
    assert np.allclose(sim.results.tb.n_symptomatic.values, tbr.results.SYMPTOMATIC, rtol=1e-4) # Both at the start of each year
    if do_plot:
        sim.plot()

    return tbr, sim


@sc.timer()
def test_two_strain_abm_vs_ode():
    """ The model with resistance, as agents, is close to the same model as an ODE: a less-fit resistant strain is out-competed, and treatment selects for it """
    pars = dict(start=2000, stop=2030, dt=ss.weeks(1), unit='year', total_pop=1e5, verbose=0) # Weekly, since agents differ from the ODE by an amount of order dt
    tb_kw = dict(beta=40, fitness=0.85, strains=dict(DS=0.8, R=0.2), init=dict(infection=0.05), rr_superinfection_latent=0, rr_superinfection_non=0)
    share = sc.objdict()
    for treat in [0, 1.0]:
        for solver in ['ode', 'abm']:
            kw = dict(n_agents=20_000, rand_seed=1) if solver == 'abm' else {}
            sim = ss.Sim(tbsim.LSHTMResistant(r_treat_sym=treat, **tb_kw), solver=solver, **pars, **kw).run()
            share[f'{solver}_{treat}'] = tbsim.resistant_share(sim)[-52:].mean() # Over the last year
            if treat == 0:
                active = sim.results.tb.active.values[-52:].mean()
                share[f'{solver}_active'] = active
    print(share)
    for solver in ['ode', 'abm']:
        assert share[f'{solver}_0'] < 0.2, f'{solver}: without treatment, the less-fit resistant strain should be out-competed'
        assert share[f'{solver}_1.0'] > share[f'{solver}_0'], f'{solver}: treatment should select for resistance'
    assert 0.8 < share.abm_active/share.ode_active < 1.25
    assert abs(share['abm_1.0'] - share['ode_1.0']) < 0.1
    return share


if __name__ == '__main__':
    tbr, sim = test_ode(do_plot=True)
    test_two_strain_abm_vs_ode()

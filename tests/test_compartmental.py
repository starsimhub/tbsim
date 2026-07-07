"""
Compare pure compartmental, compartmental Starsim, and agent-based Starsim LSHTM models.
"""

import sciris as sc
import numpy as np
import starsim as ss
import tbsim.compartmental as tbc

@sc.timer()
def test_ode(do_plot=False):
    """ Run and compare the ODE models """
    start_time = 1920
    end_time = 2020
    dt = 0.1

    # Run pure Python version
    tbr = tbc.TB_ODE()
    tbr.run(start_time=start_time, end_time=end_time) # No dt since exact solver
    if do_plot:
        tbr.plot()

    # Run Starsim version
    tbrss = tbc.TB_SS()
    sim = ss.Sim(modules=tbrss, start=start_time, stop=end_time, dt=dt, n_agents=1, copy_inputs=False)
    sim.run()
    if do_plot:
        tbrss.plot()

    return tbr, sim


@sc.timer()
def test_two_strain_ode():
    """Run the two-strain ODE reference and check key observables."""
    ode = tbc.TwoStrainODE(N=10_000, beta=16.45)
    df = ode.run(start_time=2000, end_time=2005, L_A=500, L_B=50)
    coll = ode.collapse()

    assert 'frac_resist' in df.columns, 'Expected two-strain ODE to report resistant active fraction'
    assert 'frac_super' in df.columns, 'Expected two-strain ODE to report superinfected active fraction'
    assert 'prev_active' in df.columns, 'Expected two-strain ODE to report active prevalence'
    assert np.isfinite(df.prev_active).all(), 'Expected finite active prevalence values'
    assert (df.frac_resist >= 0).all() and (df.frac_resist <= 1).all(), 'Expected resistant fraction in [0, 1]'
    assert 'ASYMPTOMATIC' in coll.columns and 'SYMPTOMATIC' in coll.columns, 'Expected collapsed TB-state compartments'
    return


if __name__ == '__main__':
    tbr, sim = test_ode(do_plot=True)
    test_two_strain_ode()



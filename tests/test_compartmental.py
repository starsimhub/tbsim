"""
Compare pure compartmental, compartmental Starsim, and agent-based Starsim LSHTM models.
"""

import sciris as sc
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


if __name__ == '__main__':
    tbr, sim = test_ode(do_plot=True)



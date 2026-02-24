"""
Detailed TBsim profiling; based on starsim/tests/profile_sim.py
"""

import sciris as sc
import starsim as ss
import tbsim

def make_sim():
    tb = tbsim.TB_LSHTM()
    net = 'random'
    kw = dict(n_agents=10e3, start=1990, stop=2020, dt=0.1, diseases=tb, networks=net, demographics=True)
    sim = ss.Sim(**kw)
    return sim

to_run = [
    'profile', # Run the built-in line profiler
    'cprofile', # Run the built-in function profiler
    'time', # Simply time how long the sim takes to run
    'plot_cpu', # Plot the CPU time ()
]

if 'profile' in to_run:
    sim = make_sim()
    prf = sim.profile()

if 'cprofile' in to_run:
    sim = make_sim()
    cprf = sim.cprofile()

if 'time' in to_run:
    sim = make_sim()
    with sc.timer() as T:
        sim.run()

if 'plot_cpu' in to_run:
    sim.loop.plot_cpu()
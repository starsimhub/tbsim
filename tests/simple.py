"""
Simple TBsim run with default parameters
"""

import starsim as ss
import tbsim as tbs

tb = tbs.TB_LSHTM()
sim = ss.Sim(
    diseases=tb, 
    networks='random', 
    demographics=True,
    start = '2000-01-01',
    stop = '2010-01-01',
    dt = ss.days(7),
    n_agents = 1000,
)
sim.run()
sim.plot()
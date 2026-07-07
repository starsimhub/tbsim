"""
Simple resistance example
=========================

Demonstrates running a drug-resistance TB simulation using ResistanceSim,
comparing two scenarios side by side with tbsim.plot().
"""

import starsim as ss
import tbsim
from tbsim.resistance import ResistanceSim

# 1. Minimal resistance sim — two strains, no treatment cascade
sim1 = ResistanceSim(
    n_agents=2000,
    strain_preset='two_strain',
    cascade=False,
    label='Minimal resistance',
)

# 2. Five-strain catalog with a DST-routed care cascade
sim2 = ResistanceSim(
    n_agents=2000,
    strain_preset='standard',
    cascade='routed',
    label='With care cascade',
)

# Run together as a MultiSim and plot
msim = ss.MultiSim(sims=[sim1, sim2])
msim.run()
tbsim.plot(msim, select=dict(regex=r'prevalence_active|incidence_kpy|cum_active|cum_deaths|n_infectious$'))

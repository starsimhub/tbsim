"""
Demo: Evolving DHS household network with HouseholdStats analyzer.

Creates synthetic DHS-style household data, runs a SIS simulation on an
EvolvingHouseholdDHSNet (where pregnant females can move out and form new
households), and tracks household size / age distribution statistics over
time using tbsim.HouseholdStats.
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim


# -- 1. Construct synthetic DHS data -------------------------------------------
np.random.seed(0)
n_households = 200
hh_ids = np.arange(n_households)
age_strings = []
for _ in range(n_households):
    hh_size = np.random.randint(1, 7)
    ages = np.random.randint(0, 80, hh_size)
    age_strings.append(sc.strjoin(ages))

dhs_data = sc.dataframe(hh_id=hh_ids, ages=age_strings)

# -- 2. Build simulation components -------------------------------------------
net = ss.HouseholdNet(dhs_data=dhs_data, prob_move_out=0.7)

pregnancy = ss.Pregnancy(fertility_rate=20)
deaths = ss.Deaths(death_rate=10)
disease = ss.SIS(beta=0.1, dur_inf=10)

analyzer = tbsim.HouseholdStats(
    network_name='householdnet',
    age_bins=(0, 5, 15, 20, 30, 40, 50, 75, 100),
)

# -- 3. Create and run sim ----------------------------------------------------
sim = ss.Sim(
    n_agents=1000,
    start=2000,
    dur=30,
    dt=1,
    diseases=disease,
    networks=net,
    demographics=[pregnancy, deaths],
    analyzers=analyzer,
    copy_inputs=False,
)
sim.run()

# -- 4. Inspect results -------------------------------------------------------
res = sim.results.householdstats
print(f"\nFinal mean household size:  {float(res.mean_hh_size[-1]):.2f}")
print(f"Final number of households: {int(res.n_households[-1])}")
print(f"Final mean age:             {float(res.mean_age[-1]):.1f}")

# -- 5. Plot ------------------------------------------------------------------
analyzer.plot()
sim.plot()
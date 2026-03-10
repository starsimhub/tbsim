"""
Demo: Evolving DHS household network with HouseholdStats analyzer.

Creates synthetic DHS-style household data, runs a SIS simulation on an
EvolvingHouseholdDHSNet (where pregnant females can move out and form new
households), and tracks household size / age distribution statistics over
time using tbsim.HouseholdStats.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import tbsim


# -- 1. Construct synthetic DHS data -------------------------------------------
np.random.seed(0)

# construct household size distribution - from Nigeria 2003 DHS
hh_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
hh_probs = [0.117, 0.12, 0.141, 0.132, 0.121, 0.108, 0.084, 0.051] + [0.126 / 5] * 5  # arbitrary assignment for the ">9" HH size

# construct age distribution - from Nigeria 1995/starsim demographics tutorial
age_data = pd.read_csv(Path(__file__).parents[1] / "tbsim" / "data" / "nigeria_age.csv")
ages = age_data["age"].tolist()
age_probs = (age_data["value"] / sum(age_data["value"])).tolist()

# loop over ages and household sizes to generate households
n_households = 200
hh_ids = np.arange(n_households)
age_strings = []
for _ in range(n_households):
    hh_size = np.random.choice(hh_sizes, size=1, p=hh_probs)
    hh_ages = np.random.choice(ages, size=hh_size, p=age_probs)
    age_strings.append(sc.strjoin(hh_ages))
dhs_data = sc.dataframe(hh_id=hh_ids, ages=age_strings)
# there is still lots of room for improvement here
# for example, we shouldn't have any children under 15 in households of size 1, and we should have more children in larger households

n_agents = sum(len(s.split()) for s in age_strings)  # make this match the HH structure

# -- 2. Build simulation components -------------------------------------------
net = ss.HouseholdNet(dhs_data=dhs_data, prob_move_out=0.7)

fertility_rates = pd.read_csv(Path(__file__).parents[1] / "tbsim" / "data" / "nigeria_asfr.csv")
pregnancy = ss.Pregnancy(fertility_rate=fertility_rates)

death_rates = pd.read_csv(Path(__file__).parents[1] / "tbsim" / "data" / "nigeria_deaths.csv")
deaths = ss.Deaths(death_rate=death_rates, rate_units=1)

disease = ss.SIS(beta=0.1, dur_inf=10)

analyzer = tbsim.HouseholdStats(
    network_name="householdnet",
    age_bins=(0, 5, 15, 20, 30, 40, 50, 75, 100),
)

# -- 3. Create and run sim ----------------------------------------------------
sim = ss.Sim(
    n_agents=n_agents,
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

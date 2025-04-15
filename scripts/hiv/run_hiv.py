import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np


def build_hivsim() -> ss.Sim:
    # --- Simulation Parameters ---
    sim_pars = dict(
        unit='day',
        dt=7,
        start=ss.date('2000-01-01'),
        stop=ss.date('2035-12-31'),
    )

    # --- Population Setup ---
    n_agents = 1000
    extra_states = [
        ss.FloatArr('SES', default=ss.bernoulli(p=0.3)),  # ~30% get 0 (low SES), ~70% get 1
        ss.Arr(name="CustomField", dtype=str, default="Any Value"),  # Custom string field
    ]
    people = ss.People(n_agents=n_agents, extra_states=extra_states)

    # --- HIV Disease Model ---
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.30),
        init_onart=ss.bernoulli(p=0.50),
        dt=7,
    )
    hiv = mtb.HIV(pars=hiv_pars)

    # --- Network Setup ---
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))

    # --- Demographics (Optional) ---
    births = ss.Births(pars=dict(birth_rate=2))
    deaths = ss.Deaths(pars=dict(death_rate=0.08))

    # --- HIV Intervention ---
    intervention_pars = dict(
        mode='both',
        prevalence=0.2,     # 20% of the population infected
        percent_on_ART=0.5, # 50% of the infected population on ART
        minimum_age=15,
        max_age=49,
    )
    interventions = [mtb.HivInterventions(pars=intervention_pars)]

    # --- Build Simulation ---
    sim = ss.Sim(
        people=people,
        diseases=hiv,
        interventions=interventions,
        networks=net,
        # demographics=[births, deaths],  # Uncomment if demographics are needed
        pars=sim_pars,
    )

    # --- Logging Frequency ---
    sim.pars.verbose = 7/365  # Log once per week

    return sim


if __name__ == '__main__':
    sim = build_hivsim()
    sim.run()
    sim.plot()
    plt.show()

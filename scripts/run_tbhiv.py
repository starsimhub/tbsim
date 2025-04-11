import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt

def build_tbhiv_sim(simpars=None, tbpars=None, hivinv_pars=None) -> ss.Sim:
    """Build a TB-HIV simulation with current disease and intervention models."""

    # --- Simulation Parameters ---
    default_simpars = dict(
        unit='day',
        dt=7,
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        rand_seed=123,
    )
    if simpars:
        default_simpars.update(simpars)

    # --- Population ---
    n_agents = 1000
    # Optional: add extra states (e.g., SES)
    extra_states = [
        ss.FloatArr('SES', default=ss.bernoulli(p=0.3)),    # ~30% get 0 (low SES), ~70% get 1
    ]
    people = ss.People(n_agents=n_agents, extra_states=extra_states)

    # --- TB Model ---
    pars = dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
    )
    tb = mtb.TB(pars=pars)

    # --- HIV Disease Model ---
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.10),     # 10% of the population is infected (in case not using intervention)
        init_onart=ss.bernoulli(p=0.50),    # 50% of the infected population is on ART (in case not using intervention)
    )
    hiv = mtb.HIV(pars=hiv_pars)

    # --- HIV Intervention ---
    hivinv_pars = hivinv_pars or dict(
        mode='both',
        prevalence=0.20,
        percent_on_ART=0.20,
        minimum_age=15,
        max_age=49,
        start=ss.date('2000-01-01'),
        stop=ss.date('2010-12-31'),
    )
    hiv_intervention = mtb.HivInterventions(pars=hivinv_pars)
    
    # # --- Demographics ---
    # demographics = [
    #     ss.Pregnancy(pars=dict(fertility_rate=20)),
    #     ss.Deaths(pars=dict(death_rate=20)),
    # ]

    # --- Network ---
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))

    # --- Connector ---
    connector = mtb.TB_HIV_Connector()

    # --- Assemble Simulation ---
    sim = ss.Sim(
        people=people,
        diseases=[tb, hiv],
        interventions=[hiv_intervention],
        # demographics=demographics,
        networks=network,
        connectors=[connector],
        pars=default_simpars,
    )

    sim.pars.verbose = 7 / 365  # One update per year
    return sim


if __name__ == '__main__':
    sim = build_tbhiv_sim()
    sim.run()
    sim.plot()
    plt.show()

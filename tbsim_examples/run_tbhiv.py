"""
TB-HIV coinfection example.

Runs two scenarios: one without HIV interventions and one with,
demonstrating the TB-HIV connector and HIV intervention modules.
"""

import tbsim
import starsim as ss
import matplotlib.pyplot as plt


def build_tbhiv_sim(interventions=True):
    """Build a TB-HIV simulation with current disease and intervention models."""

    sim_pars = dict(
        dt=ss.days(7),
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        rand_seed=123,
        verbose=0,
    )

    # TB disease
    tb = tbsim.TB(pars=dict(
        beta=ss.peryear(0.025),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
    ))

    # HIV disease
    hiv = tbsim.HIV(pars=dict(
        init_prev=ss.bernoulli(p=0.00),
        init_onart=ss.bernoulli(p=0.00),
    ))

    # Network
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))

    # HIV interventions (optional)
    intvs = None
    if interventions:
        intvs = [tbsim.HivInterventions(pars=dict(
            use_prevalence=True,
            use_art=True,
            prevalence=0.30,
            percent_on_ART=0.50,
            min_age=15, max_age=60,
            start=ss.date('2000-01-01'),
            stop=ss.date('2035-12-31'),
        ))]

    # TB-HIV connector
    connector = tbsim.TB_HIV_Connector()

    sim = ss.Sim(
        people=ss.People(n_agents=1000),
        diseases=[tb, hiv],
        interventions=intvs,
        networks=network,
        connectors=connector,
        pars=sim_pars,
    )
    return sim


if __name__ == '__main__':
    scenarios = [
        dict(interventions=False),
        dict(interventions=True),
    ]

    results = {}
    for args in scenarios:
        label = 'With HIV interventions' if args['interventions'] else 'No HIV interventions'
        print(f"Running: {label}")
        sim = build_tbhiv_sim(**args)
        sim.run()
        results[label] = sim.results.flatten()

    tbsim.plot_combined(results, title='TB-HIV Coinfection Model')
    plt.show()

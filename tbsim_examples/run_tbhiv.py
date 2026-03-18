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

    # HIV disease
    hiv = tbsim.HIV(pars=dict(
        init_prev=ss.bernoulli(p=0.00),
        init_onart=ss.bernoulli(p=0.00),
    ))

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

    sim = tbsim.Sim(
        n_agents=1000,
        start='1980',
        stop='2035',
        rand_seed=123,
        beta=ss.peryear(0.025),
        init_prev=ss.bernoulli(p=0.25),
        tb_pars=dict(),
        diseases=[hiv],
        networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0)),
        interventions=intvs,
        connectors=connector,
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

    tbsim.plot(results, title='TB-HIV Coinfection Model')
    plt.show()

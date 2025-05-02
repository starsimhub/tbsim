import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import shared_functions as sf


def build_tbhiv_sim(include_intv=False, hiv_pars=None, intv_pars=None, Demgs=False) -> ss.Sim:
    """Construct a TB-HIV simulation with optional interventions."""
    
    sim_pars = dict(
        unit='day',
        dt=7,
        start=ss.date('1980-01-01'),
        stop=ss.date('2030-12-31'),
        rand_seed=123,
        verbose=0,
    )

    people = ss.People(n_agents=500)
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))

    tb = sf.make_tb()
    hiv = sf.make_hiv(hiv_pars=hiv_pars)
    pars = dict(
                acute_multiplier     = 1.2222111,
                latent_multiplier    = 1.9999999999,
                aids_multiplier      = 2.7777,
        )
    connector = sf.make_tb_hiv_connector(include=tb, pars=pars)
    interventions = sf.make_interventions(include=include_intv, pars=intv_pars) if include_intv else None
    
    return ss.Sim(
        people=people,
        diseases=[tb, hiv],
        networks=network,
        interventions=interventions,
        connectors=connector,
        pars=sim_pars,
    )


def get_scenarios():
    """Define simulation scenarios."""
    return {
        "No HIV": dict(hiv_pars=dict(
            init_prev=ss.bernoulli(p=0.00),
            init_onart=ss.bernoulli(p=0.00)
        )),
        "HIV prevalence = 20%": dict(
            include_intv=False, 
            hiv_pars=dict(
                init_prev=ss.bernoulli(p=0.20),
                init_onart=ss.bernoulli(p=0.00)
        )),
        "Controlled by intv. 30% prev.": dict(
            include_intv=True,
            intv_pars=dict(
                prevalence=0.30,
                percent_on_ART=0.00,
                start=ss.date('1980-05-01'),
                stop=ss.date('2000-12-31'),
            )
        ),
    }


def main():
    scenarios = get_scenarios()
    results = {}

    for name, kwargs in scenarios.items():
        print(f"\nRunning scenario: {name}")
        sim = build_tbhiv_sim(**kwargs)
        sim.run()
        results[name] = sim.results.flatten()

    sf.plot_results(results)


if __name__ == '__main__':
    main()

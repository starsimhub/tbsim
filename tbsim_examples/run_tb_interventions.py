"""
TB interventions example: scenarios with BCG, TPT, BetaByYear, and Dx/Tx cascade.
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim
import matplotlib.pyplot as plt

# Simple default parameters
DEFAULT_SPARS = dict(
    dt=ss.days(7),
    start=ss.date('1975-01-01'),
    stop=ss.date('2030-12-31'),
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = dict(
    beta=ss.peryear(0.0025),
    init_prev=ss.bernoulli(p=0.25),
)

# Simple age distribution
age_data = sc.dataframe({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]
})


def build_sim(scenario=None, spars=None):
    """
    Build a TB simulation with optional interventions.

    Args:
        scenario (dict, optional): Scenario-specific components including:
            - 'tbpars' (dict): TB-specific simulation parameters.
            - 'bcgintervention' (dict/list): BCG intervention parameters.
            - 'tptintervention' (dict/list): TPT intervention parameters.
            - 'betabyyear' (dict/list): BetaByYear intervention parameters.
        spars (dict, optional): Sim parameter overrides.

    Returns:
        tbsim.Sim: A fully initialized simulation.
    """
    scenario = scenario or {}

    # Merge parameters
    spars = sc.objdict({**DEFAULT_SPARS, **(spars or {})})
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}

    # Create interventions list
    interventions = []

    # Add BCG interventions (can be single or multiple)
    bcg_params = scenario.get('bcgintervention')
    if bcg_params:
        if isinstance(bcg_params, dict):
            interventions.append(tbsim.BCGRoutine(pars=bcg_params))
        elif isinstance(bcg_params, list):
            for i, params in enumerate(bcg_params):
                params['name'] = f'BCG_{i}'
                interventions.append(tbsim.BCGRoutine(pars=params))

    # Add TPT interventions (can be single or multiple)
    tpt_params = scenario.get('tptintervention')
    if tpt_params:
        if isinstance(tpt_params, dict):
            interventions.append(tbsim.TPTSimple(pars=tpt_params))
        elif isinstance(tpt_params, list):
            for i, params in enumerate(tpt_params):
                params['name'] = f'TPT_{i}'
                interventions.append(tbsim.TPTSimple(pars=params))

    # Add Beta interventions (can be single or multiple)
    beta_params = scenario.get('betabyyear')
    if beta_params:
        if isinstance(beta_params, dict):
            interventions.append(tbsim.BetaByYear(pars=beta_params))
        elif isinstance(beta_params, list):
            for i, params in enumerate(beta_params):
                params['name'] = f'Beta_{i}'
                interventions.append(tbsim.BetaByYear(pars=params))

    # Create simulation using tbsim.Sim
    networks = [
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        tbsim.HouseholdNet(),
    ]

    spars.n_agents = 500

    return tbsim.Sim(
        people=ss.People(n_agents=spars.n_agents, age_data=age_data),
        networks=networks,
        interventions=interventions,
        tb_pars=tbpars,
        sim_pars=spars,
    )


def get_scenarios():
    """Define simulation scenarios for evaluating TB interventions."""
    return {
        'Baseline': {
            'name': 'No interventions',
        },
        'Baseline and BetaByYear': {
            'name': 'No interventions',
            'betabyyear': dict(years=[1990, 2000], x_beta=[0.5, 1.4]),
        },
        'Single BCG': {
            'name': 'Single BCG intervention',
            'bcgintervention': dict(
                coverage=0.8,
                start=ss.date('1980-01-01'),
                stop=ss.date('2020-12-31'),
                age_range=[1, 5],
            ),
        },
        # NOTE: Multiple BCG scenario currently broken (duplicate product module registration)
        # 'Multiple BCG': {
        #     'name': 'Multiple BCG interventions',
        #     'bcgintervention': [
        #         dict(coverage=0.9, start=ss.date('1980-01-01'), stop=ss.date('2020-12-31'), age_range=[0, 2]),
        #         dict(coverage=0.3, start=ss.date('1985-01-01'), stop=ss.date('2015-12-31'), age_range=[25, 40]),
        #     ],
        # },
    }


def run_scenarios(plot=True):
    """Run all scenarios and optionally plot results."""
    scenarios = get_scenarios()
    results = {}

    for name, scenario in scenarios.items():
        print(f"\nRunning: {name}")
        sim = build_sim(scenario=scenario)
        sim.run()
        results[name] = sim.results.flatten()

    if plot:
        tbsim.plot(results, row_height=2)
        plt.show()

    return results


def run_dx_tx_cascade():
    """
    Demonstrate the Dx/Tx product + delivery API.

    Cascade: HealthSeekingBehavior -> CXR screen -> Xpert confirm -> DOTS treat
    """
    # Cascade interventions
    hsb = tbsim.HealthSeekingBehavior()
    screen = tbsim.DxDelivery(
        name='screen',
        product=tbsim.CAD(),
        coverage=0.9,
        result_state='screen_positive',
        result_validity=ss.days(180),  # Screen result valid for 6 months
    )
    confirm = tbsim.DxDelivery(
        name='confirm',
        product=tbsim.Xpert(),
        coverage=0.8,
        eligibility=lambda sim: (
            sim.people.screen.screen_positive
            & ~sim.people.confirm.tested
        ).uids,
        result_state='diagnosed',
        result_validity=ss.days(365),  # Diagnosis valid for 1 year
    )
    treat = tbsim.TxDelivery(product=tbsim.DOTS())

    sim = tbsim.Sim(
        n_agents=5000,
        start='2000',
        stop='2020',
        rand_seed=42,
        init_prev=ss.bernoulli(p=0.30),
        beta=ss.peryear(0.05),
        interventions=[hsb, screen, confirm, treat],
    )
    sim.run()

    # Print summary
    r = sim.results
    print(f"\nDx/Tx Cascade Results:")
    print(f"  Screened:  {r.screen.n_tested.values.sum()}")
    print(f"  Confirmed: {r.confirm.n_tested.values.sum()}")
    print(f"  Treated:   {r.txdelivery.n_treated.values.sum()}")
    print(f"  Cured:     {r.txdelivery.cum_success.values[-1]}")
    print(f"  Failed:    {r.txdelivery.cum_failure.values[-1]}")
    return sim



def run_tpt_cascade():
    """
    Full TPT cascade: HSB → Dx → household contact tracing → triage.

    Index case pathway:
        HSB → CXR screen → Xpert confirm → DOTS treatment

    Contact tracing pathway:
        HouseholdContactTracing detects new treatment starts →
        DxDelivery screens contacts →
        Active TB contacts → TxDelivery (via diagnosed flag) →
        Non-active contacts → TPTDelivery (preventive therapy)
    """
    # Create synthetic DHS household data
    n_households = 200
    hh_ids = np.arange(n_households)
    age_strings = []
    for _ in range(n_households):
        hh_size = np.random.randint(2, 6)
        ages = np.random.randint(1, 70, hh_size)
        age_strings.append(sc.strjoin(ages))
    dhs_data = sc.dataframe(hh_id=hh_ids, ages=age_strings)

    # Networks
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)
    community_net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=3), dur=0))

    # --- Index case pathway ---
    hsb = tbsim.HealthSeekingBehavior()
    screen = tbsim.DxDelivery(
        name='screen',
        product=tbsim.CAD(),
        coverage=0.9,
        result_state='screen_positive',
        result_validity=ss.days(180),
    )
    confirm = tbsim.DxDelivery(
        name='confirm',
        product=tbsim.Xpert(),
        coverage=0.8,
        eligibility=lambda sim: (
            sim.people.screen.screen_positive
            & ~sim.people.confirm.tested
            & sim.people.alive
        ).uids,
        result_state='diagnosed',
    )
    treat = tbsim.TxDelivery(product=tbsim.DOTS())

    # --- Contact tracing pathway ---
    hh_tracing = tbsim.HouseholdContactTracing(coverage=0.8)

    # Screen contacts for active TB
    contact_screen = tbsim.DxDelivery(
        name='contact_screen',
        product=tbsim.Xpert(),
        coverage=1.0,
        eligibility=lambda sim: (
            sim.people.householdcontacttracing.contact_identified
            & ~sim.people.contact_screen.tested
            & sim.people.alive
        ).uids,
        result_state='diagnosed',  # positive contacts → TxDelivery picks them up
    )

    # TPT for contacts screened negative (no active disease)
    tpt = tbsim.TPTDelivery(
        product=tbsim.TPTTx(),
        contact_tracing='householdcontacttracing',
        contact_screen='contact_screen',
    )

    sim = tbsim.Sim(
        n_agents=5000,
        start='2000',
        stop='2020',
        rand_seed=42,
        init_prev=ss.bernoulli(p=0.30),
        beta=ss.peryear(0.05),
        networks=[hh_net, community_net],
        interventions=[hsb, screen, confirm, treat, hh_tracing, contact_screen, tpt],
    )
    sim.run()

    # Print summary
    r = sim.results
    ct_r = r.householdcontacttracing
    cs_r = r.contact_screen
    tx_r = r.txdelivery
    tpt_r = r.tptdelivery
    print(f"\nTPT Cascade Results:")
    print(f"  Index cases followed up:  {ct_r.n_index_followed_up.values.sum()}")
    print(f"  Contacts identified:      {ct_r.n_contacts_identified.values.sum()}")
    print(f"  Contacts screened:        {cs_r.n_tested.values.sum()}")
    print(f"  Contacts diagnosed (active TB): {cs_r.n_positive.values.sum()}")
    print(f"  Total treated:            {tx_r.n_treated.values.sum()}")
    print(f"  TPT initiated:            {tpt_r.n_tpt_initiated.values.sum()}")
    print(f"  TPT protected (final):    {tpt_r.n_protected.values[-1]}")
    return sim


if __name__ == '__main__':
    # Run all scenarios
    # results = run_scenarios(plot=True)

    # Run Dx/Tx cascade example
    print("\n--- Dx/Tx Cascade Example ---")
    sim = run_dx_tx_cascade()


    # Run full TPT cascade example
    print("\n--- Full TPT Cascade Example ---")
    sim_cascade = run_tpt_cascade()

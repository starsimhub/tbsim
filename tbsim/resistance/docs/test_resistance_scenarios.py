"""Scenario-level epidemiological checks for the resistance overlay."""

import numpy as np
import starsim as ss

import tbsim

from resistance_helpers import (
    cum_new_inh_r,
    final_active_resistant_share,
    make_scenario_sim,
)


def test_burden_metrics_remain_comparable_when_resistance_layer_added():
    """Core burden channels should stay within a broad envelope with resistance on."""
    base = make_scenario_sim(with_resistance=False, rand_seed=7)
    with_res = make_scenario_sim(with_resistance=True, rand_seed=7, p_selective=0.0,
                                 include_treatment=False)

    base.run()
    with_res.run()

    tb_base = tbsim.get_tb(base)
    tb_res = tbsim.get_tb(with_res)

    base_prev = float(np.mean(tb_base.results['prevalence_active'][-5:]))
    res_prev = float(np.mean(tb_res.results['prevalence_active'][-5:]))
    if base_prev > 0:
        ratio = res_prev / base_prev
        assert 0.4 <= ratio <= 1.6, f'prevalence ratio {ratio:.2f} out of [0.4, 1.6]'

    base_inc = float(tb_base.results['cum_active'][-1])
    res_inc = float(tb_res.results['cum_active'][-1])
    if base_inc > 0:
        ratio = res_inc / base_inc
        assert 0.4 <= ratio <= 1.6, f'incidence ratio {ratio:.2f} out of [0.4, 1.6]'

    base_mort = float(tb_base.results['cum_deaths'][-1])
    res_mort = float(tb_res.results['cum_deaths'][-1])
    if base_mort > 0:
        ratio = res_mort / base_mort
        assert 0.4 <= ratio <= 1.6, f'mortality ratio {ratio:.2f} out of [0.4, 1.6]'


def test_higher_selective_acquisition_increases_resistant_share():
    low = make_scenario_sim(with_resistance=True, rand_seed=11, include_treatment=False,
                            resistant_init_prev=0.0, p_random_acquisition={'INH': 0.0},
                            n_agents=800, stop=ss.date('2006-01-01'))
    high = make_scenario_sim(with_resistance=True, rand_seed=11, include_treatment=False,
                             resistant_init_prev=0.0, p_random_acquisition={'INH': 0.6},
                             n_agents=800, stop=ss.date('2006-01-01'))

    low.run()
    high.run()

    assert cum_new_inh_r(high) > cum_new_inh_r(low)


def test_higher_fitness_cost_reduces_resistant_strain_accumulation():
    low_cost = make_scenario_sim(
        with_resistance=True, rand_seed=13, include_treatment=False,
        resistant_fitness=1.0, resistant_init_prev=0.02,
        n_agents=1500, stop=ss.date('2015-01-01'),
    )
    high_cost = make_scenario_sim(
        with_resistance=True, rand_seed=13, include_treatment=False,
        resistant_fitness=0.3, resistant_init_prev=0.02,
        n_agents=1500, stop=ss.date('2015-01-01'),
    )

    low_cost.run()
    high_cost.run()

    share_low_cost = final_active_resistant_share(low_cost)
    share_high_cost = final_active_resistant_share(high_cost)
    assert share_low_cost >= share_high_cost, (
        f'Expected no-cost ({share_low_cost:.3f}) >= '
        f'high-cost ({share_high_cost:.3f}) resistant share of active TB'
    )


def test_treatment_eliminates_susceptible_strains_increasing_resistant_share():
    no_tx = make_scenario_sim(
        with_resistance=True, rand_seed=17, include_treatment=False,
        resistant_fitness=1.0, resistant_init_prev=0.005,
        n_agents=1000, stop=ss.date('2010-01-01'),
    )
    with_tx = make_scenario_sim(
        with_resistance=True, rand_seed=17, include_treatment=True,
        resistant_fitness=1.0, resistant_init_prev=0.005,
        p_selective=0.0, p_random_acquisition=None,
        n_agents=1000, stop=ss.date('2010-01-01'),
    )

    no_tx.run()
    with_tx.run()

    share_no_tx = final_active_resistant_share(no_tx)
    share_with_tx = final_active_resistant_share(with_tx)

    assert share_with_tx >= share_no_tx, (
        f'Expected treated ({share_with_tx:.3f}) >= untreated ({share_no_tx:.3f}) '
        'resistant share when regimen is ineffective against resistant strain'
    )

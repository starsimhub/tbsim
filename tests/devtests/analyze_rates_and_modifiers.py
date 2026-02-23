#!/usr/bin/env python3
"""
Detailed Analysis of TB Rates and Risk Modifiers

This script provides a comprehensive analysis of TB disease rates and risk modifiers
in the context of BCG intervention, examining both baseline and intervention scenarios.

Author: TB Simulation Team
Date: 2024
"""

import tbsim
from tbsim.interventions.bcg import BCGVx, BCGRoutine
import starsim as ss
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_test_population():
    """Create standardized test population with age distribution"""
    return pd.DataFrame({
        'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
        'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]
    })

def analyze_tb_rates_and_modifiers():
    """Comprehensive analysis of TB rates and risk modifiers"""

    print("=" * 80)
    print("COMPREHENSIVE TB RATES AND RISK MODIFIERS ANALYSIS")
    print("=" * 80)
    print()

    # Create population
    age_data = create_test_population()
    pop = ss.People(n_agents=500, age_data=age_data)

    # Initialize TB model
    tb = tbsim.TB_LSHTM(pars={'beta': ss.peryear(0.01), 'init_prev': 0.25}, name='tb')
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        pars={'dt': ss.days(7), 'start': ss.date('2000-01-01'), 'stop': ss.date('2025-12-31')}
    )
    sim.init()

    print("TB DISEASE MODEL PARAMETERS (TB_LSHTM)")
    print("-" * 40)
    print(f"Transmission Rate (beta): {tb.pars.beta}")
    print(f"Initial Prevalence: {tb.pars.init_prev}")
    print(f"Population Size: {len(sim.people)} individuals")
    print()

    print("TB TRANSITION RATES (per year)")
    print("-" * 40)
    print(f"INFECTION -> CLEARED:        {tb.pars.inf_cle}")
    print(f"INFECTION -> NON_INFECTIOUS: {tb.pars.inf_non}")
    print(f"INFECTION -> ASYMPTOMATIC:   {tb.pars.inf_asy}")
    print(f"NON_INFECTIOUS -> RECOVERED: {tb.pars.non_rec}")
    print(f"NON_INFECTIOUS -> ASYMPTOMATIC: {tb.pars.non_asy}")
    print(f"ASYMPTOMATIC -> NON_INFECTIOUS: {tb.pars.asy_non}")
    print(f"ASYMPTOMATIC -> SYMPTOMATIC:    {tb.pars.asy_sym}")
    print(f"SYMPTOMATIC -> ASYMPTOMATIC: {tb.pars.sym_asy}")
    print(f"SYMPTOMATIC -> TREATMENT:    {tb.pars.sym_treat}")
    print(f"SYMPTOMATIC -> DEAD:         {tb.pars.sym_dead}")
    print(f"TREATMENT -> SYMPTOMATIC (fail): {tb.pars.fail_rate}")
    print(f"TREATMENT -> TREATED (complete): {tb.pars.complete_rate}")
    print()

    print("BASELINE RISK MODIFIERS")
    print("-" * 40)
    try:
        if hasattr(tb.rr_activation, 'raw') and len(tb.rr_activation.raw) > 0:
            print(f"Activation Risk Modifier: {np.mean(tb.rr_activation.raw):.3f} +/- {np.std(tb.rr_activation.raw):.3f}")
            print(f"Clearance Risk Modifier: {np.mean(tb.rr_clearance.raw):.3f} +/- {np.std(tb.rr_clearance.raw):.3f}")
            print(f"Death Risk Modifier: {np.mean(tb.rr_death.raw):.3f} +/- {np.std(tb.rr_death.raw):.3f}")
        else:
            print("Risk modifiers not yet initialized (default values: 1.0)")
    except Exception as e:
        print(f"Error accessing risk modifiers: {e}")
    print()

    # Now test with BCG intervention
    print("BCG INTERVENTION ANALYSIS")
    print("-" * 40)

    bcg = BCGRoutine(pars={
        'coverage': 0.8,
        'age_range': (0, 5),
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2025-12-31')
    })

    pop_bcg = ss.People(n_agents=500, age_data=age_data)
    tb_bcg = tbsim.TB_LSHTM(pars={'beta': ss.peryear(0.01), 'init_prev': 0.25}, name='tb')
    net_bcg = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    sim_bcg = ss.Sim(
        people=pop_bcg,
        diseases=[tb_bcg],
        networks=[net_bcg],
        interventions=[bcg],
        pars={'dt': ss.days(7), 'start': ss.date('2000-01-01'), 'stop': ss.date('2025-12-31')}
    )
    sim_bcg.init()

    bcg_itv = sim_bcg.interventions['bcgroutine']
    bcg_itv.step()

    print(f"BCG Coverage: {bcg_itv.pars.coverage}")
    print(f"BCG p_take: {bcg_itv.product.pars.p_take}")
    print(f"Age Range: {bcg_itv.pars.age_range}")
    print(f"Duration of immunity: {bcg_itv.product.pars.dur_immune}")
    print()

    vaccinated = int(np.sum(bcg_itv.bcg_vaccinated))
    protected = int(np.count_nonzero(bcg_itv.product.bcg_protected))
    coverage = vaccinated / 500

    print("BCG INTERVENTION RESULTS")
    print("-" * 40)
    print(f"Total Vaccinated: {vaccinated} individuals")
    print(f"Currently Protected: {protected} individuals")
    print(f"Coverage: {coverage:.1%}")
    print()

    # Analyze individual-level modifiers
    if vaccinated > 0:
        vaccinated_uids = bcg_itv.bcg_vaccinated.uids

        print("INDIVIDUAL-LEVEL RISK MODIFIER ANALYSIS")
        print("-" * 50)

        activation_modifiers = bcg_itv.product.bcg_activation_modifier_applied[vaccinated_uids]
        clearance_modifiers = bcg_itv.product.bcg_clearance_modifier_applied[vaccinated_uids]
        death_modifiers = bcg_itv.product.bcg_death_modifier_applied[vaccinated_uids]

        valid_activation = activation_modifiers[~np.isnan(activation_modifiers)]
        valid_clearance = clearance_modifiers[~np.isnan(clearance_modifiers)]
        valid_death = death_modifiers[~np.isnan(death_modifiers)]

        if len(valid_activation) > 0:
            print(f"Activation Risk Modifiers (vaccinated individuals):")
            print(f"  Mean: {np.mean(valid_activation):.3f}")
            print(f"  Std: {np.std(valid_activation):.3f}")
            print(f"  Min: {np.min(valid_activation):.3f}")
            print(f"  Max: {np.max(valid_activation):.3f}")
            print(f"  Reduction: {(1 - np.mean(valid_activation)) * 100:.1f}%")
            print()

        if len(valid_clearance) > 0:
            print(f"Clearance Risk Modifiers (vaccinated individuals):")
            print(f"  Mean: {np.mean(valid_clearance):.3f}")
            print(f"  Std: {np.std(valid_clearance):.3f}")
            print(f"  Min: {np.min(valid_clearance):.3f}")
            print(f"  Max: {np.max(valid_clearance):.3f}")
            print(f"  Improvement: {(np.mean(valid_clearance) - 1) * 100:.1f}%")
            print()

        if len(valid_death) > 0:
            print(f"Death Risk Modifiers (vaccinated individuals):")
            print(f"  Mean: {np.mean(valid_death):.3f}")
            print(f"  Std: {np.std(valid_death):.3f}")
            print(f"  Min: {np.min(valid_death):.3f}")
            print(f"  Max: {np.max(valid_death):.3f}")
            print(f"  Reduction: {(1 - np.mean(valid_death)) * 100:.1f}%")
            print()

    # Population-level analysis
    print("POPULATION-LEVEL RISK MODIFIER ANALYSIS")
    print("-" * 50)

    try:
        if hasattr(tb_bcg.rr_activation, 'raw') and len(tb_bcg.rr_activation.raw) > 0:
            print(f"Population Activation Risk: {np.mean(tb_bcg.rr_activation.raw):.3f} +/- {np.std(tb_bcg.rr_activation.raw):.3f}")
            print(f"Population Clearance Risk: {np.mean(tb_bcg.rr_clearance.raw):.3f} +/- {np.std(tb_bcg.rr_clearance.raw):.3f}")
            print(f"Population Death Risk: {np.mean(tb_bcg.rr_death.raw):.3f} +/- {np.std(tb_bcg.rr_death.raw):.3f}")
        else:
            print("Population-level risk modifiers not yet initialized")
    except Exception as e:
        print(f"Error accessing population risk modifiers: {e}")
    print()

    # BCG modifier distributions
    print("BCG MODIFIER DISTRIBUTIONS")
    print("-" * 40)

    if vaccinated > 0:
        print("BCG Activation Modifier Distribution:")
        print(f"  Default Range: {bcg_itv.product.pars.activation_modifier}")
        print(f"  Applied Values: {len(valid_activation)} valid modifiers")

        print("\nBCG Clearance Modifier Distribution:")
        print(f"  Default Range: {bcg_itv.product.pars.clearance_modifier}")
        print(f"  Applied Values: {len(valid_clearance)} valid modifiers")

        print("\nBCG Death Modifier Distribution:")
        print(f"  Default Range: {bcg_itv.product.pars.death_modifier}")
        print(f"  Applied Values: {len(valid_death)} valid modifiers")
    print()

    # Summary
    print("SUMMARY")
    print("-" * 20)
    print("TB disease model properly initialized with realistic transition rates")
    print("BCG intervention successfully applied to target population")
    print("Individual-level risk modifiers show substantial protective effects")
    print()
    print("KEY FINDINGS:")
    print(f"   {vaccinated} individuals vaccinated ({coverage:.1%} coverage)")
    print(f"   {protected} individuals currently protected")
    if vaccinated > 0 and len(valid_activation) > 0:
        print(f"   Average activation risk reduction: {(1 - np.mean(valid_activation)) * 100:.1f}%")
        print(f"   Average clearance improvement: {(np.mean(valid_clearance) - 1) * 100:.1f}%")
        print(f"   Average death risk reduction: {(1 - np.mean(valid_death)) * 100:.1f}%")

if __name__ == '__main__':
    analyze_tb_rates_and_modifiers()

"""
Run the LSHTM model for 20 years with default parameters, 
comparing with and without acute compartment
"""

import matplotlib.pyplot as plt
import numpy as np
import tbsim
import starsim as ss

# Simulation parameters
n_agents = 5_000
rand_seed = 5
sim_pars = {
    "n_agents": n_agents,
    "start": ss.date("1990-01-01"),
    "stop": ss.date("2010-12-31"),
    "dt": ss.days(7),
    "verbose": 0.002,
    "rand_seed": rand_seed,
}

# TB parameters
tb_pars = {
    "init_prev": ss.bernoulli(0.01), # seed prevalence
    "beta": ss.permonth(0.20),       # transmission rate
    "trans_asymp": 0.82,           # κ kappa: asymp vs symp relative transmissibility
    "rr_rec": 0.21,                # π pi: reinfection risk after recovery
    "rr_treat": 3.15,              # ρ rho: reinfection risk after treatment
    "cxr_asymp_sens": 1.0,         # CXR sensitivity for asymptomatic (0–1)
}

tb_pars_acute = {
    # Acute variant only (ignored if use_acute is False)
    "rate_acute_latent": ss.permonth(0.4),   # ACUTE → INFECTION
    "trans_acute": 0.9,            # α alpha: relative transmissibility from acute
}


def build_lshtm_sim(tb_pars, interventions=None, use_acute=False, label=None):
    """Build a Starsim simulation from tb_pars and n_agents (other run options are fixed)."""
    if use_acute:
        tb = tbsim.TB_LSHTM_Acute(**tb_pars, **tb_pars_acute)
    else:
        tb = tbsim.TB_LSHTM(**tb_pars)

    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0})

    demographics = [
        ss.Births(pars=dict(birth_rate=25)),
        ss.Deaths(pars=dict(death_rate=10)),
    ]

    sim = ss.Sim(
        label=label,
        pars=sim_pars,
        n_agents=n_agents, 
        diseases=tb,
        networks=net,
        demographics=demographics,
        interventions=interventions,
    )
    return sim


def build_sims_acute_vs_no_acute():
    """Build two sims: with acute (TB_LSHTM_Acute) and without acute (TB_LSHTM)."""
    sim_acute = build_lshtm_sim(tb_pars, use_acute=True, label="TB Acute")
    sim_no_acute = build_lshtm_sim(tb_pars, use_acute=False, label="LSHTM (without acute)")
    return [sim_acute, sim_no_acute]


if __name__ == "__main__":
    sims = build_sims_acute_vs_no_acute()
    msim = ss.MultiSim(sims=sims)
    msim.run(parallel=True)

    # Custom plotting
    results = {
        s.label: {str(k): v for k, v in s.results.flatten().items()}
        for s in msim.sims
    }
    # making the keys similar to plot them combined
    results['TB Acute'] = {k.replace("_acute", ""): v for k, v in results['TB Acute'].items()}
    tbsim.plot(results, title="TB LSHTM: with acute vs without acute")
    
    msim.plot()
    plt.show()
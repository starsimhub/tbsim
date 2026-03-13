"""
Run the LSHTM model for 20 years with default parameters,
comparing with and without acute compartment.
"""

import matplotlib.pyplot as plt
import starsim as ss
import tbsim

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
    "init_prev": ss.bernoulli(0.01),  # seed prevalence
    "beta": ss.peryear(0.20),         # transmission rate
    "trans_asymp": 0.82,              # κ kappa: asymp vs symp relative transmissibility
    "rr_reinfection_rec": 0.21,       # π pi: reinfection risk after NON_INFECTIOUS → CLEARED
    "rr_reinfection_treat": 3.15,     # ρ rho: reinfection risk after TREATMENT → CLEARED
    "cxr_asymp_sens": 1.0,            # CXR sensitivity for asymptomatic (0–1)
}

tb_pars_acute = {
    "rate_acute_latent": ss.peryear(4.0),  # ACUTE → INFECTION
    "trans_acute": 0.9,                    # α alpha: relative transmissibility from acute
}


def build_lshtm_sim(tb_pars, interventions=None, use_acute=False, label=None):
    """Build a Starsim simulation from tb_pars (other run options are fixed)."""
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
    """Build two sims: with acute (TB_LSHTM_Acute) and without (TB_LSHTM)."""
    sim_acute = build_lshtm_sim(tb_pars, use_acute=True, label="TB Acute")
    sim_no_acute = build_lshtm_sim(tb_pars, use_acute=False, label="LSHTM (without acute)")
    return [sim_acute, sim_no_acute]


if __name__ == "__main__":
    sims = build_sims_acute_vs_no_acute()
    msim = ss.MultiSim(sims=sims)
    msim.run(parallel=True)

    # Plot all sims together – pandas-like metric selection
    tbsim.plot(msim, select=dict(like='in'), n_cols=3,
               title="BOTH SIMS – filtered metrics (like='in')",
               savefig=True, filename="scenarios_filtered.png")

    # Plot all metrics across both sims
    tbsim.plot(msim, title="BOTH SIMS – all metrics", savefig=True, filename="scenarios_all.png")

    # Plot each sim individually via the disease's plot() method
    tbsim.get_tb(msim.sims[0]).plot(n_cols=6,
                                    title="TB LSHTM ACUTE SIM ONLY",
                                    savefig=True, filename="scenarios_acute.png")

    tbsim.get_tb(msim.sims[1]).plot(n_cols=6,
                                    title="TB LSHTM (NON-ACUTE) SIM ONLY",
                                    savefig=True, filename="scenarios_no_acute.png")

    plt.show()

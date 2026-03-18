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
sim_pars = dict(
    n_agents=n_agents,
    start=ss.date("1990-01-01"),
    stop=ss.date("2010-12-31"),
    dt=ss.days(7),
    verbose=0.002,
    rand_seed=rand_seed,
)

# TB parameters
tb_pars = dict(
    init_prev=ss.bernoulli(0.01),
    beta=ss.peryear(0.20),
    trans_asymp=0.82,
    rr_reinfection_rec=0.21,
    rr_reinfection_treat=3.15,
    cxr_asymp_sens=1.0,
)

tb_pars_acute = dict(
    rate_acute_latent=ss.peryear(4.0),
    trans_acute=0.9,
)


def build_sim(tb_pars, interventions=None, use_acute=False, label=None):
    """Build a tbsim simulation comparing acute vs non-acute TB models."""
    tb_model = 'acute' if use_acute else 'default'
    all_tb_pars = {**tb_pars, **(tb_pars_acute if use_acute else {})}

    sim = tbsim.Sim(
        label=label,
        tb_model=tb_model,
        tb_pars=all_tb_pars,
        sim_pars=sim_pars,
        interventions=interventions,
    )
    return sim


def build_sims_acute_vs_no_acute():
    """Build two sims: with acute (TBAcute) and without (TB)."""
    sim_acute = build_sim(tb_pars, use_acute=True, label="TB Acute")
    sim_no_acute = build_sim(tb_pars, use_acute=False, label="TB (without acute)")
    return [sim_acute, sim_no_acute]


if __name__ == "__main__":
    sims = build_sims_acute_vs_no_acute()
    msim = ss.MultiSim(sims=sims)
    msim.run(parallel=True)

    # Plot all sims together – pandas-like metric selection
    tbsim.plot(msim, select=dict(like='in'), n_cols=3,
               title="BOTH SIMS – filtered metrics (like='in')",
               filename="scenarios_filtered.png")

    # Plot all metrics across both sims
    tbsim.plot(msim, title="BOTH SIMS – all metrics", filename="scenarios_all.png")

    # Plot each sim individually via the disease's plot() method
    tbsim.get_tb(msim.sims[0]).plot(n_cols=6,
                                    title="TB LSHTM ACUTE SIM ONLY",
                                    filename="scenarios_acute.png")

    tbsim.get_tb(msim.sims[1]).plot(n_cols=6,
                                    title="TB LSHTM (NON-ACUTE) SIM ONLY",
                                    filename="scenarios_no_acute.png")

    plt.show()

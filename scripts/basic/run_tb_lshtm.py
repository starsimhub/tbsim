
import matplotlib.pyplot as plt
import numpy as np
import tbsim as mtb
import starsim as ss

n_agents = 5_000
rand_seed = 5
sp = {
    "start": ss.date("1990-01-01"),
    "stop": ss.date("2010-12-31"),
    "dt": ss.days(7),
    "verbose": 0.002,
    "rand_seed": rand_seed,
}

tb_params = {
    "use_acute": True,             # True → TB_LSHTM_Acute
    "init_prev": ss.bernoulli(0.01), # seed prevalence
    "beta": ss.peryear(0.20), # transmission rate
    "kappa": 0.82,            # asymp vs symp relative transmissibility      
    "pi": 0.21,               # reinfection risk after recovery (non-infectious)
    "rho": 3.15,              # reinfection risk after treatment
    "cxr_asymp_sens": 1.0,    # CXR sensitivity for asymptomatic (0–1)  
    
    # Acute variant only (ignored if use_acute is False)
    "acu_inf": ss.years(ss.expon(1 / 4.0)),  # 1/4 years to infectious
    "alpha": 0.9,                             # relative transmissibility from acute
}


def build_lshtm_sim(tb_params, n_agents, interventions=None):
    """Build a Starsim simulation from tb_params and n_agents (other run options are fixed)."""
    use_acute = tb_params.get("use_acute", False)
    skip = {"use_acute"}
    if not use_acute:
        skip |= {"acu_inf", "alpha"}  # base model doesn't use these
    infection_pars = {k: v for k, v in tb_params.items() if k not in skip}
    if use_acute:
        infection = mtb.TB_LSHTM_Acute(pars=infection_pars)
    else:
        infection = mtb.TB_LSHTM(pars=infection_pars)

    pop = ss.People(n_agents=n_agents)
    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0})

    kwargs = dict(people=pop, networks=net, diseases=infection, pars=sp)
    if interventions is not None:
        kwargs["interventions"] = interventions
    demographics = [
        ss.Births(pars=dict(birth_rate=25)),
        ss.Deaths(pars=dict(death_rate=10)),
    ]
    kwargs["demographics"] = demographics
    sim = ss.Sim(**kwargs)
    return sim

def build_sims_acute_vs_no_acute():
    """Build two sims: with acute (TB_LSHTM_Acute) and without acute (TB_LSHTM)."""
    params_with_acute = {**tb_params, "use_acute": True}
    params_no_acute = {**tb_params, "use_acute": False}
    sim_acute = build_lshtm_sim(params_with_acute, n_agents)
    sim_acute.label = "TB Acute"
    sim_no_acute = build_lshtm_sim(params_no_acute, n_agents)
    sim_no_acute.label = "LSHTM (without acute)"
    return [sim_acute, sim_no_acute]

if __name__ == "__main__":
    sims = build_sims_acute_vs_no_acute()
    msim = ss.MultiSim(sims=sims, label="TB_LSHTM_acute_vs_no_acute")
    msim.run(parallel=True, shrink=False, reseed=False)
    results = {
        s.label: {str(k): v for k, v in s.results.flatten().items()}
        for s in msim.sims
    }
    # making the keys similar to plot them combined
    results['TB Acute'] = {k.replace("_acute", ""): v for k, v in results['TB Acute'].items()}
    mtb.plot_combined(results, title="TB LSHTM: with acute vs without acute")
    plt.show()
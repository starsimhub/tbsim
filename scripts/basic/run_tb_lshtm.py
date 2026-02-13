
import matplotlib.pyplot as plt
import numpy as np
import tbsim as mtb
import starsim as ss

# "default" | "risk_modifiers" | "both" (run both and plot combined)
RUN_MODE = "both"  # "default" | "risk_modifiers" | "both"  

tb_params = {
    "use_acute": False,             # True → TB_LSHTM_Acute
    "init_prev": ss.bernoulli(0.01), # seed prevalence
    "beta": ss.peryear(0.25), # transmission rate
    "kappa": 0.82,            # asymp vs symp relative transmissibility      
    "pi": 0.21,               # reinfection risk after recovery (non-infectious)
    "rho": 3.15,              # reinfection risk after treatment
    "cxr_asymp_sens": 1.0,    # CXR sensitivity for asymptomatic (0–1)  
    
    # Acute variant only (ignored if use_acute is False)
    "acu_inf": ss.years(ss.expon(1 / 4.0)),  # 1/4 years to infectious
    "alpha": 0.9,                             # relative transmissibility from acute
}

n_agents = 5_000
rand_seed = 5

# Pleae note, this is not a recommended way to apply risk modifiers. 
# It is only used for demonstration purposes.
modifier_fraction = 0.8  # fraction of agents to apply risk modifiers to
modifier_rr = dict(rr_activation=1.5, rr_clearance=2.5, rr_death=0.5)


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
    sp = {
        "start": ss.date("1990-01-01"),
        "stop": ss.date("2010-12-31"),
        "dt": ss.days(7),
        "verbose": 0.002,
        "rand_seed": rand_seed,
    }
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

def apply_risk_modifiers(tb):
    """Set rr_activation, rr_clearance, rr_death on a random fraction of agents."""
    n = len(tb.state)
    rng = np.random.default_rng(rand_seed)
    mask = rng.random(n) < modifier_fraction
    tb.rr_activation[mask] = modifier_rr["rr_activation"]
    tb.rr_clearance[mask] = modifier_rr["rr_clearance"]
    tb.rr_death[mask] = modifier_rr["rr_death"]


def run_default():
    """Run LSHTM with default parameters."""
    sim = build_lshtm_sim(tb_params, n_agents)
    sim.run()
    res = sim.diseases[0].results
    print(f"Default: infectious={res['n_infectious'][-1]:.0f}, prevalence={res['prevalence_active'][-1]:.4f}, incidence_kpy={res['incidence_kpy'][-1]:.2f}")
    return sim


def run_risk_modifiers_sample():
    """Run LSHTM with same params as default, plus risk modifiers on a fraction of agents."""
    sim = build_lshtm_sim(tb_params, n_agents)
    sim.init()
    apply_risk_modifiers(sim.diseases[0])
    sim.run()
    res = sim.diseases[0].results
    print(f"Risk modifiers: n_infectious={res['n_infectious'][-1]:.0f}, prevalence={res['prevalence_active'][-1]:.4f}, cum_active={res['cum_active'][-1]:.0f}, cum_deaths={res['cum_deaths'][-1]:.0f}")
    return sim


def build_sims_for_both():
    """Build two sims with identical tb_params and n_agents; second has risk modifiers applied after init."""
    sim_default = build_lshtm_sim(tb_params, n_agents)
    sim_default.label = "LSHTM (default)"
    sim_modifiers = build_lshtm_sim(tb_params, n_agents)
    sim_modifiers.label = "LSHTM (risk modifiers)"
    sim_modifiers.init()
    apply_risk_modifiers(sim_modifiers.diseases[0])
    return [sim_default, sim_modifiers]


if __name__ == "__main__":
    if RUN_MODE == "both":
        sims = build_sims_for_both()
        msim = ss.MultiSim(sims=sims, label="TB_LSHTM_both")
        msim.run(parallel=True, shrink=False, reseed=False)
        results = {
            s.label: {str(k): v for k, v in s.results.flatten().items()}
            for s in msim.sims
        }
        mtb.plot_combined(results, title="TB LSHTM: default vs risk modifiers", dark=False, heightfold=1.5)
        plt.show()
    elif RUN_MODE == "risk_modifiers":
        sim = run_risk_modifiers_sample()
        results = {"LSHTM (risk modifiers)": {str(k): v for k, v in sim.results.flatten().items()}}
        mtb.plot_combined(results, title="TB LSHTM (risk modifiers)", dark=False, heightfold=1.5)
        plt.show()
    else:
        sim = run_default()
        results = {"LSHTM": {str(k): v for k, v in sim.results.flatten().items()}}
        mtb.plot_combined(results, title="TB LSHTM", dark=False, heightfold=1.5)
        plt.show()
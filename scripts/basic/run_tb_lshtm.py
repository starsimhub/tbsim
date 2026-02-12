"""
LSHTM TB sample script (TB_LSHTM / TB_LSHTM_Acute).

Edit tb_params and sim_params below; build_lshtm_sim(tb_params, sim_params) uses them.
"""

import matplotlib.pyplot as plt

import tbsim as mtb
import starsim as ss


# =============================================================================
# tb_params — passed to TB_LSHTM(pars=...) or TB_LSHTM_Acute(pars=...)
# =============================================================================

tb_params = {
    "use_acute": False,             # True → TB_LSHTM_Acute
    "init_prev": ss.bernoulli(0.01), # seed prevalence
    "beta": ss.peryear(0.25), # transmission rate
    "kappa": 0.82,            # asymp vs symp relative transmissibility      
    "pi": 0.21,               # reinfection risk after recovery (unconfirmed)
    "rho": 3.15,              # reinfection risk after treatment
    "cxr_asymp_sens": 1.0,    # CXR sensitivity for asymptomatic (0–1)  
    # Acute variant only (ignored if use_acute is False)
    "acuinf": ss.years(ss.expon(1 / 4.0)),   # 1/4 years to infectious
    "alpha": 0.9,                            # reinfection risk after treatment
}

# =============================================================================
# sim_params — run setup, network, Sim pars, demographics
# =============================================================================

sim_params = {
    # Run
    "n_agents": 5_000,
    "start_year": 1990,
    "years": 20,
    "rand_seed": 5,
    "plot": True,
    # Network (RandomNet)
    "n_contacts": ss.poisson(lam=5),
    "dur": 0,
    # Sim
    "dt": ss.days(7),
    "verbose": 0.002,
    # Demographics (Births, Deaths); set with_demographics False to disable
    "with_demographics": True,
    "birth_rate": 25,
    "death_rate": 10,
}


def build_lshtm_sim(tb_params, sim_params):
    """Build a Starsim simulation from tb_params and sim_params."""
    use_acute = tb_params.get("use_acute", False)
    skip = {"use_acute"}
    if not use_acute:
        skip |= {"acuinf", "alpha"}  # base model doesn't use these
    infection_pars = {k: v for k, v in tb_params.items() if k not in skip}
    if use_acute:
        infection = mtb.TB_LSHTM_Acute(pars=infection_pars)
    else:
        infection = mtb.TB_LSHTM(pars=infection_pars)

    pop = ss.People(n_agents=sim_params["n_agents"])
    net = ss.RandomNet(pars={"n_contacts": sim_params["n_contacts"], "dur": sim_params["dur"]})

    start_year = sim_params["start_year"]
    stop_year = start_year + sim_params["years"]
    sp = {
        "start": ss.date(f"{start_year}-01-01"),
        "stop": ss.date(f"{stop_year}-12-31"),
        "dt": sim_params["dt"],
        "verbose": sim_params["verbose"],
    }
    if sim_params.get("rand_seed") is not None:
        sp["rand_seed"] = sim_params["rand_seed"]

    if sim_params.get("with_demographics", True):
        demographics = [
            ss.Births(pars=dict(birth_rate=sim_params["birth_rate"])),
            ss.Deaths(pars=dict(death_rate=sim_params["death_rate"])),
        ]
        sim = ss.Sim(people=pop, networks=net, diseases=infection, demographics=demographics, pars=sp)
    else:
        sim = ss.Sim(people=pop, networks=net, diseases=infection, pars=sp)

    return sim


def main():
    """Build sim from tb_params and sim_params, run, print summary; show plot if sim_params['plot']."""
    sim = build_lshtm_sim(tb_params, sim_params)

    sim.run()
    print("Done.")

    model = sim.diseases[0]
    res = model.results
    idx = -1
    print(f"  Infectious (final): {res['n_infectious'][idx]:.0f}")
    print(f"  Prevalence active (final): {res['prevalence_active'][idx]:.4f}")
    print(f"  Incidence (final, per 1000 py): {res['incidence_kpy'][idx]:.2f}")

    if sim_params["plot"]:
        flat = sim.results.flatten()
        flat_str = {str(k): v for k, v in flat.items()}
        results = {"TB LSHTM": flat_str}
        mtb.plot_combined(results, title="TB LSHTM model", dark=False, heightfold=1.5)
        plt.show()


if __name__ == "__main__":
    main()

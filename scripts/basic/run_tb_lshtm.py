"""
LSHTM TB sample script (TB_LSHTM / TB_LSHTM_Acute).

Runs a Starsim simulation with the LSHTM TB module, optional Births/Deaths,
and a random contact network. All parameters are defined at the top:

  RUN_PARS          — run behavior (variant, n_agents, years, plot, …)
  DEFAULT_TB_PARS   — TB model (init_prev, beta, kappa, …)
  DEFAULT_ACUTE_PARS — acute variant (acuinf, alpha)
  DEFAULT_NET_PARS  — RandomNet (n_contacts, dur)
  DEFAULT_SIM_PARS  — dt, verbose
  DEFAULT_DEMO_PARS — birth_rate, death_rate

Usage: edit RUN_PARS (and optionally DEFAULT_*), then run:
  python run_tb_lshtm.py
"""

import matplotlib.pyplot as plt

import tbsim as mtb
import starsim as ss


# -----------------------------------------------------------------------------
# Run parameters (edit these to change run behavior)
# -----------------------------------------------------------------------------

RUN_PARS = {
    "use_acute": False,
    "with_demographics": True,
    "n_agents": 5_000,
    "start_year": 1990,
    "years": 20,
    "rand_seed": None,
    "plot": True,
}

# -----------------------------------------------------------------------------
# TB and network/sim/demographics defaults (merged with user overrides in build_lshtm_sim)
# -----------------------------------------------------------------------------
# TB model parameters; names and types must match tbsim.models.tb_lshtm.TB_LSHTM.define_pars.
DEFAULT_TB_PARS = {
    "init_prev": ss.bernoulli(0.01),   # Initial prevalence (seed infections)
    "beta": ss.peryear(0.25),          # Transmission rate per year
    "kappa": 0.82,                     # Relative transmission (asymptomatic vs symptomatic)
    "pi": 0.21,                        # Relative risk reinfection after recovery (unconfirmed)
    "rho": 3.15,                       # Relative risk reinfection after treatment completion
    "cxr_asymp_sens": 1.0,             # CXR sensitivity for screening asymptomatic (0–1)
}

# Extra parameters for TB_LSHTM_Acute only; merged when use_acute=True.
DEFAULT_ACUTE_PARS = {
    "acuinf": ss.years(ss.expon(1 / 4.0)),  # Rate ACUTE → INFECTION per year
    "alpha": 0.9,                           # Relative transmission from acute vs symptomatic
}

# ss.RandomNet parameters: n_contacts (e.g. ss.poisson), dur (contact duration).
DEFAULT_NET_PARS = {
    "n_contacts": ss.poisson(lam=5),
    "dur": 0,
}

# Sim pars: dt (time step), verbose (0–1). start/stop set from start_year/stop_year in build_lshtm_sim.
DEFAULT_SIM_PARS = {
    "dt": ss.days(7),
    "verbose": 0.1,
}

# Demographics for ss.Births and ss.Deaths: rates per 1000 population per year.
DEFAULT_DEMO_PARS = {
    "birth_rate": 25,
    "death_rate": 10,
}


def build_lshtm_sim(
    n_agents=10_000,
    start_year=1990,
    stop_year=2010,
    dt=None,
    use_acute=False,
    with_demographics=True,
    tb_pars=None,
    net_pars=None,
    sim_pars=None,
    demo_pars=None,
    verbose=0.1,
):
    """Build a Starsim simulation with TB_LSHTM or TB_LSHTM_Acute. *pars are merged with DEFAULT_*; user overrides win."""
    # TB disease: merge defaults with user overrides (user overrides last)
    tb = dict(DEFAULT_TB_PARS)
    if use_acute:
        tb.update(DEFAULT_ACUTE_PARS)
    tb.update(tb_pars or {})
    if use_acute:
        infection = mtb.TB_LSHTM_Acute(pars=tb)
    else:
        infection = mtb.TB_LSHTM(pars=tb)

    pop = ss.People(n_agents=n_agents)

    net_p = dict(DEFAULT_NET_PARS)
    net_p.update(net_pars or {})
    net = ss.RandomNet(pars=net_p)

    sp = dict(DEFAULT_SIM_PARS, verbose=verbose)
    sp["start"] = ss.date(f"{start_year}-01-01")
    sp["stop"] = ss.date(f"{stop_year}-12-31")
    if dt is not None:
        sp["dt"] = dt
    sp.update(sim_pars or {})

    if with_demographics:
        dp = dict(DEFAULT_DEMO_PARS)
        dp.update(demo_pars or {})
        demographics = [
            ss.Births(pars=dict(birth_rate=dp["birth_rate"])),
            ss.Deaths(pars=dict(death_rate=dp["death_rate"])),
        ]
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=infection,
            demographics=demographics,
            pars=sp,
        )
    else:
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=infection,
            pars=sp,
        )

    return sim


def main():
    """Build sim from RUN_PARS and DEFAULT_*, run, print summary; if RUN_PARS['plot'], call mtb.plot_combined and plt.show()."""
    r = RUN_PARS
    stop_year = r["start_year"] + r["years"]
    sim_pars = {}
    if r["rand_seed"] is not None:
        sim_pars["rand_seed"] = r["rand_seed"]

    sim = build_lshtm_sim(
        n_agents=r["n_agents"],
        start_year=r["start_year"],
        stop_year=stop_year,
        use_acute=r["use_acute"],
        with_demographics=r["with_demographics"],
        sim_pars=sim_pars if sim_pars else None,
        verbose=DEFAULT_SIM_PARS["verbose"],
    )

    sim.run()
    print("Done.")

    model = sim.diseases[0]
    res = model.results
    idx = -1
    print(f"  Infectious (final): {res['n_infectious'][idx]:.0f}")
    print(f"  Prevalence active (final): {res['prevalence_active'][idx]:.4f}")
    print(f"  Incidence (final, per 1000 py): {res['incidence_kpy'][idx]:.2f}")

    if r["plot"]:
        flat = sim.results.flatten()
        flat_str = {str(k): v for k, v in flat.items()}
        results = {"TB LSHTM": flat_str}
        mtb.plot_combined(results, title="TB LSHTM model", dark=False, heightfold=1.5)
        plt.show()


if __name__ == "__main__":
    main()

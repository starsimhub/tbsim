"""
Sample script: run the LSHTM TB model with modified transition rates.

Rates in TB_LSHTM are exponential waiting times (in years). Each rate is
built as ss.years(ss.expon(mean_years)), so the mean time until transition
is mean_years. To change a rate, pass a replacement in the pars dict when
creating TB_LSHTM(pars={...}).

This script defines a helper to build rates from mean years, then runs
two short simulations: default rates vs modified rates (faster treatment
completion, lower TB mortality), and prints a simple comparison.
"""

import tbsim as mtb
import starsim as ss


def rate_from_mean_years(mean_years):
    """
    Build an LSHTM rate (exponential waiting time) with given mean in years.

    The model uses ss.years(ss.expon(1/mean_years)), so the mean waiting time
    is mean_years. Use this to override model rates. Example: faster treatment
    completion (mean 1 year instead of 2):
        pars["delta"] = rate_from_mean_years(1.0)
    """
    return ss.years(ss.expon(1 / mean_years))


# Default mean times (years) for each transition rate (from tb_lshtm.py docstring)
DEFAULT_MEAN_YEARS = {
    "infcle": 1.90,   # latent → cleared
    "infunc": 0.16,   # latent → unconfirmed
    "infasy": 0.06,   # latent → asymptomatic
    "uncrec": 0.18,   # unconfirmed → recovered
    "uncasy": 0.25,   # unconfirmed → asymptomatic
    "asyunc": 1.66,   # asymptomatic → unconfirmed
    "asysym": 0.88,   # asymptomatic → symptomatic
    "symasy": 0.54,   # symptomatic → asymptomatic
    "theta": 0.46,    # symptomatic → start treatment
    "mutb": 0.34,     # symptomatic → TB death
    "phi": 0.63,      # treatment → failure (symptomatic)
    "delta": 2.00,    # treatment → completion (treated)
    "mu": 0.014,      # background mortality
}


def make_pars_with_modified_rates(overrides):
    """
    Build a pars dict that replaces only the specified rates.

    overrides : dict
        Map rate name -> new mean years.
        Example: {"delta": 1.0, "mutb": 0.5} for faster treatment, lower TB mortality.
    """
    pars = {}
    for name, mean_y in overrides.items():
        if name not in DEFAULT_MEAN_YEARS:
            raise KeyError(f"Unknown rate {name}; known: {list(DEFAULT_MEAN_YEARS.keys())}")
        pars[name] = rate_from_mean_years(mean_y)
    return pars


def run_sim(pars=None, n_agents=2000, years=10, seed=42):
    """Run TB_LSHTM and return the sim and disease module."""
    pop = ss.People(n_agents=n_agents)
    tb = mtb.TB_LSHTM(pars=pars)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        pars=dict(
            dt=ss.days(7),
            start=ss.date("2000-01-01"),
            stop=ss.date(f"{2000 + years}-12-31"),
            rand_seed=seed,
            verbose=0,
        ),
    )
    sim.run()
    return sim, sim.diseases[0]


def main():
    print("LSHTM TB model: modifying transition rates")
    print("=" * 60)

    # 1) Default rates
    print("\n1) Running with default rates ...")
    sim_default, tb_default = run_sim(pars=None)
    res_d = tb_default.results
    print(f"   Final n_infectious:    {res_d['n_infectious'][-1]:.0f}")
    print(f"   Final prevalence:      {res_d['prevalence_active'][-1]:.4f}")
    print(f"   Cumulative active:    {res_d['cum_active'][-1]:.0f}")
    print(f"   Cumulative TB deaths: {res_d['cum_deaths'][-1]:.0f}")

    # 2) Modified rates: faster treatment completion, lower TB mortality
    modified_pars = make_pars_with_modified_rates({
        "delta": 1.0,   # mean 1 year to complete treatment (default 2)
        "mutb": 0.50,   # mean 0.5 year to TB death (default ~0.34 → slightly slower mortality)
    })
    # Also set transmission so runs are comparable
    modified_pars["init_prev"] = ss.bernoulli(0.02)
    modified_pars["beta"] = ss.peryear(0.25)

    print("\n2) Running with modified rates (faster delta, slower mutb) ...")
    sim_mod, tb_mod = run_sim(pars=modified_pars)
    res_m = tb_mod.results
    print(f"   Final n_infectious:    {res_m['n_infectious'][-1]:.0f}")
    print(f"   Final prevalence:      {res_m['prevalence_active'][-1]:.4f}")
    print(f"   Cumulative active:    {res_m['cum_active'][-1]:.0f}")
    print(f"   Cumulative TB deaths: {res_m['cum_deaths'][-1]:.0f}")

    print("\n3) Example: build custom pars with only some rates changed")
    custom = make_pars_with_modified_rates({"theta": 0.25, "delta": 1.5})
    print("   custom = make_pars_with_modified_rates({'theta': 0.25, 'delta': 1.5})")
    print("   tb = mtb.TB_LSHTM(pars=custom)")
    print("   (theta = faster start treatment; delta = faster treatment completion)")
    print("\nDone.")


if __name__ == "__main__":
    main()

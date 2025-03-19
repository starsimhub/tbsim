import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import scipy.stats as stats

TBS = mtb.TBS

def build_tbsim(sim_pars=None):
    """Build and return a TB simulation with default or custom parameters, including demographics."""
    default_pars = {
        "start": sc.date("2013-01-01"),
        "stop": sc.date("2040-12-31"),
        "rand_seed": 123,
        "unit": "day",
        "dt": 7,
    }
    sim_pars = {**default_pars, **(sim_pars or {})}

    # Define population
    pop = ss.People(n_agents=1000)

    # Define TB parameters and model
    tb_params = dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
        unit="day"
    )
    tb = mtb.TB(tb_params)

    # Define network
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))

    # Add demographics
    births = ss.Births(pars=dict(birth_rate=5))
    deaths = ss.Deaths(pars=dict(death_rate=5))

    # Define Analyzer
    dwell_analyzer = mtb.DwtAnalyzer(adjust_to_unit=True, unit=1.0, scenario_name="run_TB_Dwell_analyzer")

    # Build the simulation
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[births, deaths],  # Add demographic processes
        pars=sim_pars,
        analyzers=dwell_analyzer,
    )
    sim.pars.verbose = 30 / 365

    return sim


def calculate_expected_distributions(start, stop):
    """Precompute truncated exponential CDFs for TB state durations."""
    duration_days = (stop - start).days
    min_value = 150  # Minimum dwell time
    max_value = duration_days  # Maximum dwell time

    scales = {
        TBS.NONE: 126,
        TBS.LATENT_SLOW: 365,
        TBS.LATENT_FAST: 200,
        TBS.ACTIVE_PRESYMP: 100,
        TBS.ACTIVE_SMPOS: 150,
        TBS.ACTIVE_SMNEG: 300,
        TBS.ACTIVE_EXPTB: 250,
        TBS.DEAD: 400,
    }

    # Precompute the truncated exponential distributions
    distributions = {
        state: stats.truncexpon(
            b=(max_value - min_value) / scale, loc=min_value, scale=scale
        )
        for state, scale in scales.items()
    }

    # Return precomputed CDFs as functions
    return {state: dist.cdf for state, dist in distributions.items()}


if __name__ == "__main__":
    # Run TB simulation with demographics
    sim_tb = build_tbsim()
    sim_tb.run()

    # Extract start/stop and compute expected distributions
    expected_distributions = calculate_expected_distributions(sim_tb.pars.start, sim_tb.pars.stop)

    # Access dwell time analyzer and generate visualizations
    ana: mtb.DwtAnalyzer = sim_tb.analyzers[0]
    ana.graph_state_transitions()
    ana.sankey_agents_by_age_subplots(bins=[0, 5, 200])

    # Process results from saved analyzer file
    plotter = mtb.DwtPlotter(file_path=ana.file_path)
    plotter.histogram_with_kde()

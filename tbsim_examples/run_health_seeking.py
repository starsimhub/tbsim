"""
Health-seeking behaviour with the LSHTM TB model.

Three scenarios are run over 20 years and compared:

  Baseline                  – no intervention
  Low rate (10 %/day)       – HealthSeekingBehavior at 10 % daily seek probability
  High rate (40 %/day)      – 40 % daily, with a retry window (~1 month)
"""

import matplotlib.pyplot as plt
import starsim as ss
import tbsim as mtb
from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior


N_AGENTS  = 3_000
RAND_SEED = 42

sim_pars = dict(
    start     = ss.date("1990-01-01"),
    stop      = ss.date("2010-12-31"),
    dt        = ss.days(7),
    rand_seed = RAND_SEED,
    verbose   = 0,
)

tb_pars = dict(
    init_prev   = ss.bernoulli(0.05),
    beta        = ss.peryear(0.20),
    trans_asymp = 0.82,
    rr_rec      = 0.21,
    rr_treat    = 3.15,
)


def build_sim(hsb=None):
    tb  = mtb.TB_LSHTM(pars=tb_pars)
    pop = ss.People(n_agents=N_AGENTS)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    kwargs = dict(
        people       = pop,
        networks     = net,
        diseases     = tb,
        demographics = [ss.Births(pars=dict(birth_rate=20)), ss.Deaths(pars=dict(death_rate=10))],
        pars         = sim_pars,
    )
    if hsb is not None:
        kwargs["interventions"] = hsb
    return ss.Sim(**kwargs)


scenarios = {
    "Baseline": build_sim(),

    "Low rate (10 %/day)": build_sim(
        HealthSeekingBehavior(pars=dict(
            initial_care_seeking_rate = ss.perday(0.10),
        ))
    ),

    "High rate (40 %/day) + retry": build_sim(
        HealthSeekingBehavior(pars=dict(
            initial_care_seeking_rate = ss.perday(0.40),
            care_retry_steps          = 4,
        ))
    ),
}


if __name__ == "__main__":
    msim = ss.MultiSim(sims=list(scenarios.values()))
    msim.run(parallel=True, reseed=False)

    results = {
        label: {str(k): v for k, v in sim.results.flatten().items()}
        for label, sim in zip(scenarios.keys(), msim.sims)
    }

    mtb.plot(
        results,
        select = dict(like=["symptomatic", "sought_care", "notifications", "prevalence", "incidence", "eligible"]),
        title    = "Health-seeking behaviour – LSHTM TB model",
        n_cols   = 3,
        output_dir   = "results",
        theme     = 'light',
    )
    plt.show()
    

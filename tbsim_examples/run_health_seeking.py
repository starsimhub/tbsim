"""Health-seeking sample using TB_LSHTM_Acute with prevalence maintenance."""

import matplotlib.pyplot as plt
import starsim as ss
import tbsim as mtb

N_AGENTS = 5_000

SIM_PARS = dict(
    start     = ss.date("1990-01-01"),
    stop      = ss.date("2010-12-31"),
    dt        = ss.days(7),
    verbose   = 0.002,
    rand_seed = 5,
)

TB_PARS = dict(
    init_prev         = ss.bernoulli(0.25),
    beta              = ss.peryear(0.9),
    rate_acute_latent = ss.years(ss.expon(1 / 4.0)),
    trans_acute       = 0.9,
)

def build_sim() -> ss.Sim:
    return ss.Sim(
        people=ss.People(n_agents=N_AGENTS),
        networks=ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0}),
        diseases=mtb.TB_LSHTM_Acute(pars=TB_PARS),
        interventions=[
            mtb.HealthSeekingBehavior(pars={
                "initial_care_seeking_rate": ss.perday(0.03),
                "care_retry_steps":          8,
                # "custom_states":           [mtb.TBSL.ASYMPTOMATIC], # could be used for ACF  
            }),
        ],
        demographics=[
            ss.Births(pars={"birth_rate": 25}),
            ss.Deaths(pars={"death_rate": 10}),
        ],
        pars=SIM_PARS,
    )


def print_summary(sim: ss.Sim) -> None:
    hs = sim.interventions["healthseekingbehavior"]
    r  = hs.results
    print("\n=== Health-seeking summary ===")
    print(f"  total seekers     : {int(r['n_ever_sought_care'][-1])}")
    print(f"  total new seekers : {int(sum(r['new_sought_care']))}")
    print(f"  peak eligible     : {int(max(r['n_eligible']))}")


def run_and_plot() -> None:
    sim = build_sim()
    sim.run()
    print_summary(sim)
    sim.plot()
    plt.show()


if __name__ == "__main__":
    run_and_plot()

"""Run TB_LSHTM with and without immigration (and plot)."""

import argparse
import numpy as np
import starsim as ss

import tbsim
from tbsim.interventions.immigration import Immigration


def build_sim(*, use_acute: bool, label: str, include_immigration: bool, n_agents: int, rand_seed: int):
    tb = tbsim.TB_LSHTM_Acute() if use_acute else tbsim.TB_LSHTM()
    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0})

    demographics = []
    if include_immigration:
        # TB state for new arrivals (TBSL state names)
        tb_state_distribution = dict(
            SUSCEPTIBLE=0.70,
            INFECTION=0.295,
            NON_INFECTIOUS=0.001,
            ASYMPTOMATIC=0.002,
            SYMPTOMATIC=0.002,
            CLEARED=0.0,
            RECOVERED=0.0,
            TREATED=0.0,
            TREATMENT=0.0,
        )
        if use_acute:
            # Only valid for TB_LSHTM_Acute
            tb_state_distribution["ACUTE"] = 0.0

        demographics = [
            Immigration(
                pars=dict(
                    immigration_rate=1_000,  # people/year
                    tb_state_distribution=tb_state_distribution,
                )
            )
        ]

    sim = ss.Sim(
        people=ss.People(n_agents=n_agents),
        diseases=tb,
        networks=net,
        demographics=demographics,
        pars=dict(
            start=ss.date("2020-01-01"),
            stop=ss.date("2021-01-01"),
            dt=ss.days(7),
            rand_seed=rand_seed,
            verbose=0.0,
        ),
        label=label,
    )
    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acute", action="store_true")
    parser.add_argument("--n-agents", type=int, default=5_000)
    parser.add_argument("--rand-seed", type=int, default=1)
    args = parser.parse_args()

    baseline = build_sim(
        use_acute=args.acute,
        label="Baseline",
        include_immigration=False,
        n_agents=args.n_agents,
        rand_seed=args.rand_seed,
    )
    with_imm = build_sim(
        use_acute=args.acute,
        label="With immigration",
        include_immigration=True,
        n_agents=args.n_agents,
        rand_seed=args.rand_seed,
    )

    msim = ss.MultiSim([baseline, with_imm], label="lshtm_immigration")
    msim.run(parallel=True, reseed=False, shrink=False)

    base, scen = msim.sims
    tb_base = next(iter(base.diseases.values()))
    tb_scen = next(iter(scen.diseases.values()))
    n_imm = int(np.sum(scen.demographics[0].results["n_immigrants"]))

    model = "TB_LSHTM_Acute" if args.acute else "TB_LSHTM"
    base_inf_end = int(np.count_nonzero(tb_base.infectious))
    scen_inf_end = int(np.count_nonzero(tb_scen.infectious))
    print(f"{model} | baseline infectious(end)={base_inf_end} | scenario infectious(end)={scen_inf_end} | immigrants added={n_imm}")

    flat_results = {
        "Baseline": base.results.flatten(),
        "With immigration": scen.results.flatten(),
    }
    tbsim.plot_combined(
        flat_results,
        filter=[
            "tb_lshtm_prevalence_active",
            "tb_lshtm_n_infectious",
            "immigration_n_immigrants",
            "immigration_n_is_immigrant",
        ],
        title="TB_LSHTM: baseline vs immigration",
        savefig=True,
    )


if __name__ == "__main__":
    main()


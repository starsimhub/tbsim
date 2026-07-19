"""
High Gap 02: Latent treatment divergence in TxDeliveryR behavior modes.

Purpose
- Demonstrate how latent-agent handling differs between treat_latent=False and
  treat_latent=True in resistance treatment delivery.
- Provide an easy, shareable script that prints mode-specific outcomes.

How to run
- From repo root:
  /Users/mine/git/tbsimFINAL/venv/bin/python \
    tbsim/resistance/codex_review/high_gap_scripts/high_gap_02_latent_treatment_divergence.py

What this script prints
- For each mode, total treated and total successful courses.
- End-of-run latent count.
- A concise interpretation to share with another developer.
"""

import starsim as ss
import tbsim
from tbsim.tb import TBS, get_tb


def run_mode(treat_latent):
    """Run one latent-targeted treatment scenario for a chosen treat_latent mode."""

    # SECTION A: Build a resistant TB model with no transmission to isolate treatment behavior.
    tb = tbsim.TBResistant(
        drugs=["TX"],
        rel_fitness={"TX": 0.9},
        pars=dict(beta=ss.permonth(0.0), init_prev=ss.bernoulli(0.5), init_strains=[0.5, 0.5]),
    )

    # SECTION B: Target latent agents directly via custom eligibility.
    tx = tbsim.TxDeliveryR(
        name="tx",
        eligibility=lambda sim: get_tb(sim, which=tbsim.TBResistant).latent.uids,
        product=tbsim.TxR(strains=tb.strains, base_efficacy=0.0, adherence=1.0),
        treat_latent=treat_latent,
    )

    # SECTION C: Run and collect outputs.
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=1), dur=0))
    sim = ss.Sim(
        n_agents=2000,
        networks=net,
        diseases=tb,
        interventions=[tx],
        dt=ss.days(30),
        start=ss.date("2000-01-01"),
        stop=ss.date("2001-06-30"),
        rand_seed=0,
        verbose=0,
    )
    sim.run()

    tb_live = sim.diseases["tb"]
    results = sim.results["tx"]

    return {
        "treated_total": int(results.n_treated.sum()),
        "success_total": int(results.n_success.sum()),
        "latent_end": int((tb_live.state == TBS.INFECTION).count()),
    }


def main():
    print("=== High Gap 02: Latent treatment divergence ===")

    # SECTION D1: Mode that clears latent immediately (no course count).
    mode_false = run_mode(treat_latent=False)
    print("\n[treat_latent=False]")
    print(f"Total treated courses: {mode_false['treated_total']}")
    print(f"Total successful courses: {mode_false['success_total']}")
    print(f"Latent agents at end: {mode_false['latent_end']}")

    # SECTION D2: Mode that routes latent agents through full course logic.
    mode_true = run_mode(treat_latent=True)
    print("\n[treat_latent=True]")
    print(f"Total treated courses: {mode_true['treated_total']}")
    print(f"Total successful courses: {mode_true['success_total']}")
    print(f"Latent agents at end: {mode_true['latent_end']}")

    # SECTION E: High-level interpretation for quick sharing.
    print("\n[Interpretation]")
    print("- The two modes produce materially different bookkeeping and dynamics.")
    print("- This demonstrates the latent-treatment behavior divergence gap.")


if __name__ == "__main__":
    main()

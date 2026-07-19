"""
High Gap 01: Default DST-linked retreatment still targets CLEARED agents.

Purpose
- Reproduce the high-severity issue where DST-routed treatment eligibility can
  repeatedly select agents who are already CLEARED.
- Show the default behavior and an optional guarded behavior side by side.

How to run
- From repo root:
  /Users/mine/git/tbsimFINAL/venv/bin/python \
    tbsim/resistance/codex_review/high_gap_scripts/high_gap_01_dst_default_retreatment.py

What this script prints
- CLEARED selection fraction under default settings.
- CLEARED selection fraction with optional safeguards enabled
  (result_validity + retreat_after).
- A brief interpretation block for quick sharing.
"""

from collections import Counter

import starsim as ss
import tbsim
from tbsim.tb import TBS


def run_scenario(use_guards=False):
    """Run one DST-routed scenario and return a state-count summary of selected starts."""
    log = {"states": []}
    original_initiate = tbsim.TxDeliveryR._initiate

    def probed_initiate(self):
        tb = tbsim.get_tb(self.sim, which=tbsim.TBResistant)

        # SECTION A: Record the states selected by eligibility for this step.
        if self.eligibility is not None:
            selected = ss.uids(self.eligibility(self.sim))
            selected = selected[tb.state[selected] != TBS.TREATMENT]
            log["states"].extend(int(tb.state[u]) for u in selected)

        return original_initiate(self)

    tbsim.TxDeliveryR._initiate = probed_initiate

    try:
        # SECTION B: Build disease, DST, and two DST-routed treatment lines.
        tb = tbsim.TBResistant(
            drugs=["RIF"],
            rel_fitness={"RIF": 0.9},
            pars=dict(
                beta=ss.permonth(0.35),
                init_prev=ss.bernoulli(0.12),
                init_strains=[0.8, 0.2],
                rr_reinfection_inf=1.0,
                rr_reinfection_non=1.0,
            ),
        )

        dst_kwargs = {}
        first_kwargs = {}
        second_kwargs = {}
        if use_guards:
            # Optional mitigation knobs available on this branch.
            dst_kwargs["result_validity"] = ss.months(1)
            first_kwargs["retreat_after"] = ss.months(1)
            second_kwargs["retreat_after"] = ss.months(1)

        dst = tbsim.DSTDelivery(
            name="dst",
            product=tbsim.DST(strains=tb.strains, sens=0.95, spec=0.98),
            eligibility=lambda sim: tbsim.get_tb(sim, which=tbsim.TBResistant).active_tb.uids,
            **dst_kwargs,
        )

        first = tbsim.TxDeliveryR(
            name="first",
            eligibility=dst.matches(RIF=False),
            product=tbsim.TxR(strains=tb.strains, base_efficacy=0.85, resist_penalty={"RIF": 0.1}),
            **first_kwargs,
        )

        second = tbsim.TxDeliveryR(
            name="second",
            eligibility=dst.matches(RIF=True),
            product=tbsim.TxR(strains=tb.strains, base_efficacy=0.8, regimen_drugs=["RIF"]),
            **second_kwargs,
        )

        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=8), dur=0))

        # SECTION C: Run long horizon to expose repeated-selection behavior.
        sim = ss.Sim(
            n_agents=2000,
            networks=net,
            diseases=tb,
            interventions=[dst, first, second],
            dt=ss.days(30),
            start=ss.date("2000-01-01"),
            stop=ss.date("2030-12-31"),
            rand_seed=0,
            verbose=0,
        )
        sim.run()

        counts = Counter(log["states"])
        total = sum(counts.values())
        cleared = counts.get(int(TBS.CLEARED), 0)
        frac_cleared = 0.0 if total == 0 else cleared / total

        return {
            "counts_by_state": {TBS(k).name: v for k, v in sorted(counts.items())},
            "total_selected": total,
            "cleared_selected": cleared,
            "cleared_fraction": frac_cleared,
        }
    finally:
        tbsim.TxDeliveryR._initiate = original_initiate


def main():
    print("=== High Gap 01: DST-linked retreatment behavior ===")

    # SECTION D1: Default behavior on this branch.
    default = run_scenario(use_guards=False)
    print("\n[Default configuration]")
    print(f"Selected counts by state: {default['counts_by_state']}")
    print(f"Total selected starts: {default['total_selected']}")
    print(f"Selected CLEARED starts: {default['cleared_selected']}")
    print(f"CLEARED selection fraction: {default['cleared_fraction']:.4f}")

    # SECTION D2: Optional guarded behavior for comparison.
    guarded = run_scenario(use_guards=True)
    print("\n[Optional guarded configuration]")
    print(f"Selected counts by state: {guarded['counts_by_state']}")
    print(f"Total selected starts: {guarded['total_selected']}")
    print(f"Selected CLEARED starts: {guarded['cleared_selected']}")
    print(f"CLEARED selection fraction: {guarded['cleared_fraction']:.4f}")

    # SECTION E: High-level interpretation for quick sharing.
    print("\n[Interpretation]")
    print("- If default CLEARED fraction is high, the DST retreatment gap is present.")
    print("- If default CLEARED fraction is near zero, the default behavior is protected.")
    print("- Guarded configuration demonstrates behavior with explicit freshness/retreatment controls.")


if __name__ == "__main__":
    main()

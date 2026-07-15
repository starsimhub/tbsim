"""
Time-varying progression example for developers.

Why this update exists
- Time-varying progression is needed to represent front-loaded progression after infection.
- The default model remains constant-hazard (k_asy=0, k_non=0) for backward compatibility.
- TBResistant now uses the same time-varying progression-rate pathway as TB in latent exits.

What this script demonstrates
1) Case A (default): constant progression hazards (k_asy=0, k_non=0)
2) Case B: front-loaded progression (k_asy=6)
3) Side-by-side results for both TB and TBResistant
4) Overlay against Ferebee/Sutherland reference points

What this script does not do
- It does not assert correctness/validation checks at runtime.
- Those checks live in automated tests under tests/.

Design notes
- Closed cohort (no transmission, networks, demographics) so time-since-infection equals sim time.
- ASYMPTOMATIC is made absorbing to make the "ever reached ASYMPTOMATIC" curve explicit.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import tbsim


# Empirical progression anchors (cumulative ever-active by year since infection, %)
FEREBEE = {1: 4.6, 2: 4.8, 3: 5.2, 5: 5.7, 10: 6.2}
SUTHERLAND = {1: 5.8, 2: 8.0, 3: 8.8, 5: 9.3, 10: 9.6}

DEVELOPER_NOTES = [
    "Default behavior remains constant-hazard (k_asy=0, k_non=0).",
    "Front-loading is opt-in via k_asy>0 (and optional k_non>0).",
    "Validation checks moved to tests: tests/test_time_varying.py and tests/test_resistance.py.",
]


def make_closed_cohort(tb_model, n_agents=12000, years=10, seed=1):
    """Build a closed-cohort sim where everyone starts in INFECTION."""
    sim = tbsim.Sim(
        tb_model=tb_model,
        n_agents=n_agents,
        networks=[],
        demographics=[],
        dt=ss.days(30),
        start=ss.date("2000-01-01"),
        stop=ss.date(f"{2000 + years}-01-01"),
        rand_seed=seed,
    )
    sim.pars.verbose = 0
    return sim


def run_curve(tb_model, n_agents=12000, years=10, seed=1):
    """Run sim and return time axis (years) and ever-ASY curve proxy."""
    sim = make_closed_cohort(tb_model=tb_model, n_agents=n_agents, years=years, seed=seed)
    sim.run()
    tb = sim.get_tb()

    n0 = sum(int(tb.results[f"n_{state.name}"][0]) for state in tbsim.TBS)
    curve = np.asarray(tb.results["n_ASYMPTOMATIC"][:], dtype=float) / n0
    years_since_infection = np.arange(len(curve)) * sim.t.dt_year

    i1 = int(round(1.0 / sim.t.dt_year))
    y1_share = float(curve[i1] / curve[-1]) if curve[-1] > 0 else np.nan
    final_total = float(curve[-1])

    return years_since_infection, curve, y1_share, final_total


def build_tb_case(k_asy):
    return tbsim.TB(
        init_prev=ss.bernoulli(1.0),
        init_prev_active=ss.bernoulli(0.0),
        beta=ss.peryear(0.0),
        inf_cle=ss.peryear(0.1),
        inf_non=ss.peryear(0.0),
        inf_asy=ss.peryear(0.3),
        asy_non=ss.peryear(0.0),
        asy_sym=ss.peryear(0.0),
        k_asy=k_asy,
    )


def build_tbr_case(k_asy):
    return tbsim.TBResistant(
        pars=dict(
            init_prev=ss.bernoulli(1.0),
            init_prev_active=ss.bernoulli(0.0),
            beta=ss.peryear(0.0),
            inf_cle=ss.peryear(0.1),
            inf_non=ss.peryear(0.0),
            inf_asy=ss.peryear(0.3),
            asy_non=ss.peryear(0.0),
            asy_sym=ss.peryear(0.0),
            k_asy=k_asy,
        )
    )


def make_plot(tb_const, tb_front, tbr_const, tbr_front, outpath):
    yrs_tb, curve_tb_const, share_tb_const, total_tb_const = tb_const
    _, curve_tb_front, share_tb_front, total_tb_front = tb_front

    yrs_tbr, curve_tbr_const, share_tbr_const, total_tbr_const = tbr_const
    _, curve_tbr_front, share_tbr_front, total_tbr_front = tbr_front

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=False)

    axes[0].plot(yrs_tb, 100 * curve_tb_const, lw=2, label="TB constant (default k_asy=0)")
    axes[0].plot(yrs_tb, 100 * curve_tb_front, lw=2, label="TB front-loaded (k_asy=6)")
    axes[0].plot(list(FEREBEE.keys()), list(FEREBEE.values()), 'ko', ms=6, label='Ferebee 1970')
    axes[0].plot(list(SUTHERLAND.keys()), list(SUTHERLAND.values()), 'ks', ms=6, mfc='white', mew=1.2, label='Sutherland 1968')
    axes[0].set_title("TB: ever reached ASYMPTOMATIC")
    axes[0].set_xlabel("Years since infection")
    axes[0].set_ylabel("Cumulative (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    axes[0].set_xlim(0, 10.5)

    axes[1].plot(yrs_tbr, 100 * curve_tbr_const, lw=2, label="TBResistant constant (default k_asy=0)")
    axes[1].plot(yrs_tbr, 100 * curve_tbr_front, lw=2, label="TBResistant front-loaded (k_asy=6)")
    axes[1].plot(list(FEREBEE.keys()), list(FEREBEE.values()), 'ko', ms=6, label='Ferebee 1970')
    axes[1].plot(list(SUTHERLAND.keys()), list(SUTHERLAND.values()), 'ks', ms=6, mfc='white', mew=1.2, label='Sutherland 1968')
    axes[1].set_title("TBResistant: ever reached ASYMPTOMATIC")
    axes[1].set_xlabel("Years since infection")
    axes[1].set_ylabel("Cumulative (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    axes[1].set_xlim(0, 10.5)

    fig.suptitle(
        "Time-varying progression comparison\n"
        f"TB year-1 share: {share_tb_const:.3f} -> {share_tb_front:.3f}, total: {100*total_tb_const:.2f}% -> {100*total_tb_front:.2f}% | "
        f"TBResistant year-1 share: {share_tbr_const:.3f} -> {share_tbr_front:.3f}, total: {100*total_tbr_const:.2f}% -> {100*total_tbr_front:.2f}%",
        fontsize=10,
    )

    # Reserve space on the right for outside legends
    fig.subplots_adjust(right=0.78, wspace=0.35)
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    return fig


def main(show=False):
    print("Running time-varying progression developer example...")
    print("Case A: constant hazards (default, k_asy=0)")
    print("Case B: front-loaded hazards (k_asy=6)")
    print("\nDeveloper notes:")
    for note in DEVELOPER_NOTES:
        print(f"- {note}")

    tb_const_model = build_tb_case(k_asy=0.0)
    tb_front_model = build_tb_case(k_asy=6.0)
    tbr_const_model = build_tbr_case(k_asy=0.0)
    tbr_front_model = build_tbr_case(k_asy=6.0)

    tb_const = run_curve(tb_const_model, seed=11)
    tb_front = run_curve(tb_front_model, seed=11)

    tbr_const = run_curve(tbr_const_model, seed=11)
    tbr_front = run_curve(tbr_front_model, seed=11)

    outpath = Path(__file__).with_name("time_varying_progression_comparison.png")
    fig = make_plot(tb_const, tb_front, tbr_const, tbr_front, outpath)
    print(f"\nSaved plot: {outpath}")


    if show:
        plt.show()
    else:
        plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TB/TBResistant time-varying progression example')
    parser.add_argument('--show', action='store_true', help='Display plot window in addition to saving PNG')
    args = parser.parse_args()
    main(show=args.show)

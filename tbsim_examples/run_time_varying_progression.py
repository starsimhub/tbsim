"""
Front-loaded (time-varying) progression from latent infection to active TB.

Why this matters
- Most people who ever progress to active TB do so within a year or two of infection.
- With constant rates, progression is instead spread evenly over time since infection.

How the model does it
- With k_latent > 0, a recent latent infection becomes remote at the rate k_latent (per
  year), after which it progresses at the slower rates inf_asy_remote and inf_non_remote.
  The hazard of progression thus falls with time since infection, as a two-stage
  (recent, then remote) approximation of a smooth decline.
- The default (k_latent=0) has no remote stage: constant rates.
- The model with drug-resistant strains (drugs=...) has the same remote stage.

What this script shows
1) Case A (default): constant progression rates (k_latent=0)
2) Case B: front-loaded progression (k_latent=6, with a small remote rate)
3) Side-by-side results for TB, and TB with a rifampicin-resistant strain
4) Overlay against Ferebee/Sutherland reference points

Design notes
- A closed cohort (no transmission, births, or deaths), so time since infection is the
  time since the start, and everyone starts with a latent infection.
- Asymptomatic TB is made absorbing (no way out of it), so the share of people in it is
  the share who have ever reached it.
- With no network the sims run as ODEs, so the curves are exact and take a moment to run.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import tbsim


# Empirical progression anchors (cumulative ever-active by year since infection, %)
FEREBEE = {1: 4.6, 2: 4.8, 3: 5.2, 5: 5.7, 10: 6.2}
SUTHERLAND = {1: 5.8, 2: 8.0, 3: 8.8, 5: 9.3, 10: 9.6}

CASES = dict(
    const = dict(k_latent=0),                     # Constant rates (the default)
    front = dict(k_latent=6, inf_asy_remote=0.002), # Front-loaded: most progression within a year, then a slow trickle
)


def run_curve(drugs=None, years=10, **pars):
    """ Run a closed cohort, and return the years since infection and the share who have ever reached asymptomatic TB """
    tb = tbsim.TB(
        beta = 0,
        mu = 0,
        init = dict(infection=1.0),
        inf_cle = 0.1,
        inf_non = 0,
        inf_asy = 0.3,
        asy_non = 0,
        asy_sym = 0,
        drugs = drugs,
        **pars,
    )
    sim = tbsim.Sim(tb=tb, start=2000, stop=2000+years, verbose=0)
    sim.run()
    res = sim.results.tb
    curve = res.asymptomatic/res.infection[0]
    t = np.arange(len(curve))*sim.t.dt.years
    year1 = curve[np.searchsorted(t, 1.0)]/curve[-1] # The share of 10-year progression that happens in the first year
    return t, curve, year1, curve[-1]


def make_plot(results, outpath):
    """ Plot the curves for each model and case, with the reference points """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    labels = dict(const='Constant (k_latent=0)', front='Front-loaded (k_latent=6)')
    summary = []
    for ax,(model,cases) in zip(axes, results.items()):
        for case,(t, curve, year1, total) in cases.items():
            ax.plot(t, 100*curve, lw=2, label=labels[case])
        summary.append(f'{model} year-1 share: {cases.const[2]:.3f} -> {cases.front[2]:.3f}, total: {100*cases.const[3]:.2f}% -> {100*cases.front[3]:.2f}%')
        ax.plot(list(FEREBEE.keys()), list(FEREBEE.values()), 'ko', ms=6, label='Ferebee 1970')
        ax.plot(list(SUTHERLAND.keys()), list(SUTHERLAND.values()), 'ks', ms=6, mfc='white', mew=1.2, label='Sutherland 1968')
        ax.set_title(f'{model}: ever reached asymptomatic TB')
        ax.set_xlabel('Years since infection')
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 10.5)
    axes[0].set_ylabel('Cumulative (%)')

    fig.suptitle('Time-varying progression comparison\n' + ' | '.join(summary), fontsize=10, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02)) # Shared legend below the plots
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.25)
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    return fig


def main(show=False):
    print('Running the front-loaded progression example...')
    models = {'TB': None, 'TB (RIF-resistant strain)': ['RIF']}
    results = {}
    for model,drugs in models.items():
        results[model] = sc.objdict({case:run_curve(drugs=drugs, **pars) for case,pars in CASES.items()})
        for case,(t, curve, year1, total) in results[model].items():
            print(f'{model}, {case}: {100*total:.2f}% ever reached asymptomatic TB, {100*year1:.0f}% of them in the first year')

    outpath = Path(__file__).with_name('time_varying_progression_comparison.png')
    fig = make_plot(results, outpath)
    print(f'\nSaved plot: {outpath}')

    if show:
        plt.show()
    else:
        plt.close(fig)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the front-loaded progression example')
    parser.add_argument('--show', action='store_true', help='Display the plot window as well as saving the PNG')
    args = parser.parse_args()
    main(show=args.show)

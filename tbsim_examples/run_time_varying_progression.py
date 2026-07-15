#!/usr/bin/env python
"""
Prototype demo for issue #427 — time-varying progression by time since infection.

Compares a seeded latent cohort across several ``k_asy`` values:

    inf_asy(τ) = inf_asy · exp(−k_asy · τ),   τ = years since infection

``k_asy=0`` is the constant-hazard baseline. For ``k_asy>0``, τ=0 ``inf_asy`` is
raised so early levels stay near the household-contact data (joint-fit style).

Typical calibrated values: Ferebee-like ≈6 /yr, Sutherland-like ≈0.1–1 /yr,
joint-fit mid ≈2–3 /yr.

Burden re-calibration (India single-sim): wire ``k_asy`` into
https://github.com/starsimhub/tb_LAI_TPT (`run_one_sim.py` + ``utils.build_sim``).
"""

import numpy as np
import matplotlib.pyplot as plt
import starsim as ss
import tbsim
from tbsim import TBS


# Empirical anchors (radiographic / ever-subclinical ≈ model curve B)
FEREBEE    = {1: 4.6, 2: 4.8, 3: 5.2, 5: 5.7, 10: 6.2}
SUTHERLAND = {1: 5.8, 2: 8.0, 3: 8.8, 5: 9.3, 10: 9.6}

# Sweep: constant → gentle (Sutherland-ish) → mid → sharp (Ferebee-ish).
# Demonstrates dynamic k_asy inputs too: scalar, [low, high] range, and ss distribution.
SCENARIOS = (
    # (k_asy spec, inf_asy, short label for legend / table)
    (0.0, 0.020, 'k=0 constant'),
    (ss.uniform(6.0, 8.0), 0.158, 'k~U(6,8)'),
)


def run_cohort(k_asy=0.0, inf_asy=0.15, n=20_000, years=10, seed=1):
    """Seed everyone into INFECTION; no transmission. Return (t_years, cum_ever_asy)."""
    sim = ss.Sim(
        n_agents=n,
        start=ss.date('2000-01-01')
        stop=ss.date(f'{2000 + years}-01-01'),
        dt='month',
        networks=ss.RandomNet(lam=6, dur=0),
        diseases=tbsim.TB(pars=dict(
            init_prev=0,
            beta=ss.peryear(1.9),
            sym_dead=ss.peryear(0.0),
            k_asy=k_asy,
            k_non=0.0,          # optional; leave off (handoff recommendation)
            inf_cle=ss.peryear(0.62),
            inf_non=ss.peryear(0.04),
            inf_asy=ss.peryear(inf_asy),  # τ=0 start rate when k_asy>0
        )),
        rand_seed=seed,
        verbose=0,
    )
    sim.init()
    tb = tbsim.get_tb(sim)
    u = ss.uids(np.arange(n))
    tb.state[u] = TBS.INFECTION
    tb.infected[u] = True
    tb.ever_infected[u] = True
    tb.susceptible[u] = False
    tb.ti_infected[u] = tb.ti
    sim.run()

    cum = np.cumsum(tb.results['new_active'][:]) / n
    t_years = np.arange(len(cum)) * sim.t.dt_year
    return t_years, cum


def plot_k_sweep(curves, filename=None, show=True, title=None):
    """
    Plot cumulative ever-ASYMPTOMATIC for several k_asy scenarios vs data.

    Args:
        curves (dict): {label: (t_years, cum_frac, k_asy), ...} in display order
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(curves)))
    for color, (label, (t, cum, k)) in zip(cmap, curves.items()):
        # Constant baseline: dashed + dark so it reads clearly vs declining curves
        is_const = np.isscalar(k) and float(k) == 0.0
        ax.plot(
            t, 100 * cum,
            color='0.25' if is_const else color,
            lw=2.6 if is_const else 2.0,
            ls='--' if is_const else '-',
            label=label,
            zorder=4 if is_const else 3,
        )

    ax.plot(list(FEREBEE.keys()), list(FEREBEE.values()), 'ko', ms=7, zorder=5, label='Ferebee 1970')
    ax.plot(list(SUTHERLAND.keys()), list(SUTHERLAND.values()), 'ks', ms=7, mfc='white', mew=1.5, zorder=5, label='Sutherland 1968')

    ax.set_xlabel('Years since infection')
    ax.set_ylabel('Ever reached ASYMPTOMATIC (%)')
    ax.set_title(title or 'TB progression by time since infection — k_asy sweep (#427)')
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc='lower right', ncols=2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
        print(f'Saved {filename}')
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    print('Running cohorts (scalar + dynamic k_asy specs; k=0 = constant hazard)…')
    curves = {}
    for k, inf_asy, label in SCENARIOS:
        print(f'  {label}: inf_asy={inf_asy:g}/yr')
        t, c = run_cohort(k_asy=k, inf_asy=inf_asy)
        # Legend: make constant obvious; include start rate for declining curves
        if np.isscalar(k) and float(k) == 0.0:
            legend = f'{label} (inf_asy={inf_asy:g}/yr)'
        else:
            legend = f'{label}, inf_asy={inf_asy:g}/yr'
        curves[legend] = (t, c, k)

    # Integer-year table
    years = range(1, 11)
    hdr = f'{"year":>4}' + ''.join(f'  {lab:>14}' for _, _, lab in SCENARIOS)
    hdr += f'  {"Ferebee":>8}  {"Sutherland":>10}'
    print('\nEver reached ASYMPTOMATIC (% of cohort)\n')
    print(hdr)
    for y in years:
        row = f'{y:4d}'
        for legend, (t, c, _) in curves.items():
            row += f'  {100 * c[np.argmin(np.abs(t - y))]:14.2f}'
        row += f'  {FEREBEE.get(y, float("nan")):8.1f}  {SUTHERLAND.get(y, float("nan")):10.1f}'
        print(row)

    print('\nk=0 is constant hazard. Declining curves use higher τ=0 inf_asy so early')
    print('levels stay near data; absolute rates still need joint burden re-calibration.')

    plot_k_sweep(
        curves,
        filename='tbsim_examples/progression_k_asy_sweep.png',
        show=True,
        title='TB progression vs time since infection (k=0 = constant)',
    )

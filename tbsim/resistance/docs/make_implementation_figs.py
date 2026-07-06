"""
Generate the figures embedded in ``tbsim-resistance-implementation.md``.

Run from the repo root::

    /software/conda/bin/python tbsim/resistance/docs/make_implementation_figs.py

Writes PNGs to ``docs/assets/`` (the committable location for doc images). Reuses the ABM↔ODE comparison helpers in
``tbsim.resistance.devtests.ode_utils`` so the validation figure matches the devtest suite.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import starsim as ss
import tbsim
from tbsim.resistance.devtests import ode_utils as ou

HERE = os.path.dirname(__file__)
FIGS = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'docs', 'assets'))  # committable doc-image location
os.makedirs(FIGS, exist_ok=True)
BETA_ODE, YEARS, NSEEDS = 45.0, 80, 2


def fig_validation():
    """ABM vs two-strain ODE: reduction, competitive exclusion, treatment-driven selection."""
    be = ou.calibrate_beta(BETA_ODE)
    # (A) single-strain reduction — endemic compartment fractions
    o1 = ou.run_ode(BETA_ODE, years=YEARS, rr_reinfection_inf=0, rr_reinfection_non=0)
    a1 = [ou.run_abm(be, years=YEARS, seed=s, rr_reinfection_inf=0.0, rr_reinfection_non=0.0) for s in range(NSEEDS)]
    # (B) competitive exclusion — resistant fraction declines
    seeds = dict(L_A=0.75 * 0.05 * 1e5, L_B=0.25 * 0.05 * 1e5)
    o2 = ou.run_ode(BETA_ODE, years=YEARS, fit_b=0.7, seeds=seeds, rr_reinfection_inf=0, rr_reinfection_non=0)
    a2 = [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': 0.7}, init_strains=[0.75, 0.25],
                     rr_reinfection_inf=0.0, rr_reinfection_non=0.0) for s in range(NSEEDS)]
    # (C) treatment-driven selection — resistant fraction rises
    treat = dict(eff_a=0.75, eff_b=0.25, q_treat=0.03, r_treat_sym=1.0, r_treat_asym=0.05)
    seeds3 = dict(L_A=0.9 * 0.05 * 1e5, L_B=0.1 * 0.05 * 1e5)
    o3 = ou.run_ode(BETA_ODE, years=YEARS, fit_b=0.575, seeds=seeds3, treat=treat,
                    rr_reinfection_inf=1.0, rr_reinfection_non=1.0)
    a3 = [ou.run_abm(be, years=YEARS, seed=s, rel_fitness={'TX': 0.575}, init_strains=[0.9, 0.1],
                     treat=treat, rr_reinfection_inf=1.0, rr_reinfection_non=1.0) for s in range(NSEEDS)]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(ou.STATES))
    ax[0].bar(x - 0.2, [o1.fracs[s][-1] for s in ou.STATES], 0.4, label='ODE', color='0.4')
    ax[0].bar(x + 0.2, [np.mean([r.fracs[s][-1] for r in a1]) for s in ou.STATES], 0.4, label='ABM', color='tab:blue')
    ax[0].set_xticks(x); ax[0].set_xticklabels([s[:4] for s in ou.STATES], rotation=30, ha='right')
    ax[0].set(title='(A) Single-strain reduction', ylabel='endemic fraction'); ax[0].legend(frameon=False)

    for r in a2:
        ax[1].plot(r.t, r.frac_resist, color='tab:green', alpha=0.35, lw=1)
    ax[1].plot(a2[0].t, np.mean([r.frac_resist for r in a2], axis=0), color='tab:green', lw=2, label='ABM')
    ax[1].plot(o2.t, o2.frac_resist, 'k--', lw=2, label='ODE')
    ax[1].set(title='(B) Competitive exclusion', xlabel='year', ylabel='resistant fraction'); ax[1].legend(frameon=False)

    for r in a3:
        ax[2].plot(r.t, r.frac_resist, color='tab:red', alpha=0.35, lw=1)
    ax[2].plot(a3[0].t, np.mean([r.frac_resist for r in a3], axis=0), color='tab:red', lw=2, label='ABM')
    ax[2].plot(o3.t, o3.frac_resist, 'k--', lw=2, label='ODE')
    ax[2].set(title='(C) Treatment selects resistance', xlabel='year', ylabel='resistant fraction'); ax[2].legend(frameon=False)
    fig.suptitle('Agent-based TBResistant vs. two-strain reference ODE', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'resistance_abm_vs_ode.png'), dpi=110)
    print('wrote resistance_abm_vs_ode.png')


def _build(tb, interventions=None, analyzers=None, n=3000, stop='2045-12-31', seed=0):
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=10), dur=0))
    return ss.Sim(n_agents=n, networks=net, diseases=tb, interventions=interventions, analyzers=analyzers,
                  dt=ss.days(30), start=ss.date('2000-01-01'), stop=ss.date(stop), rand_seed=seed, verbose=0)


def fig_origins():
    """Three-way resistance-origin decomposition (ResistanceStats)."""
    tb = tbsim.TBResistant(rel_fitness={'TX': 0.9},
                           pars=dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.12),
                                     init_strains=[0.9, 0.1], p_rand={'TX': 0.01},
                                     rr_reinfection_inf=1.0, rr_reinfection_non=1.0))
    tx = tbsim.TxDeliveryR(product=tbsim.TxR(strains=tb.strains, base_efficacy=0.8,
                           resist_penalty={'TX': 0.2}, q_acq={'TX': 0.05}), rate_sym=ss.peryear(1.0))
    stats = tbsim.ResistanceStats()
    sim = _build(tb, interventions=tx, analyzers=stats); sim.run()
    df = stats.to_df(sim)
    origins = {'de-novo\nmutation': df.flux_denovo.sum(), 'treatment-\nacquired': df.flux_txacq.sum(),
               'transmitted': df.flux_transmitted.sum()}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(origins.keys(), origins.values(), color=['tab:blue', 'tab:orange', 'tab:green'])
    ax.set(title='Origin of new resistant infections', ylabel='cumulative events')
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'resistance_origins.png'), dpi=110)
    print('wrote resistance_origins.png')


def fig_tpt():
    """INH preventive therapy unmasks a less-fit INH-resistant strain."""
    def run(with_tpt):
        tb = tbsim.TBResistant(drugs=['INH'], rel_fitness={'INH': 0.9},
                               pars=dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.15),
                                         init_strains=[0.8, 0.2], rr_reinfection_inf=0.0, rr_reinfection_non=0.0))
        ivs = None
        if with_tpt:
            ivs = tbsim.TPTSimple(product=tbsim.TPTRx(strains=tb.strains, regimen_drugs=['INH'],
                                  pars=dict(efficacy=ss.bernoulli(0.9), p_sterilize=ss.bernoulli(1.0))),
                                  pars=dict(coverage=ss.bernoulli(0.5)))
        sim = _build(tb, interventions=ivs, n=4000, stop='2035-12-31'); sim.run()
        return sim.results.timevec, sim.results.tb['frac_resist']
    fig, ax = plt.subplots(figsize=(7, 4))
    for with_tpt, lbl, c in [(False, 'no TPT', 'tab:blue'), (True, 'INH TPT (50% coverage)', 'tab:red')]:
        t, fr = run(with_tpt)
        ax.plot(t, fr, label=lbl, color=c, lw=2)
    ax.set(title='TPT unmasks INH resistance (Mills–Cohen dynamic)', xlabel='year',
           ylabel='resistant fraction of active TB'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'resistance_tpt_unmasking.png'), dpi=110)
    print('wrote resistance_tpt_unmasking.png')


if __name__ == '__main__':
    fig_validation()
    fig_origins()
    fig_tpt()
    print('done')

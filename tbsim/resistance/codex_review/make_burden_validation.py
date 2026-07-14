"""
Generate the resistance-off vs resistance-on burden-comparison artifact requested by the tech spec
(§"Testing", first bullet): "Tests comparing key burden results output by models before vs. after
resistance is added ... overall TB disease prevalence per 100,000, annual incidence of new
asymptomatic disease per 100,000, annual TB mortality per 100,000."

We run three matched, endemic configurations on the same natural history / transmission / demographics
and write ``lai_tpt_burden_validation.csv`` next to this script:

* ``resistance_off``        — single-strain ``tbsim.TB`` (the "before").
* ``resistance_on_agnostic``— ``TBResistant.agnostic()``: the multistrain machinery enabled but
  configured single-strain-equivalent (no resistant strain ever arises). Isolates the pure overhead of
  the strain overlay — it should match ``resistance_off`` to within stochastic noise.
* ``resistance_on``         — ``TBResistant`` with a genuinely circulating resistant strain
  (fitness cost, resistant seed, de-novo acquisition, superinfection). The "after".

The parameter set is a plausible endemic configuration in the spirit of tb_LAI_TPT (natural-history
rates are TBsim defaults; ``beta``/``init_prev``/demographics chosen to sit at a stable endemic
prevalence). No treatment is applied, so the three scenarios differ *only* in whether resistance is
present — the fair "does multistrain shift overall burden?" comparison the spec asks for.

Run:  python -m tbsim.resistance.codex_review.make_burden_validation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

import tbsim

# --- Shared, matched configuration ---------------------------------------------------------------
N_AGENTS = 6000
SEEDS = (0, 1, 2)
DT = ss.days(30)
START = ss.date('2000-01-01')
STOP = ss.date('2080-12-31')     # long enough to reach an endemic equilibrium
LATE_YEARS = 20                  # averaging window at the end of the run
MU = 1 / 70                      # background mortality per year (matched births)
PARS = dict(beta=ss.permonth(0.2), init_prev=ss.bernoulli(0.05))


def _net():
    return ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))


def _demog():
    return [ss.Births(birth_rate=1000 * MU), ss.Deaths(death_rate=1000 * MU)]


def _run(tb, seed):
    sim = tbsim.Sim(n_agents=N_AGENTS, diseases=tb, networks=_net(), demographics=_demog(),
                    dt=DT, start=START, stop=STOP, rand_seed=seed, verbose=0)
    sim.run()
    return sim


def _burden(sim):
    """Return the three spec burden metrics (per 100k) averaged over the late window of one run."""
    res = sim.results.tb
    n_alive = np.asarray(sim.results.n_alive, dtype=float)
    steps = len(n_alive)
    win = int(LATE_YEARS / (30 / 365.25))          # number of 30-day steps in the late window
    sl = slice(max(0, steps - win), steps)
    years = (sl.stop - (sl.start or 0)) * (30 / 365.25)
    alive_mean = n_alive[sl].mean()
    prevalence = np.asarray(res['prevalence_active'])[sl].mean() * 1e5
    # Annualized rates: events in the window / person-years-ish (mean alive × years) × 100k.
    asymp_inc = np.asarray(res['new_active'])[sl].sum() / years / alive_mean * 1e5
    tb_mort = np.asarray(res['new_deaths'])[sl].sum() / years / alive_mean * 1e5
    frac_resist = float(np.asarray(res['frac_resist'])[sl].mean()) if 'frac_resist' in res else 0.0
    return dict(prevalence_per_100k=prevalence, asymp_incidence_per_100k_yr=asymp_inc,
                tb_mortality_per_100k_yr=tb_mort, frac_resist_active=frac_resist)


def _scenario(name, make_tb):
    rows = [_burden(_run(make_tb(), s)) for s in SEEDS]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    mean['scenario'] = name
    mean['n_seeds'] = len(SEEDS)
    return mean


def build():
    def on_tb():
        # Genuinely circulating resistance: 10% resistant seed, fitness cost, de-novo acquisition at
        # progression, and superinfection (default spec coupling via rr_reinfection_rec).
        return tbsim.TBResistant(drugs=['RIF'], rel_fitness={'RIF': 0.9},
                                 init_strains=[0.9, 0.1], p_rand={'RIF': 0.02}, **PARS)

    scenarios = [
        _scenario('resistance_off', lambda: tbsim.TB(pars=dict(PARS))),
        _scenario('resistance_on_agnostic', lambda: tbsim.TBResistant.agnostic(pars=dict(PARS))),
        _scenario('resistance_on', on_tb),
    ]
    df = pd.DataFrame(scenarios)[['scenario', 'n_seeds', 'prevalence_per_100k',
                                  'asymp_incidence_per_100k_yr', 'tb_mortality_per_100k_yr',
                                  'frac_resist_active']]
    out = Path(__file__).resolve().parent / 'lai_tpt_burden_validation.csv'
    df.to_csv(out, index=False, float_format='%.2f')
    print(df.to_string(index=False))
    print(f'\nWrote {out}')
    return df


if __name__ == '__main__':
    sc.tic()
    build()
    sc.toc()

"""
Shared ABM ↔ two-strain-ODE comparison helpers for the resistance devtests.

The deterministic ``tbsim.compartmental.TwoStrainODE`` (a port of ``ode.r``, verified
bit-identical to the R implementation) is the validation reference for the agent-based
``tbsim.TBResistant``. The two models share the LSHTM natural-history rates exactly; two
things must be reconciled to line them up:

* **Force of infection** — the ODE is frequency-dependent ``β/N`` mass action; the ABM is a
  per-edge ``β`` on a random network of mean degree ``K`` (so ``β_edge ≈ β_ode/K``, refined by
  matching single-strain endemic prevalence). :func:`calibrate_beta` does this once and caches.
* **Demographics** — uniform mortality ``μ = 1/70`` with matched births holding ``N`` roughly constant.

Rather than compare centuries-long equilibria, tests seed both models from the *same* initial
condition and compare transient trajectories / endpoints over a shared window; the ABM is a
stochastic realization that should track the deterministic ODE curve. For the "questions of
interest" (model-tests.md §10) the most robust check is *directional*: null vs. variant, does the
ABM shift the observable the same way the ODE does?
"""

import numpy as np
import sciris as sc
import starsim as ss

import tbsim
import tbsim.compartmental as tc

# Shared configuration
K = 10                    # ABM random-network mean degree
N_AGENTS = 8000
DT = ss.days(30)
MU = 1 / 70               # background mortality (per year), matches the ODE
START = 2000
STATES = ['SUSCEPTIBLE', 'INFECTION', 'NON_INFECTIOUS', 'ASYMPTOMATIC', 'SYMPTOMATIC', 'CLEARED']

# ODE param names that the ABM exposes under the same name (superinfection / progression knobs).
_SHARED = ('rr_reinfection_inf', 'rr_reinfection_non', 'rr_reinfection_asy', 'rr_reinfection_sym',
           'p_multi', 'rr_prog_super', 'rr_clear_super')


def run_ode(beta, years=80, n=1e5, fit_b=1.0, seeds=None, treat=None, denovo=None, **shared):
    """Integrate the two-strain ODE and return observables as a tidy objdict.

    Args:
        beta: ODE transmission rate (β).
        years: length of the run from ``START``.
        fit_b: strain-B relative fitness (r_b); ``1.0`` = neutral.
        seeds: dict of seed counts by ODE state (default: mono-A latent seed).
        treat: optional dict ``eff_a, eff_b, q_treat, r_treat_asym, r_treat_sym``.
        denovo: optional dict ``q_prog, prog_resist_mix``.
        **shared: any of ``_SHARED`` plus ``prog_select_fitness``, ``transmission_independent``.
    """
    kw = dict(beta=beta, mu=MU, N=n, fit_b=fit_b)
    kw.update({k: v for k, v in shared.items()})
    if treat:
        kw.update(treat)
    else:
        kw.update(r_treat_asym=0.0, r_treat_sym=0.0, q_treat=0.0)
    if denovo:
        kw.update(denovo)
    else:
        kw.update(q_prog=0.0)
    ode = tc.TwoStrainODE(**kw)
    seeds = seeds or dict(L_A=0.05 * n)
    ode.run(start_time=START, end_time=START + years, **seeds)
    df, coll = ode.df, ode.collapse()
    fracs = {s: coll[s].values / n for s in STATES if s != 'CLEARED'}
    fracs['CLEARED'] = (coll.CLEARED.values + coll.RECOVERED.values + coll.TREATED.values) / n
    return sc.objdict(t=df.time.values, prev_active=df.prev_active.values,
                      frac_resist=df.frac_resist.values, frac_super=df.frac_super.values, fracs=fracs)


def run_abm(beta_edge, years=80, seed=0, n=N_AGENTS, rel_fitness=None, init_strains=None,
            treat=None, denovo=None, demog=True, **shared):
    """Run the agent-based ``TBResistant`` and return the same observables as :func:`run_ode`.

    Args mirror :func:`run_ode`: ``rel_fitness={'TX': fit_b}``, ``init_strains`` seed weights,
    ``treat``/``denovo`` dicts, and shared superinfection/progression knobs.
    """
    tb_pars = dict(beta=ss.peryear(beta_edge), init_prev=ss.bernoulli(0.05),
                   init_strains=init_strains if init_strains is not None else [1.0, 0.0])
    tb_pars.update({k: v for k, v in shared.items() if k in _SHARED})
    if 'prog_select_fitness' in shared:
        tb_pars['prog_select'] = 'fitness' if shared['prog_select_fitness'] else 'random'
    if denovo:
        tb_pars['p_rand'] = {'TX': denovo.get('q_prog', 0.0)}
        tb_pars['prog_resist_mode'] = 'mixed' if denovo.get('prog_resist_mix', 1) else 'replacement'

    tb = tbsim.TBResistant(rel_fitness=rel_fitness or {'TX': 1.0}, pars=tb_pars)

    interventions = None
    if treat:
        interventions = tbsim.TxDeliveryR(
            product=tbsim.TxR(strains=tb.strains, base_efficacy=treat['eff_a'],
                              resist_penalty={'TX': treat['eff_b'] / treat['eff_a']},
                              adherence=1.0, q_acq={'TX': treat.get('q_treat', 0.0)}),
            rate_sym=ss.peryear(treat.get('r_treat_sym', 0.0)),
            rate_asym=ss.peryear(treat.get('r_treat_asym', 0.0)),
            dur_treatment=ss.months(6))

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=K), dur=0))
    demographics = [ss.Births(birth_rate=1000 * MU), ss.Deaths(death_rate=1000 * MU)] if demog else None
    sim = ss.Sim(n_agents=n, networks=net, diseases=tb, interventions=interventions,
                 demographics=demographics, dt=DT, start=ss.date(f'{START}-01-01'),
                 stop=ss.date(f'{START + years}-01-01'), rand_seed=seed, verbose=0)
    sim.run()
    r = sim.results.tb
    n_alive = np.array(sim.results.n_alive)
    t = START + np.arange(len(n_alive)) * (30 / 365.25)
    return sc.objdict(t=t, prev_active=np.array(r['prevalence_active']),
                      frac_resist=np.array(r['frac_resist']), frac_super=np.array(r['frac_super']),
                      fracs={s: np.array(r[f'n_{s}']) / n_alive for s in STATES}, sim=sim)


_beta_cache = {}


def calibrate_beta(beta_ode, years=180):
    """Return the ABM per-edge β whose single-strain endemic prevalence matches the ODE (cached)."""
    if beta_ode in _beta_cache:
        return _beta_cache[beta_ode]
    target = run_ode(beta_ode, years=250).prev_active[-1]
    best = None
    for be in [beta_ode / K * f for f in (0.85, 1.0, 1.15, 1.3, 1.5)]:
        prev = run_abm(be, years=years, seed=0).prev_active[-1]
        if best is None or abs(prev - target) < best[0]:
            best = (abs(prev - target), be)
    _beta_cache[beta_ode] = best[1]
    return best[1]


def final_mean(runs, key):
    """Mean of the final value of ``key`` across a list of ABM run objdicts."""
    return float(np.mean([r[key][-1] for r in runs]))


def late_mean(run, key, frac=0.2):
    """Mean of ``key`` over the last ``frac`` of a single run (smooths stochastic endpoints)."""
    v = run[key]
    k = max(1, int(len(v) * frac))
    return float(np.mean(v[-k:]))

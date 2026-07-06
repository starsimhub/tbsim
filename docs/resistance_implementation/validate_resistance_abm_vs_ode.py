"""
Validation: full agent-based model (``tbsim.TBResistant``) vs the literal two-strain
ODE (``tbsim.compartmental.TwoStrainODE`` — a direct port of ``ode.r``, verified
bit-identical to the R implementation; a plain ``scipy.odeint`` system, *not* a Starsim model).

The two models share the LSHTM natural-history rates exactly (``tbsim.TB`` defaults ==
``two_strain_defaults()``). Two things are reconciled:
  * **Force of infection** — the ODE is frequency-dependent ``β/N`` mass action; the ABM is a
    per-edge ``β`` on a random network of mean degree ``k`` (so ``β_edge ≈ β_ode/k``, refined by
    matching the single-strain endemic prevalence). Calibrated once in `calibrate_beta`.
  * **Demographics** — uniform mortality ``μ=1/70`` with births holding N ≈ constant.

Rather than compare centuries-long equilibria (the coexistence equilibrium equilibrates very
slowly — which is why ``ode.r`` integrates for ~500 years), we seed **both models from the same
initial condition** and overlay their **transient trajectories** over a shared window. The ABM is
then a stochastic realization that should track the deterministic ODE curve. Three scenarios:
  1. Single strain, no treatment — natural history + transmission (prevalence & compartments).
  2. Two strains, no treatment, resistant strain less fit — competitive decline of resistance.
  3. Two strains + treatment — treatment-driven rise of resistance.

Run:  python scripts/validate_resistance_abm_vs_ode.py
Outputs results/validate_resistance_abm_vs_ode.png and a printed metrics table.
"""

import os
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

import tbsim
import tbsim.compartmental as tc

# --- Shared configuration -------------------------------------------------------
K = 10                    # ABM random-network mean degree
N_AGENTS = 25000
DT = ss.days(30)
START, YEARS = 2000, 100  # shared simulation window
MU = 1 / 70               # background mortality (per year), matches the ODE
BETA_ODE = 45.0
FIT_B = 0.85              # resistant-strain transmission fitness (r_b)
INIT_LATENT = 0.05        # initial latent prevalence (rest susceptible)
INIT_B_FRAC = 0.20        # fraction of the initial latent seed that is strain B

STATES = ['SUSCEPTIBLE', 'INFECTION', 'NON_INFECTIOUS', 'ASYMPTOMATIC', 'SYMPTOMATIC', 'CLEARED']
# Treatment operating point (σ=0 → mono-strain competition).
TX = dict(eff_a=0.85, eff_b=0.5, q_treat=0.03, q_prog=5e-4, r_treat_asym=0.05)


# --- ODE side -------------------------------------------------------------------
def run_ode(beta, two_strain, r_treat_sym, years=YEARS, n=1e5):
    kw = dict(beta=beta, mu=MU, rr_reinfection_inf=0, rr_reinfection_non=0, N=n)  # σ=0
    if two_strain:
        kw.update(fit_b=FIT_B, r_treat_sym=r_treat_sym, r_treat_asym=TX['r_treat_asym'],
                  eff_a=TX['eff_a'], eff_b=TX['eff_b'], q_treat=TX['q_treat'], q_prog=TX['q_prog'])
        seeds = dict(L_A=(1 - INIT_B_FRAC) * INIT_LATENT * n, L_B=INIT_B_FRAC * INIT_LATENT * n)
    else:
        kw.update(fit_b=1.0, r_treat_sym=0.0, r_treat_asym=0.0, q_treat=0.0, q_prog=0.0)
        seeds = dict(L_A=INIT_LATENT * n)
    ode = tc.TwoStrainODE(**kw)
    ode.run(start_time=START, end_time=START + years, **seeds)
    df, coll = ode.df, ode.collapse()
    # tbsim.TB folds cleared-from-latent, recovered-from-non-infectious, and post-treatment into
    # one CLEARED state; the ODE keeps them separate (CLE/REC/TRD) — sum them to compare like-for-like.
    fracs = {s: coll[s].values / n for s in STATES if s != 'CLEARED'}
    fracs['CLEARED'] = (coll.CLEARED.values + coll.RECOVERED.values + coll.TREATED.values) / n
    return sc.objdict(t=df.time.values, prev_active=df.prev_active.values,
                      frac_resist=df.frac_resist.values, frac_super=df.frac_super.values, fracs=fracs)


# --- ABM side -------------------------------------------------------------------
def run_abm(beta_edge, two_strain, r_treat_sym, seed=0, years=YEARS):
    tb_kw = dict(beta=ss.peryear(beta_edge), init_prev=ss.bernoulli(INIT_LATENT),
                 rr_reinfection_inf=0.0, rr_reinfection_non=0.0)  # σ=0
    interventions = None
    if two_strain:
        tb = tbsim.TBResistant(rel_fitness={'TX': FIT_B},
                               pars=dict(tb_kw, init_strains=[1 - INIT_B_FRAC, INIT_B_FRAC],
                                         q_prog=TX['q_prog'], prog_resist_mode='mixed'))
        interventions = tbsim.TxDeliveryR(
            product=tbsim.TxR(strains=tb.strains, base_efficacy=TX['eff_a'],
                              resist_penalty={'TX': TX['eff_b'] / TX['eff_a']},
                              adherence=1.0, q_acq=TX['q_treat']),
            rate_sym=ss.peryear(r_treat_sym), rate_asym=ss.peryear(TX['r_treat_asym']),
            dur_treatment=ss.months(6))
    else:
        tb = tbsim.TBResistant(rel_fitness={'TX': 1.0}, pars=dict(tb_kw, init_strains=[1.0, 0.0]))

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=K), dur=0))
    demog = [ss.Births(birth_rate=1000 * MU), ss.Deaths(death_rate=1000 * MU)]
    sim = ss.Sim(n_agents=N_AGENTS, networks=net, diseases=tb, interventions=interventions,
                 demographics=demog, dt=DT, start=ss.date(f'{START}-01-01'),
                 stop=ss.date(f'{START + years}-01-01'), rand_seed=seed, verbose=0)
    sim.run()
    r = sim.results.tbresistant
    n_alive = np.array(sim.results.n_alive)
    t = START + np.arange(len(n_alive)) * (30 / 365.25)
    return sc.objdict(t=t, prev_active=np.array(r['prevalence_active']),
                      frac_resist=np.array(r['frac_resist']), frac_super=np.array(r['frac_super']),
                      fracs={s: np.array(r[f'n_{s}']) / n_alive for s in STATES})


def mean_of(runs, key):
    m = min(len(r[key]) for r in runs)
    return np.mean([r[key][:m] for r in runs], axis=0), runs[0].t[:m]


# --- Calibrate β_edge to the ODE single-strain endemic prevalence ---------------
def calibrate_beta():
    target = run_ode(BETA_ODE, two_strain=False, r_treat_sym=0, years=250).prev_active[-1]
    print(f'\n[calibrate] ODE single-strain endemic prevalence = {target:.4f}')
    best = None
    for be in [BETA_ODE / K * f for f in (0.9, 1.1, 1.3)]:
        prev = run_abm(be, two_strain=False, r_treat_sym=0, years=150).prev_active[-1]
        print(f'[calibrate] β_edge={be:.2f} (β_eff≈{be*K:.0f}) → ABM prev={prev:.4f}')
        if best is None or abs(prev - target) < best[0]:
            best = (abs(prev - target), be)
    print(f'[calibrate] chosen β_edge={best[1]:.2f}')
    return best[1]


# --- Main -----------------------------------------------------------------------
def main():
    beta_edge = calibrate_beta()
    nseeds = 3

    # Scenario 1: single strain, no treatment (natural history)
    o1 = run_ode(BETA_ODE, two_strain=False, r_treat_sym=0)
    a1 = [run_abm(beta_edge, two_strain=False, r_treat_sym=0, seed=s) for s in range(nseeds)]

    # Scenario 2: two strains, no treatment — resistant strain out-competed
    o2 = run_ode(BETA_ODE, two_strain=True, r_treat_sym=0)
    a2 = [run_abm(beta_edge, two_strain=True, r_treat_sym=0, seed=s) for s in range(nseeds)]

    # Scenario 3: two strains + treatment — resistance selected for
    o3 = run_ode(BETA_ODE, two_strain=True, r_treat_sym=1.0)
    a3 = [run_abm(beta_edge, two_strain=True, r_treat_sym=1.0, seed=s) for s in range(nseeds)]

    # --- Metrics table (final-time comparison, same window) ---
    def final(runs, key):
        return np.mean([r[key][-1] for r in runs])

    def row(label, ode_val, abm_val):
        print(f'{label:36s}{ode_val:10.4f}{abm_val:10.4f}')

    print('\n=== ABM vs ODE at year %d (matched IC & window) ===' % (START + YEARS))
    print(f'{"scenario / observable":36s}{"ODE":>10s}{"ABM":>10s}')
    row('1 single-strain prev_active', o1.prev_active[-1], final(a1, 'prev_active'))
    for s in STATES:
        row('  frac ' + s, o1.fracs[s][-1], np.mean([r.fracs[s][-1] for r in a1]))
    row('2 no-tx frac_resist (collapses)', o2.frac_resist[-1], final(a2, 'frac_resist'))
    row('3 +tx  frac_resist (sustained)', o3.frac_resist[-1], final(a3, 'frac_resist'))
    row('3 +tx  prev_active', o3.prev_active[-1], final(a3, 'prev_active'))

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) single-strain prevalence trajectory
    ax = axes[0, 0]
    for r in a1:
        ax.plot(r.t, r.prev_active, color='tab:blue', alpha=0.35, lw=1)
    m, t = mean_of(a1, 'prev_active')
    ax.plot(t, m, color='tab:blue', lw=2, label='ABM (mean of %d)' % nseeds)
    ax.plot(o1.t, o1.prev_active, 'k--', lw=2, label='ODE')
    ax.set_title('(a) Single strain, no treatment: active prevalence')
    ax.set_xlabel('year'); ax.set_ylabel('active-TB prevalence'); ax.legend(frameon=False)

    # (b) single-strain endemic compartments (final)
    ax = axes[0, 1]
    x = np.arange(len(STATES))
    ax.bar(x - 0.2, [o1.fracs[s][-1] for s in STATES], 0.4, label='ODE', color='0.4')
    ax.bar(x + 0.2, [np.mean([r.fracs[s][-1] for r in a1]) for s in STATES], 0.4, label='ABM', color='tab:blue')
    ax.set_xticks(x); ax.set_xticklabels([s[:5] for s in STATES], rotation=30, ha='right')
    ax.set_title('(b) Single strain: endemic compartment fractions')
    ax.set_ylabel('fraction of population'); ax.legend(frameon=False)

    # (c) competitive exclusion (no treatment): frac_resist declines
    ax = axes[1, 0]
    for r in a2:
        ax.plot(r.t, r.frac_resist, color='tab:green', alpha=0.35, lw=1)
    m, t = mean_of(a2, 'frac_resist')
    ax.plot(t, m, color='tab:green', lw=2, label='ABM (mean of %d)' % nseeds)
    ax.plot(o2.t, o2.frac_resist, 'k--', lw=2, label='ODE')
    ax.axhline(INIT_B_FRAC, color='0.7', ls=':', lw=1, label='initial B fraction')
    ax.set_title('(c) Two strains, no treatment: resistance out-competed')
    ax.set_xlabel('year'); ax.set_ylabel('resistant fraction of active TB'); ax.legend(frameon=False)

    # (d) treatment-driven selection: frac_resist rises
    ax = axes[1, 1]
    for r in a3:
        ax.plot(r.t, r.frac_resist, color='tab:red', alpha=0.35, lw=1)
    m, t = mean_of(a3, 'frac_resist')
    ax.plot(t, m, color='tab:red', lw=2, label='ABM (mean of %d)' % nseeds)
    ax.plot(o3.t, o3.frac_resist, 'k--', lw=2, label='ODE')
    ax.axhline(INIT_B_FRAC, color='0.7', ls=':', lw=1, label='initial B fraction')
    ax.set_title('(d) Two strains + treatment: resistance selected for')
    ax.set_xlabel('year'); ax.set_ylabel('resistant fraction of active TB'); ax.legend(frameon=False)

    fig.suptitle('TBsim resistance: agent-based model vs two-strain ODE (matched IC & window)', fontweight='bold')
    fig.tight_layout()
    os.makedirs('results', exist_ok=True)
    path = 'results/validate_resistance_abm_vs_ode.png'
    fig.savefig(path, dpi=120)
    print(f'\nFigure saved to {path}')
    return fig


if __name__ == '__main__':
    main()

"""
Compare the agent-based TBsim model against the compartmental ODE version.

Motivation
----------
It has been claimed that the TB epidemic dies out in the agent-based model (ABM),
which would indicate a bug (the natural-history parameters admit an endemic
equilibrium, as the ODE demonstrates). This script runs both models over a shared
window (``START``-``STOP``) with *comparable* parameters and overlays their trends
so the two can be judged side by side.

The two models cannot use *identical* parameters because the ODE assumes
homogeneous mixing (mass-action force of infection) while the ABM transmits over
a contact network. The mapping choices below are documented inline; each is
chosen so that the two models have the same expected per-year fluxes at low
prevalence, which is the regime where a linear (ODE) and a network (ABM) model
should agree most closely.

Run with::

    /software/conda/bin/python ode_compare.py
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

import tbsim
from tbsim.compartmental.lshtm_ode import TB_ODE, default_pars


# --- Shared simulation window --------------------------------------------------
START = 1950
STOP  = 2100

# --- Parameter-mapping choices (ODE <-> ABM) -----------------------------------
#
# 1. TRANSMISSION (beta).
#    The ODE force of infection is  foi = (beta_ode / N) * (trans_asymp*A + S),
#    i.e. beta_ode is a *per-person-per-year* contact-times-transmissibility
#    product under homogeneous mixing (default beta_ode = 9/year).
#    The ABM instead applies a *per-edge* rate over a RandomNet: a susceptible
#    with k contacts sees, at low prevalence, an expected FOI of
#        beta_edge * k * (trans_asymp*A + S) / N.
#    Matching the two therefore requires  beta_edge = beta_ode / k, where k is
#    the mean contact degree. We fix the degree to a constant k = 5 (rather than
#    the Poisson(5) that tbsim.Sim uses by default) to remove one source of
#    network heterogeneity and make the mapping exact in expectation.
#
#    We use k = 20 rather than tbsim's default of ~5. A degree sweep (holding
#    beta_edge = beta_ode/k fixed) showed that the ABM approaches the ODE as k grows
#    -- mass-action mixing is the k->infinity limit -- and that the improvement
#    saturates by ~k=20. So k=20 is the most faithful mean-field match worth paying
#    for; k=5 systematically under-transmits due to finite-degree network effects.
K_CONTACTS = 10
BETA_ODE   = default_pars.beta               # 9 / year (homogeneous-mixing beta)
#
#    BETA_SCALE is a *diagnostic* multiplier on the ABM's per-edge beta. With
#    BETA_SCALE = 1 the ABM's mean-field FOI equals the ODE's by construction, yet
#    the ABM still under-shoots the ODE's endemic level by ~2x even at high degree.
#    Setting BETA_SCALE = 2 tests whether that residual gap is simply a constant
#    factor in how beta is defined/used (a units/definition mismatch) -- in which
#    case doubling beta should line the two models up -- versus a deeper dynamical
#    difference, in which case the shapes will still disagree (e.g. the ABM keeps
#    drifting down while the ODE holds a stable endemic equilibrium).
BETA_SCALE = 1.12
BETA_EDGE  = BETA_SCALE * BETA_ODE / K_CONTACTS   # per-edge rate (per year)
#
# 2. DEMOGRAPHY.
#    The ODE holds N constant: it adds mu*N births/year into SUSCEPTIBLE, applies
#    background mortality mu = 1/70 per year to every compartment, and *recycles*
#    TB deaths back into SUSCEPTIBLE. Susceptible replenishment is what sustains
#    an endemic equilibrium, so the ABM must replenish too. We give the ABM equal
#    crude birth and death rates of mu = 1/70 per year (14.29 per 1000). This
#    keeps the population ~constant in the absence of TB. It differs from the ODE
#    only in that ABM TB deaths genuinely remove agents rather than being recycled
#    to susceptible; at endemic TB prevalence that is a <0.1%/year effect on N and
#    does not change the qualitative trend.
MU        = default_pars.mu                  # 1/70 per year
RATE_1000 = 1000 * MU                        # Starsim births/deaths use per-1000 units (rate_units=0.001)
#
# 3. INITIAL CONDITIONS.
#    The ODE seeds 1% of the population directly into SYMPTOMATIC active TB. We match
#    this exactly using the ABM's ``init_prev_active`` parameter (which seeds the
#    SYMPTOMATIC compartment) and set the latent ``init_prev`` to 0. Both models
#    therefore start from identical active-TB prevalence, so any divergence afterwards
#    is dynamics, not initial conditions.
INIT_PREV        = 0.0    # latent seed (INFECTION) -- off, to mirror the ODE
INIT_PREV_ACTIVE = 0.01   # active seed (SYMPTOMATIC) -- 1%, matching the ODE's yini
#
# 4. POPULATION SIZE / TIME STEP.
#    The ODE uses N = 1e5. The ABM uses 50,000 agents (results are compared as
#    population *fractions*, so absolute N only affects stochastic noise, not the
#    trend). We use a monthly step: small enough that the competing-risk transition
#    probabilities stay ~linear (matching the continuous ODE rates), large enough
#    that the full run stays fast.
N_AGENTS = 10_000
DT       = ss.months(1)
N_RUNS   = 10 # How many random seeds to run
#
# 5. TREATMENT.
#    Off in both models (ODE theta=0 by default; the ABM has no TxDelivery, so no
#    agent is ever diagnosed/treated). The reinfection machinery is equivalent:
#    the ODE splits CLEARED into CLEARED/RECOVERED/TREATED to carry pathway-specific
#    reinfection multipliers, while the ABM stores the same multipliers per agent
#    on entry to its single CLEARED state. With treatment off, the TREATED pathway
#    is unused in both.


def run_ode():
    """Run the compartmental ODE with default (natural-history) parameters."""
    ode = TB_ODE()  # default_pars: treatment off, beta=9, mu=1/70
    ode.run(start_time=START, end_time=STOP)
    return ode


def run_abm():
    """Run the agent-based model with parameters mapped from the ODE (see above)."""
    tb = tbsim.TB(pars=dict(
        beta             = ss.peryear(BETA_EDGE),
        init_prev        = ss.bernoulli(INIT_PREV),
        init_prev_active = ss.bernoulli(INIT_PREV_ACTIVE),
        trans_asymp      = default_pars.trans_asymp,
        # Natural-history rates below are already identical to the ODE defaults
        # (tbsim.TB and default_pars share the same numeric values), but we set
        # them explicitly so the mapping is self-documenting.
        inf_cle=ss.peryear(default_pars.inf_cle), inf_non=ss.peryear(default_pars.inf_non), inf_asy=ss.peryear(default_pars.inf_asy),
        non_rec=ss.peryear(default_pars.non_rec), non_asy=ss.peryear(default_pars.non_asy),
        asy_non=ss.peryear(default_pars.asy_non), asy_sym=ss.peryear(default_pars.asy_sym),
        sym_asy=ss.peryear(default_pars.sym_asy), sym_dead=ss.peryear(default_pars.sym_dead),
        rr_reinfection_rec=default_pars.rr_reinfection_rec,
        rr_reinfection_treat=default_pars.rr_reinfection_treat,
        rr_reinfection_cleared=default_pars.rr_reinfection_cleared,
    ))

    # Fixed-degree RandomNet so the mean contact number matches K_CONTACTS exactly.
    net = ss.RandomNet(pars=dict(n_contacts=ss.constant(K_CONTACTS), dur=0))

    # Balanced births/deaths at the ODE's background rate mu = 1/70 per year.
    births = ss.Births(pars=dict(birth_rate=RATE_1000))
    deaths = ss.Deaths(pars=dict(death_rate=RATE_1000))

    sim = ss.Sim(
        diseases     = tb,
        networks     = net,
        demographics = [births, deaths],
        start        = START,
        stop         = STOP,
        dt           = DT,
        n_agents     = N_AGENTS,
        rand_seed    = 1,
        verbose      = 0.02,
    )
    msim = ss.MultiSim(base_sim=sim, n_runs=N_RUNS)
    msim.run()
    msim.mean()  # populates msim.base_sim.results with the across-run mean
    return msim


def ode_fractions(ode):
    """Return ODE state-group fractions (divide compartments by the constant N)."""
    o = ode.results
    N = default_pars.N
    return sc.objdict(
        time        = o.time,
        susceptible = o.SUSCEPTIBLE / N,
        latent      = o.INFECTION / N,
        # ABM lumps all post-infection states into one CLEARED; sum the ODE's three sub-states to match.
        cleared     = (o.CLEARED + o.RECOVERED + o.TREATED) / N,
        active      = (o.NON_INFECTIOUS + o.ASYMPTOMATIC + o.SYMPTOMATIC) / N,
        infectious  = (o.ASYMPTOMATIC + o.SYMPTOMATIC) / N,
    )


def abm_fractions(sim):
    """Return ABM state-group fractions for a single sim (counts / living population each step)."""
    r = sim.results.tb
    nalive = sim.results.n_alive.astype(float)
    nalive[nalive == 0] = np.nan
    return sc.objdict(
        time        = np.asarray(sim.results.timevec, dtype=float),
        susceptible = r['n_SUSCEPTIBLE'] / nalive,
        latent      = r['n_INFECTION'] / nalive,
        cleared     = r['n_CLEARED'] / nalive,
        active      = (r['n_NON_INFECTIOUS'] + r['n_ASYMPTOMATIC'] + r['n_SYMPTOMATIC']) / nalive,
        infectious  = r['n_infectious'] / nalive,
    )


def plot(ode_frac, mean_frac, sim_fracs, filename='ode_compare.png'):
    """Overlay the ODE (thick black) and the ABM ensemble: each run faint, the mean thick solid."""
    panels = [
        ('susceptible', 'Susceptible'),
        ('latent',      'Latent infection'),
        ('cleared',     'Cleared / recovered'),
        ('active',      'Active TB (all)'),
        ('infectious',  'Infectious (asymp + symp)'),
    ]

    def draw_abm(ax, key):
        """Faint line per individual run, then the bold ensemble mean on top."""
        for f in sim_fracs:
            ax.plot(f.time, f[key], lw=1, alpha=0.2, color='C3')
        ax.plot(mean_frac.time, mean_frac[key], lw=2, ls='-', color='C3', label='ABM (tbsim, mean)')

    with sc.options.with_style('fancy'):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        axs = axs.flatten()
        for ax, (key, title) in zip(axs, panels):
            ax.plot(ode_frac.time, ode_frac[key], lw=2.5, color='k', label='ODE (compartmental)')
            draw_abm(ax, key)
            ax.set_title(title)
            ax.set_xlabel('Year')
            ax.set_ylabel('Fraction of population')
            ax.set_ylim(bottom=0)
            sc.boxoff(ax)
            ax.legend(frameon=False, fontsize=9)

        # Sixth panel: endemic-prevalence summary (the crux of the "dies out" claim).
        ax = axs[5]
        ax.plot(ode_frac.time, ode_frac.active, lw=2.5, color='k', label='ODE active')
        draw_abm(ax, 'active')
        ax.set_yscale('log')
        ax.set_title('Active TB prevalence (log scale)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Fraction (log)')
        sc.boxoff(ax)
        ax.legend(frameon=False, fontsize=9)

        fig.suptitle(f'TBsim ABM vs. compartmental ODE, {START}-{STOP}', fontweight='bold')
        sc.figlayout()
    fig.savefig(filename, dpi=120)
    print(f'Saved figure to {filename}')
    return fig


def summarize(ode_frac, abm_frac):
    """Print active-TB prevalence at the start, midpoint, and end for a quick verdict."""
    years = [START, (START + STOP) // 2, STOP]
    print('\n=== Active TB prevalence (fraction of population) ===')
    print(f'{"":18s}' + ''.join(f'{y:>10d}' for y in years))
    for name, f in [('ODE', ode_frac), ('ABM (mean)', abm_frac)]:
        def at(year):
            i = int(np.argmin(np.abs(f.time - year)))
            return f.active[i]
        print(f'{name:18s}' + ''.join(f'{at(y):>10.4f}' for y in years))
    endemic = abm_frac.active[-1]
    verdict = 'DIES OUT' if endemic < 1e-4 else 'endemic (does NOT die out)'
    print(f'\nABM final active prevalence = {endemic:.4g}  ->  {verdict}')
    return


if __name__ == '__main__':
    ode = run_ode()
    msim = run_abm()
    ode_frac = ode_fractions(ode)
    sim_fracs = [abm_fractions(s) for s in msim.sims]  # one per random-seed run
    mean_frac = abm_fractions(msim.base_sim)           # across-run mean (from msim.mean())
    summarize(ode_frac, mean_frac)
    plot(ode_frac, mean_frac, sim_fracs)
    plt.show()

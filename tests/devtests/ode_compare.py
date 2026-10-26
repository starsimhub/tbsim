"""
Compare the TB model (tbsim.TB) as an ODE and as agents on a contact network

The same declared model runs under both solvers. As an ODE, people mix homogeneously, with
`beta` contacts per person per year; as agents, they transmit over a random network of
`K_CONTACTS` contacts each, at `beta/K_CONTACTS` per contact, which gives the same force of
infection at low prevalence. The network is the only difference, so the agents approach the
ODE as `K_CONTACTS` grows (mass action is its limit), and fall below it on a sparse network.
Births replace every death in both (the model's `mu`), so the population stays the same size.

Run with::

    python ode_compare.py
"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
import tbsim

START, STOP = 1950, 2100
K_CONTACTS = 10 # The number of contacts each agent has
N_AGENTS = 10_000
N_RUNS = 10 # The number of runs as agents
init = dict(symptomatic=0.01) # As the reference ODE (lshtm_ode.R)
groups = dict(
    susceptible = ['susceptible'],
    latent      = ['infection'],
    cleared     = ['cleared', 'recovered', 'treated'],
    active      = ['non_infectious', 'asymptomatic', 'symptomatic'],
    infectious  = ['asymptomatic', 'symptomatic'],
)


def make(solver):
    kw = dict(start=START, stop=STOP, dt=ss.months(1), total_pop=N_AGENTS, verbose=0)
    if solver == 'ode':
        return ss.Sim(tbsim.TB(init=init), solver='ode', **kw)
    beta = tbsim.tb_pars().beta.raw.value/K_CONTACTS
    net = ss.RandomNet(n_contacts=ss.constant(K_CONTACTS))
    return ss.Sim(tbsim.TB(beta=beta, init=init), networks=net, n_agents=N_AGENTS, **kw)


def fractions(res, alive):
    """ The share of people in each group of states """
    return sc.objdict({key:sum(np.asarray(res[s]) for s in states)/alive for key,states in groups.items()})


if __name__ == '__main__':
    ode = make('ode').run()
    msim = make('abm').run(n_runs=N_RUNS)
    t = ode.t.yearvec
    ode_frac = fractions(ode.results.tb, ode.results.alive.values)
    abm_frac = fractions(msim.results.tb, msim.results.alive.values) # One row per run

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, key in zip(axs.flat, list(groups) + ['active']):
        ax.plot(t, abm_frac[key].T, lw=1, alpha=0.2, color='C3')
        ax.plot(t, abm_frac[key].mean(axis=0), lw=2, color='C3', label=f'Agents (mean of {N_RUNS})')
        ax.plot(t, ode_frac[key], lw=2.5, color='k', label='ODE')
        ax.set_title(key.capitalize() + (' (log scale)' if ax is axs.flat[-1] else ''))
        ax.set_xlabel('Year')
        ax.set_ylabel('Share of people')
        if ax is axs.flat[-1]:
            ax.set_yscale('log')
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f'tbsim.TB as an ODE and as agents ({K_CONTACTS} contacts each), {START}–{STOP}')
    sc.figlayout()
    fig.savefig('ode_compare.png', dpi=120)

    for name, f in [('ODE', ode_frac.active), ('Agents (mean)', abm_frac.active.mean(axis=0))]:
        print(f'{name:14s} active TB: ' + ', '.join(f'{f[i]:.4f}' for i in [0, len(f)//2, -1]) + f' (at {START}, {(START + STOP)//2}, {STOP})')

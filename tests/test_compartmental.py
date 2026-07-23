"""
Compare pure compartmental, compartmental Starsim, and agent-based Starsim LSHTM models,
and validate the multi-strain agent-based model against the two-strain reference ODE.
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim
import tbsim.compartmental as tbc

@sc.timer()
def test_ode(do_plot=False):
    """ Run and compare the ODE models """
    start_time = 1920
    end_time = 2020
    dt = 0.1

    # Run pure Python version
    tbr = tbc.TB_ODE()
    tbr.run(start_time=start_time, end_time=end_time) # No dt since exact solver
    if do_plot:
        tbr.plot()

    # Run Starsim version
    tbrss = tbc.TB_SS()
    sim = ss.Sim(modules=tbrss, start=start_time, stop=end_time, dt=dt, n_agents=1, copy_inputs=False)
    sim.run()
    if do_plot:
        tbrss.plot()

    return tbr, sim


# --------------------------------------------------------------------------- Two-strain ABM vs ODE
# The agent-based tbsim.TBResistant and the deterministic tbsim.compartmental.TwoStrainODE share the
# LSHTM natural-history rates exactly. Seeded from the same initial condition over a shared window, the
# ABM is a stochastic realization that should track the ODE. Three scenarios exercise the resistance
# dynamics: single-strain natural history, competitive exclusion of a less-fit resistant strain, and
# treatment-driven selection *for* resistance. (A CI-sized version of a fuller ABM↔ODE validation study.)
K = 10                    # ABM random-network mean degree
N_AGENTS = 8000
DT = ss.days(30)
START, YEARS = 2000, 60   # shared simulation window
MU = 1 / 70               # background mortality (per year), matches the ODE
BETA_ODE = 45.0
FIT_B = 0.85              # resistant-strain transmission fitness
INIT_LATENT = 0.05        # initial latent prevalence (rest susceptible)
INIT_B_FRAC = 0.20        # fraction of the initial latent seed that is strain B
TX = dict(eff_a=0.85, eff_b=0.5, q_treat=0.03, q_prog=5e-4, r_treat_asym=0.05)


def _run_ode(two_strain, r_treat_sym, years=YEARS, n=1e5):
    kw = dict(beta=BETA_ODE, mu=MU, rr_reinfection_inf=0, rr_reinfection_non=0, N=n)  # σ=0
    if two_strain:
        kw.update(fit_b=FIT_B, r_treat_sym=r_treat_sym, r_treat_asym=TX['r_treat_asym'],
                  eff_a=TX['eff_a'], eff_b=TX['eff_b'], q_treat=TX['q_treat'], q_prog=TX['q_prog'])
        seeds = dict(L_A=(1 - INIT_B_FRAC) * INIT_LATENT * n, L_B=INIT_B_FRAC * INIT_LATENT * n)
    else:
        kw.update(fit_b=1.0, r_treat_sym=0.0, r_treat_asym=0.0, q_treat=0.0, q_prog=0.0)
        seeds = dict(L_A=INIT_LATENT * n)
    ode = tbc.TwoStrainODE(**kw)
    df = ode.run(start_time=START, end_time=START + years, **seeds)
    return df


def _run_abm(two_strain, r_treat_sym, seed=0, years=YEARS):
    beta_edge = BETA_ODE / K   # per-edge β on a mean-degree-K network approximates the ODE's β/N mass action
    tb_kw = dict(beta=ss.peryear(beta_edge), init_prev=ss.bernoulli(INIT_LATENT),
                 rr_reinfection_inf=0.0, rr_reinfection_non=0.0)  # σ=0
    interventions = None
    if two_strain:
        tb = tbsim.TBResistant(rel_fitness={'TX': FIT_B},
                               pars=dict(tb_kw, init_strains=[1 - INIT_B_FRAC, INIT_B_FRAC],
                                         p_rand={'TX': TX['q_prog']}, prog_resist_mode='mixed'))
        interventions = tbsim.TxDeliveryR(
            product=tbsim.TxR(strains=tb.strains, base_efficacy=TX['eff_a'],
                              resist_penalty={'TX': TX['eff_b'] / TX['eff_a']},
                              adherence=1.0, q_acq={'TX': TX['q_treat']}),
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
    r = sim.results.tb
    return sc.objdict(prev_active=np.array(r['prevalence_active']),
                      frac_resist=np.array(r['frac_resist']))


@sc.timer()
def test_two_strain_abm_vs_ode(do_plot=False):
    """The multi-strain ABM reproduces the two-strain ODE across the three reference scenarios."""
    # Scenario 1: single strain, no treatment — endemic active prevalence should agree in magnitude.
    o1 = _run_ode(two_strain=False, r_treat_sym=0)
    a1 = _run_abm(two_strain=False, r_treat_sym=0)
    ode_prev, abm_prev = o1.prev_active.values[-1], a1.prev_active[-1]
    assert ode_prev > 0.001 and abm_prev > 0.001, 'both models should reach endemic active TB'
    assert 0.4 < abm_prev / ode_prev < 2.5, f'ABM prevalence {abm_prev:.4f} vs ODE {ode_prev:.4f}'

    # Scenario 2: two strains, no treatment — a less-fit resistant strain is out-competed (frac_resist falls).
    o2 = _run_ode(two_strain=True, r_treat_sym=0)
    a2 = _run_abm(two_strain=True, r_treat_sym=0)
    o2_fr, a2_fr = o2.frac_resist.values[-1], a2.frac_resist[-1]
    assert o2_fr < INIT_B_FRAC, 'ODE: resistance out-competed without treatment'
    assert a2_fr < INIT_B_FRAC, 'ABM: resistance out-competed without treatment'

    # Scenario 3: two strains + treatment — treatment selects *for* resistance, so the resistant fraction
    # is higher than the untreated case in both models (a paired comparison at the shared end-time).
    o3 = _run_ode(two_strain=True, r_treat_sym=1.0)
    a3 = _run_abm(two_strain=True, r_treat_sym=1.0)
    assert o3.frac_resist.values[-1] > o2_fr, 'ODE: treatment selects for resistance vs no treatment'
    assert a3.frac_resist[-1] > a2_fr, 'ABM: treatment selects for resistance vs no treatment'

    if do_plot:
        import matplotlib.pyplot as plt
        t_abm = START + np.arange(len(a2.frac_resist)) * (30 / 365.25)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(o1.time, o1.prev_active, 'k--', label='ODE')
        axes[0].plot(START + np.arange(len(a1.prev_active)) * (30 / 365.25), a1.prev_active, label='ABM')
        axes[0].set_title('Single strain: active prevalence'); axes[0].legend(frameon=False)
        axes[1].plot(o2.time, o2.frac_resist, 'k--', label='ODE (no tx)')
        axes[1].plot(t_abm, a2.frac_resist, label='ABM (no tx)')
        axes[1].plot(o3.time, o3.frac_resist, 'r--', label='ODE (+tx)')
        axes[1].plot(START + np.arange(len(a3.frac_resist)) * (30 / 365.25), a3.frac_resist, 'r', label='ABM (+tx)')
        axes[1].axhline(INIT_B_FRAC, color='0.7', ls=':'); axes[1].set_title('Resistant fraction of active TB')
        axes[1].legend(frameon=False)
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    tbr, sim = test_ode(do_plot=True)
    test_two_strain_abm_vs_ode(do_plot=True)

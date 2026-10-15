"""
TB interventions example: scenarios with BCG, preventive therapy (TPT), a change in transmission, and the care cascade

Each scenario runs as agents, with people mixing homogeneously; the care cascade is also
shown step by step (screening, confirmation, and treatment).
"""
import sciris as sc
import starsim as ss
import tbsim

# Default parameters
sim_pars = dict(start='1975-01-01', stop='2030-12-31', dt=ss.months(1), solver='abm', n_agents=5_000, rand_seed=1, verbose=0)
tb_pars = dict(init=dict(infection=0.25, symptomatic=0.005), **tbsim.regimen('dots'))


def make_sim(scenario='Baseline'):
    """
    Make a TB simulation for one scenario

    Args:
        scenario (str): 'Baseline', 'BCG', 'TPT', 'Beta change', or 'Care cascade'
    """
    ppl = ss.People(n_agents=sim_pars['n_agents'])
    tb = tbsim.TB(**tb_pars).stratify(age=[0, 15]) # By age, for the tests' sensitivity in children and adults
    if scenario == 'Baseline':
        intv = None
    elif scenario == 'BCG': # Each child under 5 is offered BCG once
        intv = ss.vaccinate(tbsim.BCG(), eligible=ppl.age < 5, coverage=0.8, start=1980, stop=2020, name='bcg_vx')
    elif scenario == 'TPT': # Yearly rounds of TPT for adults with a latent infection
        intv = ss.treat(tbsim.TPT(), eligible=tb.infection & (ppl.age >= 15), coverage=0.7, interval=ss.years(1), start=1990, stop=2025, name='tpt_tx')
    elif scenario == 'Beta change': # Transmission halves in 1990, then is 1.4 times the original from 2000
        intv = ss.change(tb.pars.beta, at=[1990, 2000], by=[0.5, 1.4], name='beta_change')
    elif scenario == 'Care cascade': # People with symptomatic TB seek care, are tested with Xpert, and start treatment
        intv = tbsim.care(tb, seeking=ss.peryear(1), every=ss.months(1), start=1990)
    else:
        errormsg = f'Unknown scenario "{scenario}"'
        raise ValueError(errormsg)
    return tbsim.Sim(people=ppl, tb=tb, interventions=intv, label=scenario, **sim_pars)


def run_scenarios(scenarios=('Baseline', 'BCG', 'TPT', 'Beta change', 'Care cascade'), plot=True):
    """ Run the scenarios one after another, and plot them together """
    msim = ss.MultiSim([make_sim(scenario) for scenario in scenarios])
    msim.run(parallel=False)
    for sim in msim.sims:
        res = sim.results.tb
        print(f'{sim.label:13s}: {res.cum_tb_death[-1]:6,.0f} TB deaths, active TB prevalence in 2030 {res.prevalence_active[-1]:.4f}')
    if plot:
        msim.plot(key=['tb_infectious', 'tb_prevalence_active', 'tb_new_infections', 'tb_cum_tb_death'])
    return msim


def run_cascade():
    """
    The care cascade, step by step: people with symptomatic TB seek care and are screened with CAD,
    those who screen positive are confirmed with Xpert, and those confirmed start DOTS
    """
    tb = tbsim.TB(**tb_pars).stratify(age=[0, 15])
    start_tx = ss.treat(ss.Treatment(efficacy=1.0, transition='diagnosis'), coverage=0.9, name='start_tx') # Starts treatment: the model's diagnosis transition
    confirm = ss.screen(tbsim.Xpert(), coverage=0.8, on_positive=start_tx, name='confirm')
    screen = ss.screen(tbsim.CAD(), eligible=tb.symptomatic, coverage=0.15, interval=ss.months(1), start=1990, on_positive=confirm, name='screen')
    sim = tbsim.Sim(tb=tb, interventions=screen, **sim_pars)
    sim.run()

    r = sim.results
    print('\nCare cascade:')
    print(f'  Screened:  {r.screen.delivered.sum():,.0f}')
    print(f'  Confirmed: {r.confirm.diagnosed.sum():,.0f} of {r.confirm.delivered.sum():,.0f} tested')
    print(f'  Treated:   {r.start_tx.delivered.sum():,.0f}')
    print(f'  Completed: {r.tb.completion.sum():,.0f}')
    print(f'  Failed:    {r.tb.failure.sum():,.0f}')
    return sim


if __name__ == '__main__':
    T = sc.timer()
    msim = run_scenarios()
    sim = run_cascade()
    T.toc()

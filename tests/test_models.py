"""
Test the TB models declared as states and transitions (tbsim.TB and tbsim.LSHTMResistant), which run as an ODE or as agents
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim
import tbsim.compartmental as tbc

kw = dict(unit='year', total_pop=1e5, verbose=0)


@sc.timer()
def test_lshtm_ode():
    """ As an ODE, the declared model is the reference ODE (lshtm_ode.R, as ported to Python) """
    ref = tbc.TB_ODE().run(start_time=1920, end_time=2020)
    sim = ss.Sim(tbsim.TB(init=dict(symptomatic=1000)), start=1920, stop=2020, **kw).run()
    res = sim.results.tb
    for key in ['SUSCEPTIBLE', 'INFECTION', 'CLEARED', 'RECOVERED', 'NON_INFECTIOUS', 'ASYMPTOMATIC', 'SYMPTOMATIC']:
        model = res[f'n_{key.lower()}'].values # Both at the start of each year
        diff = np.abs(model - ref[key]).max()/ref[key].max()
        print(f'{key:15s} differs from the reference by {diff:0.1e}')
        assert diff < 1e-4
    return sim


@sc.timer()
def test_resistant_ode():
    """ As an ODE, the model with resistance is the reference ODE (two_strain_ode.R, as ported to Python), in every compartment and for each strain """
    names = dict(SUS='susceptible', CLE='cleared', REC='recovered', TRD='treated', L='infection', N='non_infectious', AS='asymptomatic', SY='symptomatic', TA='treat_asy', TY='treat_sym')
    level = dict(A=0, B=1, AB=2) # The levels of the resistance dimension: DS, R, DS+R
    flags = dict(transmission_independent=True, prog_resist_mix=False, prog_select_fitness=True, p_multi=0.5)
    for refpars,pars in [(dict(), dict()), (dict(rr_reinfection_asy=0.5, **flags), dict(rr_superinfection_asy=0.5, **flags))]:
        ref = tbc.TwoStrainODE(**refpars).run(start_time=1920, end_time=2020)
        sim = ss.Sim(tbsim.LSHTMResistant(**pars), start=1920, stop=2020, **kw).run()
        res = sim.results.tb
        diffs = []
        for key in tbc.two_strain_ode.STATES[:-1]: # Not the deaths, which are an accumulator
            prefix, _, strain = key.partition('_')
            result = res[f'n_{names[prefix]}']
            model = result.by('resistance').values[:, level[strain]] if strain else result.values
            diffs.append(np.abs(model - ref[key].values).max()/max(ref[key].max(), 1)) # Both at the start of each year
        print(f'{pars}: largest difference from the reference {max(diffs):0.1e}')
        assert max(diffs) < 1e-3
    return sim


@sc.timer()
def test_agents():
    """ The same models, as agents (one agent per person), are close to the ODE (within the difference of order dt that agents have) """
    pars = dict(start=2000, stop=2010, dt=ss.weeks(1), **kw)
    for Model in [tbsim.TB, tbsim.LSHTMResistant]:
        init = dict(infection=0.03, symptomatic=0.005) if Model is tbsim.TB else None
        ode = ss.Sim(Model(init=init), **pars).run().results.tb.active.values
        abm = ss.Sim(Model(init=init), solver='abm', n_agents=100_000, **pars).run().results.tb.active.values
        ratio = abm[-104:].mean()/ode[-104:].mean() # Over the last two years
        print(f'{Model.__name__}: active TB as agents is {ratio:0.2f} times that of the ODE')
        assert 0.8 < ratio < 1.25
    return abm


@sc.timer()
def test_network():
    """ As agents, the model runs over a network, with births and deaths from demographics modules (mu=0) """
    tb = tbsim.TB(beta=10, mu=0, init=dict(infection=0.1, symptomatic=0.01))
    sim = tbsim.Sim(tb=tb, networks=ss.RandomNet(n_contacts=5), demographics=[ss.Births(birth_rate=20), ss.Deaths(death_rate=15)], n_agents=2000, verbose=0)
    sim.run()
    assert sim.solver == 'abm'
    assert sim.results.tb.transmission.values.sum() > 0
    assert sim.results.tb.active.values[-1] > 0
    return sim


if __name__ == '__main__':
    test_lshtm_ode()
    test_resistant_ode()
    test_agents()
    test_network()
